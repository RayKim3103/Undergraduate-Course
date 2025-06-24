import triton
import triton.language as tl
from mgp import empty

@triton.jit
def _im2col_kernel(
    x_ptr, col_ptr,
    N, C, H, W,
    H_out, W_out,
    kH, kW, sH, sW, pH, pW, dH, dW,
    K,
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // (H_out * W_out)
    idx = pid % (H_out * W_out)
    ho = idx // W_out
    wo = idx % W_out
    row = pid

    offs = tl.arange(0, BLOCK_K)
    for i in range(0, K, BLOCK_K):
        k_idx = i + offs
        ci = k_idx // (kH * kW)
        rem = k_idx % (kH * kW)
        kh = rem // kW
        kw = rem % kW
        hi = ho * sH - pH + kh * dH
        wi = wo * sW - pW + kw * dW
        cond_hi = (0 <= hi) & (hi < H)
        cond_wi = (0 <= wi) & (wi < W)
        is_valid = cond_hi & cond_wi
        offset = ((n * C + ci) * H + hi) * W + wi
        x_vals = tl.load(x_ptr + offset, mask=is_valid, other=0.0)
        offset_col = row * K + k_idx
        tl.store(col_ptr + offset_col, x_vals, mask=k_idx < K)


@triton.jit
def _transpose_kernel(
    weight_ptr, trans_ptr,
    C_out, C_in, kH, kW, K,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # (K,)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # (C_out,)

    kk = offs_k[:, None]  # (BLOCK_K, 1)
    co = offs_c[None, :]  # (1, BLOCK_C)

    # Compute ci, kh, kw for each K
    ci = kk // (kH * kW)
    rem = kk % (kH * kW)
    kh = rem // kW
    kw = rem % kW

    # Compute the input weight offset
    weight_offset = ((co * C_in + ci) * kH + kh) * kW + kw  # shape (BLOCK_K, BLOCK_C)

    # Bounds check
    mask_k = kk < K
    mask_c = co < C_out
    mask = mask_k & mask_c

    vals = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)

    # Store at transposed location: (K, C_out)
    trans_offset = kk * C_out + co
    tl.store(trans_ptr + trans_offset, vals, mask=mask)


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((offs_k[None, :] + k) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((offs_k[:, None] + k) < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def reshape_output_kernel(
    input_ptr, output_ptr,
    N, C_out, H_out, W_out,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_out * H_out * W_out

    mask = offs < total

    n = offs // (C_out * H_out * W_out)
    rem1 = offs % (C_out * H_out * W_out)
    c = rem1 // (H_out * W_out)
    rem2 = rem1 % (H_out * W_out)
    h = rem2 // W_out
    w = rem2 % W_out

    m = n * H_out * W_out + h * W_out + w
    input_offset = m * C_out + c
    output_offset = offs

    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, val, mask=mask)


def cdiv(a, b):
    return (a + b - 1) // b


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    K = C_in * kH * kW
    M = N * H_out * W_out
    N_ = C_out

    x_col = empty((M, K), device=x.device, dtype=x.dtype)

    _im2col_kernel[(M,)](
        x, x_col,
        N, C_in, H, W,
        H_out, W_out,
        kH, kW, sH, sW, pH, pW, dH, dW,
        K,
        BLOCK_K=64
    )

    weight_trans = empty((K, C_out), device=x.device, dtype=weight.dtype)

    _transpose_kernel[
        (cdiv(K, 64), cdiv(C_out, 64))
    ](
        weight, weight_trans,
        C_out, C_in, kH, kW, K,
        BLOCK_K=64, BLOCK_C=64
    )

    out = empty((M, N_), device=x.device, dtype=x.dtype)

    _matmul_kernel[
        (cdiv(M, 64), cdiv(N_, 64))
    ](
        x_col, weight_trans, out,
        M, N_, K,
        stride_am=K, stride_ak=1,
        stride_bk=C_out, stride_bn=1,
        stride_cm=N_, stride_cn=1,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
    )

    final_out = empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    total = N * C_out * H_out * W_out

    reshape_output_kernel[
        (cdiv(total, 128),)
    ](
        out, final_out,
        N, C_out, H_out, W_out,
        BLOCK_SIZE=128
    )

    return final_out
