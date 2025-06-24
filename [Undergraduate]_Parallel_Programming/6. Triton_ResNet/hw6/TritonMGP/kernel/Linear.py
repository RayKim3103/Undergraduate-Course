import triton
import triton.language as tl
from mgp import empty

@triton.jit
def _linear_tiled_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    has_bias: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row block
    pid_n = tl.program_id(1)  # col block

    # Compute ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to blocks
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled dot-product over K dimension
    for k in range(0, K, BLOCK_K):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mask_w = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

        x_block = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(x_block, w_block)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if has_bias:
        b_ptrs = b_ptr + offs_n[None, :]
        bias = tl.load(b_ptrs, mask=(offs_n[None, :] < N), other=0.0)
        acc += bias

    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, tl.cast(acc, tl.float16), mask=mask_out)


def triton_linear(x, weight_T, bias=None):
    M, K = x.shape
    _, N = weight_T.shape

    out = empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    def grid(meta):
        return (
            (M + BLOCK_M - 1) // BLOCK_M,
            (N + BLOCK_N - 1) // BLOCK_N,
        )

    has_bias = 1 if bias is not None else 0

    _linear_tiled_kernel[grid](
        x, weight_T, bias if has_bias else x, out,
        M, K, N,
        has_bias,
        stride_xm=K, stride_xk=1,
        stride_wk=N, stride_wn=1,
        stride_om=N, stride_on=1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out
