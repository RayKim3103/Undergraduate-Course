import triton
import triton.language as tl
from mgp import empty

@triton.jit
def _maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    kH, kW, sH, sW, pH, pW,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    WB = tl.cdiv(W_out, BLOCK_SIZE)
    line_index = pid // WB
    block_index = pid % WB

    per_channel_row = H_out
    nc = line_index // per_channel_row
    ho = line_index % per_channel_row
    n = nc // C
    c = nc % C

    w0 = block_index * BLOCK_SIZE
    offs_wo = w0 + tl.arange(0, BLOCK_SIZE)
    mask_wo = offs_wo < W_out

    # Calculate starting h and w in the padded input
    h0 = ho * sH - pH
    w0 = offs_wo * sW - pW

    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)

    for kh in range(kH):
        hi = h0 + kh
        valid_hi = (0 <= hi) & (hi < H)

        for kw in range(kW):
            wi = w0 + kw
            valid_wi = (0 <= wi) & (wi < W)
            mask = valid_hi & valid_wi & mask_wo

            base_offset = ((n * C + c) * H + hi) * W + wi
            x_vals = tl.load(x_ptr + base_offset, mask=mask, other=-float("inf"))
            max_val = tl.where(x_vals > max_val, x_vals, max_val)

    out_offset = ((n * C + c) * H_out + ho) * W_out + offs_wo
    tl.store(out_ptr + out_offset, max_val, mask=mask_wo)


def triton_maxpool2d(x, kernel_size, stride, padding):
    N, C, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    out = empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 256
    WB = (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs = N * C * H_out * WB

    def grid(meta):
        return (num_programs,)

    _maxpool2d_kernel[grid](
        x, out,
        N, C, H, W,
        H_out, W_out,
        kH, kW, sH, sW, pH, pW,
        BLOCK_SIZE
    )

    return out

