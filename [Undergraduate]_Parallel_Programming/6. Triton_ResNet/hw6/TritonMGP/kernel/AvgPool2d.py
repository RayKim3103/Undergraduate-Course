import triton
import triton.language as tl
from mgp import empty

@triton.jit
def _avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    kH, kW, sH, sW,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    # Number of width-blocks per row: WB = ceil(W_out / BLOCK_SIZE)
    WB = tl.cdiv(W_out, BLOCK_SIZE)

    # pid into -> line_index, block_index
    line_index = pid // WB          # which (n, c, ho) combination
    block_index = pid % WB          # which horizontal block within that row

    # Recover (n, c, ho)
    per_channel_row = H_out
    nc = line_index // per_channel_row
    ho = line_index % per_channel_row
    n = nc // C
    c = nc % C

    # Compute the starting wo for this block
    w0 = block_index * BLOCK_SIZE
    offs_wo = w0 + tl.arange(0, BLOCK_SIZE)   # [w0, w0+1, …]
    mask_wo = offs_wo < W_out

    # Compute the starting input height for output ho (no padding)
    h0 = ho * sH

    # Initialize accumulator vector (float32)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over the kH × kW pooling window
    for kh in range(kH):
        hi = h0 + kh
        valid_hi = (0 <= hi) & (hi < H)

        for kw in range(kW):
            wi = offs_wo * sW + kw            
            valid_wi = (0 <= wi) & (wi < W)   
            mask_hw = valid_hi & valid_wi & mask_wo

            # Compute flattened input offsets
            base_hi = ((n * C + c) * H + hi) * W
            offs_x = base_hi + wi               

            # Load x values, with masking
            x_vals = tl.load(x_ptr + offs_x, mask=mask_hw, other=0.0)
            sum_val += tl.cast(x_vals, tl.float32)

    # Compute average
    denom = kH * kW
    avg_vals = sum_val / denom

    # Store results
    out_base = ((n * C + c) * H_out + ho) * W_out    
    offs_out = out_base + offs_wo                    
    tl.store(out_ptr + offs_out, avg_vals, mask=mask_wo)


def triton_avgpool2d(x, pool_size, stride):
    N, C, H, W = x.shape
    kH, kW = pool_size
    sH, sW = stride

    # Compute output dimensions
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    # Allocate output tensor
    out = empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Choose block size for width
    BLOCK_SIZE = 256
    WB = (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE  # ceil(W_out / BLOCK_SIZE)
    num_programs = N * C * H_out * WB

    # Define Triton grid
    def grid(meta):
        return (num_programs,)

    # Launch the avgpool kernel
    _avgpool2d_kernel[grid](
        x, out,
        N, C, H, W,
        H_out, W_out,
        kH, kW, sH, sW,
        BLOCK_SIZE
    )
    return out

