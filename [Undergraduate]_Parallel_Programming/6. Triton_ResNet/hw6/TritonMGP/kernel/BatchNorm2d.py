import triton
import triton.language as tl
from mgp import empty

@triton.jit
def _bn2d_inference_kernel(
    x_ptr, out_ptr,
    rm_ptr, rv_ptr, w_ptr, b_ptr,
    N, C, H, W, eps, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    # Total spatial elements per channel
    T = H * W
    # Number of blocks along the spatial dimension
    NB = tl.cdiv(T, BLOCK_SIZE)

    # Decode pid into (nc_index, tile_index)
    nc_index = pid // NB          # flatten index for (n, c)
    tile_index = pid % NB         # which BLOCK of spatial elements

    # Recover batch index n and channel index c
    n = nc_index // C
    c = nc_index % C

    # Compute the starting spatial index t0 for this tile
    t0 = tile_index * BLOCK_SIZE
    offs_t = t0 + tl.arange(0, BLOCK_SIZE)        # vector of length BLOCK_SIZE
    mask_t = offs_t < T                           # mask for valid spatial positions

    # Compute flattened input/output offsets for these spatial positions
    base = nc_index * T                           # base offset = (n*C + c)*T
    offs = base + offs_t                          # vector of input/output offsets

    # Load x values (vector) with masking
    x_vals = tl.load(x_ptr + offs, mask=mask_t, other=0.0)  # float32

    # Load per-channel parameters (scalar)
    rm = tl.load(rm_ptr + c)        
    rv = tl.load(rv_ptr + c)        
    wv = tl.load(w_ptr + c)        
    bv = tl.load(b_ptr + c)       

    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(rv + eps)

    # Normalize input values
    norm = (x_vals - rm) * inv_std

    # Scale and shift
    y_vals = norm * wv + bv

    # Store results into output, masking out-of-bounds lanes
    tl.store(out_ptr + offs, y_vals, mask=mask_t)


def triton_bn2d(input, weight, bias, running_mean, running_var, momentum, eps):

    N, C, H, W = input.shape

    # Allocate output tensor
    out = empty((N, C, H, W), device=input.device, dtype=input.dtype)

    # Total spatial elements per channel
    T = H * W
    # Choose block size for spatial dimension (compile-time constant)
    BLOCK_SIZE = 256
    # Number of blocks per (n, c)
    NB = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Total number of Triton programs = (N * C) * NB
    num_programs = N * C * NB

    # Launch Triton kernel
    def grid(meta):
        return (num_programs,)

    _bn2d_inference_kernel[grid](
        input, out,
        running_mean, running_var, weight, bias,
        N, C, H, W, eps, BLOCK_SIZE
    )
    return out
