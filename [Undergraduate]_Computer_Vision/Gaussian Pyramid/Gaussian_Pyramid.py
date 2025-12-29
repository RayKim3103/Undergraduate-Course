import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ---------- utils ----------
def to_float01(img):
    img = img.astype(np.float32)
    if img.max() > 1.0: img /= 255.0
    return np.clip(img, 0.0, 1.0)

def gaussian_kernel_1d():
    """
    2D Gaussian kernel can be separated into two 1D kernels.
    For example, a 3x3 Gaussian kernel with sigma=1 is:
    [ [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1] ] / 16
    This can be separated into two 1D kernels:
    [1, 2, 1] / 4  (horizontal)
    ([1, 2, 1] / 4)^T  (vertical)
    [1, 2, 1] / 4 * [[0, 0, 0], [0, 1, 0], [0, 0, 0]] 
    = [ [0, 0, 0], 
        [1, 2, 1], 
        [0, 0, 0] ] / 4
    ([1, 2, 1] / 4)^T * [[0, 0, 0], [1, 2, 1], [0, 0, 0]] / 4 
    = [ [1, 2, 1], 
        [2, 4, 2], 
        [1, 2, 1] ] / 16
    [ [1, 2, 1],             [ [0, 0, 0],      [ [1, 2, 1],
      [2, 4, 2],        *      [0, 1, 0],   =    [2, 4, 2],
      [1, 2, 1] ] / 16         [0, 0, 0] ]       [1, 2, 1] ] / 16  
    ( All convolition is calculated with zero padding)
    """
    ######
    #Fill#
    # 1D Gaussian kernel with size 1x3 and sigma=1
    k = np.array([1, 2, 1], dtype=np.float32) / 4.0
    ######
    return k

def sep_conv(img, k1d):
    img = img.astype(np.float32)
    ######
    #Fill#
    # radius of the kernel, in 1x3 kernel, 3//2=1
    r = len(k1d) // 2
    ######
    P = np.pad(img, ((0,0),(r,r)), mode='reflect')
    tmp = np.zeros_like(img, dtype=np.float32)
    for dx in range(-r, r+1):
        ######
        #Fill#
        # By applying horizontal & vertical 1D convolution we can get 2D convolution
        # I explained this in the Report mathematically
        # horizontal 1D convolution, pad by 1 to have same size output
        tmp += k1d[dx + r] * P[:, r+dx:P.shape[1]-(r-dx)]
        ######
    P2 = np.pad(tmp, ((r,r),(0,0)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(-r, r+1):
        ######
        #Fill#
        # vertical 1D convolution, pad by 1 to have same size output
        out += k1d[dy + r] * P2[r+dy:P2.shape[0]-(r-dy), :]
        ######
    return out

def downsample2(img, k1d):
    return sep_conv(img, k1d)[::2, ::2] # downsample by 2 (each Height and Width)

def upsample2(img, k1d):
    H, W = img.shape
    ######
    #Fill#
    # scale by 2 and insert zeros in the new samples, it is the starting of interpolation
    # we need to make Xp(x, y) to do interpolation, I have explained this in the Report using 1D signal
    # it is very similar to 1D case, just expanding dimensions
    up = np.zeros((H*2, W*2), dtype=np.float32)
    up[::2, ::2] = img
    up = sep_conv(up, k1d) * 4.0 # multiply by 4 to compensate for the zeros inserted (Interpolation needs gain)
    # this convolution above act as Reconstruction filter
    ######
    return up

# ---------- pyramids ----------
def build_gaussian_pyr(img, levels=4):
    k1d = gaussian_kernel_1d()
    G = [to_float01(img)]
    for _ in range(levels):
        ######
        #Fill#
        # iteratively downsample the image, iterate amount = levels
        # make a list of images which is Gaussian Pyramids
        G.append(downsample2(G[-1], k1d))
        ######
    return G

def build_laplacian_pyr(G):
    k1d = gaussian_kernel_1d()
    Ls = []
    for i in range(len(G)-1):
        ######
        #Fill#
        # upsample G[i+1] and subtract it from G[i]
        # it extracts the high-frequency components
        up = upsample2(G[i+1], k1d)
        target_shape = G[i].shape
        # to match the shape with G[i] (ex. G[3] = 240x135, G[4] = 120x68 -> up = 240x136)
        up = up[:target_shape[0], :target_shape[1]] 
        L = G[i] - up
        # make a list of Laplacian images which is Laplacian Pyramids
        Ls.append(L)
        ######
    return Ls, G[-1]

def reconstruct_from_lap(Ls, G_top):
    k1d = gaussian_kernel_1d()
    current = G_top
    for i in reversed(range(len(Ls))):
        ######
        #Fill#
        # upsample current and add Ls[i] to reconstruct image
        current = upsample2(current, k1d)
        target_shape = Ls[i].shape
        # to match the shape with Ls[i] (ex. G[3] = 240x135, G[4] = 120x68 -> up = 240x136)
        current = current[:target_shape[0], :target_shape[1]]
        # restore high-frequency components of upsampled gaussian pyramid image
        current = current + Ls[i]
        ######
    return np.clip(current, 0.0, 1.0)

# ---------- filtering via Laplacian pyramid ----------
def lap_pyr_filter(img, gains, levels=4):
    G = build_gaussian_pyr(img, levels)
    Ls, Gtop = build_laplacian_pyr(G)
    # pad/trim gains to len(Ls)

    ######
    #Fill#
    # if gains is shorter than Ls, pad with 1.0
    # if longer, trim it
    # apply gains to each level of Laplacian pyramid, it scales the high-frequency components
    # by scaling high-frequency components, it can reduce noise
    gains = gains + [1.0] * (len(Ls) - len(gains)) if len(gains) < len(Ls) else gains[:len(Ls)]
    Ls_mod = [L * g for L, g in zip(Ls, gains)]
    ######
    out = reconstruct_from_lap(Ls_mod, Gtop)
    return out

###########################DO NOT TOUCH##############################
def add_gaussian_noise(img, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + n, 0.0, 1.0)
    return noisy, sigma

def add_salt_pepper(img, p=0.02, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32).copy()
    U = rng.random(img.shape)
    noisy[U < (p/2)] = 0.0
    noisy[U > 1 - (p/2)] = 1.0
    return noisy

def psnr(x, y):
    x = x.astype(np.float32); y = y.astype(np.float32)
    mse = np.mean((x - y)**2, dtype=np.float32)
    return 99.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)

def load_gray(path):
    img = Image.open(path).convert('RGB') 
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return np.clip(gray.astype(np.float32), 0.0, 1.0)

def show_side_by_side(imgs, titles=None):
    n = len(imgs)
    plt.figure(figsize=(4*n, 4))
    for i, im in enumerate(imgs):
        ax = plt.subplot(1, n, i+1)
        ax.imshow(im, cmap='gray', vmin=0, vmax=1) 
        if titles:
            ax.set_title(titles[i], fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_pyramids(G, Ls, G_top=None, recon=None):
    nG = len(G)
    plt.figure(figsize=(4*nG, 4))
    for i, g in enumerate(G):
        ax = plt.subplot(1, nG, i+1)
        ax.imshow(g, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Gaussian L{i}  ({g.shape[1]}x{g.shape[0]})', fontsize=10)
        ax.axis('off')
    plt.suptitle('Gaussian Pyramid', fontsize=14)
    plt.tight_layout()
    plt.show()

    nL = len(Ls)
    if nL > 0:
        plt.figure(figsize=(4*nL, 4))
        for i, L in enumerate(Ls):
            ax = plt.subplot(1, nL, i+1)
            a = np.max(np.abs(L)) + 1e-8
            ax.imshow(L, cmap='gray', vmin=-a, vmax=+a)
            ax.set_title(f'Laplacian L{i}  (±{a:.3f})', fontsize=10)
            ax.axis('off')
        plt.suptitle('Laplacian Pyramid (zero-centered)', fontsize=14)
        plt.tight_layout()
        plt.show()

#####################################################################

# Choose one of the following settings:
##########################################################
# Setting1
levels   = 5
gains    = [0.35, 0.50, 0.75, 0.95, 1.00]
# Setting2
# levels   = 4
# gains    = [0.60, 0.75, 0.90, 1.00]
##########################################################
HERE = Path(__file__).resolve().parent
img_path = HERE / "test_image.jpg"
if not img_path.exists():
    raise FileNotFoundError(f"Could not find: {img_path}")
img_gray = load_gray(str(img_path))


G = build_gaussian_pyr(img_gray, levels)
Ls, Gtop = build_laplacian_pyr(G)
rec = reconstruct_from_lap(Ls, Gtop)
mse = np.mean((rec - to_float01(img_gray))**2)
print("MSE:", mse)  # expect ≤ 1e-4


# Make some changes to sigma
noisy, sigma = add_gaussian_noise(img_gray, sigma=0.1, seed=42)
denoised = lap_pyr_filter(noisy, gains=gains, levels=levels)

visualize_pyramids(G, Ls, G_top=Gtop, recon=rec)

# pyr_to_grid(G, "gaussian")
# pyr_to_grid(Ls, "laplacian")

print("PSNR(noisy):   ", psnr(img_gray, noisy))
print("PSNR(denoised):", psnr(img_gray, denoised))

show_side_by_side(
    [noisy, denoised],
    titles=[f"Noisy (σ={sigma}, PSNR={psnr(img_gray,noisy):.2f} dB)",
            f"Denoised (PSNR={psnr(img_gray,denoised):.2f} dB)"]
)
