import os, random, time
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# ---------------------- Config ---------------------- #
@dataclass
class Cfg:
    data_dir: str = None  
    epochs: int = 10
    batch_size: int = 128
    val_size: int = 5000
    seed: int = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_data_root(dirname="data") -> str:
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    dr = (base / dirname).expanduser()
    dr.mkdir(parents=True, exist_ok=True)
    return str(dr)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------- Data: transforms to implement ---------------------- #
class ToTensorOnly:
    def __call__(self, img):
        return TF.to_tensor(img)

def compute_mean_std(dataset, batch_size=256):
    """
    TODO: 
      Implement via running sums(sum, sum_sq) on a DataLoader(dataset, batch_size).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    n_pixels = 0
    sum = None
    sum_sq = None
    for imgs, _ in loader:
        b, c, h, w = imgs.shape
        if sum is None:
            sum = torch.zeros(c, dtype=torch.float64)       # shape: (C,)
            sum_sq = torch.zeros(c, dtype=torch.float64)    # shape: (C,)
        imgs64 = imgs.to(dtype=torch.float64)
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # imgs64 is PyTorch tensor of shape (B, C, H, W)
        # PyTorch have sum & sqrt function as a method
        #=============================
        sum += torch.sum(imgs64, dim=[0, 2, 3])             # sum of pixel values per channel (shape: [B C H W] -> [C])
        sum_sq += torch.sum(imgs64 ** 2, dim=[0, 2, 3])     # sum of squared pixel values per channel (shape: [B C H W] -> [C])
        n_pixels += b * h * w                               # total number of pixels

    mean = sum / n_pixels
    std = torch.sqrt(sum_sq / n_pixels - mean ** 2)
    
    return mean, std

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: torch.Tensor):
        """
        TODO: 
          With prob p, flip horizontally (reverse width dim) * Use random.random() *
        """
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # use random lib -> random.random()
        # PyTorch have flip function as a method
        #=============================
        if random.random() <= self.p:
            return torch.flip(x, dims=[2])  # flipping width dim ([C H W] -> width is dim of index 2)
        return x

class RandomCrop:
    def __init__(self, size=32, padding=4):
        self.size = size
        self.padding = padding

    def __call__(self, x: torch.Tensor):
        """
        TODO:
          1) zero-pad by 'padding' on all sides using F.pad
          2) take random size x crop
        """
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # use random lib -> random.randint()
        # torch.nn.functional have pad function as a method
        #=============================
        
        # add padding (zero-pad by 'padding' on all sides -> left, top, right, bottom)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        
        # take random crop
        _, h, w = x.shape
        top = random.randint(0, h - self.size)  # crop (h-self.size) ~ (h)
        left = random.randint(0, w - self.size) # crop (w-self.size) ~ (w)
        
        return x[:, top:top+self.size, left:left+self.size]

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean).view(-1,1,1)  # shape mean as [C, 1, 1]
        self.std = torch.as_tensor(std).view(-1,1,1)    # shape std as [C, 1, 1]

    def __call__(self, x: torch.Tensor):
        """
        TODO: 
          Return (x - mean) / std with broadcasting. * Watch device and dtype *
        """
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # x has shape of [C H W]
        # mean has shape of [C 1 1]
        # std has shape of [C 1 1]
        #=============================
        if x is None:
            # exception handling: create a dummy tensor (32x32 RGB)
            x = torch.zeros(3, 32, 32, dtype=torch.float32)

        # match device and dtype for safety
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std  = self.std.to(device=x.device, dtype=x.dtype)
        
        # x has shape of 32x32x3 but mean and std have shape of 3x1x1 -> python do broadcasting
        return (x - mean) / std

class Compose:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

# ---------------------- Model: ConvBlock to implement ---------------------- #    
class ResBlock(nn.Module):
    """
    TODO: 
      Residual Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + skip -> ReLU
      All convs have padding=1 and bias=False 
      In case in_ch == out_ch and stride=1, skip is identity, else it should modify the input x's shape
    """
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # need to define functions used in Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + skip -> ReLU forward
        # Do strided convolution in the first conv layer -> do downsampling
        
        # if stride = 1 & channels are the same, skip connection is identity
        # else, use 1x1 conv with input x to match in channel & out channel
        #=============================
        self.cv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.cv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        if in_ch == out_ch and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        # Implement Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + skip forward path
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.cv2(out)
        out = self.bn2(out)
        out += self.skip(x)
        out = self.relu(out)
        return out

class SmallResNet(nn.Module):
    """
    TODO:
      Small ResNet for CIFAR-10 classification
      stem : Conv3x3(3->32) -> BN -> ReLU
      body_1 : ResBlock layer(C : 32 -> 32, (H, W) -> (H, W))
      body_2 : ResBlock layer(C : 32 -> 64, (H, W) -> (H/2, W/2))
      body_3 : ResBlock layer(C : 64 -> 128, (H/2, W/2) -> (H/4, W/4))
      body_4 : ResBlock layer(C : 128 -> 128, (H/4, W/4) -> (H/8, W/8))
      head_gap : Global Average Pooling(nn.AdaptiveAvgPool2d(1))
      head_fc : Linear(128 -> num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # need to define functions used in stem -> body_1 -> body_2 -> body_3 -> body_4 -> head_gap -> head_fc forward
        #=============================
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.body_1 = ResBlock(32, 32, stride=1)
        self.body_2 = ResBlock(32, 64, stride=2)
        self.body_3 = ResBlock(64, 128, stride=2)
        self.body_4 = ResBlock(128, 128, stride=2)
        self.head_gap = nn.AdaptiveAvgPool2d(1)
        self.head_fc = nn.Linear(128, num_classes)

    def forward(self, x):
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        #=============================
        # Implement stem -> body_1 -> body_2 -> body_3 -> body_4 -> head_gap -> head_fc forward
        # Need to flatten before head_fc, because it is linear gemv operation
        #=============================
        out = self.stem(x)
        out = self.body_1(out)
        out = self.body_2(out)
        out = self.body_3(out)
        out = self.body_4(out)
        out = self.head_gap(out)
        out = torch.flatten(out, 1)
        out = self.head_fc(out)
        
        return out

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    TODO: 
      Implement training loop for one epoch * Refer to evaluate function *
      Use own scheduler function and call scheduler.step() after optimizer.step()
    """
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
    #=============================
    # In main function
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = cosine_warmup_scheduler
    #----------------------------
    # Train steps
    # 1. Set model to training mode, BatchNorm is enabled
    # 2. Iterate over datasets, each iteration gives a batch of images and labels
    # 2.1. Move images and labels to device
    # 2.2. Clear gradients, for independent updates with new batch
    # 2.3. Forward pass -> compute predicted outputs
    # 2.4. Compute CrossEntropyLoss
    # 2.5. Backward pass -> compute gradient of the loss
    # 2.6. Update parameters using optimizer
    # 2.7. Update learning rate using scheduler
    # 2.8. Accumulate batch loss
    # 2.9. Compute batch accuracy
    #=============================
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n = 0
    
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        bs = imgs.size(0)
        
        # loss is scalar tensor -> using .item() to get Python float
        running_loss += loss.item() * bs
        
        # logits is tensor of shape (batch_size, num_classes) 
        # -> predicted class is argmax along dim 1
        # -> compute accuracy by comparing predicted and true labels
        # -> sum all of correct predictions in the batch
        # running_acc += (logits.argmax(1) == labels).float().sum().item()
        running_acc += (logits.argmax(1) == labels).sum().item()
        
        n += bs
        
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return running_loss / n, running_acc / n

def cosine_warmup_scheduler(
    optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 1,
    min_lr_ratio: float = 0.1,
):
    total_steps  = steps_per_epoch * epochs
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(global_step: int): 
        """
        TODO:
          1) linear warm-up : global_step(num of current step) < warmup_steps -> return linearly increasing lr_factor from 0 to 1
          2) cosine decay : global_step >= warmup_steps -> return lr_factor following cosine decay lr_factor from 1 to min_lr_ratio
        """
        ##############################
        #                            #
        # ===== YOUR CODE HERE ===== #
        #                            #
        ##############################
        if(global_step < warmup_steps):
            # warm-up: linearly increase 0 to 1 
            return global_step / warmup_steps
        else:
            #------------------------------------------------------------------------
            # # step decay: 1 to min_lr_ratio for experiment
            # progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            # if progress < 0.3:
            #     return 1.0
            # elif progress < 0.6:
            #     return 0.5
            # elif progress < 0.9:
            #     return 0.2
            # else:
            #     return min_lr_ratio
            #------------------------------------------------------------------------
            # cosine decay: 1 to min_lr_ratio, using function in lecture5 pdf
            progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
            #------------------------------------------------------------------------

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ---------------------- Main (single training loop) ---------------------- #
def main():
    cfg = Cfg()
    cfg.data_dir = get_data_root("data")
    set_seed(cfg.seed)

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"[device] CPU, threads={torch.get_num_threads()}")

    # 1) Load raw train/test with ToTensorOnly to compute stats
    train_raw = datasets.CIFAR10(root=cfg.data_dir, train=True,  download=True, transform=ToTensorOnly())

    # 2) Split train/val
    total_len = len(train_raw)  # 50,000
    train_len = total_len - cfg.val_size
    g = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(train_raw, [train_len, cfg.val_size], generator=g)

    # 3) Compute mean/std on train subset
    mean, std = compute_mean_std(train_subset)
    print("[mean]", mean.tolist()) # standard mean value : [0.4914, 0.4822, 0.4465] / Do not need to same as these exactly
    print("[std ]", std.tolist()) # standard std value : [0.2470, 0.2435, 0.2616] / Values between 0.24 ~ 0.26 would be workable

    # 4) Build final transforms 
    train_transform = Compose([
        lambda img: TF.to_tensor(img),
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(p=0.5),
        Normalize(mean, std),
    ])
    eval_transform = Compose([
        lambda img: TF.to_tensor(img),
        Normalize(mean, std),
    ])

    # 5) Rebuild datasets with final transforms (preserve split indices)
    train_ds = datasets.CIFAR10(root=cfg.data_dir, train=True, download=False, transform=train_transform)
    train_ds = Subset(train_ds, train_subset.indices)
    val_ds = datasets.CIFAR10(root=cfg.data_dir, train=True, download=False, transform=eval_transform)
    val_ds = Subset(val_ds, val_subset.indices)
    test_ds = datasets.CIFAR10(root=cfg.data_dir, train=False, download=False, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 6) Model / criterion / optimizer
    model = SmallResNet(num_classes=10).to(device)
    print("[params]", count_params(model)) # ex. 620714

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    steps_per_epoch = len(train_loader)
    scheduler = cosine_warmup_scheduler(
                    optimizer,
                    steps_per_epoch=steps_per_epoch,
                    epochs=cfg.epochs,
                    warmup_epochs=1,
                    min_lr_ratio=0.1
                )

    # 7) Training loop (inline, no helper)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []} # track history for plotting
    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        # ---- train ---- #
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)

        # ---- validate ---- #
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        dt = time.time() - t0
        print(f"[{epoch:02d}/{cfg.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": best_val_acc
            }, "best_cifar10_smallresnet_cpu.pth")
            print(f"[ckpt] saved (val acc={best_val_acc:.3f})")

    # === Save curves === #
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy")
    plt.legend(); plt.savefig("acc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
    plt.legend(); plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    # 8) Test evaluation
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"[test] loss {te_loss:.4f} acc {te_acc:.3f}")

if __name__ == "__main__":
    main()
