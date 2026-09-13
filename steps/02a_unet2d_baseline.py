# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Resize

OUT_PATH = os.getenv("NEUROEXPLAIN_OUTPUT_PATH", "outputs")

SPLIT_JSON = os.path.join(OUT_PATH, "split.json")
PREP_PATH = os.path.join(OUT_PATH, "preprocessed_npz") # Explicitly defining PREP_PATH to ensure correctness

assert os.path.exists(PREP_PATH), "PREP_PATH not found. Check OUT_PATH / Step1 outputs."
assert os.path.exists(SPLIT_JSON), "split.json not found. Check Step1 outputs."

print("✅ PREP_PATH:", PREP_PATH)
print("✅ SPLIT_JSON:", SPLIT_JSON)
print("NPZ files:", len([f for f in os.listdir(PREP_PATH) if f.endswith('.npz')]))

with open(SPLIT_JSON, "r") as f:
    split = json.load(f)

train_files = split["train"]
val_files   = split["val"]
test_files  = split["test"]

print("Train / Val / Test:", len(train_files), len(val_files), len(test_files))
print("Example:", train_files[0])

# Map BraTS label 4 to class 3 for four-class model training and analysis.
def remap_brats_labels(mask_2d):
    mask = mask_2d.copy()
    mask[mask == 4] = 3
    return mask


from monai.transforms import Resize

# STEP 2A: Define the initial 2D U-Net baseline experiment.
# 2D dataset that samples tumor-containing slices more often during baseline training.
class Brats2DSliceDataset(Dataset):
    """
    Loads .npz created in Step 1:
      img: (4, H, W, D)
      seg: (H, W, D)

    Returns random slices:
      x: (4, H, W)
      y: (H, W)
    """
    def __init__(self, npz_dir, file_list, tumor_frac=0.8, spatial_size=(160, 160)):
        self.npz_dir = npz_dir
        self.files = file_list
        self.tumor_frac = tumor_frac
        self.spatial_size = spatial_size

        self.resize_image = Resize(spatial_size=self.spatial_size, mode="bilinear")
        self.resize_label = Resize(spatial_size=self.spatial_size, mode="nearest")

        self.index = []
        for fname in self.files:
            path = os.path.join(self.npz_dir, fname)
            data = np.load(path, allow_pickle=True)
            seg = data["seg"]   # (H, W, D)
            D = seg.shape[2]
            tumor_slices = np.where((seg != 0).reshape(-1, D).any(axis=0))[0]
            self.index.append({"fname": fname, "D": D, "tumor_slices": tumor_slices.astype(np.int32)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        info = self.index[idx]
        fname, D, tumor_slices = info["fname"], info["D"], info["tumor_slices"]

        if len(tumor_slices) > 0 and random.random() < self.tumor_frac:
            z = int(random.choice(tumor_slices))
        else:
            z = random.randrange(D)

        path = os.path.join(self.npz_dir, fname)
        data = np.load(path, allow_pickle=True)

        img = data["img"]  # (4,H,W,D)
        seg = data["seg"]  # (H,W,D)

        x_np = img[:, :, :, z].astype(np.float32)     # (4,H,W)
        y_np = seg[:, :, z].astype(np.int64)          # (H,W)
        y_np = remap_brats_labels(y_np)

        x = self.resize_image(x_np)
        y = self.resize_label(y_np[None, ...])[0].long()

        return x, y

BATCH_SIZE = 6
NUM_WORKERS = 0  # set to 0 for debugging DataLoader issues

train_ds = Brats2DSliceDataset(PREP_PATH, train_files, tumor_frac=0.85)
val_ds   = Brats2DSliceDataset(PREP_PATH, val_files, tumor_frac=0.50)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

xb, yb = next(iter(train_loader))
print("X batch:", xb.shape, xb.dtype)  # (B,4,H,W)
print("Y batch:", yb.shape, yb.dtype)  # (B,H,W)
print("Unique labels:", torch.unique(yb))

# Reusable two-convolution block used by the 2D U-Net baseline.
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

# 2D U-Net baseline for early slice-based segmentation experiments.
class UNet2D(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

# Calculate per-class Dice scores from model logits and target masks.
def dice_per_class(logits, targets, num_classes=4, eps=1e-6):
    preds = torch.argmax(logits, dim=1)  # (B,H,W)
    dices = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        targ_c = (targets == c).float()
        inter = (pred_c * targ_c).sum(dim=(1,2))
        denom = pred_c.sum(dim=(1,2)) + targ_c.sum(dim=(1,2))
        dice = (2*inter + eps) / (denom + eps)
        dices.append(dice.mean())
    return torch.stack(dices)  # (C,)

# Differentiable multi-class Dice loss used with cross-entropy for 2D training.
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=4, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
        onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)

        inter = (probs * onehot).sum(dim=(0,2,3))
        denom = (probs + onehot).sum(dim=(0,2,3))
        dice = (2*inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()

ce_loss = nn.CrossEntropyLoss()
dice_loss = SoftDiceLoss(num_classes=4)

# Combine cross-entropy and soft-Dice loss for 2D training.
def combined_loss(logits, targets, alpha=0.5):
    return alpha * ce_loss(logits, targets) + (1 - alpha) * dice_loss(logits, targets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2D(in_ch=4, num_classes=4, base=32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Run one 2D training or validation epoch and aggregate loss and Dice scores.
def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    total_dice = torch.zeros(4, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = combined_loss(logits, y, alpha=0.5)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dice += dice_per_class(logits, y, num_classes=4)

    n = len(loader)
    return total_loss / n, (total_dice / n).detach().cpu().numpy()

EPOCHS = 6
best_fg = -1.0
best_path = os.path.join(OUT_PATH, "unet2d_best.pt")

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_dice = run_epoch(train_loader, train=True)
    va_loss, va_dice = run_epoch(val_loader, train=False)
    va_fg = float(va_dice[1:].mean())  # exclude background

    print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | "
          f"val dice [bg,1,2,3] {np.round(va_dice,3)} | val FG dice {va_fg:.3f}")

    if va_fg > best_fg:
        best_fg = va_fg
        torch.save(model.state_dict(), best_path)
        print("✅ Saved best model:", best_path)

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

x, y = next(iter(val_loader))
x = x.to(device)

with torch.no_grad():
    logits = model(x)
pred = torch.argmax(logits, dim=1).cpu().numpy()

x = x.cpu().numpy()
y = y.numpy()

i = 0
flair = x[i, 0]      # channel 0 (FLAIR)
gt = y[i]
pd_ = pred[i]

plt.figure(figsize=(14,4))
plt.subplot(1,3,1); plt.imshow(flair, cmap="gray"); plt.title("Input (FLAIR)"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(flair, cmap="gray"); plt.imshow(gt, alpha=0.5); plt.title("GT Overlay"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(flair, cmap="gray"); plt.imshow(pd_, alpha=0.5); plt.title("Pred Overlay"); plt.axis("off")
plt.tight_layout()
plt.show()

print("GT labels:", np.unique(gt))
print("Pred labels:", np.unique(pd_))

print("✅ Installed MONAI + tqdm")



from monai.transforms import (
    Compose, EnsureTyped,
    RandSpatialCropd, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("✅ Imports ready")


