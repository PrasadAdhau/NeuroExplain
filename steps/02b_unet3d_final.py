# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, EnsureTyped, RandSpatialCropd, RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

OUT_PATH = os.getenv("NEUROEXPLAIN_OUTPUT_PATH", "outputs")
PREP_PATH = os.path.join(OUT_PATH, "preprocessed_npz")
SPLIT_JSON = os.path.join(OUT_PATH, "split.json")

assert os.path.exists(PREP_PATH), f"❌ PREP_PATH not found: {PREP_PATH}"
assert os.path.exists(SPLIT_JSON), f"❌ split.json not found: {SPLIT_JSON}"

VIS_DIR = os.path.join(OUT_PATH, "step3_visuals")
os.makedirs(VIS_DIR, exist_ok=True)

print("✅ PREP_PATH:", PREP_PATH)
print("✅ SPLIT_JSON:", SPLIT_JSON)
print("✅ VIS_DIR:", VIS_DIR)

with open(SPLIT_JSON, "r") as f:
    split = json.load(f)

train_files = split["train"]
val_files   = split["val"]
test_files  = split["test"]

print("Train/Val/Test:", len(train_files), len(val_files), len(test_files))
print("Example:", train_files[0])

# Map BraTS label 4 to class 3 for 3D segmentation tensors.
def remap_brats_labels_3d(seg):
    seg = seg.copy()
    seg[seg == 4] = 3
    return seg

# STEP 2B: Define the final 3D MONAI U-Net training workflow.
# 3D dataset that loads preprocessed multimodal volumes and segmentation labels.
class Brats3DNPZDataset(Dataset):
    def __init__(self, npz_dir, file_list):
        self.npz_dir = npz_dir
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.npz_dir, fname)
        data = np.load(path, allow_pickle=True)

        img = data["img"].astype(np.float32)   # (4,H,W,D)
        seg = data["seg"].astype(np.int64)     # (H,W,D)
        seg = remap_brats_labels_3d(seg)
        seg = np.expand_dims(seg, axis=0)      # (1,H,W,D)

        if img.shape[1:] != seg.shape[1:]:
            raise ValueError(f"Shape mismatch {fname}: img {img.shape} vs seg {seg.shape}")

        return {"image": img, "label": seg, "fname": fname}

BATCH_SIZE = 1
NUM_WORKERS = 2  # set 0 if you see worker issues

train_ds = Brats3DNPZDataset(PREP_PATH, train_files)
val_ds   = Brats3DNPZDataset(PREP_PATH, val_files)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

b = next(iter(train_loader))
print("Loader batch image:", b["image"].shape)  # (1,4,H,W,D)
print("Loader batch label:", b["label"].shape)  # (1,1,H,W,D)

PATCH_SIZE = (96, 96, 96)  # if OOM -> (80,80,80)

train_tfms = Compose([
    EnsureTyped(keys=["image", "label"], track_meta=False),
    RandSpatialCropd(keys=["image", "label"], roi_size=PATCH_SIZE, random_size=False),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
])

val_tfms = Compose([
    EnsureTyped(keys=["image", "label"], track_meta=False),
])

# Remove the batch dimension before dictionary-based MONAI transforms.
def unbatch(batch_dict):
    """(1,...) -> (...) for tensors; keeps strings/lists as-is."""
    out = {}
    for k, v in batch_dict.items():
        if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out

s = unbatch(b)                 # (4,H,W,D), (1,H,W,D)
t = train_tfms(s)              # (4,ps,ps,ps), (1,ps,ps,ps)
print("After transform image:", t["image"].shape)
print("After transform label:", t["label"].shape)
print("✅ Transforms OK")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

CKPT_PATH = os.path.join(OUT_PATH, "unet3d_best.pt")
print("✅ CKPT_PATH:", CKPT_PATH)

# Train the 3D U-Net for one epoch using random augmented patches.
def train_one_epoch():
    model.train()
    total_loss, steps = 0.0, 0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        sample = unbatch(batch)        # remove batch dim
        sample = train_tfms(sample)    # apply transforms on single sample

        x = sample["image"].unsqueeze(0).to(device)  # add batch back
        y = sample["label"].unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)

@torch.no_grad()
# Evaluate the 3D U-Net with sliding-window inference and foreground Dice.
def validate_one_epoch():
    model.eval()
    dice_metric.reset()
    total_loss, n = 0.0, 0

    for batch in tqdm(val_loader, desc="Val", leave=False):
        sample = unbatch(batch)
        sample = val_tfms(sample)

        x = sample["image"].unsqueeze(0).to(device)  # (1,4,H,W,D)
        y = sample["label"].unsqueeze(0).to(device)  # (1,1,H,W,D)

        logits = sliding_window_inference(x, PATCH_SIZE, 1, model)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        n += 1

        dice_metric(y_pred=logits, y=y)

    fg_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return total_loss / max(n, 1), fg_dice

history = {"epoch": [], "train_loss": [], "val_loss": [], "val_fg_dice": []}
best_dice = -1.0
EPOCHS = 3

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch()
    va_loss, va_dice = validate_one_epoch()

    history["epoch"].append(epoch)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["val_fg_dice"].append(va_dice)

    print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | val FG dice {va_dice:.4f}")

    if va_dice > best_dice:
        best_dice = va_dice
        torch.save(model.state_dict(), CKPT_PATH)
        print("✅ Saved best model")

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_loss"], label="val loss")
    plt.title("Loss Curves"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["epoch"], history["val_fg_dice"], label="val FG Dice")
    plt.title("Validation Dice (FG)"); plt.xlabel("Epoch"); plt.ylabel("Dice")
    plt.ylim(0, 1); plt.legend()
    plt.tight_layout()
    plt.show()

print("✅ Training done. Best val FG Dice:", best_dice)

# Save multi-slice ground-truth and prediction overlays for one 3D case.
def overlay_vis_3d_case(model, npz_path, patch_size, device, out_dir, num_slices=6):
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    img = data["img"].astype(np.float32)  # (4,H,W,D)
    seg = data["seg"].astype(np.int64)    # (H,W,D)
    seg = remap_brats_labels_3d(seg)

    x = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,4,H,W,D)

    model.eval()
    with torch.no_grad():
        logits = sliding_window_inference(x, patch_size, 1, model)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H,W,D)

    flair = img[0]
    D = flair.shape[2]

    tumor_slices = np.where((seg != 0).reshape(-1, D).any(axis=0))[0]
    if len(tumor_slices) > 0:
        zs = np.linspace(tumor_slices.min(), tumor_slices.max(), num_slices).astype(int)
    else:
        zs = np.linspace(0, D-1, num_slices).astype(int)

    fig, axs = plt.subplots(num_slices, 3, figsize=(12, 4*num_slices))
    for i, z in enumerate(zs):
        axs[i,0].imshow(flair[:,:,z], cmap="gray")
        axs[i,0].set_title(f"FLAIR z={z}")
        axs[i,0].axis("off")

        axs[i,1].imshow(flair[:,:,z], cmap="gray")
        axs[i,1].imshow(seg[:,:,z], alpha=0.5)
        axs[i,1].set_title("GT overlay")
        axs[i,1].axis("off")

        axs[i,2].imshow(flair[:,:,z], cmap="gray")
        axs[i,2].imshow(pred[:,:,z], alpha=0.5)
        axs[i,2].set_title("Pred overlay")
        axs[i,2].axis("off")

    plt.tight_layout()
    out_file = os.path.join(out_dir, os.path.basename(npz_path).replace(".npz", "_overlay.png"))
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ Saved:", out_file)
    print("GT labels:", np.unique(seg))
    print("Pred labels:", np.unique(pred))

model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
print("✅ Loaded best model:", CKPT_PATH)

num_cases = min(3, len(val_files))
for fname in random.sample(val_files, num_cases):
    npz_path = os.path.join(PREP_PATH, fname)
    print("\nVisualizing:", fname)
    overlay_vis_3d_case(model, npz_path, PATCH_SIZE, device, VIS_DIR, num_slices=6)

print("\n✅ Visuals saved to:", VIS_DIR)


