# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_PATH, PREP_PATH, PATCH_SIZE, load_split, load_final_model
from monai.inferers import sliding_window_inference
try:
    from IPython.display import display
except ImportError:
    display = print

split = load_split()
val_files = split["val"]
model, device = load_final_model()

# STEP 4B: Run enhanced XAI analysis, overlap metrics, and modality importance.
STEP4_OUT = os.path.join(OUT_PATH, "step4_xai_enhanced")
STEP4_VIS = os.path.join(STEP4_OUT, "visuals")
STEP4_CSV = os.path.join(STEP4_OUT, "tables")

os.makedirs(STEP4_OUT, exist_ok=True)
os.makedirs(STEP4_VIS, exist_ok=True)
os.makedirs(STEP4_CSV, exist_ok=True)

print("✅ Step 4 output:", STEP4_OUT)

# Map BraTS label 4 to class 3 for four-class model training and analysis.
def remap_brats_labels(seg):
    seg = seg.copy()
    seg[seg == 4] = 3
    return seg

# Convert an NPZ filename to the corresponding case identifier.
def patient_id_from_npz(fname):
    return fname.replace(".npz", "")

# Load preprocessed image channels and the remapped ground-truth mask.
def load_case_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    img = data["img"].astype(np.float32)   # (4,H,W,D)
    gt = data["seg"].astype(np.int64)      # (H,W,D)
    gt = remap_brats_labels(gt)
    return img, gt

# Run model inference for an image volume used by the enhanced XAI workflow.
def forward_predict(img_4chw):
    x = torch.from_numpy(img_4chw).unsqueeze(0).to(device)  # (1,4,H,W,D)
    model.eval()
    with torch.no_grad():
        original_spatial_shape = x.shape[2:]
        x_padded, _, padding_info = pad_to_multiple(x, multiple=16)

        logits_padded = model(x_padded)
        logits = crop_to_original(logits_padded, original_spatial_shape, padding_info)

        pred_padded = torch.argmax(logits_padded, dim=1)
        pred = crop_to_original(pred_padded.unsqueeze(0), original_spatial_shape, padding_info).squeeze(0).cpu().numpy()

    return logits, pred

# Select the final suitable convolution layer for Grad-CAM.
def find_target_layer(model):
    candidate = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            if module.kernel_size != (1, 1, 1):
                candidate = module
    if candidate is None:
        raise ValueError("No suitable Conv3d layer found.")
    return candidate

target_layer = find_target_layer(model)
print("✅ Target layer:", target_layer)

# Compute binary Dice overlap between two masks.
def dice_binary(mask1, mask2, eps=1e-8):
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)
    inter = (mask1 * mask2).sum()
    denom = mask1.sum() + mask2.sum()
    return (2 * inter + eps) / (denom + eps)

# Compute binary intersection-over-union between two masks.
def iou_binary(mask1, mask2, eps=1e-8):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return (inter + eps) / (union + eps)

# Grad-CAM implementation that captures convolution activations and gradients for 3D explanations.
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x, target_class):
        """
        x: (1,4,H,W,D)
        target_class: int in [1,2,3]
        returns:
          cam_3d: (H,W,D) normalized Grad-CAM heatmap
          pred:   (H,W,D) predicted mask
          logits: raw logits (cropped to original size)
        """
        self.model.eval()
        self.model.zero_grad()

        original_spatial_shape = x.shape[2:] # (H,W,D)
        x_padded, _, padding_info = pad_to_multiple(x, multiple=16)

        logits_padded = self.model(x_padded)                      # (1,C,H_padded,W_padded,D_padded)

        logits = crop_to_original(logits_padded, original_spatial_shape, padding_info)
        pred_padded = torch.argmax(logits_padded, dim=1) # (1, H_padded, W_padded, D_padded)
        pred = crop_to_original(pred_padded.unsqueeze(0), original_spatial_shape, padding_info).squeeze() # (H,W,D)

        class_map_padded = logits_padded[:, target_class, :, :, :]          # (1,H_padded,W_padded,D_padded)
        pred_mask_padded = (pred_padded == target_class).float()            # (1,H_padded,W_padded,D_padded)

        if pred_mask_padded.sum() == 0:
            score = class_map_padded.mean()
        else:
            score = (class_map_padded * pred_mask_padded).sum() / (pred_mask_padded.sum() + 1e-8)

        score.backward(retain_graph=True)

        grads = self.gradients[0]      # (C,h,w,d)
        acts  = self.activations[0]    # (C,h,w,d)

        weights = grads.mean(dim=(1,2,3), keepdim=True)       # (C,1,1,1)
        cam = (weights * acts).sum(dim=0)                     # (h,w,d) at target layer
        cam = F.relu(cam)

        cam_padded = cam.unsqueeze(0).unsqueeze(0)                   # (1,1,h,w,d)
        cam_padded = F.interpolate(
            cam_padded,
            size=x_padded.shape[2:], # Target size is the padded input size
            mode="trilinear",
            align_corners=False
        ).squeeze() # (H_padded,W_padded,D_padded)

        cam_final = crop_to_original(cam_padded.unsqueeze(0).unsqueeze(0), original_spatial_shape, padding_info).squeeze().cpu().numpy()

        cam_final = cam_final - cam_final.min()
        if cam_final.max() > 0:
            cam_final = cam_final / cam_final.max()

        return cam_final, pred.cpu().numpy(), logits.detach().cpu()

# Create a Grad-CAM heatmap for each requested tumor class.
def generate_per_class_gradcam(npz_fname, class_list=[1, 2, 3]):
    npz_path = os.path.join(PREP_PATH, npz_fname)
    patient_id = patient_id_from_npz(npz_fname)

    img, gt = load_case_npz(npz_path)
    x = torch.from_numpy(img).unsqueeze(0).to(device)

    gradcam = GradCAM3D(model, target_layer)

    cams = {}
    with torch.enable_grad():
        for cls in class_list:
            cam, pred, _ = gradcam.generate(x, target_class=cls)
            cams[cls] = cam

    gradcam.remove_hooks()

    return {
        "patient_id": patient_id,
        "img": img,
        "gt": gt,
        "pred": pred,
        "cams": cams
    }

# Plot multi-slice Grad-CAM panels for each tumor class.
def plot_per_class_multislice_gradcam(case_data, save_dir, num_slices=6):
    patient_id = case_data["patient_id"]
    img = case_data["img"]
    gt = case_data["gt"]
    pred = case_data["pred"]
    cams = case_data["cams"]

    flair = img[0]
    D = flair.shape[2]

    tumor_slices = np.where((pred > 0).reshape(-1, D).any(axis=0))[0]
    if len(tumor_slices) > 0:
        z_list = np.linspace(tumor_slices.min(), tumor_slices.max(), num_slices).astype(int)
    else:
        z_list = np.linspace(0, D-1, num_slices).astype(int)

    fig, axs = plt.subplots(num_slices, 6, figsize=(22, 4 * num_slices))

    if num_slices == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, z in enumerate(z_list):
        axs[i, 0].imshow(flair[:, :, z], cmap="gray")
        axs[i, 0].set_title(f"FLAIR z={z}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(flair[:, :, z], cmap="gray")
        axs[i, 1].imshow(gt[:, :, z], alpha=0.5)
        axs[i, 1].set_title("GT")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(flair[:, :, z], cmap="gray")
        axs[i, 2].imshow(pred[:, :, z], alpha=0.5)
        axs[i, 2].set_title("Pred")
        axs[i, 2].axis("off")

        for j, cls in enumerate([1, 2, 3]):
            axs[i, 3 + j].imshow(flair[:, :, z], cmap="gray")
            axs[i, 3 + j].imshow(cams[cls][:, :, z], cmap="jet", alpha=0.5)
            axs[i, 3 + j].set_title(f"Grad-CAM Class {cls}")
            axs[i, 3 + j].axis("off")

    plt.suptitle(f"{patient_id} | Per-class Multi-slice Grad-CAM", y=1.0, fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{patient_id}_perclass_multislice_gradcam.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("✅ Saved:", out_path)

# Convert a continuous Grad-CAM map into low, medium, and high attention regions.
def threshold_cam(cam, low=0.3, high=0.6):
    low_mask = ((cam >= low) & (cam < high)).astype(np.uint8)
    high_mask = (cam >= high).astype(np.uint8)
    return low_mask, high_mask

# Visualize thresholded Grad-CAM attention on a representative slice.
def plot_thresholded_cam(case_data, target_class=3, save_dir=None):
    patient_id = case_data["patient_id"]
    img = case_data["img"]
    pred = case_data["pred"]
    cam = case_data["cams"][target_class]

    flair = img[0]
    D = flair.shape[2]
    tumor_slices = np.where((pred > 0).reshape(-1, D).any(axis=0))[0]
    z = int(np.median(tumor_slices)) if len(tumor_slices) > 0 else D // 2

    low_mask, high_mask = threshold_cam(cam)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(flair[:, :, z], cmap="gray")
    axs[0].set_title("MRI")
    axs[0].axis("off")

    axs[1].imshow(flair[:, :, z], cmap="gray")
    axs[1].imshow(cam[:, :, z], cmap="jet", alpha=0.5)
    axs[1].set_title("Raw Grad-CAM")
    axs[1].axis("off")

    axs[2].imshow(flair[:, :, z], cmap="gray")
    axs[2].imshow(low_mask[:, :, z], cmap="Blues", alpha=0.5)
    axs[2].set_title("Medium Attention")
    axs[2].axis("off")

    axs[3].imshow(flair[:, :, z], cmap="gray")
    axs[3].imshow(high_mask[:, :, z], cmap="Reds", alpha=0.6)
    axs[3].set_title("High Attention")
    axs[3].axis("off")

    plt.tight_layout()
    if save_dir:
        out_path = os.path.join(save_dir, f"{patient_id}_thresholded_cam_class{target_class}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print("✅ Saved:", out_path)
    plt.show()

# Compare ground truth, prediction, and Grad-CAM across slices.
def plot_gt_pred_cam_comparison(case_data, target_class=3, save_dir=None, num_slices=4):
    patient_id = case_data["patient_id"]
    img = case_data["img"]
    gt = case_data["gt"]
    pred = case_data["pred"]
    cam = case_data["cams"][target_class]

    flair = img[0]
    D = flair.shape[2]
    tumor_slices = np.where((pred > 0).reshape(-1, D).any(axis=0))[0]
    if len(tumor_slices) > 0:
        z_list = np.linspace(tumor_slices.min(), tumor_slices.max(), num_slices).astype(int)
    else:
        z_list = np.linspace(0, D-1, num_slices).astype(int)

    fig, axs = plt.subplots(num_slices, 4, figsize=(16, 4 * num_slices))

    if num_slices == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, z in enumerate(z_list):
        axs[i, 0].imshow(flair[:, :, z], cmap="gray")
        axs[i, 0].set_title(f"FLAIR z={z}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(flair[:, :, z], cmap="gray")
        axs[i, 1].imshow(gt[:, :, z], alpha=0.5)
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(flair[:, :, z], cmap="gray")
        axs[i, 2].imshow(pred[:, :, z], alpha=0.5)
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

        axs[i, 3].imshow(flair[:, :, z], cmap="gray")
        axs[i, 3].imshow(cam[:, :, z], cmap="jet", alpha=0.5)
        axs[i, 3].set_title(f"Grad-CAM Class {target_class}")
        axs[i, 3].axis("off")

    plt.tight_layout()
    if save_dir:
        out_path = os.path.join(save_dir, f"{patient_id}_gt_pred_cam_comparison_class{target_class}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print("✅ Saved:", out_path)
    plt.show()

# Quantify overlap between Grad-CAM attention, prediction, and ground truth.
def compute_cam_overlap_scores(case_data, target_class=3, cam_threshold=0.6):
    gt = case_data["gt"]
    pred = case_data["pred"]
    cam = case_data["cams"][target_class]

    cam_bin = (cam >= cam_threshold).astype(np.uint8)
    pred_bin = (pred == target_class).astype(np.uint8)
    gt_bin = (gt == target_class).astype(np.uint8)

    scores = {
        "target_class": target_class,
        "cam_threshold": cam_threshold,
        "cam_vs_pred_dice": dice_binary(cam_bin, pred_bin),
        "cam_vs_pred_iou": iou_binary(cam_bin, pred_bin),
        "cam_vs_gt_dice": dice_binary(cam_bin, gt_bin),
        "cam_vs_gt_iou": iou_binary(cam_bin, gt_bin),
    }
    return scores

# Estimate modality importance by measuring prediction changes after ablation.
def modality_importance_analysis(npz_fname, target_class=3):
    npz_path = os.path.join(PREP_PATH, npz_fname)
    patient_id = patient_id_from_npz(npz_fname)

    img, gt = load_case_npz(npz_path)

    _, pred_full = forward_predict(img)
    full_count = int((pred_full == target_class).sum())

    modality_names = ["FLAIR", "T1", "T1CE", "T2"]
    rows = []

    rows.append({
        "patient_id": patient_id,
        "setting": "Full Input",
        "target_class": target_class,
        "predicted_voxels": full_count,
        "relative_to_full": 1.0
    })

    for ch, mod_name in enumerate(modality_names):
        img_mod = img.copy()
        img_mod[ch] = 0.0

        _, pred_mod = forward_predict(img_mod)
        mod_count = int((pred_mod == target_class).sum())

        rows.append({
            "patient_id": patient_id,
            "setting": f"Without {mod_name}",
            "target_class": target_class,
            "predicted_voxels": mod_count,
            "relative_to_full": mod_count / full_count if full_count > 0 else 0.0
        })

    df = pd.DataFrame(rows)
    return df

# Plot the modality-ablation importance results.
def plot_modality_importance(df_mod, save_dir=None):
    patient_id = df_mod["patient_id"].iloc[0]
    target_class = df_mod["target_class"].iloc[0]

    plt.figure(figsize=(8, 4))
    plt.bar(df_mod["setting"], df_mod["relative_to_full"])
    plt.title(f"{patient_id} | Modality Importance (Class {target_class})")
    plt.ylabel("Relative Predicted Voxels")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, f"{patient_id}_modality_importance_class{target_class}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print("✅ Saved:", out_path)

    plt.show()

XAI_TARGET_FILES = val_files[:2]   # or random.sample(val_files, 2)

all_overlap_rows = []
all_modality_rows = []

for fname in XAI_TARGET_FILES:
    print("\n==============================")
    print("Running enhanced XAI for:", fname)

    case_data = generate_per_class_gradcam(fname, class_list=[1, 2, 3])

    plot_per_class_multislice_gradcam(case_data, STEP4_VIS, num_slices=6)

    for cls in [1, 2, 3]:
        plot_gt_pred_cam_comparison(case_data, target_class=cls, save_dir=STEP4_VIS, num_slices=4)

    for cls in [1, 2, 3]:
        plot_thresholded_cam(case_data, target_class=cls, save_dir=STEP4_VIS)

    for cls in [1, 2, 3]:
        scores = compute_cam_overlap_scores(case_data, target_class=cls, cam_threshold=0.6)
        scores["patient_id"] = case_data["patient_id"]
        all_overlap_rows.append(scores)

    dominant_class = max([1, 2, 3], key=lambda c: int((case_data["pred"] == c).sum()))
    df_mod = modality_importance_analysis(fname, target_class=dominant_class)
    all_modality_rows.append(df_mod)
    plot_modality_importance(df_mod, save_dir=STEP4_VIS)

print("\n✅ Enhanced Step 4 completed")

df_overlap = pd.DataFrame(all_overlap_rows)
df_overlap.to_csv(os.path.join(STEP4_CSV, "gradcam_overlap_scores.csv"), index=False)

df_modality = pd.concat(all_modality_rows, ignore_index=True)
df_modality.to_csv(os.path.join(STEP4_CSV, "modality_importance.csv"), index=False)

print("✅ Saved overlap table:", os.path.join(STEP4_CSV, "gradcam_overlap_scores.csv"))
print("✅ Saved modality table:", os.path.join(STEP4_CSV, "modality_importance.csv"))

display(df_overlap.head())
display(df_modality.head())

plt.figure(figsize=(10, 5))
for metric in ["cam_vs_pred_dice", "cam_vs_gt_dice"]:
    plt.plot(df_overlap.index, df_overlap[metric], marker="o", label=metric)

plt.title("Grad-CAM Overlap Scores")
plt.xlabel("Case-Class Index")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(STEP4_VIS, "gradcam_overlap_summary.png"), dpi=180, bbox_inches="tight")
plt.show()

pivot_mod = df_modality.pivot(index="patient_id", columns="setting", values="relative_to_full")

pivot_mod.plot(kind="bar", figsize=(10, 5))
plt.title("Modality Importance Summary")
plt.ylabel("Relative Predicted Voxels")
plt.tight_layout()
plt.savefig(os.path.join(STEP4_VIS, "modality_importance_summary.png"), dpi=180, bbox_inches="tight")
plt.show()

print("✅ Installed sentence-transformers + faiss")

import faiss

from sentence_transformers import SentenceTransformer
