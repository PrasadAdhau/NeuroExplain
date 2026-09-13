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

split = load_split()
val_files = split["val"]
model, device = load_final_model()

# STEP 4A: Generate initial 3D Grad-CAM explanation visuals.
STEP4_OUT = os.path.join(OUT_PATH, "step4_xai")
STEP4_VIS = os.path.join(STEP4_OUT, "visuals")
os.makedirs(STEP4_OUT, exist_ok=True)
os.makedirs(STEP4_VIS, exist_ok=True)

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

# Run direct model inference and return logits for explanation workflows.
def predict_case_logits(npz_path):
    img, gt = load_case_npz(npz_path)
    x = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,4,H,W,D)

    model.eval()
    with torch.no_grad():
        logits = model(x)   # direct forward for XAI
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return img, gt, logits, pred

# Pad a 3D tensor so U-Net downsampling levels divide its spatial dimensions.
def pad_to_multiple(image_tensor, multiple=16):
    """Pads a tensor (N, C, D1, D2, D3) to ensure spatial dimensions are multiples of 'multiple'."""
    original_spatial_shape = image_tensor.shape[2:]  # (D1, D2, D3)
    padded_spatial_shape = [((dim + multiple - 1) // multiple) * multiple for dim in original_spatial_shape]

    padding_values_for_fpad = [] # This will be in the order (d3_left, d3_right, d2_left, d2_right, d1_left, d1_right)

    for orig_d, pad_d in zip(original_spatial_shape, padded_spatial_shape):
        pad_total = pad_d - orig_d
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padding_values_for_fpad.extend([pad_before, pad_after])

    padding_values_for_fpad = padding_values_for_fpad[::-1]

    padded_tensor = F.pad(image_tensor, padding_values_for_fpad, "constant", 0)
    return padded_tensor, original_spatial_shape, padding_values_for_fpad # Return original_spatial_shape and padding info for cropping

# Remove padding and restore an output tensor to its original spatial shape.
def crop_to_original(padded_tensor, original_spatial_shape, padding_values_for_fpad):
    """Crops a tensor (..., D1_padded, D2_padded, D3_padded) back to its original spatial dimensions.
       Assumes the tensor has spatial dimensions at the end.
    """
    num_spatial_dims = len(original_spatial_shape)
    num_non_spatial_dims = padded_tensor.ndim - num_spatial_dims

    slices = [slice(None)] * num_non_spatial_dims # Slices for non-spatial dimensions (N, C, etc.)

    for i in range(num_spatial_dims):

        pad_idx_in_fpad = (num_spatial_dims - 1 - i) * 2
        pad_left = padding_values_for_fpad[pad_idx_in_fpad]

        start_idx = pad_left
        end_idx = start_idx + original_spatial_shape[i]

        slices.append(slice(start_idx, end_idx))

    return padded_tensor[tuple(slices)]

# Select the final suitable convolution layer for Grad-CAM.
def find_target_layer(model):
    candidate = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            if module.kernel_size != (1, 1, 1):
                candidate = module
    if candidate is None:
        raise ValueError("No suitable Conv3d layer found for Grad-CAM.")
    return candidate

target_layer = find_target_layer(model)
print("✅ Target layer selected for Grad-CAM:")
print(target_layer)

# Grad-CAM implementation that captures convolution activations and gradients for 3D explanations.
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
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

# Choose the most represented predicted foreground class for explanation.
def choose_target_class(pred):
    counts = {c: int((pred == c).sum()) for c in [1, 2, 3]}
    target_class = max(counts, key=counts.get)
    return target_class, counts

# Generate Grad-CAM visualizations and summaries for one validation case.
def run_gradcam_on_case(npz_fname, save_dir):
    patient_id = patient_id_from_npz(npz_fname)
    npz_path = os.path.join(PREP_PATH, npz_fname)

    img, gt = load_case_npz(npz_path)
    x = torch.from_numpy(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits_swi = sliding_window_inference(x, PATCH_SIZE, 1, model)
        pred_swi = torch.argmax(logits_swi, dim=1).squeeze(0).cpu().numpy()

    target_class, class_counts = choose_target_class(pred_swi)

    gradcam = GradCAM3D(model, target_layer)
    with torch.enable_grad():
        cam, pred, _ = gradcam.generate(x, target_class) # pred from gradcam.generate is also cropped and padded
    gradcam.remove_hooks()

    flair = img[0]   # visualize FLAIR
    D = flair.shape[2]

    tumor_slices = np.where((pred > 0).reshape(-1, D).any(axis=0))[0]
    if len(tumor_slices) > 0:
        z_main = int(np.median(tumor_slices))
        z_list = np.linspace(tumor_slices.min(), tumor_slices.max(), 3).astype(int)
    else:
        z_main = D // 2
        z_list = np.array([D//4, D//2, 3*D//4])

    result = {
        "patient_id": patient_id,
        "target_class": int(target_class),
        "class_counts": class_counts,
        "main_slice": int(z_main)
    }

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
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
        axs[i, 3].set_title("Grad-CAM")
        axs[i, 3].axis("off")

    plt.suptitle(f"{patient_id} | Target Class: {target_class}", y=1.02, fontsize=14)
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, f"{patient_id}_gradcam_grid.png")
    plt.savefig(fig1_path, dpi=180, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(6,6))
    plt.imshow(flair[:, :, z_main], cmap="gray")
    plt.imshow(pred[:, :, z_main], alpha=0.35)
    plt.imshow(cam[:, :, z_main], cmap="jet", alpha=0.35)
    plt.title(f"{patient_id} | Pred + Grad-CAM (z={z_main})")
    plt.axis("off")
    fig2_path = os.path.join(save_dir, f"{patient_id}_pred_gradcam_overlay.png")
    plt.savefig(fig2_path, dpi=180, bbox_inches="tight")
    plt.show()

    cam_profile = cam.mean(axis=(0,1))
    plt.figure(figsize=(8,4))
    plt.plot(cam_profile)
    plt.axvline(z_main, linestyle="--")
    plt.title(f"{patient_id} | Slice-wise Mean Grad-CAM Intensity")
    plt.xlabel("Slice index (z)")
    plt.ylabel("Mean CAM intensity")
    fig3_path = os.path.join(save_dir, f"{patient_id}_cam_profile.png")
    plt.savefig(fig3_path, dpi=180, bbox_inches="tight")
    plt.show()

    cx, cy, cz = np.array(np.where(pred > 0)).mean(axis=1).astype(int) if (pred > 0).sum() > 0 else (pred.shape[0]//2, pred.shape[1]//2, pred.shape[2]//2)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0,0].imshow(flair[:, :, cz], cmap="gray")
    axs[0,0].set_title(f"Axial MRI z={cz}")
    axs[0,0].axis("off")

    axs[1,0].imshow(flair[:, :, cz], cmap="gray")
    axs[1,0].imshow(cam[:, :, cz], cmap="jet", alpha=0.5)
    axs[1,0].set_title("Axial Grad-CAM")
    axs[1,0].axis("off")

    axs[0,1].imshow(flair[:, cy, :], cmap="gray")
    axs[0,1].set_title(f"Coronal MRI y={cy}")
    axs[0,1].axis("off")

    axs[1,1].imshow(flair[:, cy, :], cmap="gray")
    axs[1,1].imshow(cam[:, cy, :], cmap="jet", alpha=0.5)
    axs[1,1].set_title("Coronal Grad-CAM")
    axs[1,1].axis("off")

    axs[0,2].imshow(flair[cx, :, :], cmap="gray")
    axs[0,2].set_title(f"Sagittal MRI x={cx}")
    axs[0,2].axis("off")

    axs[1,2].imshow(flair[cx, :, :], cmap="gray")
    axs[1,2].imshow(cam[cx, :, :], cmap="jet", alpha=0.5)
    axs[1,2].set_title("Sagittal Grad-CAM")
    axs[1,2].axis("off")

    plt.tight_layout()
    fig4_path = os.path.join(save_dir, f"{patient_id}_orthogonal_gradcam.png")
    plt.savefig(fig4_path, dpi=180, bbox_inches="tight")
    plt.show()

    print("Patient ID:", patient_id)
    print("Target class:", target_class)
    print("Predicted voxel counts:", class_counts)
    print("Saved:")
    print(" -", fig1_path)
    print(" -", fig2_path)
    print(" -", fig3_path)
    print(" -", fig4_path)

    return {
        "patient_id": patient_id,
        "target_class": target_class,
        "class_counts": class_counts,
        "cam_profile": cam_profile
    }

XAI_TARGET_FILES = val_files[:2]   # or random.sample(val_files, 2)

xai_results = []
for fname in XAI_TARGET_FILES:
    print("\nRunning Grad-CAM for:", fname)
    res = run_gradcam_on_case(fname, STEP4_VIS)
    xai_results.append(res)

print("\n✅ Step 4 completed for", len(xai_results), "cases")

fig, axs = plt.subplots(len(xai_results), 1, figsize=(10, 4 * len(xai_results)))

if len(xai_results) == 1:
    axs = [axs]

for i, res in enumerate(xai_results):
    axs[i].plot(res["cam_profile"])
    axs[i].set_title(f"{res['patient_id']} | Slice-wise Grad-CAM Intensity")
    axs[i].set_xlabel("Slice index (z)")
    axs[i].set_ylabel("Mean CAM intensity")

plt.tight_layout()
plt.savefig(os.path.join(STEP4_VIS, "two_case_gradcam_profiles.png"), dpi=180, bbox_inches="tight")
plt.show()



