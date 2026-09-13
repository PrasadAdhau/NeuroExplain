# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_PATH, PREP_PATH, PATCH_SIZE, load_split, load_final_model
from monai.inferers import sliding_window_inference

split = load_split()
train_files, val_files, test_files = split["train"], split["val"], split["test"]
model, device = load_final_model()

# STEP 3: Extract structured findings from predicted tumor segmentations.
STEP3_OUT = os.path.join(OUT_PATH, "step3_structured_findings")
JSON_DIR = os.path.join(STEP3_OUT, "json")
VIS_DIR = os.path.join(STEP3_OUT, "visuals")

os.makedirs(STEP3_OUT, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print("✅ Step 3 output:", STEP3_OUT)

# Map BraTS label 4 to class 3 for four-class model training and analysis.
def remap_brats_labels(seg):
    seg = seg.copy()
    seg[seg == 4] = 3
    return seg

# Convert an NPZ filename to the corresponding case identifier.
def patient_id_from_npz(fname):
    return fname.replace(".npz", "")

# Run sliding-window segmentation inference for one preprocessed case.
def predict_case(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    img = data["img"].astype(np.float32)   # (4,H,W,D)
    gt = data["seg"].astype(np.int64)      # (H,W,D)
    gt = remap_brats_labels(gt)

    x = torch.from_numpy(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = sliding_window_inference(x, PATCH_SIZE, 1, model)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return img, gt, pred

# Calculate the centroid of a binary tumor mask.
def compute_centroid(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)

# Assign a left, right, or midline proxy from the tumor centroid.
def get_laterality(cx, width):
    mid = width / 2.0
    if cx < mid - 5:
        return "left"
    elif cx > mid + 5:
        return "right"
    return "midline"

# Assign an anterior/posterior and superior/inferior location proxy.
def get_region_proxy(cy, cz, shape):
    _, w, d = shape
    ap = "anterior" if cy < (w / 2) else "posterior"
    si = "superior" if cz < (d / 2) else "inferior"
    return f"{ap}-{si}"

# Apply the project rule-based severity score from tumor burden and enhancement.
def severity_rule(whole_tumor_vox, enhancing_ratio):
    score = 0

    if whole_tumor_vox > 80000:
        score += 2
    elif whole_tumor_vox > 30000:
        score += 1

    if enhancing_ratio > 0.20:
        score += 2
    elif enhancing_ratio > 0.08:
        score += 1

    if score <= 1:
        level = "low"
    elif score <= 3:
        level = "medium"
    else:
        level = "high"

    return level, score

# Convert a predicted segmentation mask into structured JSON-ready findings.
def extract_structured_findings(patient_id, pred):
    wt_mask = pred > 0
    core_mask = pred == 1
    edema_mask = pred == 2
    enhancing_mask = pred == 3

    wt_vox = int(wt_mask.sum())
    core_vox = int(core_mask.sum())
    edema_vox = int(edema_mask.sum())
    enhancing_vox = int(enhancing_mask.sum())

    core_ratio = core_vox / wt_vox if wt_vox > 0 else 0.0
    edema_ratio = edema_vox / wt_vox if wt_vox > 0 else 0.0
    enhancing_ratio = enhancing_vox / wt_vox if wt_vox > 0 else 0.0

    centroid = compute_centroid(wt_mask)
    if centroid is None:
        cx, cy, cz = None, None, None
        laterality = "none"
        region_proxy = "none"
    else:
        cx, cy, cz = [float(v) for v in centroid]
        laterality = get_laterality(cx, pred.shape[0])
        region_proxy = get_region_proxy(cy, cz, pred.shape)

    severity_level, severity_score = severity_rule(wt_vox, enhancing_ratio)

    result = {
        "patient_id": patient_id,
        "volumes_voxels": {
            "whole_tumor": wt_vox,
            "tumor_core": core_vox,
            "edema": edema_vox,
            "enhancing_tumor": enhancing_vox
        },
        "proportions": {
            "core_ratio": core_ratio,
            "edema_ratio": edema_ratio,
            "enhancing_ratio": enhancing_ratio
        },
        "centroid_voxel": {
            "x": cx,
            "y": cy,
            "z": cz
        },
        "location": {
            "laterality": laterality,
            "region_proxy": region_proxy
        },
        "severity": {
            "level": severity_level,
            "score": severity_score
        }
    }
    return result

TARGET_FILES = val_files   # change to test_files if needed

all_results = []

for fname in TARGET_FILES:
    npz_path = os.path.join(PREP_PATH, fname)
    patient_id = patient_id_from_npz(fname)

    img, gt, pred = predict_case(npz_path)
    result = extract_structured_findings(patient_id, pred)
    all_results.append(result)

    with open(os.path.join(JSON_DIR, f"{patient_id}.json"), "w") as f:
        json.dump(result, f, indent=2)

print(f"✅ Structured findings extracted for {len(all_results)} cases")
print("✅ JSON files saved in:", JSON_DIR)

rows = []
for r in all_results:
    rows.append({
        "patient_id": r["patient_id"],
        "whole_tumor_vox": r["volumes_voxels"]["whole_tumor"],
        "tumor_core_vox": r["volumes_voxels"]["tumor_core"],
        "edema_vox": r["volumes_voxels"]["edema"],
        "enhancing_tumor_vox": r["volumes_voxels"]["enhancing_tumor"],
        "core_ratio": r["proportions"]["core_ratio"],
        "edema_ratio": r["proportions"]["edema_ratio"],
        "enhancing_ratio": r["proportions"]["enhancing_ratio"],
        "centroid_x": r["centroid_voxel"]["x"],
        "centroid_y": r["centroid_voxel"]["y"],
        "centroid_z": r["centroid_voxel"]["z"],
        "laterality": r["location"]["laterality"],
        "region_proxy": r["location"]["region_proxy"],
        "severity_level": r["severity"]["level"],
        "severity_score": r["severity"]["score"]
    })

df_step3 = pd.DataFrame(rows)
csv_path = os.path.join(STEP3_OUT, "structured_findings_summary.csv")
df_step3.to_csv(csv_path, index=False)

print("✅ Summary CSV saved:", csv_path)
df_step3.head()

plt.figure(figsize=(8,5))
plt.hist(df_step3["whole_tumor_vox"], bins=20)
plt.title("Whole Tumor Volume Distribution")
plt.xlabel("Whole Tumor Volume (voxels)")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "whole_tumor_volume_distribution.png"), dpi=180)
plt.show()

severity_counts = df_step3["severity_level"].value_counts().reindex(["low", "medium", "high"], fill_value=0)

plt.figure(figsize=(6,4))
plt.bar(severity_counts.index, severity_counts.values)
plt.title("Severity Level Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "severity_distribution.png"), dpi=180)
plt.show()

laterality_counts = df_step3["laterality"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(laterality_counts.index, laterality_counts.values)
plt.title("Tumor Laterality Distribution")
plt.xlabel("Laterality")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "laterality_distribution.png"), dpi=180)
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(df_step3["edema_ratio"], df_step3["enhancing_ratio"], alpha=0.8)
plt.xlabel("Edema Ratio")
plt.ylabel("Enhancing Tumor Ratio")
plt.title("Tumor Subregion Proportions")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "subregion_proportion_scatter.png"), dpi=180)
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df_step3["centroid_x"], df_step3["centroid_y"], df_step3["centroid_z"], s=20)
ax.set_title("Predicted Tumor Centroid Locations")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "centroid_scatter_3d.png"), dpi=180)
plt.show()

# Visualize findings with MRI, ground-truth, and prediction overlays.
def visualize_structured_case(npz_fname):
    patient_id = patient_id_from_npz(npz_fname)
    npz_path = os.path.join(PREP_PATH, npz_fname)

    img, gt, pred = predict_case(npz_path)
    result = extract_structured_findings(patient_id, pred)

    flair = img[0]
    D = flair.shape[2]

    tumor_slices = np.where((pred > 0).reshape(-1, D).any(axis=0))[0]
    z = int(np.median(tumor_slices)) if len(tumor_slices) > 0 else D // 2

    fig, axs = plt.subplots(1, 3, figsize=(14,4))

    axs[0].imshow(flair[:, :, z], cmap="gray")
    axs[0].set_title(f"FLAIR z={z}")
    axs[0].axis("off")

    axs[1].imshow(flair[:, :, z], cmap="gray")
    axs[1].imshow(gt[:, :, z], alpha=0.5)
    axs[1].set_title("GT Overlay")
    axs[1].axis("off")

    axs[2].imshow(flair[:, :, z], cmap="gray")
    axs[2].imshow(pred[:, :, z], alpha=0.5)
    axs[2].set_title("Pred Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{patient_id}_structured_case.png"), dpi=180, bbox_inches="tight")
    plt.show()

    print(json.dumps(result, indent=2))

visualize_structured_case(TARGET_FILES[0])

import torch.nn.functional as F

