# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.








import os
try:
    from IPython.display import display
except ImportError:
    display = print

# PROJECT CONFIGURATION: Read dataset location from an environment variable so local paths stay private.
DATASET_PATH = os.getenv("BRATS_DATASET_PATH", "path/to/BraTS2020_TrainingData")

patients = sorted(os.listdir(DATASET_PATH))
print("Number of patients:", len(patients))
print("First 5 folders:", patients[:5])


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

patient_id = patients[0]
patient_path = os.path.join(DATASET_PATH, patient_id)

print("Loading:", patient_id)


print(f"Contents of {patient_path}:")
if os.path.exists(patient_path):
    for item in os.listdir(patient_path):
        print(f"  - {item}")
else:
    print(f"  - Directory does not exist: {patient_path}")

flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii")).get_fdata()
t1 = nib.load(os.path.join(patient_path, f"{patient_id}_t1.nii")).get_fdata()
t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii")).get_fdata()
t2 = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii")).get_fdata()
seg = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii")).get_fdata()

print("FLAIR shape:", flair.shape)
print("Segmentation shape:", seg.shape)


slice_index = flair.shape[2] // 2

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(flair[:,:,slice_index], cmap='gray')
plt.title("FLAIR")

plt.subplot(1,2,2)
plt.imshow(seg[:,:,slice_index])
plt.title("Ground Truth Segmentation")

plt.show()


import json, random
import pandas as pd
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



OUT_PATH = os.getenv("NEUROEXPLAIN_OUTPUT_PATH", "outputs")
os.makedirs(OUT_PATH, exist_ok=True)

# STEP 1: Create folders for exploratory data analysis and preprocessing artifacts.
EDA_PATH = os.path.join(OUT_PATH, "eda")
EDA_EXTRA_PATH = os.path.join(OUT_PATH, "eda_extras")
PREP_PATH = os.path.join(OUT_PATH, "preprocessed_npz")

for p in [EDA_PATH, EDA_EXTRA_PATH, PREP_PATH]:
    os.makedirs(p, exist_ok=True)

print("DATASET_PATH:", DATASET_PATH)
print("OUT_PATH:", OUT_PATH)

# Return the available BraTS case folders from the dataset directory.
def list_patient_dirs(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

# Build and validate the expected MRI-modality and segmentation file paths for one case.
def find_modality_files(patient_dir, pid):
    expected = {
        "flair": os.path.join(patient_dir, f"{pid}_flair.nii"),
        "t1":    os.path.join(patient_dir, f"{pid}_t1.nii"),
        "t1ce":  os.path.join(patient_dir, f"{pid}_t1ce.nii"),
        "t2":    os.path.join(patient_dir, f"{pid}_t2.nii"),
        "seg":   os.path.join(patient_dir, f"{pid}_seg.nii"),
    }
    status = {k: os.path.exists(v) for k, v in expected.items()}
    return expected, status

# Load a NIfTI volume as a float32 NumPy array and retain its affine metadata.
def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine

# Calculate non-background intensity statistics for EDA and quality checks.
def intensity_summary(vol):
    nonzero = vol[vol != 0]
    if nonzero.size == 0:
        return dict(mean=0.0, std=0.0, p1=0.0, p99=0.0, min=0.0, max=0.0)
    return dict(
        mean=float(nonzero.mean()),
        std=float(nonzero.std()),
        p1=float(np.percentile(nonzero, 1)),
        p99=float(np.percentile(nonzero, 99)),
        min=float(nonzero.min()),
        max=float(nonzero.max())
    )

# Plot the intensity distribution for non-background voxels in one volume.
def plot_hist(vol, title):
    nonzero = vol[vol != 0]
    plt.figure(figsize=(6,4))
    plt.hist(nonzero.flatten(), bins=80)
    plt.title(title)
    plt.xlabel("Intensity (non-zero voxels)")
    plt.ylabel("Count")
    plt.show()

# Count segmentation voxels by class label.
def seg_label_counts(seg):
    labels, counts = np.unique(seg.astype(np.int32), return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}

# Measure how many axial slices contain tumor voxels.
def tumor_slice_stats(seg):
    D = seg.shape[2]
    tumor_per_slice = (seg != 0).reshape(-1, D).any(axis=0)
    tumor_slices = int(tumor_per_slice.sum())
    total_slices = int(D)
    ratio = tumor_slices / total_slices if total_slices else 0.0
    return tumor_slices, total_slices, ratio

# Display all four MRI modalities with the ground-truth segmentation for one case.
def show_case(pid, slice_strategy="tumor_if_possible"):
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)

    flair, _ = load_nii(paths["flair"])
    t1, _    = load_nii(paths["t1"])
    t1ce, _  = load_nii(paths["t1ce"])
    t2, _    = load_nii(paths["t2"])
    seg, _   = load_nii(paths["seg"])

    D = flair.shape[2]
    if slice_strategy == "tumor_if_possible":
        tumor_slices = np.where((seg != 0).reshape(-1, D).any(axis=0))[0]
        z = int(np.median(tumor_slices)) if len(tumor_slices) else D // 2
    else:
        z = D // 2

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.ravel()

    axs[0].imshow(flair[:,:,z], cmap="gray"); axs[0].set_title("FLAIR")
    axs[1].imshow(t1[:,:,z], cmap="gray");    axs[1].set_title("T1")
    axs[2].imshow(t1ce[:,:,z], cmap="gray");  axs[2].set_title("T1CE")
    axs[3].imshow(t2[:,:,z], cmap="gray");    axs[3].set_title("T2")

    axs[4].imshow(flair[:,:,z], cmap="gray")
    axs[4].imshow(seg[:,:,z], alpha=0.5)
    axs[4].set_title("FLAIR + SEG overlay")

    axs[5].imshow(seg[:,:,z]); axs[5].set_title("SEG labels")

    for ax in axs: ax.axis("off")
    plt.suptitle(f"{pid} | slice z={z}", y=0.98)
    plt.tight_layout()
    plt.show()

patients = list_patient_dirs(DATASET_PATH)
print("Patients found:", len(patients))
print("Example:", patients[:3])

rows = []
for pid in tqdm(patients):
    pdir = os.path.join(DATASET_PATH, pid)
    _, status = find_modality_files(pdir, pid)
    rows.append({
        "patient_id": pid,
        **{f"has_{k}": v for k, v in status.items()},
        "all_present": all(status.values())
    })

inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(OUT_PATH, "inventory.csv"), index=False)

print(inv["all_present"].value_counts())
good_patients = inv[inv["all_present"] == True]["patient_id"].tolist()
print("Valid cases:", len(good_patients))

inv[inv["all_present"] == False].head()

sample = random.choice(good_patients)
print("Sample patient:", sample)
show_case(sample, slice_strategy="tumor_if_possible")

pdir = os.path.join(DATASET_PATH, sample)
paths, _ = find_modality_files(pdir, sample)
seg, _ = load_nii(paths["seg"])
print("Unique seg labels in sample:", np.unique(seg).astype(int))

from mpl_toolkits.mplot3d import Axes3D

# STEP 1A: Generate required dataset-level EDA figures and tables.
EDA_REQ_PATH = os.path.join(OUT_PATH, "eda_required")
os.makedirs(EDA_REQ_PATH, exist_ok=True)

EDA_N = min(150, len(good_patients))
EDA_PATIENTS = random.sample(good_patients, EDA_N)

# Validate and return the input paths needed for a single case.
def get_case_paths(pid):
    pdir = os.path.join(DATASET_PATH, pid)
    paths, status = find_modality_files(pdir, pid)
    if not all(status.values()):
        raise FileNotFoundError(f"Missing files for {pid}")
    return paths

# Return non-background segmentation labels present in a mask.
def safe_unique_labels(seg):
    labels = np.unique(seg.astype(np.int32))
    labels = labels[labels != 0]
    return [int(x) for x in labels]

# Compute the voxel-space centroid for the whole tumor or a specific label.
def tumor_centroid(seg, label=None):
    if label is None:
        coords = np.argwhere(seg != 0)
    else:
        coords = np.argwhere(seg == label)
    if coords.size == 0:
        return None
    c = coords.mean(axis=0)  # (x,y,z)
    return tuple(c.tolist())

# Save a multimodal MRI montage with segmentation overlays for EDA.
def save_multimodal_overlay(pid, out_dir, slice_strategy="tumor_if_possible"):
    paths = get_case_paths(pid)
    flair, _ = load_nii(paths["flair"])
    t1, _    = load_nii(paths["t1"])
    t1ce, _  = load_nii(paths["t1ce"])
    t2, _    = load_nii(paths["t2"])
    seg, _   = load_nii(paths["seg"])

    D = flair.shape[2]
    if slice_strategy == "tumor_if_possible":
        tumor_slices = np.where((seg != 0).reshape(-1, D).any(axis=0))[0]
        z = int(np.median(tumor_slices)) if len(tumor_slices) else D // 2
    else:
        z = D // 2

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()

    imgs = [flair[:,:,z], t1[:,:,z], t1ce[:,:,z], t2[:,:,z]]
    titles = ["FLAIR", "T1", "T1CE", "T2"]

    for i in range(4):
        axs[i].imshow(imgs[i], cmap="gray")
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    for i in range(4):
        axs[i+4].imshow(imgs[i], cmap="gray")
        axs[i+4].imshow(seg[:,:,z], alpha=0.5)
        axs[i+4].set_title(f"{titles[i]} + SEG")
        axs[i+4].axis("off")

    plt.suptitle(f"{pid} | z={z}", y=0.98)
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"{pid}_multimodal_overlay.png")
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return out_file

for pid in random.sample(good_patients, min(3, len(good_patients))):
    f = save_multimodal_overlay(pid, EDA_REQ_PATH)
    print("Saved:", f)

vol_rows = []
all_labels = set()

for pid in tqdm(EDA_PATIENTS):
    paths = get_case_paths(pid)
    seg, _ = load_nii(paths["seg"])
    labels = safe_unique_labels(seg)
    all_labels.update(labels)

    counts = seg_label_counts(seg)  # {label: voxels}
    row = {"patient_id": pid}
    for l, c in counts.items():
        if l != 0:
            row[f"label_{l}"] = c
    vol_rows.append(row)

vol_df = pd.DataFrame(vol_rows).fillna(0)
vol_df.to_csv(os.path.join(EDA_REQ_PATH, "subregion_voxel_counts.csv"), index=False)

labels_sorted = sorted(list(all_labels))
data_for_plot = [vol_df[f"label_{l}"].values for l in labels_sorted]

plt.figure(figsize=(10,5))
plt.boxplot(data_for_plot, tick_labels=[str(l) for l in labels_sorted], showfliers=True)
plt.title("Tumor sub-region voxel distributions (boxplot)")
plt.xlabel("Segmentation Label (sub-region)")
plt.ylabel("Voxel Count")
plt.tight_layout()
plt.savefig(os.path.join(EDA_REQ_PATH, "subregion_volume_boxplot.png"), dpi=180, bbox_inches="tight")
plt.show()

prop_rows = []
for _, row in vol_df.iterrows():
    tumor_total = sum(row.get(f"label_{l}", 0) for l in labels_sorted)
    if tumor_total == 0:
        continue
    props = {l: row.get(f"label_{l}", 0) / tumor_total for l in labels_sorted}
    prop_rows.append(props)

prop_df = pd.DataFrame(prop_rows).fillna(0)
avg_props = prop_df.mean(axis=0)  # index = labels
avg_props = avg_props.sort_index()

plt.figure(figsize=(10,3))
left = 0.0
for l in avg_props.index:
    val = float(avg_props[l])
    plt.barh([0], [val], left=left, label=f"Label {l}")
    left += val

plt.xlim(0, 1)
plt.yticks([])
plt.title("Average proportion of tumor sub-regions (stacked bar)")
plt.xlabel("Proportion within tumor voxels")
plt.legend(ncol=min(4, len(avg_props.index)), bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(EDA_REQ_PATH, "avg_subregion_proportion_stackedbar.png"), dpi=180, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6,6))
plt.pie([float(avg_props[l]) for l in avg_props.index],
        labels=[f"Label {l}" for l in avg_props.index],
        autopct="%1.1f%%")
plt.title("Average proportion of tumor sub-regions (pie)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_REQ_PATH, "avg_subregion_proportion_pie.png"), dpi=180, bbox_inches="tight")
plt.show()

mods = ["flair", "t1", "t1ce", "t2"]
sample_per_case = 20000  # voxels sampled per modality per case (tweak for speed)

collected = {m: [] for m in mods}

for pid in tqdm(EDA_PATIENTS):
    paths = get_case_paths(pid)
    for m in mods:
        vol, _ = load_nii(paths[m])
        nz = vol[vol != 0]
        if nz.size == 0:
            continue
        if nz.size > sample_per_case:
            idx = np.random.choice(nz.size, sample_per_case, replace=False)
            nz = nz[idx]
        collected[m].append(nz.astype(np.float32))

samples = {m: np.concatenate(collected[m]) if len(collected[m]) else np.array([]) for m in mods}

plt.figure(figsize=(10,5))
for m in mods:
    x = samples[m]
    if x.size == 0:
        continue
    plt.hist(x, bins=120, alpha=0.35, density=True, label=m)

plt.title("Intensity distribution comparison across modalities (non-zero voxels)")
plt.xlabel("Intensity")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EDA_REQ_PATH, "intensity_distribution_modalities.png"), dpi=180, bbox_inches="tight")
plt.show()

centroids = []
for pid in tqdm(EDA_PATIENTS):
    paths = get_case_paths(pid)
    seg, _ = load_nii(paths["seg"])
    c = tumor_centroid(seg, label=None)  # centroid of all tumor voxels
    if c is None:
        continue
    centroids.append({"patient_id": pid, "cx": c[0], "cy": c[1], "cz": c[2]})

cent_df = pd.DataFrame(centroids)
cent_df.to_csv(os.path.join(EDA_REQ_PATH, "tumor_centroids.csv"), index=False)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(cent_df["cx"], cent_df["cy"], cent_df["cz"], s=12)

ax.set_title("3D scatter: tumor centroid locations (voxel coordinates)")
ax.set_xlabel("X (voxel)")
ax.set_ylabel("Y (voxel)")
ax.set_zlabel("Z (voxel)")

plt.tight_layout()
plt.savefig(os.path.join(EDA_REQ_PATH, "tumor_centroids_3d_scatter.png"), dpi=180, bbox_inches="tight")
plt.show()

pdir = os.path.join(DATASET_PATH, sample)
paths, _ = find_modality_files(pdir, sample)

mods = ["flair","t1","t1ce","t2"]

for m in mods:
    vol, _ = load_nii(paths[m])
    stats = intensity_summary(vol)
    print(f"{sample} | {m} stats:", stats)
    plot_hist(vol, f"{sample} - {m} intensity histogram (non-zero)")

eda_rows = []
for pid in tqdm(good_patients[:370]):  # increase later
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)
    seg, _ = load_nii(paths["seg"])

    counts = seg_label_counts(seg)
    total_tumor = int((seg != 0).sum())

    eda_rows.append({
        "patient_id": pid,
        "total_tumor_voxels": total_tumor,
        **{f"label_{k}_voxels": v for k, v in counts.items()}
    })

tumor_df = pd.DataFrame(eda_rows).fillna(0)
tumor_df.to_csv(os.path.join(EDA_PATH, "tumor_volume_stats_sample.csv"), index=False)

plt.figure(figsize=(7,4))
plt.hist(tumor_df["total_tumor_voxels"], bins=50)
plt.title("Tumor volume proxy distribution (voxel count)")
plt.xlabel("Tumor voxels (seg != 0)")
plt.ylabel("Patients")
plt.show()

print("Patients with zero tumor voxels:", int((tumor_df["total_tumor_voxels"]==0).sum()))
tumor_df.head()

rows = []
for pid in tqdm(good_patients[:370]):  # increase later
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)
    seg, _ = load_nii(paths["seg"])

    tumor_slices, total_slices, ratio = tumor_slice_stats(seg)
    rows.append({
        "patient_id": pid,
        "tumor_slices": tumor_slices,
        "total_slices": total_slices,
        "tumor_slice_ratio": ratio,
        "total_tumor_voxels": int((seg != 0).sum())
    })

slice_df = pd.DataFrame(rows)
slice_df.to_csv(os.path.join(EDA_EXTRA_PATH, "slice_tumor_stats_sample.csv"), index=False)

plt.figure(figsize=(7,4))
plt.hist(slice_df["tumor_slices"], bins=40)
plt.title("Tumor slices per patient")
plt.xlabel("Number of slices containing tumor")
plt.ylabel("Patients")
plt.show()

plt.figure(figsize=(7,4))
plt.hist(slice_df["tumor_slice_ratio"], bins=40)
plt.title("Tumor slice ratio per patient")
plt.xlabel("Tumor slices / total slices")
plt.ylabel("Patients")
plt.show()

slice_df.head()

global_counts = {}
per_patient_counts = []

for pid in tqdm(good_patients[:370]):  # increase later
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)
    seg, _ = load_nii(paths["seg"])

    c = seg_label_counts(seg)
    per_patient_counts.append({"patient_id": pid, **{f"label_{k}": v for k,v in c.items()}})

    for k,v in c.items():
        global_counts[k] = global_counts.get(k, 0) + v

per_label_df = pd.DataFrame(per_patient_counts).fillna(0)
per_label_df.to_csv(os.path.join(EDA_EXTRA_PATH, "per_patient_label_counts_sample.csv"), index=False)

global_counts_sorted = dict(sorted(global_counts.items(), key=lambda x: x[0]))
print("Global label voxel counts (sample):", global_counts_sorted)

plt.figure(figsize=(6,4))
plt.bar(list(global_counts_sorted.keys()), list(global_counts_sorted.values()))
plt.title("Global label voxel counts (sample)")
plt.xlabel("Label")
plt.ylabel("Voxel count")
plt.show()

# Standardize a numeric series for outlier detection.
def zscore_series(x):
    return (x - x.mean()) / (x.std() + 1e-8)

rows = []
for pid in tqdm(good_patients[:370]):  # increase later
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)

    row = {"patient_id": pid}
    for m in mods:
        v, _ = load_nii(paths[m])
        s = intensity_summary(v)
        for k,val in s.items():
            row[f"{m}_{k}"] = val
    rows.append(row)

int_df = pd.DataFrame(rows)
int_df.to_csv(os.path.join(EDA_EXTRA_PATH, "intensity_stats_sample.csv"), index=False)

flag_df = pd.DataFrame({"patient_id": int_df["patient_id"]})
for m in mods:
    for metric in ["mean","std","p99"]:
        col = f"{m}_{metric}"
        z = zscore_series(int_df[col])
        flag_df[f"outlier_{col}"] = (z.abs() > 3.0)

flag_df["any_outlier"] = flag_df.drop(columns=["patient_id"]).any(axis=1)
outliers = flag_df[flag_df["any_outlier"]].merge(int_df, on="patient_id")
outliers.to_csv(os.path.join(EDA_EXTRA_PATH, "intensity_outliers_sample.csv"), index=False)

print("Outlier cases found:", len(outliers))
outliers.head(10)

# Compute cross-modality correlation using shared non-background voxels.
def modality_corr(pid):
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)
    vols = {m: load_nii(paths[m])[0] for m in mods}

    mask = np.ones_like(vols["flair"], dtype=bool)
    for m in mods:
        mask &= (vols[m] != 0)

    if mask.sum() < 1000:
        return None

    X = np.stack([vols[m][mask] for m in mods], axis=0)  # (4, N)
    return np.corrcoef(X)

corr_rows = []
sample_ids = random.sample(good_patients, min(20, len(good_patients)))
for pid in sample_ids:
    c = modality_corr(pid)
    if c is not None:
        corr_rows.append(c)

if corr_rows:
    avg_corr = np.mean(corr_rows, axis=0)

    plt.figure(figsize=(5,4))
    plt.imshow(avg_corr, vmin=-1, vmax=1)
    plt.xticks(range(len(mods)), mods, rotation=45)
    plt.yticks(range(len(mods)), mods)
    plt.title("Average modality correlation (non-zero voxels)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    np.save(os.path.join(EDA_EXTRA_PATH, "avg_modality_corr.npy"), avg_corr)
else:
    print("Not enough valid overlap to compute correlations.")

MONTAGE_DIR = os.path.join(EDA_EXTRA_PATH, "montages")
os.makedirs(MONTAGE_DIR, exist_ok=True)

# Save a representative six-panel MRI and segmentation montage.
def save_case_montage(pid, out_dir):
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)

    flair, _ = load_nii(paths["flair"])
    t1, _    = load_nii(paths["t1"])
    t1ce, _  = load_nii(paths["t1ce"])
    t2, _    = load_nii(paths["t2"])
    seg, _   = load_nii(paths["seg"])

    D = flair.shape[2]
    tumor_slices = np.where((seg != 0).reshape(-1, D).any(axis=0))[0]
    z = int(np.median(tumor_slices)) if len(tumor_slices) else D // 2

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.ravel()

    axs[0].imshow(flair[:,:,z], cmap="gray"); axs[0].set_title("FLAIR")
    axs[1].imshow(t1[:,:,z], cmap="gray");    axs[1].set_title("T1")
    axs[2].imshow(t1ce[:,:,z], cmap="gray");  axs[2].set_title("T1CE")
    axs[3].imshow(t2[:,:,z], cmap="gray");    axs[3].set_title("T2")

    axs[4].imshow(flair[:,:,z], cmap="gray")
    axs[4].imshow(seg[:,:,z], alpha=0.5)
    axs[4].set_title("FLAIR + SEG overlay")

    axs[5].imshow(seg[:,:,z]); axs[5].set_title("SEG labels")

    for ax in axs: ax.axis("off")
    plt.suptitle(f"{pid} | slice z={z}", y=0.98)
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"{pid}_montage.png")
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_file

montage_ids = random.sample(good_patients, min(10, len(good_patients)))
saved_files = [save_case_montage(pid, MONTAGE_DIR) for pid in tqdm(montage_ids)]

print("Saved montages:", len(saved_files))
print("Folder:", MONTAGE_DIR)

# Normalize only non-background MRI voxels while preserving zero background.
def zscore_nonzero(x, eps=1e-8):
    mask = x != 0
    if mask.sum() == 0:
        return x.astype(np.float32)
    mean = x[mask].mean()
    std  = x[mask].std()
    return np.where(mask, (x - mean) / (std + eps), 0).astype(np.float32)

# Find a padded 3D bounding box around non-background voxels.
def bbox_nonzero(vol, margin=5):
    coords = np.array(np.where(vol != 0))
    if coords.size == 0:
        return (0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2])
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin + 1, np.array(vol.shape))
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

# Normalize, crop, and save one multimodal case as a compressed NPZ file.
def preprocess_case(pid, save_dir, do_crop=True):
    pdir = os.path.join(DATASET_PATH, pid)
    paths, _ = find_modality_files(pdir, pid)

    flair, _ = load_nii(paths["flair"])
    t1, _    = load_nii(paths["t1"])
    t1ce, _  = load_nii(paths["t1ce"])
    t2, _    = load_nii(paths["t2"])
    seg, _   = load_nii(paths["seg"])

    flair = zscore_nonzero(flair)
    t1    = zscore_nonzero(t1)
    t1ce  = zscore_nonzero(t1ce)
    t2    = zscore_nonzero(t2)

    img = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.float32)  # (4,H,W,D)
    seg = seg.astype(np.int16)

    if do_crop:
        bb = bbox_nonzero(flair, margin=5)
        x1,x2,y1,y2,z1,z2 = bb
        img = img[:, x1:x2, y1:y2, z1:z2]
        seg = seg[x1:x2, y1:y2, z1:z2]
    else:
        bb = None

    out_file = os.path.join(save_dir, f"{pid}.npz")
    np.savez_compressed(out_file, img=img, seg=seg, bbox=bb)
    return out_file, img.shape, seg.shape

out_file, img_shape, seg_shape = preprocess_case(sample, PREP_PATH, do_crop=True)
print("Saved:", out_file)
print("Image shape:", img_shape, "| Seg shape:", seg_shape)

data = np.load(out_file, allow_pickle=True)
img = data["img"]   # (4,H,W,D)
seg = data["seg"]

z = img.shape[-1] // 2
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img[0,:,:,z], cmap="gray"); plt.title("FLAIR (norm+crop)"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(img[2,:,:,z], cmap="gray"); plt.title("T1CE (norm+crop)"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(img[0,:,:,z], cmap="gray"); plt.imshow(seg[:,:,z], alpha=0.5); plt.title("Overlay"); plt.axis("off")
plt.tight_layout(); plt.show()

print("Unique seg labels:", np.unique(seg).astype(int))

# STEP 1B: Preprocess all valid cases and persist compressed model-ready volumes.
to_process = good_patients[:370]  # ✅ change to good_patients when confident

log = []
for pid in tqdm(to_process):
    try:
        out_file, img_shape, seg_shape = preprocess_case(pid, PREP_PATH, do_crop=True)
        log.append({"patient_id": pid, "out_file": out_file, "img_shape": str(img_shape), "seg_shape": str(seg_shape)})
    except Exception as e:
        log.append({"patient_id": pid, "error": str(e)})

log_df = pd.DataFrame(log)
log_df.to_csv(os.path.join(OUT_PATH, "preprocess_log.csv"), index=False)

print("Saved npz:", int(log_df["out_file"].notna().sum()))
if "error" in log_df.columns:
    print("Errors:", int(log_df["error"].notna().sum()))
else:
    print("Errors:", 0)

# STEP 1C: Create a reproducible train, validation, and test split.
all_npz = sorted([f for f in os.listdir(PREP_PATH) if f.endswith(".npz")])
random.shuffle(all_npz)

n = len(all_npz)
train = all_npz[:int(0.7*n)]
val   = all_npz[int(0.7*n):int(0.85*n)]
test  = all_npz[int(0.85*n):]

split = {"train": train, "val": val, "test": test}
with open(os.path.join(OUT_PATH, "split.json"), "w") as f:
    json.dump(split, f, indent=2)

print({k: len(v) for k, v in split.items()})
print("Saved:", os.path.join(OUT_PATH, "split.json"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

