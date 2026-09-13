"""NeuroExplain Streamlit dashboard.

Run locally with: streamlit run NeuroExplain_dashboard.py
Set NEUROEXPLAIN_OUTPUT_PATH to the folder created by the pipeline.
Research prototype only: not for diagnosis, treatment, or patient care.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NeuroExplain Dashboard", layout="wide")

OUT_PATH = os.getenv("NEUROEXPLAIN_OUTPUT_PATH", "outputs")
PREP_PATH = os.path.join(OUT_PATH, "preprocessed_npz")
CKPT_PATH = os.path.join(OUT_PATH, "unet3d_best.pt")

STEP3_JSON_DIR = os.path.join(OUT_PATH, "step3_structured_findings", "json")
STEP5_REPORT_DIR = os.path.join(OUT_PATH, "step5_rag", "detailed_reports")
STEP5_TABLE_DIR = os.path.join(OUT_PATH, "step5_rag", "tables")
STEP6_JSON_DIR = os.path.join(OUT_PATH, "step6_consistency", "json")

PATCH_SIZE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIVISOR = 16  # UNet strides=(2,2,2,2)

# =========================================================
# VALIDATION
# =========================================================
if not os.path.exists(PREP_PATH):
    st.error(f"❌ preprocessed_npz not found at:\n{PREP_PATH}")
    st.stop()

if not os.path.exists(CKPT_PATH):
    st.error(f"❌ Model checkpoint not found:\n{CKPT_PATH}")
    st.stop()

# =========================================================
# HELPERS
# =========================================================
def remap_brats_labels(seg):
    seg = seg.copy()
    seg[seg == 4] = 3
    return seg

def patient_id_from_npz(fname):
    return fname.replace(".npz", "")

def pad_to_multiple_3d(x, multiple=16):
    """
    x: torch tensor (1,C,H,W,D)
    pads H,W,D to nearest multiple of `multiple`
    returns padded tensor and crop metadata
    """
    _, _, h, w, d = x.shape

    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_d = ((d + multiple - 1) // multiple) * multiple

    pad_h = new_h - h
    pad_w = new_w - w
    pad_d = new_d - d

    # F.pad for 5D uses (D_left, D_right, W_left, W_right, H_left, H_right)
    pads = (0, pad_d, 0, pad_w, 0, pad_h)
    x_pad = F.pad(x, pads, mode="constant", value=0)

    meta = {"orig_shape": (h, w, d)}
    return x_pad, meta

def crop_back_3d(arr, orig_shape):
    h, w, d = orig_shape
    return arr[:h, :w, :d]

@st.cache_resource
def load_model():
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_data
def list_cases():
    files = sorted([f for f in os.listdir(PREP_PATH) if f.endswith(".npz")])
    return [patient_id_from_npz(f) for f in files]

@st.cache_data
def load_case(case_id):
    data = np.load(os.path.join(PREP_PATH, case_id + ".npz"), allow_pickle=True)
    img = data["img"].astype(np.float32)
    seg = remap_brats_labels(data["seg"].astype(np.int64))
    return img, seg

def get_last_conv(model):
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            last_conv = m
    return last_conv

class GradCAM3D:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.activations = None
        self.gradients = None

        self.h1 = layer.register_forward_hook(self._forward_hook)
        self.h2 = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x, cls):
        """
        x: (1,C,H,W,D) original tensor
        returns cam in original spatial size
        """
        x_pad, meta = pad_to_multiple_3d(x, multiple=DIVISOR)

        self.model.zero_grad()
        out = self.model(x_pad)  # safe now

        # class score
        score = out[:, cls].mean()
        score.backward()

        acts = self.activations           # (1,C,h,w,d)
        grads = self.gradients            # (1,C,h,w,d)

        weights = grads.mean(dim=(2, 3, 4), keepdim=True)   # (1,C,1,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)     # (1,1,h,w,d)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x_pad.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam = crop_back_3d(cam, meta["orig_shape"])

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

def choose_target_class(pred):
    counts = {c: int((pred == c).sum()) for c in [1, 2, 3]}
    return max(counts, key=counts.get)

@st.cache_data
def load_structured_json(case_id):
    path = os.path.join(STEP3_JSON_DIR, f"{case_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_report_text(case_id):
    candidates = [
        os.path.join(STEP5_REPORT_DIR, f"{case_id}_detailed_report.txt"),
        os.path.join(STEP5_REPORT_DIR, f"{case_id}_report.txt"),
        os.path.join(OUT_PATH, "step5_rag", "reports", f"{case_id}_report.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                return f.read()
    return "Report not found."

@st.cache_data
def load_consistency_json(case_id):
    path = os.path.join(STEP6_JSON_DIR, f"{case_id}_consistency.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_retrieval_table(case_id):
    candidates = [
        os.path.join(STEP5_TABLE_DIR, "retrieval_results.csv"),
        os.path.join(OUT_PATH, "step5_rag", "enhanced", f"{case_id}_comparison.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "patient_id" in df.columns:
                return df[df["patient_id"] == case_id]
            return df
    return pd.DataFrame()

def make_overlay_fig(base, overlay=None, cmap=None, alpha=0.45, title=""):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(base, cmap="gray")
    if overlay is not None:
        if cmap is None:
            ax.imshow(overlay, alpha=alpha)
        else:
            ax.imshow(overlay, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    return fig

def make_subregion_bar(structured):
    vols = structured["volumes_voxels"]
    labels = ["Tumor Core", "Edema", "Enhancing"]
    values = [vols["tumor_core"], vols["edema"], vols["enhancing_tumor"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(labels, values)
    ax.set_ylabel("Voxels")
    ax.set_title("Tumor Subregion Volumes")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig

# =========================================================
# LOAD CASE + MODEL
# =========================================================
st.title("🧠 NeuroExplain Dashboard")

cases = list_cases()
if len(cases) == 0:
    st.error("No NPZ cases found.")
    st.stop()

case = st.sidebar.selectbox("Select Case", cases)

modality_map = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
modality_name = st.sidebar.selectbox("MRI Modality", list(modality_map.keys()))
modality_idx = modality_map[modality_name]

img, gt = load_case(case)
model = load_model()

x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = sliding_window_inference(x, PATCH_SIZE, 1, model)
    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

default_class = choose_target_class(pred)
target_class = st.sidebar.selectbox("Grad-CAM Target Class", [1, 2, 3], index=[1, 2, 3].index(default_class))

gradcam = GradCAM3D(model, get_last_conv(model))
cam = gradcam.generate(x, cls=target_class)
gradcam.remove()

D = img.shape[3]
tumor_slices = np.where((pred > 0).reshape(-1, D).any(axis=0))[0]
default_slice = int(np.median(tumor_slices)) if len(tumor_slices) > 0 else D // 2
z = st.sidebar.slider("Slice", 0, D - 1, default_slice)

base = img[modality_idx, :, :, z]
structured = load_structured_json(case)
report_text = load_report_text(case)
consistency = load_consistency_json(case)
retrieval_df = load_retrieval_table(case)

# =========================================================
# MAIN VIEW
# =========================================================
st.markdown(f"**Case:** {case}  |  **Modality:** {modality_name}  |  **Slice:** {z}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.pyplot(make_overlay_fig(base, title="MRI"))
with c2:
    st.pyplot(make_overlay_fig(base, gt[:, :, z], alpha=0.45, title="Ground Truth Overlay"))
with c3:
    st.pyplot(make_overlay_fig(base, pred[:, :, z], alpha=0.45, title="Prediction Overlay"))
with c4:
    st.pyplot(make_overlay_fig(base, cam[:, :, z], cmap="jet", alpha=0.45, title=f"Grad-CAM Class {target_class}"))

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Structured Findings")
    if structured:
        st.metric("Whole Tumor Volume", structured["volumes_voxels"]["whole_tumor"])
        st.metric("Laterality", structured["location"]["laterality"])
        st.metric("Region Proxy", structured["location"]["region_proxy"])
        st.metric("Severity", structured["severity"]["level"])
        st.pyplot(make_subregion_bar(structured))

        centroid = structured["centroid_voxel"]
        st.markdown(f"**Centroid:** ({centroid['x']}, {centroid['y']}, {centroid['z']})")

        props = structured["proportions"]
        st.markdown(
            f"""
            **Subregion Proportions**
            - Core Ratio: {props['core_ratio']:.3f}
            - Edema Ratio: {props['edema_ratio']:.3f}
            - Enhancing Ratio: {props['enhancing_ratio']:.3f}
            """
        )
    else:
        st.warning("Step 3 JSON not found.")

with right:
    st.subheader("Generated Report")
    st.text_area("Report", report_text, height=420)

    st.download_button(
        label="Download Report (.txt)",
        data=report_text,
        file_name=f"{case}_report.txt",
        mime="text/plain"
    )

left2, right2 = st.columns([1, 1])

with left2:
    st.subheader("Consistency Verification")
    if consistency:
        st.metric("Consistency Score", f"{consistency['consistency_score']:.2f}")
        st.metric("Consistent Fields", consistency["consistent_count"])
        st.metric("Missing Fields", consistency["missing_count"])
        st.metric("Inconsistent Fields", consistency["inconsistent_count"])

        with st.expander("Checks"):
            st.json(consistency["checks"])
    else:
        st.warning("Step 6 consistency output not found.")

with right2:
    st.subheader("Retrieved Similar Cases")
    if not retrieval_df.empty:
        st.dataframe(retrieval_df, use_container_width=True)
        if "score" in retrieval_df.columns and "retrieved_doc_id" in retrieval_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(retrieval_df["retrieved_doc_id"], retrieval_df["score"])
            ax.set_title("Retrieval Similarity Scores")
            ax.set_ylabel("Score")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Step 5 retrieval table not found.")

st.subheader("Combined Prediction + Grad-CAM")
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(base, cmap="gray")
ax.imshow(pred[:, :, z], alpha=0.30)
ax.imshow(cam[:, :, z], cmap="jet", alpha=0.30)
ax.axis("off")
ax.set_title(f"{modality_name} with Prediction + Grad-CAM")
st.pyplot(fig)
