"""Shared configuration and artifact-loading helpers for the split NeuroExplain pipeline."""
from __future__ import annotations

import json
import os

import torch
from monai.networks.nets import UNet

OUT_PATH = os.getenv("NEUROEXPLAIN_OUTPUT_PATH", "outputs")
PREP_PATH = os.path.join(OUT_PATH, "preprocessed_npz")
SPLIT_JSON = os.path.join(OUT_PATH, "split.json")
CKPT_PATH = os.path.join(OUT_PATH, "unet3d_best.pt")
PATCH_SIZE = (96, 96, 96)


def load_split() -> dict:
    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(f"Missing split file: {SPLIT_JSON}. Run Step 1 first.")
    with open(SPLIT_JSON, "r", encoding="utf-8") as file:
        return json.load(file)


def load_final_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing final 3D checkpoint: {CKPT_PATH}. Run Step 2B first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model, device
