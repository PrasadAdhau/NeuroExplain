# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_PATH

JSON_DIR = os.path.join(OUT_PATH, "step3_structured_findings", "json")
DETAILED_REPORT_DIR = os.path.join(OUT_PATH, "step5_rag", "detailed_reports")

# STEP 6: Verify report claims against the structured findings.
STEP6_OUT = os.path.join(OUT_PATH, "step6_consistency")
STEP6_JSON = os.path.join(STEP6_OUT, "json")
STEP6_TABLES = os.path.join(STEP6_OUT, "tables")
STEP6_VIS = os.path.join(STEP6_OUT, "visuals")

os.makedirs(STEP6_OUT, exist_ok=True)
os.makedirs(STEP6_JSON, exist_ok=True)
os.makedirs(STEP6_TABLES, exist_ok=True)
os.makedirs(STEP6_VIS, exist_ok=True)

print("✅ Step 6 output:", STEP6_OUT)
print("✅ Structured input:", JSON_DIR)
print("✅ Report input:", DETAILED_REPORT_DIR)

structured_map = {}
for fname in sorted(os.listdir(JSON_DIR)):
    if fname.endswith(".json"):
        with open(os.path.join(JSON_DIR, fname), "r") as f:
            case = json.load(f)
            structured_map[case["patient_id"]] = case

print("Structured cases loaded:", len(structured_map))

report_map = {}
for fname in sorted(os.listdir(DETAILED_REPORT_DIR)):
    if fname.endswith("_detailed_report.txt"):
        patient_id = fname.replace("_detailed_report.txt", "")
        with open(os.path.join(DETAILED_REPORT_DIR, fname), "r") as f:
            report_map[patient_id] = f.read()

print("Reports loaded:", len(report_map))

common_patients = sorted(list(set(structured_map.keys()) & set(report_map.keys())))
print("Patients available for Step 6:", len(common_patients))
print(common_patients[:5])

# Standardize report text before extracting verification claims.
def normalize_text(text):
    return text.lower().strip()

# Parse report fields that can be checked against structured findings.
def extract_report_claims(report_text):
    text = normalize_text(report_text)

    claims = {
        "severity": None,
        "laterality": None,
        "region_proxy": None,
        "whole_tumor_volume": None,
        "core_ratio": None,
        "edema_ratio": None,
        "enhancing_ratio": None
    }

    for sev in ["low", "medium", "high"]:
        if f"severity level: {sev}" in text or f"{sev} severity" in text:
            claims["severity"] = sev
            break

    for lat in ["left", "right", "midline"]:
        if f"{lat} hemisphere" in text:
            claims["laterality"] = lat
            break

    region_candidates = [
        "anterior-superior", "anterior-inferior",
        "posterior-superior", "posterior-inferior"
    ]
    for region in region_candidates:
        if region in text:
            claims["region_proxy"] = region
            break

    m = re.search(r"whole tumor volume of\s+(\d+)\s+voxels", text)
    if m:
        claims["whole_tumor_volume"] = int(m.group(1))

    m = re.search(r"core ratio:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        claims["core_ratio"] = float(m.group(1))

    m = re.search(r"edema ratio:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        claims["edema_ratio"] = float(m.group(1))

    m = re.search(r"enhancing tumor ratio:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        claims["enhancing_ratio"] = float(m.group(1))

    return claims

# Compare numeric claims using a relative tolerance.
def approx_equal(a, b, tolerance=0.05):
    if a is None or b is None:
        return False
    if b == 0:
        return abs(a - b) < 1e-8
    return abs(a - b) / max(abs(b), 1e-8) <= tolerance

# Check whether report claims match the source structured findings.
def verify_case_consistency(structured_case, report_text):
    claims = extract_report_claims(report_text)

    expected = {
        "severity": structured_case["severity"]["level"],
        "laterality": structured_case["location"]["laterality"],
        "region_proxy": structured_case["location"]["region_proxy"],
        "whole_tumor_volume": structured_case["volumes_voxels"]["whole_tumor"],
        "core_ratio": structured_case["proportions"]["core_ratio"],
        "edema_ratio": structured_case["proportions"]["edema_ratio"],
        "enhancing_ratio": structured_case["proportions"]["enhancing_ratio"]
    }

    checks = {}
    missing = []
    unsupported = []

    for field in ["severity", "laterality", "region_proxy"]:
        if claims[field] is None:
            checks[field] = "missing"
            missing.append(field)
        elif claims[field] == expected[field]:
            checks[field] = "consistent"
        else:
            checks[field] = "inconsistent"
            unsupported.append(field)

    for field in ["whole_tumor_volume", "core_ratio", "edema_ratio", "enhancing_ratio"]:
        if claims[field] is None:
            checks[field] = "missing"
            missing.append(field)
        elif approx_equal(claims[field], expected[field], tolerance=0.10):
            checks[field] = "consistent"
        else:
            checks[field] = "inconsistent"
            unsupported.append(field)

    total_fields = len(checks)
    consistent_count = sum(1 for v in checks.values() if v == "consistent")
    missing_count = sum(1 for v in checks.values() if v == "missing")
    inconsistent_count = sum(1 for v in checks.values() if v == "inconsistent")

    consistency_score = consistent_count / total_fields if total_fields > 0 else 0.0

    result = {
        "patient_id": structured_case["patient_id"],
        "claims": claims,
        "expected": expected,
        "checks": checks,
        "consistent_count": consistent_count,
        "missing_count": missing_count,
        "inconsistent_count": inconsistent_count,
        "consistency_score": consistency_score,
        "missing_fields": missing,
        "unsupported_fields": unsupported
    }
    return result

verification_results = []

for pid in common_patients:
    structured_case = structured_map[pid]
    report_text = report_map[pid]

    result = verify_case_consistency(structured_case, report_text)
    verification_results.append(result)

    with open(os.path.join(STEP6_JSON, f"{pid}_consistency.json"), "w") as f:
        json.dump(result, f, indent=2)

print("✅ Verified", len(verification_results), "cases")

rows = []
for r in verification_results:
    row = {
        "patient_id": r["patient_id"],
        "consistency_score": r["consistency_score"],
        "consistent_count": r["consistent_count"],
        "missing_count": r["missing_count"],
        "inconsistent_count": r["inconsistent_count"],
        "severity_check": r["checks"]["severity"],
        "laterality_check": r["checks"]["laterality"],
        "region_proxy_check": r["checks"]["region_proxy"],
        "whole_tumor_volume_check": r["checks"]["whole_tumor_volume"],
        "core_ratio_check": r["checks"]["core_ratio"],
        "edema_ratio_check": r["checks"]["edema_ratio"],
        "enhancing_ratio_check": r["checks"]["enhancing_ratio"]
    }
    rows.append(row)

df_step6 = pd.DataFrame(rows)
csv_path = os.path.join(STEP6_TABLES, "consistency_summary.csv")
df_step6.to_csv(csv_path, index=False)

print("✅ Summary CSV saved:", csv_path)
df_step6.head()

plt.figure(figsize=(8, 4))
plt.bar(df_step6["patient_id"], df_step6["consistency_score"])
plt.title("Report Consistency Score by Patient")
plt.xlabel("Patient ID")
plt.ylabel("Consistency Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(STEP6_VIS, "consistency_scores.png"), dpi=180)
plt.show()

x = np.arange(len(df_step6))
width = 0.35

plt.figure(figsize=(9, 4))
plt.bar(x - width/2, df_step6["missing_count"], width, label="Missing")
plt.bar(x + width/2, df_step6["inconsistent_count"], width, label="Inconsistent")

plt.xticks(x, df_step6["patient_id"], rotation=30)
plt.title("Missing vs Inconsistent Report Claims")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(STEP6_VIS, "missing_vs_inconsistent.png"), dpi=180)
plt.show()

field_cols = [
    "severity_check",
    "laterality_check",
    "region_proxy_check",
    "whole_tumor_volume_check",
    "core_ratio_check",
    "edema_ratio_check",
    "enhancing_ratio_check"
]

heatmap_df = df_step6[["patient_id"] + field_cols].copy()

mapping = {"consistent": 1.0, "missing": 0.5, "inconsistent": 0.0}
for col in field_cols:
    heatmap_df[col] = heatmap_df[col].map(mapping)

heatmap_matrix = heatmap_df[field_cols].values

plt.figure(figsize=(10, 4))
plt.imshow(heatmap_matrix, aspect="auto", vmin=0, vmax=1)
plt.colorbar(label="Consistency (1=consistent, 0.5=missing, 0=inconsistent)")
plt.yticks(range(len(heatmap_df)), heatmap_df["patient_id"])
plt.xticks(range(len(field_cols)), field_cols, rotation=45, ha="right")
plt.title("Field-Level Report Consistency Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(STEP6_VIS, "consistency_heatmap.png"), dpi=180)
plt.show()

# Render an easy-to-read visual consistency-check summary.
def render_consistency_card(result, save_dir):
    pid = result["patient_id"]

    text_lines = []
    text_lines.append(f"Patient ID: {pid}")
    text_lines.append("")
    text_lines.append(f"Consistency Score: {result['consistency_score']:.2f}")
    text_lines.append(f"Consistent Fields: {result['consistent_count']}")
    text_lines.append(f"Missing Fields: {result['missing_count']}")
    text_lines.append(f"Inconsistent Fields: {result['inconsistent_count']}")
    text_lines.append("")
    text_lines.append("Field Checks:")
    for k, v in result["checks"].items():
        text_lines.append(f" - {k}: {v}")
    text_lines.append("")
    text_lines.append("Missing Fields:")
    text_lines.append(", ".join(result["missing_fields"]) if result["missing_fields"] else "None")
    text_lines.append("")
    text_lines.append("Unsupported/Inconsistent Fields:")
    text_lines.append(", ".join(result["unsupported_fields"]) if result["unsupported_fields"] else "None")

    text = "\n".join(text_lines)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", wrap=True, fontsize=11, family="monospace")
    plt.title(f"{pid} - Report Consistency Card", pad=20)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{pid}_consistency_card.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()

    print("✅ Saved:", out_path)

for r in verification_results:
    render_consistency_card(r, STEP6_VIS)

verification_results[0]







