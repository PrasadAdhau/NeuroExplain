# Original NeuroExplain code is retained below and runs as this pipeline stage.
# Configure BRATS_DATASET_PATH and NEUROEXPLAIN_OUTPUT_PATH before running.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_PATH
try:
    from IPython.display import display
except ImportError:
    display = print

JSON_DIR = os.path.join(OUT_PATH, "step3_structured_findings", "json")

# STEP 5: Build the FAISS retrieval workflow and generate grounded research reports.
STEP5_OUT = os.path.join(OUT_PATH, "step5_rag")
REPORT_DIR = os.path.join(STEP5_OUT, "reports")
TABLE_DIR = os.path.join(STEP5_OUT, "tables")
VIS_DIR = os.path.join(STEP5_OUT, "visuals")

os.makedirs(STEP5_OUT, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print("✅ Step 5 output:", STEP5_OUT)
print("✅ JSON input dir:", JSON_DIR)

json_files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])
print("Structured JSON files found:", len(json_files))

structured_cases = []
for fname in json_files:
    with open(os.path.join(JSON_DIR, fname), "r") as f:
        structured_cases.append(json.load(f))

print("✅ Loaded", len(structured_cases), "cases")
structured_cases[0]

# Serialize structured findings into text for vector retrieval.
def case_to_text(case):
    pid = case["patient_id"]

    wt = case["volumes_voxels"]["whole_tumor"]
    core = case["volumes_voxels"]["tumor_core"]
    edema = case["volumes_voxels"]["edema"]
    enh = case["volumes_voxels"]["enhancing_tumor"]

    core_ratio = case["proportions"]["core_ratio"]
    edema_ratio = case["proportions"]["edema_ratio"]
    enh_ratio = case["proportions"]["enhancing_ratio"]

    lat = case["location"]["laterality"]
    region = case["location"]["region_proxy"]

    sev = case["severity"]["level"]
    score = case["severity"]["score"]

    text = (
        f"Patient {pid}. "
        f"Whole tumor volume {wt} voxels. "
        f"Tumor core volume {core} voxels. "
        f"Edema volume {edema} voxels. "
        f"Enhancing tumor volume {enh} voxels. "
        f"Core ratio {core_ratio:.3f}. "
        f"Edema ratio {edema_ratio:.3f}. "
        f"Enhancing ratio {enh_ratio:.3f}. "
        f"Tumor laterality {lat}. "
        f"Region proxy {region}. "
        f"Severity level {sev} with severity score {score}."
    )
    return text

corpus_rows = []
for case in structured_cases:
    corpus_rows.append({
        "patient_id": case["patient_id"],
        "text": case_to_text(case),
        "severity_level": case["severity"]["level"],
        "severity_score": case["severity"]["score"],
        "laterality": case["location"]["laterality"]
    })

corpus_df = pd.DataFrame(corpus_rows)
corpus_df.head()

template_docs = [
    {
        "doc_id": "template_low",
        "text": (
            "Radiology-style template: The lesion demonstrates relatively limited overall tumor burden "
            "with low enhancing tumor proportion. The findings suggest a lower severity pattern based on "
            "tumor size and composition."
        ),
        "doc_type": "template"
    },
    {
        "doc_id": "template_medium",
        "text": (
            "Radiology-style template: The lesion demonstrates moderate tumor burden with mixed internal "
            "composition, including edema and enhancing components. Findings are consistent with an "
            "intermediate severity pattern."
        ),
        "doc_type": "template"
    },
    {
        "doc_id": "template_high",
        "text": (
            "Radiology-style template: The lesion demonstrates substantial tumor burden and/or a prominent "
            "enhancing component. These imaging-derived features are consistent with a higher severity pattern."
        ),
        "doc_type": "template"
    }
]

case_docs = []
for _, row in corpus_df.iterrows():
    case_docs.append({
        "doc_id": row["patient_id"],
        "text": row["text"],
        "doc_type": "case",
        "severity_level": row["severity_level"],
        "laterality": row["laterality"]
    })

all_docs = case_docs + template_docs
len(all_docs)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_texts = [d["text"] for d in all_docs]
doc_ids = [d["doc_id"] for d in all_docs]
doc_types = [d["doc_type"] for d in all_docs]

embeddings = embed_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
print("Embedding shape:", embeddings.shape)

index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity using normalized vectors
index.add(embeddings)

print("✅ FAISS index built with", index.ntotal, "documents")

# Search the FAISS index for similar case or template documents.
def retrieve_similar_docs(query_text, top_k=5):
    query_emb = embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "doc_id": doc_ids[idx],
            "doc_type": doc_types[idx],
            "text": doc_texts[idx],
            "score": float(score)
        })
    return results

# Create a template-based report grounded in structured findings and retrieval.
def generate_grounded_report(case, retrieved_docs):
    pid = case["patient_id"]

    wt = case["volumes_voxels"]["whole_tumor"]
    core = case["volumes_voxels"]["tumor_core"]
    edema = case["volumes_voxels"]["edema"]
    enh = case["volumes_voxels"]["enhancing_tumor"]

    core_ratio = case["proportions"]["core_ratio"]
    edema_ratio = case["proportions"]["edema_ratio"]
    enh_ratio = case["proportions"]["enhancing_ratio"]

    lat = case["location"]["laterality"]
    region = case["location"]["region_proxy"]
    sev = case["severity"]["level"]
    score = case["severity"]["score"]

    retrieved_case_ids = [d["doc_id"] for d in retrieved_docs if d["doc_type"] == "case" and d["doc_id"] != pid][:3]
    retrieved_template = [d["text"] for d in retrieved_docs if d["doc_type"] == "template"]

    report = []
    report.append(f"Patient ID: {pid}")
    report.append("")
    report.append("Imaging Findings:")
    report.append(
        f"A segmented intracranial lesion is identified with an estimated whole tumor volume of {wt} voxels. "
        f"The lesion includes a tumor core of {core} voxels, edema of {edema} voxels, and enhancing tumor of {enh} voxels."
    )
    report.append(
        f"The lesion is located in the {lat} hemisphere with a {region} regional distribution based on centroid-derived localization."
    )
    report.append(
        f"Subregion composition includes core ratio {core_ratio:.3f}, edema ratio {edema_ratio:.3f}, "
        f"and enhancing tumor ratio {enh_ratio:.3f}."
    )
    report.append("")
    report.append("Interpretation:")
    report.append(
        f"Based on imaging-derived structured features, the lesion demonstrates a {sev} severity pattern "
        f"(severity score = {score})."
    )

    if retrieved_template:
        report.append(retrieved_template[0])

    if retrieved_case_ids:
        report.append(
            f"Retrieval analysis identified similar structured cases including: {', '.join(retrieved_case_ids)}."
        )

    report.append("")
    report.append("Note:")
    report.append(
        "This report is automatically generated from segmentation-derived structured findings and retrieved reference patterns. "
        "It is intended for research demonstration and not for clinical diagnosis."
    )

    return "\n".join(report)

TARGET_CASES = structured_cases[:55]   # change if needed

retrieval_rows = []
report_rows = []

for case in TARGET_CASES:
    query_text = case_to_text(case)
    retrieved = retrieve_similar_docs(query_text, top_k=5)

    retrieved_clean = []
    for r in retrieved:
        if not (r["doc_type"] == "case" and r["doc_id"] == case["patient_id"]):
            retrieved_clean.append(r)

    retrieved_clean = retrieved_clean[:4]

    report_text = generate_grounded_report(case, retrieved_clean)

    report_path = os.path.join(REPORT_DIR, f"{case['patient_id']}_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    out_json = {
        "patient_id": case["patient_id"],
        "query_text": query_text,
        "retrieved_docs": retrieved_clean,
        "generated_report": report_text
    }
    with open(os.path.join(REPORT_DIR, f"{case['patient_id']}_rag_output.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    for r in retrieved_clean:
        retrieval_rows.append({
            "patient_id": case["patient_id"],
            "retrieved_doc_id": r["doc_id"],
            "doc_type": r["doc_type"],
            "score": r["score"]
        })

    report_rows.append({
        "patient_id": case["patient_id"],
        "report_path": report_path,
        "generated_report": report_text
    })

print("✅ Step 5 completed for", len(TARGET_CASES), "cases")

df_retrieval = pd.DataFrame(retrieval_rows)
retrieval_csv = os.path.join(TABLE_DIR, "retrieval_results.csv")
df_retrieval.to_csv(retrieval_csv, index=False)

print("✅ Retrieval table saved:", retrieval_csv)
df_retrieval

for pid in df_retrieval["patient_id"].unique():
    df_pid = df_retrieval[df_retrieval["patient_id"] == pid]

    plt.figure(figsize=(8,4))
    plt.bar(df_pid["retrieved_doc_id"], df_pid["score"])
    plt.title(f"{pid} - Retrieved Document Similarity Scores")
    plt.xlabel("Retrieved Document")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{pid}_retrieval_scores.png"), dpi=180)
    plt.show()

severity_map = {}
for case in structured_cases:
    severity_map[case["patient_id"]] = case["severity"]["level"]

for pid in df_retrieval["patient_id"].unique():
    df_pid = df_retrieval[df_retrieval["patient_id"] == pid].copy()
    df_pid["retrieved_severity"] = df_pid["retrieved_doc_id"].map(severity_map)

    sev_counts = df_pid["retrieved_severity"].value_counts()

    plt.figure(figsize=(6,4))
    plt.bar(sev_counts.index, sev_counts.values)
    plt.title(f"{pid} - Severity of Retrieved Cases")
    plt.xlabel("Severity Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{pid}_retrieved_severity.png"), dpi=180)
    plt.show()

for row in report_rows:
    pid = row["patient_id"]
    text = row["generated_report"]

    plt.figure(figsize=(10,6))
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", wrap=True, fontsize=10)
    plt.title(f"{pid} - Generated RAG Report", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{pid}_report_preview.png"), dpi=180, bbox_inches="tight")
    plt.show()

for row in report_rows:
    print("\n" + "="*80)
    print("PATIENT:", row["patient_id"])
    print("="*80)
    print(row["generated_report"])

# Calculate a rule-based similarity score between two structured cases.
def structured_similarity_score(case_a, case_b):
    score = 0

    if case_a["severity"]["level"] == case_b["severity"]["level"]:
        score += 2

    if case_a["location"]["laterality"] == case_b["location"]["laterality"]:
        score += 1

    v1 = case_a["volumes_voxels"]["whole_tumor"]
    v2 = case_b["volumes_voxels"]["whole_tumor"]
    vol_diff = abs(v1 - v2) / max(v1, v2, 1)
    score += (1 - vol_diff)

    return score

# Combine embedding retrieval with structured-field similarity ranking.
def hybrid_retrieve(case, top_k=5):

    query_text = case_to_text(case)

    retrieved = retrieve_similar_docs(query_text, top_k=10)

    candidate_ids = [r["doc_id"] for r in retrieved if r["doc_type"] == "case"]

    case_map = {c["patient_id"]: c for c in structured_cases}

    scored_candidates = []

    for pid in candidate_ids:
        if pid == case["patient_id"]:
            continue

        candidate_case = case_map[pid]

        struct_score = structured_similarity_score(case, candidate_case)

        sem_score = next(r["score"] for r in retrieved if r["doc_id"] == pid)

        final_score = 0.6 * sem_score + 0.4 * struct_score

        scored_candidates.append((pid, final_score, sem_score, struct_score))

    scored_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    top_cases = scored_candidates[:top_k]

    return top_cases

# Create a table comparing the target case with retrieved cases.
def build_comparison_table(target_case, retrieved_cases):

    rows = []

    rows.append({
        "type": "target",
        "patient_id": target_case["patient_id"],
        "severity": target_case["severity"]["level"],
        "laterality": target_case["location"]["laterality"],
        "volume": target_case["volumes_voxels"]["whole_tumor"],
        "enh_ratio": target_case["proportions"]["enhancing_ratio"]
    })

    case_map = {c["patient_id"]: c for c in structured_cases}

    for pid, final_score, sem, struct in retrieved_cases:
        c = case_map[pid]

        rows.append({
            "type": "retrieved",
            "patient_id": pid,
            "severity": c["severity"]["level"],
            "laterality": c["location"]["laterality"],
            "volume": c["volumes_voxels"]["whole_tumor"],
            "enh_ratio": c["proportions"]["enhancing_ratio"],
            "final_score": final_score
        })

    df = pd.DataFrame(rows)
    return df

# Visualize comparison metrics for the target and retrieved cases.
def plot_comparison(df, save_path=None):

    df_plot = df[df["type"] == "retrieved"]

    plt.figure(figsize=(8,4))
    plt.bar(df_plot["patient_id"], df_plot["volume"])
    plt.title("Retrieved Cases vs Volume")
    plt.xlabel("Patient")
    plt.ylabel("Whole Tumor Volume")
    plt.xticks(rotation=30)

    if save_path:
        plt.savefig(save_path, dpi=180)

    plt.show()

# Create a direct structured-findings baseline report.
def generate_baseline_report(case):
    return f"""
Patient ID: {case['patient_id']}

Findings:
Whole tumor volume: {case['volumes_voxels']['whole_tumor']}
Core ratio: {case['proportions']['core_ratio']:.3f}
Enhancing ratio: {case['proportions']['enhancing_ratio']:.3f}

Interpretation:
This lesion is categorized as {case['severity']['level']} severity.

Note:
This report is generated without retrieval support.
"""

ENHANCED_OUT = os.path.join(STEP5_OUT, "enhanced")
os.makedirs(ENHANCED_OUT, exist_ok=True)

TARGET_CASES = structured_cases[:55]

for case in TARGET_CASES:

    print("\n" + "="*80)
    print("Processing:", case["patient_id"])
    print("="*80)

    retrieved_cases = hybrid_retrieve(case, top_k=3)

    df_compare = build_comparison_table(case, retrieved_cases)

    display(df_compare)

    df_compare.to_csv(
        os.path.join(ENHANCED_OUT, f"{case['patient_id']}_comparison.csv"),
        index=False
    )

    plot_comparison(
        df_compare,
        save_path=os.path.join(ENHANCED_OUT, f"{case['patient_id']}_comparison.png")
    )

    baseline_report = generate_baseline_report(case)

    retrieved_docs = retrieve_similar_docs(case_to_text(case), top_k=5)
    rag_report = generate_grounded_report(case, retrieved_docs)

    with open(os.path.join(ENHANCED_OUT, f"{case['patient_id']}_baseline.txt"), "w") as f:
        f.write(baseline_report)

    with open(os.path.join(ENHANCED_OUT, f"{case['patient_id']}_rag.txt"), "w") as f:
        f.write(rag_report)

    print("\n--- WITHOUT RETRIEVAL ---")
    print(baseline_report)

    print("\n--- WITH RETRIEVAL (RAG) ---")
    print(rag_report)

# Create the detailed research-style report from structured evidence.
def generate_detailed_medical_report(case, retrieved_docs):
    pid = case["patient_id"]

    wt = case["volumes_voxels"]["whole_tumor"]
    core = case["volumes_voxels"]["tumor_core"]
    edema = case["volumes_voxels"]["edema"]
    enh = case["volumes_voxels"]["enhancing_tumor"]

    core_ratio = case["proportions"]["core_ratio"]
    edema_ratio = case["proportions"]["edema_ratio"]
    enh_ratio = case["proportions"]["enhancing_ratio"]

    lat = case["location"]["laterality"]
    region = case["location"]["region_proxy"]

    cx = case["centroid_voxel"]["x"]
    cy = case["centroid_voxel"]["y"]
    cz = case["centroid_voxel"]["z"]

    sev = case["severity"]["level"]
    sev_score = case["severity"]["score"]

    retrieved_case_ids = [d["doc_id"] for d in retrieved_docs if d["doc_type"] == "case" and d["doc_id"] != pid][:3]
    retrieved_templates = [d["text"] for d in retrieved_docs if d["doc_type"] == "template"]

    if sev == "low":
        sev_text = (
            "The lesion demonstrates relatively limited tumor burden with a lower enhancing component, "
            "suggesting a lower imaging-derived severity pattern."
        )
    elif sev == "medium":
        sev_text = (
            "The lesion demonstrates moderate tumor burden with mixed internal composition, "
            "consistent with an intermediate imaging-derived severity pattern."
        )
    else:
        sev_text = (
            "The lesion demonstrates substantial tumor burden and/or a prominent enhancing component, "
            "consistent with a higher imaging-derived severity pattern."
        )

    dominant_component = max(
        [("core", core_ratio), ("edema", edema_ratio), ("enhancing tumor", enh_ratio)],
        key=lambda x: x[1]
    )[0]

    report = f"""
NEUROEXPLAIN AUTOMATED MRI REPORT
================================

Patient ID:
{pid}

1. FINDINGS
-----------
A segmented intracranial lesion is identified with an estimated whole tumor volume of {wt} voxels.
The lesion includes:
- Tumor core: {core} voxels
- Edema: {edema} voxels
- Enhancing tumor: {enh} voxels

2. TUMOR COMPOSITION
--------------------
The predicted lesion composition is as follows:
- Core ratio: {core_ratio:.3f}
- Edema ratio: {edema_ratio:.3f}
- Enhancing tumor ratio: {enh_ratio:.3f}

The dominant lesion component is {dominant_component}.

3. LOCATION ASSESSMENT
----------------------
The lesion centroid is located at voxel coordinates:
(x={cx}, y={cy}, z={cz})

The lesion is localized primarily to the {lat} hemisphere with a {region} spatial distribution.

4. SEVERITY ASSESSMENT
----------------------
Predicted severity level: {sev.upper()}
Severity score: {sev_score}

Interpretation:
{sev_text}
"""

    if retrieved_case_ids:
        report += f"""

5. RETRIEVED SIMILAR CASES
--------------------------
The retrieval module identified the following structurally similar reference cases:
- {chr(10).join([f"  • {rid}" for rid in retrieved_case_ids])}
"""

    if retrieved_templates:
        report += f"""

6. GROUNDED REFERENCE PATTERN
-----------------------------
{retrieved_templates[0]}
"""

    report += f"""

7. IMPRESSION
-------------
This case demonstrates a {sev} imaging-derived severity pattern with a lesion located in the {lat} hemisphere.
The internal tumor composition suggests that {dominant_component} is the most prominent component.
These findings are generated from segmentation-derived structured features and retrieval-supported pattern matching.

8. DISCLAIMER
-------------
This report is automatically generated for research and educational purposes only.
It is not intended for clinical diagnosis, treatment planning, or medical decision-making without expert review.
"""

    return report.strip()

DETAILED_REPORT_DIR = os.path.join(STEP5_OUT, "detailed_reports")
os.makedirs(DETAILED_REPORT_DIR, exist_ok=True)

detailed_report_rows = []

TARGET_CASES = structured_cases[:55]   # change if needed

for case in TARGET_CASES:
    query_text = case_to_text(case)
    retrieved = retrieve_similar_docs(query_text, top_k=5)

    retrieved_clean = []
    for r in retrieved:
        if not (r["doc_type"] == "case" and r["doc_id"] == case["patient_id"]):
            retrieved_clean.append(r)
    retrieved_clean = retrieved_clean[:4]

    detailed_report = generate_detailed_medical_report(case, retrieved_clean)

    txt_path = os.path.join(DETAILED_REPORT_DIR, f"{case['patient_id']}_detailed_report.txt")
    with open(txt_path, "w") as f:
        f.write(detailed_report)

    json_path = os.path.join(DETAILED_REPORT_DIR, f"{case['patient_id']}_detailed_report.json")
    with open(json_path, "w") as f:
        json.dump({
            "patient_id": case["patient_id"],
            "retrieved_docs": retrieved_clean,
            "report": detailed_report
        }, f, indent=2)

    detailed_report_rows.append({
        "patient_id": case["patient_id"],
        "report_path": txt_path,
        "report_text": detailed_report
    })

print("✅ Detailed reports created for", len(detailed_report_rows), "cases")

for row in detailed_report_rows:
    print("\n" + "="*100)
    print("PATIENT:", row["patient_id"])
    print("="*100)
    print(row["report_text"])

# Render a text report as an image preview for project presentation.
def save_report_preview(patient_id, report_text, save_dir):
    plt.figure(figsize=(10, 12))
    plt.axis("off")
    plt.text(0.01, 0.99, report_text, va="top", wrap=True, fontsize=10, family="monospace")
    plt.title(f"{patient_id} - Detailed Medical-Style Report", pad=20)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{patient_id}_detailed_report_preview.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()

    print("✅ Saved:", out_path)

for row in detailed_report_rows:
    save_report_preview(row["patient_id"], row["report_text"], DETAILED_REPORT_DIR)

import re

