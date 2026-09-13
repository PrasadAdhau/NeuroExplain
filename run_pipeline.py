"""Run NeuroExplain stages in dependency order from the project root."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    "steps/01_data_eda_preprocessing.py",
    "steps/02a_unet2d_baseline.py",
    "steps/02b_unet3d_final.py",
    "steps/03_structured_findings.py",
    "steps/04a_gradcam_basic.py",
    "steps/04b_gradcam_enhanced.py",
    "steps/05_rag_reporting.py",
    "steps/06_consistency_verification.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, len(STEPS) + 1))
    parser.add_argument("--to-step", type=int, default=len(STEPS), choices=range(1, len(STEPS) + 1))
    args = parser.parse_args()
    if args.from_step > args.to_step:
        parser.error("--from-step must not be greater than --to-step")
    root = Path(__file__).resolve().parent
    for position, relative_path in enumerate(STEPS, start=1):
        if args.from_step <= position <= args.to_step:
            print(f"\n{'=' * 60}\nRunning stage {position}: {relative_path}\n{'=' * 60}")
            subprocess.run([sys.executable, str(root / relative_path)], cwd=root, check=True)


if __name__ == "__main__":
    main()
