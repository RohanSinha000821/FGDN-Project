"""Summarize FGDN ablation evaluation outputs into CSV, Markdown, and plots.

This script is intentionally tolerant: it searches outputs/ablations recursively for CSV/JSON
files containing acc/auc fields. If your evaluator writes a differently named file, add its
name pattern in find_metric_files().
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FGDN ablation results.")
    parser.add_argument("--project-root", type=str, default="D:/FGDN_Project")
    parser.add_argument("--ablations-root", type=str, default="outputs/ablations")
    return parser.parse_args()


def find_metric_files(root: Path) -> Iterable[Path]:
    patterns = ["*evaluation*.json", "*metrics*.json", "*results*.json", "*evaluation*.csv", "*metrics*.csv", "*results*.csv"]
    seen = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path not in seen:
                seen.add(path)
                yield path


def parse_context(path: Path, root: Path) -> Dict[str, Any]:
    rel = path.relative_to(root).parts
    context = {"metric_file": str(path).replace("\\", "/")}
    if len(rel) >= 3:
        context["atlas"] = rel[0]
        context["study"] = rel[1]
        context["run_id"] = rel[2]
    return context


def load_json_metrics(path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def load_csv_metrics(path: Path) -> Dict[str, Any] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    # Prefer single-row evaluation files; otherwise take last row.
    return df.iloc[-1].to_dict()


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    key_map = {
        "accuracy": "acc",
        "test_accuracy": "acc",
        "test_acc": "acc",
        "auc": "auc",
        "test_auc": "auc",
        "loss": "loss",
        "test_loss": "loss",
    }
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        normalized = key_map.get(str(key), str(key))
        out[normalized] = value
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    root = project_root / args.ablations_root
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for path in find_metric_files(root):
        metrics = load_json_metrics(path) if path.suffix.lower() == ".json" else load_csv_metrics(path)
        if not metrics:
            continue
        row = parse_context(path, root)
        row.update(normalize_metrics(metrics))
        if "acc" in row or "auc" in row:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No metric files found under {root}. Run evaluation first or update file patterns.")

    df = pd.DataFrame(rows)
    csv_path = summaries / "ablation_results.csv"
    md_path = summaries / "ablation_results.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    for metric, filename in [("auc", "ablation_auc_plot.png"), ("acc", "ablation_accuracy_plot.png")]:
        if metric not in df.columns:
            continue
        plot_df = df.copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            continue
        labels = plot_df.get("run_id", plot_df.index.astype(str)).astype(str)
        plt.figure(figsize=(max(8, len(plot_df) * 0.35), 5))
        plt.bar(labels, plot_df[metric])
        plt.xticks(rotation=75, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"FGDN Ablation {metric.upper()}")
        plt.tight_layout()
        plt.savefig(summaries / filename, dpi=200)
        plt.close()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
