"""Configuration-driven ablation runner for FGDN.

Run from project root, for example:
    python -m src.ablation.run_ablations --config configs/ablation_quick_aal5_fold1.json

This script assumes train/evaluate scripts support:
    --output-root outputs/ablations/<atlas>/<study>/<run_id>
    --log-root outputs/ablations/<atlas>/<study>/<run_id>/logs
See the small training/evaluation patch in the instructions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FGDN ablation studies from a JSON config.")
    parser.add_argument("--config", required=True, type=str, help="Path to ablation JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate/summarize existing runs.")
    parser.add_argument("--skip-eval", action="store_true", help="Only train, do not evaluate.")
    return parser.parse_args()


def slug_value(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("\n" + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def module_names(template_mode: str) -> tuple[str, str]:
    if template_mode == "weighted":
        return "src.training.train_fgdn_weighted", "src.evaluation.evaluate_fgdn_weighted"
    if template_mode == "binary":
        return "src.training.train_fgdn", "src.evaluation.evaluate_fgdn"
    raise ValueError(f"Unknown template_mode: {template_mode}")


def cmd_common(run: Dict[str, Any], project_root: str, device: str | None) -> List[str]:
    cmd = [
        "--project-root", project_root,
        "--atlas", str(run["atlas"]),
        "--kind", str(run["kind"]),
        "--num-folds", str(run["num_folds"]),
        "--fold", str(run["fold"]),
    ]
    if device:
        cmd += ["--device", device]
    return cmd


def train_cmd(run: Dict[str, Any], project_root: str, device: str | None, output_root: Path, log_root: Path) -> List[str]:
    train_module, _ = module_names(run["template_mode"])
    return [sys.executable, "-m", train_module] + cmd_common(run, project_root, device) + [
        "--hidden-channels", str(run["hidden_channels"]),
        "--cheb-k", str(run["cheb_k"]),
        "--dropout", str(run["dropout"]),
        "--batch-size", str(run["batch_size"]),
        "--epochs", str(run["epochs"]),
        "--lr", str(run["lr"]),
        "--weight-decay", str(run["weight_decay"]),
        "--patience", str(run["patience"]),
        "--template-k", str(run["template_k"]),
        "--output-root", str(output_root).replace("\\", "/"),
        "--log-root", str(log_root).replace("\\", "/"),
    ]


def eval_cmd(run: Dict[str, Any], project_root: str, device: str | None, output_root: Path, checkpoint_type: str) -> List[str]:
    _, eval_module = module_names(run["template_mode"])
    cmd = [sys.executable, "-m", eval_module] + cmd_common(run, project_root, device) + [
        "--checkpoint-type", checkpoint_type,
        "--output-root", str(output_root).replace("\\", "/"),
    ]
    return cmd


def build_runs(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    base = config["base"]
    for study in config["studies"]:
        for value in study["values"]:
            for fold in base.get("folds", [1]):
                run = dict(base)
                run[study["param"]] = value
                run["fold"] = fold
                run["study"] = study["name"]
                run["param"] = study["param"]
                run["value"] = value
                run["run_id"] = f"{study['param']}_{slug_value(value)}_fold_{fold}"
                run["epochs"] = config.get("epochs", 100)
                run["patience"] = config.get("patience", 20)
                yield run


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    project_root = config.get("project_root", "D:/FGDN_Project")
    device = config.get("device")
    checkpoint_type = config.get("checkpoint_type", "best")
    rows: List[Dict[str, Any]] = []

    for run in build_runs(config):
        output_root = Path("outputs") / "ablations" / run["atlas"] / run["study"] / run["run_id"]
        log_root = output_root / "logs"
        if not args.skip_train:
            run_command(train_cmd(run, project_root, device, output_root, log_root), args.dry_run)
        if not args.skip_eval:
            run_command(eval_cmd(run, project_root, device, output_root, checkpoint_type), args.dry_run)
        rows.append({
            "atlas": run["atlas"],
            "kind": run["kind"],
            "num_folds": run["num_folds"],
            "fold": run["fold"],
            "study": run["study"],
            "param": run["param"],
            "value": run["value"],
            "template_mode": run["template_mode"],
            "run_id": run["run_id"],
            "output_root": str(output_root).replace("\\", "/"),
        })

    manifest_path = Path(project_root) / "outputs" / "ablations" / "summaries" / "ablation_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nManifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
