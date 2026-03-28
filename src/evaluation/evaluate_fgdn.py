import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch_geometric.loader import DataLoader

from src.models.fgdn_model import FGDNModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FGDN checkpoint on one fold.")
    parser.add_argument("--project-root", type=str, default="D:/FGDN_Project")
    parser.add_argument("--atlas", type=str, choices=["AAL", "HarvardOxford"], required=True)
    parser.add_argument("--num-folds", type=int, choices=[5, 10], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-type", type=str, choices=["best", "last"], default="best")
    return parser.parse_args()


def load_dataset(project_root: Path, atlas: str, num_folds: int, fold: int):
    fold_dir = (
        project_root
        / "data"
        / "processed"
        / "pyg_datasets"
        / atlas
        / f"{num_folds}_fold"
        / f"fold_{fold}"
    )
    test_dataset = torch.load(
        fold_dir / "test_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    return test_dataset


def load_checkpoint(project_root: Path, atlas: str, num_folds: int, fold: int, checkpoint_type: str):
    ckpt_dir = (
        project_root
        / "outputs"
        / "checkpoints"
        / "fgdn"
        / atlas
        / f"{num_folds}_fold"
        / f"fold_{fold}"
    )

    ckpt_name = "best_fgdn.pt" if checkpoint_type == "best" else "last_fgdn.pt"
    ckpt_path = ckpt_dir / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint, ckpt_path


def build_model_from_checkpoint(sample, checkpoint, device):
    args = checkpoint["args"]

    model = FGDNModel(
        in_channels=sample.x.size(1),
        num_nodes=sample.x.size(0),
        hidden_channels=args["hidden_channels"],
        cheb_k=args["cheb_k"],
        dropout=args["dropout"],
        num_classes=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    y_true = []
    y_pred = []
    y_prob = []
    subject_ids = []

    for batch in loader:
        batch = batch.to(device)

        logits, _ = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
        subject_ids.extend(batch.subject_id)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "subject_ids": list(subject_ids),
    }


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    test_dataset = load_dataset(
        project_root=project_root,
        atlas=args.atlas,
        num_folds=args.num_folds,
        fold=args.fold,
    )
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint, ckpt_path = load_checkpoint(
        project_root=project_root,
        atlas=args.atlas,
        num_folds=args.num_folds,
        fold=args.fold,
        checkpoint_type=args.checkpoint_type,
    )

    model = build_model_from_checkpoint(test_dataset[0], checkpoint, device)
    results = evaluate(model, loader, device)

    out_dir = (
        project_root
        / "outputs"
        / "tables"
        / "fgdn"
        / args.atlas
        / f"{args.num_folds}_fold"
        / f"fold_{args.fold}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.checkpoint_type}_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "atlas": args.atlas,
                "num_folds": args.num_folds,
                "fold": args.fold,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_type": args.checkpoint_type,
                "best_monitor_auc_from_training": checkpoint.get("best_monitor_auc", None),
                "training_epoch_saved": checkpoint.get("epoch", None),
                "split_info": checkpoint.get("split_info", None),
                "accuracy": results["accuracy"],
                "auc": results["auc"],
                "confusion_matrix": results["confusion_matrix"],
            },
            f,
            indent=2,
        )

    pred_path = out_dir / f"{args.checkpoint_type}_predictions.npz"
    np.savez(
        pred_path,
        subject_ids=np.array(results["subject_ids"], dtype=object),
        y_true=np.array(results["y_true"], dtype=np.int64),
        y_pred=np.array(results["y_pred"], dtype=np.int64),
        y_prob=np.array(results["y_prob"], dtype=np.float32),
    )

    print("=" * 72)
    print(f"Checkpoint                : {ckpt_path}")
    print(f"Best monitor AUC (train)  : {checkpoint.get('best_monitor_auc', None)}")
    print(f"Accuracy                  : {results['accuracy']:.6f}")
    print(f"AUC                       : {results['auc']:.6f}")
    print(f"Confusion Matrix          : {results['confusion_matrix']}")
    print(f"Saved JSON                : {out_path}")
    print(f"Saved preds               : {pred_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()