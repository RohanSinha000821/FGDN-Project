import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.models.fgdn_model import FGDNModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGDN on one atlas/fold.")
    parser.add_argument("--project-root", type=str, default="D:/FGDN_Project")
    parser.add_argument("--atlas", type=str, choices=["AAL", "HarvardOxford"], required=True)
    parser.add_argument("--num-folds", type=int, choices=[5, 10], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--branch-out-channels", type=int, default=64)
    parser.add_argument("--cheb-k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_fold_datasets(project_root: Path, atlas: str, num_folds: int, fold: int):
    fold_dir = (
        project_root
        / "data"
        / "processed"
        / "pyg_datasets"
        / atlas
        / f"{num_folds}_fold"
        / f"fold_{fold}"
    )

    train_dataset = torch.load(
        fold_dir / "train_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    test_dataset = torch.load(
        fold_dir / "test_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )

    summary_path = fold_dir / "dataset_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    return train_dataset, test_dataset, summary


def make_loaders(train_dataset, test_dataset, batch_size: int):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def build_model(sample, args, device):
    model = FGDNModel(
        in_channels=sample.x.size(1),
        hidden_channels=args.hidden_channels,
        branch_out_channels=args.branch_out_channels,
        cheb_k=args.cheb_k,
        dropout=args.dropout,
        num_classes=2,
    )
    return model.to(device)


def run_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)

    try:
        epoch_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        epoch_auc = float("nan")

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "auc": epoch_auc,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        batch = batch.to(device)

        logits, _ = model(batch)
        loss = criterion(logits, batch.y)

        running_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)

    try:
        epoch_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        epoch_auc = float("nan")

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "auc": epoch_auc,
    }


def save_history_csv(history: List[Dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print(f"Training FGDN | atlas={args.atlas} | folds={args.num_folds} | fold={args.fold}")
    print(f"Using device: {device}")
    print("=" * 72)

    train_dataset, test_dataset, dataset_summary = load_fold_datasets(
        project_root=project_root,
        atlas=args.atlas,
        num_folds=args.num_folds,
        fold=args.fold,
    )

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Empty train or test dataset.")

    train_loader, test_loader = make_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
    )

    model = build_model(train_dataset[0], args, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = (
        project_root
        / "outputs"
        / "checkpoints"
        / "fgdn"
        / args.atlas
        / f"{args.num_folds}_fold"
        / f"fold_{args.fold}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = checkpoint_dir / "best_fgdn.pt"
    last_ckpt_path = checkpoint_dir / "last_fgdn.pt"

    history_csv_path = (
        project_root
        / "data"
        / "processed"
        / "logs"
        / "fgdn"
        / args.atlas
        / f"{args.num_folds}_fold"
        / f"fold_{args.fold}"
        / "training_history.csv"
    )

    history = []
    best_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        current_auc = val_metrics["auc"]
        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_auc,
                    "args": vars(args),
                    "dataset_summary": dataset_summary,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_auc,
                "args": vars(args),
                "dataset_summary": dataset_summary,
            },
            last_ckpt_path,
        )

        save_history_csv(history, history_csv_path)

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print("=" * 72)
    print("Training finished")
    print("=" * 72)
    print(f"Best val AUC    : {best_auc:.6f}")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best checkpoint : {best_ckpt_path}")
    print(f"Last checkpoint : {last_ckpt_path}")
    print(f"History CSV     : {history_csv_path}")


if __name__ == "__main__":
    main()