import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torch_geometric.loader import DataLoader

from src.models.fgdn_model import FGDNModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGDN on one atlas/fold.")
    parser.add_argument("--project-root", type=str, default="D:/FGDN_Project")
    parser.add_argument("--atlas", type=str, choices=["AAL", "HarvardOxford"], required=True)
    parser.add_argument("--num-folds", type=int, choices=[5, 10], required=True)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--cheb-k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument(
        "--monitor-ratio",
        type=float,
        default=0.10,
        help="Fraction of the outer training fold reserved as monitoring set.",
    )
    parser.add_argument(
        "--monitor-seed",
        type=int,
        default=42,
        help="Random seed for inner-train / monitor split.",
    )
    parser.add_argument(
        "--template-k",
        type=int,
        default=20,
        help="K in KNN graph template construction.",
    )

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

    outer_train_dataset = torch.load(
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

    return outer_train_dataset, test_dataset, summary


def split_outer_train(
    outer_train_dataset,
    monitor_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < monitor_ratio < 1.0:
        raise ValueError(f"monitor_ratio must be in (0, 1), got {monitor_ratio}")

    labels = np.array([int(data.y.item()) for data in outer_train_dataset], dtype=np.int64)
    indices = np.arange(len(outer_train_dataset))

    if len(np.unique(labels)) < 2:
        raise ValueError("Outer training fold contains fewer than 2 classes; cannot stratify.")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=monitor_ratio,
        random_state=seed,
    )

    inner_train_idx, monitor_idx = next(sss.split(indices, labels))
    return inner_train_idx.astype(np.int64), monitor_idx.astype(np.int64)


def subset_dataset(dataset, indices: np.ndarray):
    return [dataset[int(i)] for i in indices]


def label_counts(dataset) -> Dict[str, int]:
    labels = np.array([int(d.y.item()) for d in dataset], dtype=np.int64)
    return {
        "ASD_1": int((labels == 1).sum()),
        "HC_0": int((labels == 0).sum()),
    }


def build_knn_adjacency(mean_fc: np.ndarray, k: int) -> np.ndarray:
    num_rois = mean_fc.shape[0]
    effective_k = min(k, num_rois - 1)

    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nbrs.fit(mean_fc)
    _, indices = nbrs.kneighbors(mean_fc)

    adjacency = np.zeros((num_rois, num_rois), dtype=np.int64)

    for i in range(num_rois):
        for j in indices[i]:
            if i == j:
                continue
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    np.fill_diagonal(adjacency, 0)
    return adjacency


def adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    rows, cols = np.where(adjacency > 0)
    return np.vstack([rows, cols]).astype(np.int64)


def build_templates_from_inner_train(inner_train_dataset, k: int):
    xs = np.stack([data.x.detach().cpu().numpy() for data in inner_train_dataset], axis=0)
    ys = np.array([int(data.y.item()) for data in inner_train_dataset], dtype=np.int64)

    asd_x = xs[ys == 1]
    hc_x = xs[ys == 0]

    if len(asd_x) == 0 or len(hc_x) == 0:
        raise ValueError("Inner-train split has an empty class; cannot build templates.")

    asd_mean_fc = asd_x.mean(axis=0)
    hc_mean_fc = hc_x.mean(axis=0)

    asd_adj = build_knn_adjacency(asd_mean_fc, k=k)
    hc_adj = build_knn_adjacency(hc_mean_fc, k=k)

    asd_edge_index = adjacency_to_edge_index(asd_adj)
    hc_edge_index = adjacency_to_edge_index(hc_adj)

    return asd_edge_index, hc_edge_index


def apply_templates_to_dataset(dataset, asd_edge_index_np: np.ndarray, hc_edge_index_np: np.ndarray):
    asd_edge_index = torch.from_numpy(asd_edge_index_np).long()
    hc_edge_index = torch.from_numpy(hc_edge_index_np).long()

    new_dataset = []
    for data in dataset:
        item = copy.copy(data)
        item.edge_index_asd = asd_edge_index.clone()
        item.edge_index_hc = hc_edge_index.clone()
        new_dataset.append(item)
    return new_dataset


def make_loaders(train_dataset, monitor_dataset, test_dataset, batch_size: int):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    monitor_loader = DataLoader(monitor_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, monitor_loader, test_loader


def build_model(sample, args, device):
    model = FGDNModel(
        in_channels=sample.x.size(1),
        num_nodes=sample.x.size(0),
        hidden_channels=args.hidden_channels,
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

    return {"loss": epoch_loss, "acc": epoch_acc, "auc": epoch_auc}


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

    return {"loss": epoch_loss, "acc": epoch_acc, "auc": epoch_auc}


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

    outer_train_dataset, raw_test_dataset, dataset_summary = load_fold_datasets(
        project_root=project_root,
        atlas=args.atlas,
        num_folds=args.num_folds,
        fold=args.fold,
    )

    if len(outer_train_dataset) == 0 or len(raw_test_dataset) == 0:
        raise RuntimeError("Empty outer-train or test dataset.")

    inner_train_idx, monitor_idx = split_outer_train(
        outer_train_dataset=outer_train_dataset,
        monitor_ratio=args.monitor_ratio,
        seed=args.monitor_seed,
    )

    raw_inner_train_dataset = subset_dataset(outer_train_dataset, inner_train_idx)
    raw_monitor_dataset = subset_dataset(outer_train_dataset, monitor_idx)

    asd_edge_index_np, hc_edge_index_np = build_templates_from_inner_train(
        inner_train_dataset=raw_inner_train_dataset,
        k=args.template_k,
    )

    train_dataset = apply_templates_to_dataset(
        raw_inner_train_dataset, asd_edge_index_np, hc_edge_index_np
    )
    monitor_dataset = apply_templates_to_dataset(
        raw_monitor_dataset, asd_edge_index_np, hc_edge_index_np
    )
    test_dataset = apply_templates_to_dataset(
        raw_test_dataset, asd_edge_index_np, hc_edge_index_np
    )

    print(f"Outer-train size : {len(outer_train_dataset)}")
    print(f"Inner-train size : {len(train_dataset)}")
    print(f"Monitor size     : {len(monitor_dataset)}")
    print(f"Test size        : {len(test_dataset)}")
    print(f"Inner-train labels: {label_counts(train_dataset)}")
    print(f"Monitor labels    : {label_counts(monitor_dataset)}")
    print(f"Test labels       : {label_counts(test_dataset)}")
    print(f"ASD template edges: {asd_edge_index_np.shape[1]}")
    print(f"HC template edges : {hc_edge_index_np.shape[1]}")

    train_loader, monitor_loader, test_loader = make_loaders(
        train_dataset=train_dataset,
        monitor_dataset=monitor_dataset,
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
    best_monitor_loss = float("inf")
    best_monitor_auc = float("nan")
    best_epoch = -1
    patience_counter = 0

    split_info = {
        "monitor_ratio": args.monitor_ratio,
        "monitor_seed": args.monitor_seed,
        "template_k": args.template_k,
        "outer_train_size": len(outer_train_dataset),
        "inner_train_size": len(train_dataset),
        "monitor_size": len(monitor_dataset),
        "test_size": len(test_dataset),
        "inner_train_indices_outer": inner_train_idx.tolist(),
        "monitor_indices_outer": monitor_idx.tolist(),
        "inner_train_label_distribution": label_counts(train_dataset),
        "monitor_label_distribution": label_counts(monitor_dataset),
        "test_label_distribution": label_counts(test_dataset),
        "asd_num_edges_directed": int(asd_edge_index_np.shape[1]),
        "hc_num_edges_directed": int(hc_edge_index_np.shape[1]),
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        monitor_metrics = evaluate(
            model=model,
            loader=monitor_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_auc": train_metrics["auc"],
            "monitor_loss": monitor_metrics["loss"],
            "monitor_acc": monitor_metrics["acc"],
            "monitor_auc": monitor_metrics["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"monitor_loss={monitor_metrics['loss']:.4f} monitor_acc={monitor_metrics['acc']:.4f} monitor_auc={monitor_metrics['auc']:.4f}"
        )

        current_monitor_loss = monitor_metrics["loss"]

        if current_monitor_loss < best_monitor_loss:
            best_monitor_loss = current_monitor_loss
            best_monitor_auc = monitor_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_monitor_loss": best_monitor_loss,
                    "best_monitor_auc": best_monitor_auc,
                    "args": vars(args),
                    "dataset_summary": dataset_summary,
                    "split_info": split_info,
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
                "best_monitor_loss": best_monitor_loss,
                "best_monitor_auc": best_monitor_auc,
                "args": vars(args),
                "dataset_summary": dataset_summary,
                "split_info": split_info,
            },
            last_ckpt_path,
        )

        save_history_csv(history, history_csv_path)

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    best_checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    final_test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("=" * 72)
    print("Training finished")
    print("=" * 72)
    print(f"Best monitor loss: {best_monitor_loss:.6f}")
    print(f"Best monitor AUC : {best_monitor_auc:.6f}")
    print(f"Best epoch       : {best_epoch}")
    print(f"Best checkpoint  : {best_ckpt_path}")
    print(f"Last checkpoint  : {last_ckpt_path}")
    print(f"History CSV      : {history_csv_path}")
    print(
        f"Best-ckpt test   : "
        f"loss={final_test_metrics['loss']:.4f} "
        f"acc={final_test_metrics['acc']:.4f} "
        f"auc={final_test_metrics['auc']:.4f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()