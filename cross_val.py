#!/usr/bin/env python3
"""
Cross-validation script for EdgeClassifierNetwork_Attr graph model.

Performs K-fold cross-validation on the Termo paired dataset using
the same architecture and training procedure as train.py.

Usage:
    python cross_val.py [--k-folds 5] [--epochs 100] [--lr 1e-3]
                        [--hidden-channels 128] [--batch-size 16]
                        [--seed 42] [--device cuda]

Output:
    out_Termo_both/cross_val_*/aggregate_metrics.json   (mean +- std across folds)
    out_Termo_both/cross_val_*/fold_summary.csv         (per-fold metrics table)
    out_Termo_both/cross_val_*/fold_i/                  (best model, metrics.json, tensorboard)
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from exp import _create_criterion
from src.datasets import paired_collate, prepare_data
from src.models.EdgeClassifierNetwork_Attr import EdgeClassifierNetwork_Attr
from src.utils import compute_metrics, get_str_timestamp, train, valid


def compute_binary_metrics(pred_classes, all_targets):
    """Binary metrics: all classes except last (43) are positive, class 43 is negative."""
    binary_targets = all_targets != 43
    binary_predictions = pred_classes != 43
    precision = precision_score(binary_targets, binary_predictions, zero_division=0)
    recall = recall_score(binary_targets, binary_predictions, zero_division=0)
    f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
    return {"Precision": precision, "Recall": recall, "F1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validation for EdgeClassifierNetwork_Attr")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Node hidden channels")
    parser.add_argument("--num-node-layers", type=int, default=8, help="Number of node GNN layers")
    parser.add_argument("--edge-hidden-channels", type=int, default=128, help="Edge hidden channels")
    parser.add_argument("--num-edge-layers", type=int, default=8, help="Number of edge attention layers")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="out_Termo_both", help="Output directory root")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ #
    # Configuration (mirrors train.py defaults)
    # ------------------------------------------------------------------ #
    init_lr = args.lr
    epochs_num = args.epochs
    final_lr = 1e-6

    dataset_config = dict(
        datasets_dir="datasets",
        name="Termo_model_fwd_and_bwd",
        load=True,
        fp="data_Termo_Heat.pt",
        node_attr=["pos_x", "pos_y", "types_def", "types_usr", "types_src", "P", "Temp",
                   "P_ideal", "Temp_ideal"],
        edge_attr=["d", "l", "Vid_fwd", "Vid_bwd", "Vid_usr"],
        edge_label=["graph_label"],
        scaler_fn="StandardScaler",
        num_samples=None,
        add_ideal=True,
    )

    dataloader_config = dict(
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------ #
    # Load full dataset (no pre-defined splits)
    # ------------------------------------------------------------------ #
    print("Loading dataset ...")
    dataset, _scalers = prepare_data(dataset_config, dataloader_config, args.seed,
                                      prepare_dataloaders=False)

    # Infer input dimensions from the first paired sample
    fwd_sample, _ = dataset[0]
    in_node_dim = fwd_sample.x.shape[1]
    in_edge_dim = fwd_sample.edge_attr.shape[1]
    in_global_dim = fwd_sample.global_attrs.shape[1]
    print(f"  in_node_dim={in_node_dim}, in_edge_dim={in_edge_dim}, in_global_dim    ={in_global_dim}")
    print(f"  Total paired samples: {len(dataset)}")

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    exp_name = "_".join([
        "cross_val",
        dataset_config["scaler_fn"],
        "EdgeClassifierNetwork_Attr",
        f"bs{args.batch_size}",
        get_str_timestamp(),
    ])
    out_dir = Path(args.out_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full config for reproducibility
    cfg_snapshot = {
        "dataset": copy.deepcopy(dataset_config),
        "dataloader": copy.deepcopy(dataloader_config),
        "model": {
            "name": "EdgeClassifierNetwork_Attr",
            "kwargs": {
                "in_node_dim": in_node_dim,
                "in_edge_dim": in_edge_dim,
                "out_dim": 44,
                "node_hidden_channels": args.hidden_channels,
                "num_node_layers": args.num_node_layers,
                "edge_hidden_channels": args.edge_hidden_channels,
                "num_edge_layers": args.num_edge_layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "jump_mode": "cat",
                "in_global_dim": in_global_dim,
            },
        },
        "optimizer": {"name": "Adam", "lr": init_lr},
        "train": {"num_epochs": epochs_num, "k_folds": args.k_folds, "seed": args.seed},
    }
    with open(out_dir / "params.json", "w") as f:
        json.dump(cfg_snapshot, f, indent=2)

    # ------------------------------------------------------------------ #
    # K-Fold cross-validation
    # ------------------------------------------------------------------ #
    class_weights = [1.0] * 44
    class_weights[43] = 0.5

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\n{'=' * 40} Fold {fold + 1} / {args.k_folds} {'=' * 40}")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_dir)

        # --- fold datasets & loaders ---
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=paired_collate)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=paired_collate)

        print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

        # --- fresh model per fold ---
        model = EdgeClassifierNetwork_Attr(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            out_dim=44,
            node_hidden_channels=args.hidden_channels,
            num_node_layers=args.num_node_layers,
            edge_hidden_channels=args.edge_hidden_channels,
            num_edge_layers=args.num_edge_layers,
            heads=args.heads,
            dropout=args.dropout,
            jump_mode="cat",
            in_global_dim=in_global_dim,
        ).to(device)

        # --- optimizer ---
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=init_lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6,
        )

        # --- scheduler: StepLR with geometric decay ---
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=pow(final_lr / init_lr, 1 / epochs_num)
        )

        # --- criterion (matches train.py) ---
        criterion_cfg = {
            "criterion": {
                "name": "CrossEntropyLoss",
                "kwargs": {"weight": class_weights, "label_smoothing": 0.1},
            }
        }
        criterion = _create_criterion(criterion_cfg, device)

        # --- training loop ---
        best_score = float("inf")
        best_epoch = -1
        edge_label_scaler = None

        for epoch in range(epochs_num):
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            train_metrics = train(model, train_loader, optimizer, criterion, device,
                                  scaler=edge_label_scaler, max_norm=1e-2)
            valid_metrics = valid(model, val_loader, criterion, device,
                                  scaler=edge_label_scaler)

            for k, v in train_metrics.items():
                writer.add_scalar(f"{k}/train", v, epoch)
            for k, v in valid_metrics.items():
                writer.add_scalar(f"{k}/val", v, epoch)

            if valid_metrics["Loss"] < best_score:
                best_score = valid_metrics["Loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), fold_dir / "best_model.pth")

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs_num}  "
                      f"Train Loss: {train_metrics['Loss']:.4f}  "
                      f"Val Loss: {valid_metrics['Loss']:.4f}  "
                      f"Val Acc: {valid_metrics.get('Accuracy', 0):.4f}")

        print(f"  Best epoch: {best_epoch+1} (val Loss = {best_score:.4f})")

        # --- evaluate best model on val set ---
        state_dict = torch.load(fold_dir / "best_model.pth", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    batch_fwd, batch_bwd = batch
                    batch_fwd = batch_fwd.to(device)
                    batch_bwd = batch_bwd.to(device)
                    pred_fwd = model(batch_fwd)
                    pred_bwd = model(batch_bwd)
                    preds = (pred_fwd + pred_bwd) / 2
                    targets = batch_fwd.edge_label
                else:
                    batch = batch.to(device)
                    preds = model(batch)
                    targets = batch.edge_label
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        pred_classes = all_preds.argmax(dim=1)

        metrics = {"Loss": best_score}
        metrics.update(compute_metrics(all_preds, all_targets, scaler=None))
        metrics.update(compute_binary_metrics(pred_classes.numpy(), all_targets.numpy()))

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        fold_results.append(metrics)
        writer.close()

        print(f"  -> Fold metrics: {json.dumps(metrics)}")

    # ------------------------------------------------------------------ #
    # Aggregate across folds
    # ------------------------------------------------------------------ #
    metric_names = [k for k in fold_results[0]]
    aggregated = {}
    for name in metric_names:
        values = [r[name] for r in fold_results]
        aggregated[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    print("\n" + "=" * 60)
    print(f"Cross-validation results ({args.k_folds} folds):")
    print("=" * 60)
    for name in sorted(aggregated):
        s = aggregated[name]
        print(f"  {name:15s}  {s['mean']:.4f} +- {s['std']:.4f}")

    # Save aggregated results
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # CSV summary
    with open(out_dir / "fold_summary.csv", "w", newline="") as f:
        fieldnames = ["fold"] + metric_names
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, m in enumerate(fold_results):
            w.writerow({"fold": i, **m})

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
