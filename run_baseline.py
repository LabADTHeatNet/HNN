#!/usr/bin/env python3
r"""
Baseline non-graph models for the HNN classification task.

Trains sklearn classifiers (RF, GBDT, MLP) on flat feature vectors
extracted from PyG Data objects, then evaluates using the same metrics
as the GNN pipeline.

Usage:
    python run_baseline.py --model rf
    python run_baseline.py --model gbdt
    python run_baseline.py --model mlp --seed 123

Dependencies: scikit-learn, numpy, torch, matplotlib (all in the hnn env).
"""

import argparse
import copy
import json
import os.path as osp
import pprint
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from src.datasets import prepare_data
from src.baselines import (
    create_model,
    evaluate,
    extract_dataset,
)
from src.utils import get_str_timestamp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train baseline sklearn models for HNN classification."
    )
    parser.add_argument(
        "--model",
        choices=["rf", "gbdt", "mlp"],
        default="rf",
        help="Which baseline model to train (default: rf).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42, matching train.py).",
    )
    parser.add_argument(
        "--out-dir",
        default="out_Termo_both",
        help="Root output directory (default: out_Termo_both).",
    )
    parser.add_argument(
        "--fp",
        default="data_Termo_Heat.pt",
        help="Preprocessed dataset file (default: data_Termo_Heat.pt).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Build config (same keys as train.py) ---
    node_attr = [
        "pos_x", "pos_y",
        "types_def", "types_usr", "types_src",
        "P", "Temp", "P_ideal", "Temp_ideal",
    ]
    edge_attr = ["d", "l", "Vid_fwd", "Vid_bwd", "Vid_usr"]
    in_global_dim = 5

    dataset_cfg = dict(
        datasets_dir=osp.join(".", "datasets"),
        name="Termo_model_fwd_and_bwd",
        load=True,
        fp=args.fp,
        node_attr=node_attr,
        edge_attr=edge_attr,
        edge_label=["graph_label"],
        scaler_fn="StandardScaler",
        num_samples=args.num_samples,
        add_ideal=True,
    )

    dataloader_cfg = dict(
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=16,  # not used by baselines but kept for config completeness
    )

    utils_cfg = dict(
        server_name="seth",
        out_dir=args.out_dir,
        device=device,
        seed=args.seed,
    )

    # --- Load dataset -----
    print("Loading dataset...")
    dataset, scalers = prepare_data(
        dataset_cfg, dataloader_cfg, args.seed, prepare_dataloaders=False
    )
    total = len(dataset)
    train_ratio = dataloader_cfg["train_ratio"]
    val_ratio = dataloader_cfg["val_ratio"]
    train_len = int(train_ratio * total)
    val_len = int(val_ratio * total)
    test_len = total - train_len - val_len

    torch.manual_seed(args.seed)
    train_sub, val_sub, test_sub = random_split(
        range(total), [train_len, val_len, test_len]
    )
    print(f"Train: {len(train_sub)}, Val: {len(val_sub)}, Test: {len(test_sub)}")

    # --- Extract flat features ---
    print("Extracting features...")
    X_train, y_train = extract_dataset(dataset, train_sub.indices)
    X_val, y_val = extract_dataset(dataset, val_sub.indices)
    X_test, y_test = extract_dataset(dataset, test_sub.indices)

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Training labels: {len(np.unique(y_train))} unique classes "
          f"(out of {44} total)")

    # --- Validate no NaN / inf ---
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        assert np.all(np.isfinite(X)), f"Non-finite values found in {name} features!"

    # --- Train model -----
    print(f"\nTraining {args.model}...")
    model = create_model(args.model, random_state=args.seed)

    if args.model in ("gbdt", "mlp"):
        # These support early stopping with validation data
        model.fit(X_train, y_train)
    else:
        # RandomForest doesn't use validation during fit
        model.fit(X_train, y_train)

    # --- Evaluate ---
    print("\n--- Test Set Evaluation ---")
    y_pred = model.predict(X_test).astype(int)

    # Create output directory
    exp_name = (
        f"baseline_{args.model}_seed{args.seed}_"
        f"{get_str_timestamp()}"
    )
    out_dir = Path(utils_cfg["out_dir"]) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg_dump = {
        "dataset": dataset_cfg,
        "model": {"name": args.model},
        "train": {"num_epochs": "N/A (sklearn)"},
    }
    with open(out_dir / "params.json", "w") as f:
        json.dump(cfg_dump, f, indent=4)

    # Run evaluation (same format as GNN test_exp)
    metrics = evaluate(
        y_test, y_pred,
        out_dir=out_dir, num_classes=44,
    )
    metrics["model"] = args.model
    metrics["seed"] = args.seed
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    print(f"\nResults saved to: {out_dir}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Binary F1:     {metrics['binary_f1']:.4f}")

    # --- Also report train accuracy for diagnostics ---
    y_train_pred = model.predict(X_train).astype(int)
    train_acc = (y_train_pred == y_train).mean()
    print(f"Train accuracy: {train_acc:.4f}")

    # Save model
    import joblib
    joblib.dump(model, out_dir / "model.joblib")
    print(f"Model saved to: {out_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()