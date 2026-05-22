#!/usr/bin/env python3
"""
Pair-aware feature experiments on top of selected explainer outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError

from explain_analysis_utils import (
    DIRECTION_TO_INDEX,
    build_denorm_tables,
    build_generic_node_score_table,
    ensure_dir,
    load_cfg,
    load_model,
    resolve_device,
)
from explain_visualization import METHOD_DISPLAY_NAMES
from src.datasets import prepare_data, split_dataset


DEFAULT_METHODS = [
    "gnn_object_margin",
    "gnn_object_vs_no_defect",
    "pg_explainer",
    "notebook_object_log_probs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("out_Termo_Ablation_heads/8_heads"),
        help="Experiment directory with params.json and best_model.pth",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("out_Termo_Ablation_heads/8_heads/pyg_explainer_benchmark"),
        help="Benchmark directory with explainer example folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save pair-aware experiments",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method keys to analyze",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=12,
        help="How many top nodes from each method/example to analyze",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device; default auto",
    )
    parser.add_argument(
        "--distances-csv",
        type=Path,
        default=Path("sections_distances.csv"),
        help="Section graph distance matrix",
    )
    return parser.parse_args()


def parse_incident_sections(text: str) -> list[int]:
    if not isinstance(text, str) or not text:
        return []
    return [int(value) for value in text.split(",") if value]


def get_feature_indices(feature_names: list[str]) -> dict[str, int]:
    required = ["P", "Temp", "P_ideal", "Temp_ideal"]
    mapping = {name: feature_names.index(name) for name in required}
    return mapping


def load_top_node_table(
    example_dir: Path,
    nodes_denorm_df: pd.DataFrame,
    edges_denorm_df: pd.DataFrame,
    method: str,
    true_class: int,
    pred_class: int,
    top_nodes: int,
) -> tuple[pd.DataFrame, str]:
    node_scores_path = example_dir / "node_scores.csv"
    edge_scores_path = example_dir / "edge_scores.csv"

    try:
        node_scores_df = pd.read_csv(node_scores_path)
        if not node_scores_df.empty and "score" in node_scores_df.columns:
            top_node_df = node_scores_df.sort_values(["score", "rank"], ascending=[False, True]).head(top_nodes).copy()
            return top_node_df, "node_scores"
    except EmptyDataError:
        pass

    if not edge_scores_path.exists():
        return pd.DataFrame(), "missing_edge_scores"

    try:
        edge_scores_df = pd.read_csv(edge_scores_path)
    except EmptyDataError:
        return pd.DataFrame(), "empty_edge_scores"
    if edge_scores_df.empty or "score" not in edge_scores_df.columns:
        return pd.DataFrame(), "invalid_edge_scores"

    node_id_to_idx = {
        int(node_id): int(idx)
        for idx, node_id in enumerate(nodes_denorm_df["id"].astype(int).tolist())
    }
    node_scores = np.zeros(len(nodes_denorm_df), dtype=float)
    for row in edge_scores_df.itertuples(index=False):
        score = float(getattr(row, "score", 0.0))
        for node_id in (int(getattr(row, "id_in")), int(getattr(row, "id_out"))):
            node_idx = node_id_to_idx.get(node_id)
            if node_idx is None:
                continue
            node_scores[node_idx] = max(node_scores[node_idx], score)

    reconstructed_df = build_generic_node_score_table(
        nodes_df=nodes_denorm_df,
        edges_df=edges_denorm_df,
        node_scores=node_scores,
        method=f"{method}_edge_projection",
        true_class=true_class,
        pred_class=pred_class,
    )
    top_node_df = reconstructed_df.head(top_nodes).copy()
    return top_node_df, "edge_projection"


def compute_pair_gradients(model, data, target_class: int) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    x = data.x.detach().clone().requires_grad_(True)
    logits = model(x, data.edge_index, data)
    logits[0, target_class].backward()
    return x.grad.detach().clone()


def compute_pair_node_signals(
    model,
    data,
    node_table: pd.DataFrame,
    nodes_denorm_df: pd.DataFrame,
    feature_names: list[str],
    target_class: int,
    distance_df: pd.DataFrame,
    true_class: int,
) -> pd.DataFrame:
    feature_idx = get_feature_indices(feature_names)
    gradients = compute_pair_gradients(model, data, target_class=target_class)
    base_x = data.x.detach().clone()
    with torch.no_grad():
        base_logits = model(data.x, data.edge_index, data)
    base_logit = float(base_logits[0, target_class].item())

    no_defect_class = int(len(distance_df.index))
    near_sections = set()
    if int(true_class) != no_defect_class and int(true_class) < len(distance_df.index):
        near_sections = {
            int(section_id)
            for section_id, distance in enumerate(distance_df.iloc[int(true_class)].tolist())
            if int(distance) >= 0 and int(distance) <= 1
        }

    nodes_denorm = nodes_denorm_df.copy().reset_index(drop=True)
    nodes_denorm["node_idx"] = np.arange(len(nodes_denorm), dtype=int)

    rows = []
    for row in node_table.itertuples(index=False):
        node_idx = int(row.node_idx)
        incident_sections = parse_incident_sections(getattr(row, "incident_sections", ""))
        touches_near = any(section_id in near_sections for section_id in incident_sections)

        p_idx = feature_idx["P"]
        t_idx = feature_idx["Temp"]
        p_ideal_idx = feature_idx["P_ideal"]
        t_ideal_idx = feature_idx["Temp_ideal"]

        pressure_gap_norm = float(abs(base_x[node_idx, p_idx] - base_x[node_idx, p_ideal_idx]).item())
        temperature_gap_norm = float(abs(base_x[node_idx, t_idx] - base_x[node_idx, t_ideal_idx]).item())
        pair_gap_norm = pressure_gap_norm + temperature_gap_norm

        pressure_grad_gap = float(abs((gradients[node_idx, p_idx] * (base_x[node_idx, p_idx] - base_x[node_idx, p_ideal_idx])).item()))
        temperature_grad_gap = float(abs((gradients[node_idx, t_idx] * (base_x[node_idx, t_idx] - base_x[node_idx, t_ideal_idx])).item()))
        pair_grad_gap = pressure_grad_gap + temperature_grad_gap

        perturbed_x = base_x.clone()
        perturbed_x[node_idx, p_idx] = base_x[node_idx, p_ideal_idx]
        with torch.no_grad():
            pressure_logits = model(perturbed_x, data.edge_index, data)
        pressure_to_ideal_delta = base_logit - float(pressure_logits[0, target_class].item())

        perturbed_x = base_x.clone()
        perturbed_x[node_idx, t_idx] = base_x[node_idx, t_ideal_idx]
        with torch.no_grad():
            temperature_logits = model(perturbed_x, data.edge_index, data)
        temperature_to_ideal_delta = base_logit - float(temperature_logits[0, target_class].item())

        perturbed_x = base_x.clone()
        perturbed_x[node_idx, p_idx] = base_x[node_idx, p_ideal_idx]
        perturbed_x[node_idx, t_idx] = base_x[node_idx, t_ideal_idx]
        with torch.no_grad():
            thermo_logits = model(perturbed_x, data.edge_index, data)
        thermo_to_ideal_delta = base_logit - float(thermo_logits[0, target_class].item())

        denorm_row = nodes_denorm.loc[nodes_denorm["node_idx"] == node_idx].iloc[0]
        pressure_gap_value = float(abs(denorm_row["P"] - denorm_row["P_ideal"]))
        temperature_gap_value = float(abs(denorm_row["Temp"] - denorm_row["Temp_ideal"]))

        rows.append(
            {
                "node_idx": node_idx,
                "node_id": int(row.id),
                "incident_sections": getattr(row, "incident_sections", ""),
                "touches_true_section": bool(getattr(row, "touches_true_section", False)),
                "touches_pred_section": bool(getattr(row, "touches_pred_section", False)),
                "touches_near_section": touches_near,
                "base_node_score": float(getattr(row, "score", getattr(row, "importance_max", 0.0))),
                "pressure_gap_norm": pressure_gap_norm,
                "temperature_gap_norm": temperature_gap_norm,
                "pair_gap_norm": pair_gap_norm,
                "pressure_gap_value": pressure_gap_value,
                "temperature_gap_value": temperature_gap_value,
                "pressure_grad_gap": pressure_grad_gap,
                "temperature_grad_gap": temperature_grad_gap,
                "pair_grad_gap": pair_grad_gap,
                "pressure_to_ideal_delta": pressure_to_ideal_delta,
                "temperature_to_ideal_delta": temperature_to_ideal_delta,
                "thermo_to_ideal_delta": thermo_to_ideal_delta,
                "thermo_to_ideal_abs_delta": abs(thermo_to_ideal_delta),
            }
        )
    return pd.DataFrame(rows)


def summarize_pair_signals(signals_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_example = (
        signals_df.groupby(["method", "sample_id", "requested_category"])
        .agg(
            mean_pair_gap_norm=("pair_gap_norm", "mean"),
            mean_pair_grad_gap=("pair_grad_gap", "mean"),
            mean_pressure_to_ideal_delta=("pressure_to_ideal_delta", "mean"),
            mean_temperature_to_ideal_delta=("temperature_to_ideal_delta", "mean"),
            mean_thermo_to_ideal_delta=("thermo_to_ideal_delta", "mean"),
            total_thermo_abs_delta=("thermo_to_ideal_abs_delta", "sum"),
            near_thermo_abs_delta=("thermo_to_ideal_abs_delta", lambda values: float(values.iloc[signals_df.loc[values.index, "touches_near_section"].values].sum())),
        )
        .reset_index()
    )
    per_example["near_thermo_share"] = np.where(
        per_example["total_thermo_abs_delta"] > 0.0,
        per_example["near_thermo_abs_delta"] / per_example["total_thermo_abs_delta"],
        np.nan,
    )

    summary = (
        per_example.groupby("method")
        .agg(
            num_examples=("sample_id", "nunique"),
            mean_pair_gap_norm=("mean_pair_gap_norm", "mean"),
            mean_pair_grad_gap=("mean_pair_grad_gap", "mean"),
            mean_pressure_to_ideal_delta=("mean_pressure_to_ideal_delta", "mean"),
            mean_temperature_to_ideal_delta=("mean_temperature_to_ideal_delta", "mean"),
            mean_thermo_to_ideal_delta=("mean_thermo_to_ideal_delta", "mean"),
            mean_near_thermo_share=("near_thermo_share", "mean"),
        )
        .reset_index()
        .sort_values(["mean_near_thermo_share", "mean_thermo_to_ideal_delta"], ascending=[False, False])
    )

    category_summary = (
        per_example.groupby(["method", "requested_category"])
        .agg(
            num_examples=("sample_id", "nunique"),
            mean_pair_gap_norm=("mean_pair_gap_norm", "mean"),
            mean_pair_grad_gap=("mean_pair_grad_gap", "mean"),
            mean_thermo_to_ideal_delta=("mean_thermo_to_ideal_delta", "mean"),
            mean_near_thermo_share=("near_thermo_share", "mean"),
        )
        .reset_index()
    )
    return summary, category_summary


def plot_pair_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    plotting_df = summary_df.copy()
    plotting_df["display_name"] = plotting_df["method"].map(lambda name: METHOD_DISPLAY_NAMES.get(name, name))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(plotting_df["display_name"], plotting_df["mean_thermo_to_ideal_delta"], color="#4c72b0")
    axes[0].set_title("Mean thermo->ideal delta")
    axes[0].set_xlabel("Base logit drop")

    x = np.arange(len(plotting_df))
    width = 0.35
    axes[1].bar(x - width / 2, plotting_df["mean_pressure_to_ideal_delta"], width=width, color="#dd8452", label="P -> P_ideal")
    axes[1].bar(x + width / 2, plotting_df["mean_temperature_to_ideal_delta"], width=width, color="#55a868", label="T -> T_ideal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plotting_df["display_name"], rotation=20, ha="right")
    axes[1].set_title("Pressure vs temperature repair")
    axes[1].legend(fontsize=8)

    axes[2].barh(plotting_df["display_name"], plotting_df["mean_near_thermo_share"], color="#c44e52")
    axes[2].set_title("Mean near-section thermo share")
    axes[2].set_xlabel("Share of thermo delta on true/neighbor nodes")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = ensure_dir((args.output_dir or benchmark_dir / "pair_feature_experiments").resolve())
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    manifest_path = benchmark_dir / "explainer_benchmark_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["explainer"].isin(methods)].copy()
    if manifest_df.empty:
        raise RuntimeError(f"No benchmark examples found for methods: {methods}")

    cfg = load_cfg(exp_dir)
    device = resolve_device(args.device, cfg)
    distance_df = pd.read_csv(args.distances_csv, sep="\t")

    pair_dataset, pair_scalers, _, _, _, _ = prepare_data(
        cfg["dataset"],
        cfg["dataloader"],
        cfg["utils"]["seed"],
    )
    _, _, test_dataset = split_dataset(
        pair_dataset,
        cfg["dataloader"]["train_ratio"],
        cfg["dataloader"]["val_ratio"],
        seed=cfg["utils"]["seed"],
    )

    sample_graph = pair_dataset[0][0]
    model = load_model(cfg, sample_graph, exp_dir, device)
    feature_names = cfg["dataset"]["node_attr"]

    all_rows = []
    status_rows = []
    for record in manifest_df.itertuples(index=False):
        direction = str(record.direction)
        direction_idx = DIRECTION_TO_INDEX[direction]
        data = test_dataset[int(record.test_index)][direction_idx].to(device)
        nodes_denorm_df, edges_denorm_df = build_denorm_tables(
            data,
            pair_scalers[direction_idx],
            cfg,
            direction=direction,
        )

        source_example_dir = Path(str(record.example_dir))
        top_node_df, node_source = load_top_node_table(
            example_dir=source_example_dir,
            nodes_denorm_df=nodes_denorm_df,
            edges_denorm_df=edges_denorm_df,
            method=str(record.explainer),
            true_class=int(record.true_class),
            pred_class=int(record.pred_class),
            top_nodes=args.top_nodes,
        )
        if top_node_df.empty:
            status_rows.append(
                {
                    "method": str(record.explainer),
                    "sample_id": str(record.sample_id),
                    "requested_category": str(record.requested_category),
                    "status": "skipped_empty_nodes",
                    "node_source": node_source,
                }
            )
            continue

        signal_df = compute_pair_node_signals(
            model=model,
            data=data,
            node_table=top_node_df,
            nodes_denorm_df=nodes_denorm_df,
            feature_names=feature_names,
            target_class=int(record.pred_class),
            distance_df=distance_df,
            true_class=int(record.true_class),
        )
        signal_df = signal_df.assign(
            method=str(record.explainer),
            sample_id=str(record.sample_id),
            direction=direction,
            requested_category=str(record.requested_category),
            true_class=int(record.true_class),
            pred_class=int(record.pred_class),
            node_source=node_source,
        )
        all_rows.append(signal_df)
        status_rows.append(
            {
                "method": str(record.explainer),
                "sample_id": str(record.sample_id),
                "requested_category": str(record.requested_category),
                "status": "ok",
                "node_source": node_source,
            }
        )

        target_example_dir = ensure_dir(output_dir / "methods" / str(record.explainer) / "examples" / source_example_dir.name)
        signal_df.to_csv(target_example_dir / "node_pair_signals.csv", index=False)

    all_signals_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    status_df = pd.DataFrame(status_rows)
    summary_df, category_summary_df = summarize_pair_signals(all_signals_df)

    all_signals_df.to_csv(output_dir / "pair_feature_node_signals.csv", index=False)
    status_df.to_csv(output_dir / "pair_feature_run_status.csv", index=False)
    summary_df.to_csv(output_dir / "pair_feature_method_summary.csv", index=False)
    category_summary_df.to_csv(output_dir / "pair_feature_category_summary.csv", index=False)
    plot_pair_summary(summary_df, output_dir / "pair_feature_method_summary.png")

    metadata = {
        "exp_dir": str(exp_dir),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "methods": methods,
        "top_nodes": int(args.top_nodes),
        "device": str(device),
    }
    (output_dir / "pair_feature_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved pair feature experiments to: {output_dir}")


if __name__ == "__main__":
    main()
