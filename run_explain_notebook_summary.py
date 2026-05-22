#!/usr/bin/env python3
"""
Run a corrected single-graph explainer pass similar to explain.ipynb and print
a compact textual summary for one sample.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from explain_analysis_utils import (
    DIRECTION_TO_INDEX,
    PAIR_AWARE_EXPLANATION_METHOD,
    PRIMARY_EXPLANATION_METHOD,
    build_denorm_tables,
    build_edge_table,
    build_explainer,
    build_node_table,
    build_section_scores,
    compute_edge_feature_ablation,
    compute_node_feature_ablation,
    get_tout_from_path,
    load_cfg,
    load_model,
    resolve_device,
    sample_id_from_path,
)
from src.datasets import prepare_data, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("out_Termo_Ablation_heads/8_heads"),
        help="Experiment directory with params.json and best_model.pth",
    )
    parser.add_argument(
        "--distances-csv",
        type=Path,
        default=Path("sections_distances.csv"),
        help="Section-to-section distance matrix used to infer the no-defect class",
    )
    parser.add_argument(
        "--split",
        choices=["pair", "train", "val", "test"],
        default="test",
        help="Dataset split to sample from",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index inside the selected split",
    )
    parser.add_argument(
        "--direction",
        choices=["fwd", "bwd"],
        default="fwd",
        help="Which graph direction to explain",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Default: auto",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Optimization epochs for GNNExplainer",
    )
    parser.add_argument(
        "--explainer-kind",
        choices=[PRIMARY_EXPLANATION_METHOD, PAIR_AWARE_EXPLANATION_METHOD],
        default=PAIR_AWARE_EXPLANATION_METHOD,
        help="Which explainer input space to use",
    )
    parser.add_argument(
        "--top-sections",
        type=int,
        default=5,
        help="How many highest-ranked sections to print",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=5,
        help="How many highest-ranked nodes to print",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=5,
        help="How many highest-ranked edges to print",
    )
    parser.add_argument(
        "--top-node-features",
        type=int,
        default=3,
        help="How many ablated features to print per top node",
    )
    parser.add_argument(
        "--top-edge-features",
        type=int,
        default=3,
        help="How many ablated features to print per top edge",
    )
    parser.add_argument(
        "--pair-top-nodes",
        type=int,
        default=5,
        help="How many pred-supporting nodes to print for pair-aware diagnostics",
    )
    parser.add_argument(
        "--pred-neighborhood-radius",
        type=int,
        default=1,
        help="Graph distance radius around Pred considered local evidence",
    )
    return parser.parse_args()


def format_scalar(value) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e3 or magnitude < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def format_class_label(class_id: int, no_defect_class: int) -> str:
    return "no_defect" if int(class_id) == int(no_defect_class) else f"section {int(class_id)}"


def format_attr_values(row: pd.Series, feature_names: list[str]) -> str:
    return ", ".join(
        f"{feature_name}={format_scalar(row[feature_name])}"
        for feature_name in feature_names
        if feature_name in row.index
    )


def format_node_attr_values(row: pd.Series) -> str:
    fields = [
        ("pos_x", "x"),
        ("pos_y", "y"),
        ("types_def", "def"),
        ("types_usr", "usr"),
        ("types_src", "src"),
        ("P", "P"),
        ("P_ideal", "P_ideal"),
        ("delta_P", "delta_P"),
        ("Temp", "T"),
        ("Temp_ideal", "T_ideal"),
        ("delta_T", "delta_T"),
    ]
    parts = []
    for source_name, display_name in fields:
        if source_name in row.index:
            parts.append(f"{display_name}={format_scalar(row[source_name])}")
    return ", ".join(parts)


def build_feature_summary(
    feature_df: pd.DataFrame,
    entity_col: str,
    top_n: int,
) -> dict[int, str]:
    if feature_df.empty or top_n <= 0:
        return {}

    compact_df = (
        feature_df.sort_values([entity_col, "abs_delta_logit"], ascending=[True, False])
        .groupby(entity_col)
        .head(top_n)
        .reset_index(drop=True)
    )

    summary: dict[int, str] = {}
    for entity_id, group in compact_df.groupby(entity_col, sort=False):
        parts = []
        for row in group.itertuples(index=False):
            parts.append(
                f"{row.feature_name}(dlogit={format_scalar(row.delta_logit)}, val_norm={format_scalar(row.feature_value_norm)})"
            )
        summary[int(entity_id)] = ", ".join(parts)
    return summary


def select_pair(
    pair_dataset,
    split_name: str,
    index: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    train_dataset, val_dataset, test_dataset = split_dataset(
        pair_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split_map = {
        "pair": pair_dataset,
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    split_dataset_obj = split_map[split_name]
    if index < 0 or index >= len(split_dataset_obj):
        raise IndexError(f"{split_name} index {index} is out of range for dataset of size {len(split_dataset_obj)}")

    original_indices = getattr(split_dataset_obj, "indices", None)
    original_index = int(original_indices[index]) if original_indices is not None else int(index)
    return split_dataset_obj[index], original_index


def compute_prediction_summary(
    model,
    data,
) -> tuple[int, int, float, float, float]:
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data)
        probs = logits.softmax(dim=-1)
    true_class = int(data.edge_label.view(-1)[0].item())
    pred_class = int(probs.argmax(dim=-1)[0].item())
    pred_confidence = float(probs[0, pred_class].item())
    true_confidence = float(probs[0, true_class].item())
    top2 = torch.topk(probs[0], k=min(2, probs.shape[-1]), dim=-1).values
    pred_margin = float(top2[0].item() - top2[1].item()) if top2.numel() > 1 else float(top2[0].item())
    return true_class, pred_class, pred_confidence, true_confidence, pred_margin


def compute_section_ranks(section_scores: pd.DataFrame, true_class: int, pred_class: int) -> tuple[int | None, int | None]:
    ordered_sections = section_scores["id_section"].astype(int).tolist()
    true_rank = ordered_sections.index(int(true_class)) + 1 if int(true_class) in ordered_sections else None
    pred_rank = ordered_sections.index(int(pred_class)) + 1 if int(pred_class) in ordered_sections else None
    return true_rank, pred_rank


def parse_incident_sections(section_text: str) -> list[int]:
    if not isinstance(section_text, str) or not section_text:
        return []
    return [int(value) for value in section_text.split(",") if value]


def section_neighborhood(section_id: int, distance_df: pd.DataFrame | None, radius: int = 1) -> set[int]:
    if distance_df is None or int(section_id) >= len(distance_df.index):
        return {int(section_id)}
    distances = distance_df.iloc[int(section_id)].astype(int).tolist()
    return {
        int(other_section)
        for other_section, distance in enumerate(distances)
        if distance >= 0 and distance <= int(radius)
    }


def augment_node_pair_columns(nodes_df: pd.DataFrame) -> pd.DataFrame:
    nodes_df = nodes_df.copy()
    if {"P", "P_ideal"}.issubset(nodes_df.columns):
        nodes_df["delta_P"] = nodes_df["P"] - nodes_df["P_ideal"]
        nodes_df["abs_delta_P"] = nodes_df["delta_P"].abs()
    if {"Temp", "Temp_ideal"}.issubset(nodes_df.columns):
        nodes_df["delta_T"] = nodes_df["Temp"] - nodes_df["Temp_ideal"]
        nodes_df["abs_delta_T"] = nodes_df["delta_T"].abs()
    return nodes_df


def get_raw_feature_indices(raw_feature_names: list[str]) -> dict[str, int]:
    required = ["P", "Temp", "P_ideal", "Temp_ideal"]
    return {name: raw_feature_names.index(name) for name in required}


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
    raw_feature_names: list[str],
    target_class: int,
    pred_class: int,
    distance_df: pd.DataFrame | None,
    pred_neighborhood_radius: int,
) -> pd.DataFrame:
    if node_table.empty:
        return pd.DataFrame()

    feature_idx = get_raw_feature_indices(raw_feature_names)
    gradients = compute_pair_gradients(model, data, target_class=target_class)
    base_x = data.x.detach().clone()
    with torch.no_grad():
        base_logits = model(data.x, data.edge_index, data)
    base_logit = float(base_logits[0, target_class].item())

    pred_neighborhood = section_neighborhood(pred_class, distance_df, radius=pred_neighborhood_radius)
    nodes_denorm = nodes_denorm_df.copy().reset_index(drop=True)
    nodes_denorm["node_idx"] = np.arange(len(nodes_denorm), dtype=int)

    rows = []
    for row in node_table.itertuples(index=False):
        node_idx = int(row.node_idx)
        incident_sections = parse_incident_sections(getattr(row, "incident_sections", ""))
        touches_pred_neighborhood = any(section_id in pred_neighborhood for section_id in incident_sections)

        p_idx = feature_idx["P"]
        t_idx = feature_idx["Temp"]
        p_ideal_idx = feature_idx["P_ideal"]
        t_ideal_idx = feature_idx["Temp_ideal"]

        pressure_gap_norm = float(abs(base_x[node_idx, p_idx] - base_x[node_idx, p_ideal_idx]).item())
        temperature_gap_norm = float(abs(base_x[node_idx, t_idx] - base_x[node_idx, t_ideal_idx]).item())
        pair_gap_norm = pressure_gap_norm + temperature_gap_norm

        pressure_grad_gap = float(
            abs((gradients[node_idx, p_idx] * (base_x[node_idx, p_idx] - base_x[node_idx, p_ideal_idx])).item())
        )
        temperature_grad_gap = float(
            abs((gradients[node_idx, t_idx] * (base_x[node_idx, t_idx] - base_x[node_idx, t_ideal_idx])).item())
        )
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
        pressure_delta_value = float(denorm_row["P"] - denorm_row["P_ideal"])
        temperature_delta_value = float(denorm_row["Temp"] - denorm_row["Temp_ideal"])
        pressure_gap_value = float(abs(pressure_delta_value))
        temperature_gap_value = float(abs(temperature_delta_value))
        base_node_score = float(getattr(row, "importance_max", getattr(row, "score", 0.0)))

        rows.append(
            {
                "node_idx": node_idx,
                "node_id": int(row.id),
                "incident_sections": getattr(row, "incident_sections", ""),
                "touches_pred_section": bool(getattr(row, "touches_pred_section", False)),
                "touches_pred_neighborhood": touches_pred_neighborhood,
                "base_node_score": base_node_score,
                "pressure_gap_norm": pressure_gap_norm,
                "temperature_gap_norm": temperature_gap_norm,
                "pair_gap_norm": pair_gap_norm,
                "pressure_delta_value": pressure_delta_value,
                "temperature_delta_value": temperature_delta_value,
                "pressure_gap_value": pressure_gap_value,
                "temperature_gap_value": temperature_gap_value,
                "pressure_grad_gap": pressure_grad_gap,
                "temperature_grad_gap": temperature_grad_gap,
                "pair_grad_gap": pair_grad_gap,
                "pressure_to_ideal_delta": pressure_to_ideal_delta,
                "temperature_to_ideal_delta": temperature_to_ideal_delta,
                "thermo_to_ideal_delta": thermo_to_ideal_delta,
                "thermo_to_ideal_abs_delta": abs(thermo_to_ideal_delta),
                "pred_support_score": max(0.0, thermo_to_ideal_delta) * base_node_score,
                "pred_contra_score": max(0.0, -thermo_to_ideal_delta) * base_node_score,
            }
        )
    return pd.DataFrame(rows)


def build_pair_section_summary(
    pair_signals_df: pd.DataFrame,
    section_scores: pd.DataFrame,
    pred_class: int,
    true_class: int,
    distance_df: pd.DataFrame | None,
    pred_neighborhood_radius: int,
) -> pd.DataFrame:
    if pair_signals_df.empty:
        return pd.DataFrame()

    rows = []
    for row in pair_signals_df.itertuples(index=False):
        for section_id in parse_incident_sections(row.incident_sections):
            rows.append(
                {
                    "id_section": int(section_id),
                    "node_id": int(row.node_id),
                    "base_node_score": float(row.base_node_score),
                    "positive_thermo_delta": max(0.0, float(row.thermo_to_ideal_delta)),
                    "abs_thermo_delta": float(row.thermo_to_ideal_abs_delta),
                    "weighted_positive_thermo_delta": max(0.0, float(row.thermo_to_ideal_delta)) * float(row.base_node_score),
                    "weighted_pressure_delta": max(0.0, float(row.pressure_to_ideal_delta)) * float(row.base_node_score),
                    "weighted_temperature_delta": max(0.0, float(row.temperature_to_ideal_delta)) * float(row.base_node_score),
                    "pair_gap_norm": float(row.pair_gap_norm),
                    "pair_grad_gap": float(row.pair_grad_gap),
                    "pressure_gap_value": float(row.pressure_gap_value),
                    "temperature_gap_value": float(row.temperature_gap_value),
                }
            )

    section_pair_df = (
        pd.DataFrame(rows)
        .groupby("id_section")
        .agg(
            num_signal_nodes=("node_id", "nunique"),
            sum_node_score=("base_node_score", "sum"),
            max_node_score=("base_node_score", "max"),
            positive_thermo_delta_sum=("positive_thermo_delta", "sum"),
            abs_thermo_delta_sum=("abs_thermo_delta", "sum"),
            weighted_positive_thermo_delta=("weighted_positive_thermo_delta", "sum"),
            weighted_pressure_delta=("weighted_pressure_delta", "sum"),
            weighted_temperature_delta=("weighted_temperature_delta", "sum"),
            max_pair_gap_norm=("pair_gap_norm", "max"),
            max_pair_grad_gap=("pair_grad_gap", "max"),
            max_pressure_gap_value=("pressure_gap_value", "max"),
            max_temperature_gap_value=("temperature_gap_value", "max"),
            mean_pressure_gap_value=("pressure_gap_value", "mean"),
            mean_temperature_gap_value=("temperature_gap_value", "mean"),
        )
        .reset_index()
    )

    pred_neighborhood = section_neighborhood(pred_class, distance_df, radius=pred_neighborhood_radius)
    merged = section_scores.merge(section_pair_df, on="id_section", how="left")
    numeric_columns = [
        "num_signal_nodes",
        "sum_node_score",
        "max_node_score",
        "positive_thermo_delta_sum",
        "abs_thermo_delta_sum",
        "weighted_positive_thermo_delta",
        "weighted_pressure_delta",
        "weighted_temperature_delta",
        "max_pair_gap_norm",
        "max_pair_grad_gap",
        "max_pressure_gap_value",
        "max_temperature_gap_value",
        "mean_pressure_gap_value",
        "mean_temperature_gap_value",
    ]
    for column in numeric_columns:
        merged[column] = merged[column].fillna(0.0)

    merged["dist_to_pred"] = merged["id_section"].map(
        lambda section_id: int(distance_df.iloc[int(pred_class), int(section_id)])
        if distance_df is not None and int(pred_class) < len(distance_df.index) and int(section_id) < len(distance_df.columns)
        else (0 if int(section_id) == int(pred_class) else -1)
    )
    merged["dist_to_true"] = merged["id_section"].map(
        lambda section_id: int(distance_df.iloc[int(true_class), int(section_id)])
        if distance_df is not None and int(true_class) < len(distance_df.index) and int(section_id) < len(distance_df.columns)
        else (0 if int(section_id) == int(true_class) else -1)
    )
    merged["in_pred_neighborhood"] = merged["id_section"].astype(int).isin(pred_neighborhood)
    merged = merged.sort_values(
        ["summary_score", "weighted_positive_thermo_delta", "id_section"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return merged


def compute_pred_neighborhood_mass_ratio(section_scores: pd.DataFrame, pred_neighborhood: set[int]) -> float:
    if section_scores.empty:
        return float("nan")
    total_mass = float(section_scores["explanation_mass"].sum())
    if total_mass <= 0.0:
        return float("nan")
    neighborhood_mass = float(section_scores.loc[section_scores["id_section"].isin(pred_neighborhood), "explanation_mass"].sum())
    return neighborhood_mass / total_mass


def select_competitor_section(
    pair_section_df: pd.DataFrame,
    pred_class: int,
    pred_neighborhood: set[int],
) -> int | None:
    if pair_section_df.empty:
        return None
    outside = pair_section_df[
        (~pair_section_df["id_section"].astype(int).isin(pred_neighborhood))
        & (pair_section_df["id_section"].astype(int) != int(pred_class))
    ].copy()
    if outside.empty:
        outside = pair_section_df[pair_section_df["id_section"].astype(int) != int(pred_class)].copy()
    if outside.empty:
        return None
    outside = outside.sort_values(
        ["weighted_positive_thermo_delta", "summary_score", "id_section"],
        ascending=[False, False, True],
    )
    return int(outside.iloc[0]["id_section"])


def build_display_section_scores(section_scores: pd.DataFrame) -> pd.DataFrame:
    display_scores = section_scores.copy()
    display_scores["summary_score"] = np.maximum(
        display_scores["node_importance_max"].astype(float),
        display_scores["edge_importance_max"].astype(float),
    )
    display_scores["explanation_mass"] = (
        display_scores["node_importance_sum"].astype(float)
        + display_scores["edge_importance_sum"].astype(float)
    )
    display_scores = display_scores.sort_values(
        ["summary_score", "explanation_mass", "id_section"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return display_scores


def print_summary(
    args: argparse.Namespace,
    sample_id: str,
    tout: int,
    original_index: int,
    true_class: int,
    pred_class: int,
    no_defect_class: int,
    pred_confidence: float,
    true_confidence: float,
    pred_margin: float,
    section_scores: pd.DataFrame,
    top_edge_table: pd.DataFrame,
    top_node_table: pd.DataFrame,
    edge_feature_summary: dict[int, str],
    node_feature_summary: dict[int, str],
    edge_feature_names: list[str],
    pred_neighborhood: set[int],
) -> None:
    true_rank, pred_rank = compute_section_ranks(section_scores, true_class, pred_class)

    print(
        f"Sample: split={args.split} index={args.index} original_index={original_index} "
        f"direction={args.direction} tout={tout}"
    )
    print(f"Sample id: {sample_id}")
    print(
        f"GT={format_class_label(true_class, no_defect_class)} | "
        f"Pred={format_class_label(pred_class, no_defect_class)} | "
        f"p(pred)={format_scalar(pred_confidence)} | "
        f"p(gt)={format_scalar(true_confidence)} | "
        f"margin={format_scalar(pred_margin)}"
    )
    print(
        f"Section ranks in explanation: GT={true_rank if true_rank is not None else '-'} | "
        f"Pred={pred_rank if pred_rank is not None else '-'}"
    )

    print("\nTop sections:")
    for rank, row in enumerate(section_scores.head(args.top_sections).itertuples(index=False), start=1):
        markers = []
        if int(row.id_section) == int(true_class):
            markers.append("GT")
        if int(row.id_section) == int(pred_class):
            markers.append("Pred")
        if int(row.id_section) in pred_neighborhood and int(row.id_section) != int(pred_class):
            markers.append("PredNear")
        marker_text = f" [{' / '.join(markers)}]" if markers else ""
        print(
            f"  {rank}. section {int(row.id_section)} | score={format_scalar(row.summary_score)} | "
            f"mass={format_scalar(row.explanation_mass)} | "
            f"node_max={format_scalar(row.node_importance_max)} | "
            f"edge_max={format_scalar(row.edge_importance_max)}{marker_text}"
        )

    print("\nTop edges:")
    for rank, (_, row) in enumerate(top_edge_table.iterrows(), start=1):
        edge_idx = int(row["edge_idx"])
        print(
            f"  {rank}. edge_idx={edge_idx} {int(row['id_in'])}->{int(row['id_out'])} | "
            f"section={int(row['id_section'])} | importance={format_scalar(row['importance'])}"
        )
        print(f"     attrs: {format_attr_values(row, edge_feature_names)}")
        feature_text = edge_feature_summary.get(edge_idx)
        if feature_text:
            print(f"     top edge attrs by ablation: {feature_text}")

    print("\nTop nodes:")
    for rank, (_, row) in enumerate(top_node_table.iterrows(), start=1):
        node_idx = int(row["node_idx"])
        sections = row["incident_sections"] if row["incident_sections"] else "-"
        print(
            f"  {rank}. node_idx={node_idx} node_id={int(row['id'])} | "
            f"sections={sections} | importance={format_scalar(row['importance_max'])}"
        )
        print(f"     attrs: {format_node_attr_values(row)}")
        feature_text = node_feature_summary.get(node_idx)
        if feature_text:
            print(f"     top node attrs by ablation: {feature_text}")


def print_pair_rationale_summary(
    args: argparse.Namespace,
    pair_section_df: pd.DataFrame,
    pair_signals_df: pd.DataFrame,
    pred_class: int,
    true_class: int,
    pred_neighborhood: set[int],
) -> None:
    if pair_section_df.empty or pair_signals_df.empty:
        return

    pred_mass_ratio = compute_pred_neighborhood_mass_ratio(pair_section_df, pred_neighborhood)
    competitor_section = select_competitor_section(pair_section_df, pred_class, pred_neighborhood)

    print("\nPred-Centered Pair Diagnostics:")
    print(
        f"  explainer={args.explainer_kind} | "
        f"pred_neighborhood={sorted(pred_neighborhood)} | "
        f"pred_neighborhood_mass_ratio={format_scalar(pred_mass_ratio)}"
    )

    interest_sections: list[int] = [int(pred_class)]
    if int(true_class) != int(pred_class):
        interest_sections.append(int(true_class))
    if competitor_section is not None and int(competitor_section) not in interest_sections:
        interest_sections.append(int(competitor_section))

    print("  section comparison:")
    for section_id in interest_sections:
        subset = pair_section_df.loc[pair_section_df["id_section"] == int(section_id)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        markers = []
        if int(section_id) == int(pred_class):
            markers.append("Pred")
        if int(section_id) == int(true_class):
            markers.append("GT")
        if int(section_id) == int(competitor_section):
            markers.append("Competitor")
        marker_text = "/".join(markers)
        print(
            f"    section {int(section_id)} [{marker_text}] | "
            f"dist_to_pred={int(row['dist_to_pred'])} | "
            f"score={format_scalar(row['summary_score'])} | "
            f"mass={format_scalar(row['explanation_mass'])} | "
            f"thermo_support={format_scalar(row['weighted_positive_thermo_delta'])} | "
            f"P_support={format_scalar(row['weighted_pressure_delta'])} | "
            f"T_support={format_scalar(row['weighted_temperature_delta'])}"
        )

    pred_support_nodes = pair_signals_df[pair_signals_df["touches_pred_neighborhood"]].copy()
    pred_support_nodes = pred_support_nodes.sort_values(
        ["pred_support_score", "thermo_to_ideal_delta", "base_node_score"],
        ascending=[False, False, False],
    )
    external_support_nodes = pair_signals_df[~pair_signals_df["touches_pred_neighborhood"]].copy()
    external_support_nodes = external_support_nodes.sort_values(
        ["pred_support_score", "thermo_to_ideal_delta", "base_node_score"],
        ascending=[False, False, False],
    )

    print("  strongest nodes inside Pred neighborhood:")
    for rank, row in enumerate(pred_support_nodes.head(args.pair_top_nodes).itertuples(index=False), start=1):
        print(
            f"    {rank}. node_id={int(row.node_id)} sections={row.incident_sections or '-'} | "
            f"score={format_scalar(row.base_node_score)} | "
            f"thermo->ideal={format_scalar(row.thermo_to_ideal_delta)} | "
            f"P->ideal={format_scalar(row.pressure_to_ideal_delta)} | "
            f"T->ideal={format_scalar(row.temperature_to_ideal_delta)} | "
            f"delta_P={format_scalar(row.pressure_delta_value)} | "
            f"delta_T={format_scalar(row.temperature_delta_value)}"
        )

    if not external_support_nodes.empty:
        print("  strongest nodes outside Pred neighborhood:")
        for rank, row in enumerate(external_support_nodes.head(args.pair_top_nodes).itertuples(index=False), start=1):
            print(
                f"    {rank}. node_id={int(row.node_id)} sections={row.incident_sections or '-'} | "
                f"score={format_scalar(row.base_node_score)} | "
                f"thermo->ideal={format_scalar(row.thermo_to_ideal_delta)} | "
                f"P->ideal={format_scalar(row.pressure_to_ideal_delta)} | "
                f"T->ideal={format_scalar(row.temperature_to_ideal_delta)} | "
                f"delta_P={format_scalar(row.pressure_delta_value)} | "
                f"delta_T={format_scalar(row.temperature_delta_value)}"
            )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.exp_dir)
    device = resolve_device(args.device, cfg)

    pair_dataset, pair_scalers = prepare_data(
        cfg["dataset"],
        cfg["dataloader"],
        cfg["utils"]["seed"],
        prepare_dataloaders=False,
    )
    pair, original_index = select_pair(
        pair_dataset,
        args.split,
        args.index,
        train_ratio=cfg["dataloader"]["train_ratio"],
        val_ratio=cfg["dataloader"]["val_ratio"],
        seed=cfg["utils"]["seed"],
    )
    direction_idx = DIRECTION_TO_INDEX[args.direction]
    data = pair[direction_idx].to(device)

    distance_df = None
    if args.distances_csv.exists():
        distance_df = pd.read_csv(args.distances_csv, sep="\t")
        no_defect_class = int(len(distance_df.index))
    else:
        no_defect_class = int(cfg["model"]["kwargs"]["out_dim"]) - 1

    model = load_model(cfg, pair_dataset[0][direction_idx], args.exp_dir, device)
    explainer_setup = build_explainer(
        model=model,
        epochs=args.epochs,
        node_feature_names=list(cfg["dataset"]["node_attr"]),
        explainer_kind=args.explainer_kind,
    )

    true_class, pred_class, pred_confidence, true_confidence, pred_margin = compute_prediction_summary(model, data)
    explain_x = explainer_setup.transform_node_inputs(data.x)
    explanation = explainer_setup.explainer(explain_x, data.edge_index, index=None, data=data)

    edge_scores = (
        explanation.edge_mask.detach().cpu().numpy()
        if explanation.edge_mask is not None
        else np.zeros(data.edge_index.shape[1], dtype=float)
    )
    node_mask_matrix = (
        explanation.node_mask.detach().cpu().numpy()
        if explanation.node_mask is not None
        else np.zeros((data.x.shape[0], data.x.shape[1]), dtype=float)
    )

    nodes_df, edges_df = build_denorm_tables(
        data=data,
        scalers=pair_scalers[direction_idx],
        cfg=cfg,
        direction=args.direction,
    )
    nodes_df = augment_node_pair_columns(nodes_df)
    node_feature_names = list(explainer_setup.node_feature_names)
    edge_feature_names = list(cfg["dataset"]["edge_attr"])
    raw_node_feature_names = list(cfg["dataset"]["node_attr"])
    pred_neighborhood = section_neighborhood(
        pred_class,
        distance_df,
        radius=args.pred_neighborhood_radius,
    )

    node_table = build_node_table(
        nodes_df=nodes_df,
        edges_df=edges_df,
        node_mask_matrix=node_mask_matrix,
        node_feature_names=node_feature_names,
        true_class=true_class,
        pred_class=pred_class,
    )
    edge_table = build_edge_table(
        edges_df=edges_df,
        edge_scores=edge_scores,
        true_class=true_class,
        pred_class=pred_class,
    )
    section_scores = build_section_scores(
        node_table=node_table,
        edge_table=edge_table,
        edges_df=edges_df,
        true_class=true_class,
        pred_class=pred_class,
    )
    display_section_scores = build_display_section_scores(section_scores)
    pair_signals_df = compute_pair_node_signals(
        model=model,
        data=data,
        node_table=node_table,
        nodes_denorm_df=nodes_df,
        raw_feature_names=raw_node_feature_names,
        target_class=pred_class,
        pred_class=pred_class,
        distance_df=distance_df,
        pred_neighborhood_radius=args.pred_neighborhood_radius,
    )
    pair_section_df = build_pair_section_summary(
        pair_signals_df=pair_signals_df,
        section_scores=display_section_scores,
        pred_class=pred_class,
        true_class=true_class,
        distance_df=distance_df,
        pred_neighborhood_radius=args.pred_neighborhood_radius,
    )

    top_node_table = node_table.head(args.top_nodes).copy()
    top_edge_table = edge_table.head(args.top_edges).copy()

    node_feature_df = compute_node_feature_ablation(
        model=model,
        data=data,
        target_class=pred_class,
        node_indices=top_node_table["node_idx"].tolist(),
        feature_names=node_feature_names,
        transform_node_inputs=explainer_setup.transform_node_inputs,
        inverse_node_inputs=explainer_setup.inverse_node_inputs,
    )
    edge_feature_df = compute_edge_feature_ablation(
        model=model,
        data=data,
        target_class=pred_class,
        edge_indices=top_edge_table["edge_idx"].tolist(),
        feature_names=edge_feature_names,
    )
    node_feature_summary = build_feature_summary(
        node_feature_df,
        entity_col="node_idx",
        top_n=args.top_node_features,
    )
    edge_feature_summary = build_feature_summary(
        edge_feature_df,
        entity_col="edge_idx",
        top_n=args.top_edge_features,
    )

    print_summary(
        args=args,
        sample_id=sample_id_from_path(data.edges_fp),
        tout=get_tout_from_path(data.edges_fp),
        original_index=original_index,
        true_class=true_class,
        pred_class=pred_class,
        no_defect_class=no_defect_class,
        pred_confidence=pred_confidence,
        true_confidence=true_confidence,
        pred_margin=pred_margin,
        section_scores=display_section_scores,
        top_edge_table=top_edge_table,
        top_node_table=top_node_table,
        edge_feature_summary=edge_feature_summary,
        node_feature_summary=node_feature_summary,
        edge_feature_names=edge_feature_names,
        pred_neighborhood=pred_neighborhood,
    )
    print_pair_rationale_summary(
        args=args,
        pair_section_df=pair_section_df,
        pair_signals_df=pair_signals_df,
        pred_class=pred_class,
        true_class=true_class,
        pred_neighborhood=pred_neighborhood,
    )


if __name__ == "__main__":
    main()
