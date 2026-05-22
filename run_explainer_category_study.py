#!/usr/bin/env python3
"""
Study explainer behavior on balanced test examples from four prediction categories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explain_analysis_utils import (
    DIRECTION_TO_INDEX,
    PAIR_AWARE_EXPLANATION_METHOD,
    build_denorm_tables,
    build_edge_table,
    build_explainer,
    build_node_table,
    build_section_scores,
    load_cfg,
    load_model,
    resolve_device,
    run_inference,
)
from run_explain_notebook_summary import (
    augment_node_pair_columns,
    build_display_section_scores,
    build_pair_section_summary,
    compute_pair_node_signals,
    compute_pred_neighborhood_mass_ratio,
    section_neighborhood,
    select_competitor_section,
)
from src.datasets import prepare_data, split_dataset


CATEGORY_ORDER = ["no_defect", "correct", "neighbor", "farther"]


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
        help="Section graph distance matrix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/explainer_category_study"),
        help="Output directory for study artifacts",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=20,
        help="How many examples to analyze per category",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="GNNExplainer optimization epochs",
    )
    parser.add_argument(
        "--pair-top-nodes",
        type=int,
        default=12,
        help="How many top explainer nodes to use for pair-aware diagnostics",
    )
    parser.add_argument(
        "--pred-neighborhood-radius",
        type=int,
        default=1,
        help="Graph distance radius around Pred considered local evidence",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed for category selection",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Default: auto",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_balanced_examples(
    records_df: pd.DataFrame,
    examples_per_category: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected_frames = []
    for category in CATEGORY_ORDER:
        category_df = records_df[records_df["requested_category"] == category].copy()
        if category_df.empty:
            continue

        target_bwd = examples_per_category // 2
        target_fwd = examples_per_category - target_bwd

        category_parts = []
        chosen_indices: set[int] = set()
        for direction, target in [("fwd", target_fwd), ("bwd", target_bwd)]:
            direction_df = category_df[category_df["direction"] == direction].copy()
            if direction_df.empty or target <= 0:
                continue
            sample_n = min(target, len(direction_df))
            choice = rng.choice(direction_df.index.to_numpy(), size=sample_n, replace=False)
            chosen_indices.update(int(idx) for idx in np.atleast_1d(choice).tolist())
            category_parts.append(direction_df.loc[choice])

        selected = pd.concat(category_parts, ignore_index=True) if category_parts else pd.DataFrame(columns=category_df.columns)
        remaining_need = examples_per_category - len(selected)
        if remaining_need > 0:
            remaining_pool = category_df.loc[~category_df.index.isin(list(chosen_indices))]
            if not remaining_pool.empty:
                sample_n = min(remaining_need, len(remaining_pool))
                choice = rng.choice(remaining_pool.index.to_numpy(), size=sample_n, replace=False)
                chosen_indices.update(int(idx) for idx in np.atleast_1d(choice).tolist())
                selected = pd.concat([selected, remaining_pool.loc[choice]], ignore_index=True)

        selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=int)
        selected_frames.append(selected.head(examples_per_category))

    if not selected_frames:
        return pd.DataFrame(columns=records_df.columns)
    return pd.concat(selected_frames, ignore_index=True)


def distance_lookup(distance_df: pd.DataFrame | None, src: int, dst: int) -> int | None:
    if distance_df is None:
        return 0 if int(src) == int(dst) else None
    if int(src) >= len(distance_df.index) or int(dst) >= len(distance_df.columns):
        return None
    return int(distance_df.iloc[int(src), int(dst)])


def extract_section_value(section_df: pd.DataFrame, section_id: int, column: str) -> float:
    subset = section_df.loc[section_df["id_section"] == int(section_id), column]
    return float(subset.iloc[0]) if not subset.empty else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def build_example_metrics(
    record: pd.Series,
    display_section_scores: pd.DataFrame,
    pair_section_df: pd.DataFrame,
    pair_signals_df: pd.DataFrame,
    distance_df: pd.DataFrame | None,
    pred_neighborhood_radius: int,
) -> dict:
    true_class = int(record.true_class)
    pred_class = int(record.pred_class)
    pred_neighborhood = section_neighborhood(pred_class, distance_df, radius=pred_neighborhood_radius)

    total_pair_support = float(pair_section_df["weighted_positive_thermo_delta"].sum()) if not pair_section_df.empty else 0.0
    total_pressure_support = float(pair_section_df["weighted_pressure_delta"].sum()) if not pair_section_df.empty else 0.0
    total_temperature_support = float(pair_section_df["weighted_temperature_delta"].sum()) if not pair_section_df.empty else 0.0
    pred_support = extract_section_value(pair_section_df, pred_class, "weighted_positive_thermo_delta")
    pred_pressure_support = extract_section_value(pair_section_df, pred_class, "weighted_pressure_delta")
    pred_temperature_support = extract_section_value(pair_section_df, pred_class, "weighted_temperature_delta")
    true_support = extract_section_value(pair_section_df, true_class, "weighted_positive_thermo_delta")
    true_pressure_support = extract_section_value(pair_section_df, true_class, "weighted_pressure_delta")
    true_temperature_support = extract_section_value(pair_section_df, true_class, "weighted_temperature_delta")
    competitor_section = select_competitor_section(pair_section_df, pred_class, pred_neighborhood)
    competitor_support = (
        extract_section_value(pair_section_df, competitor_section, "weighted_positive_thermo_delta")
        if competitor_section is not None
        else float("nan")
    )
    competitor_pressure_support = (
        extract_section_value(pair_section_df, competitor_section, "weighted_pressure_delta")
        if competitor_section is not None
        else float("nan")
    )
    competitor_temperature_support = (
        extract_section_value(pair_section_df, competitor_section, "weighted_temperature_delta")
        if competitor_section is not None
        else float("nan")
    )

    pred_neighborhood_support = float(
        pair_section_df.loc[
            pair_section_df["id_section"].astype(int).isin(pred_neighborhood),
            "weighted_positive_thermo_delta",
        ].sum()
    ) if not pair_section_df.empty else 0.0
    pred_neighborhood_support_ratio = (
        pred_neighborhood_support / total_pair_support if total_pair_support > 0.0 else np.nan
    )

    top_support_section = None
    top_support_value = np.nan
    top_support_dist_to_pred = np.nan
    top_support_dist_to_true = np.nan
    top_support_in_pred_neighborhood = np.nan
    if not pair_section_df.empty:
        top_support_row = pair_section_df.sort_values(
            ["weighted_positive_thermo_delta", "summary_score", "id_section"],
            ascending=[False, False, True],
        ).iloc[0]
        top_support_section = int(top_support_row["id_section"])
        top_support_value = float(top_support_row["weighted_positive_thermo_delta"])
        top_support_dist_to_pred = distance_lookup(distance_df, pred_class, top_support_section)
        top_support_dist_to_true = distance_lookup(distance_df, true_class, top_support_section)
        top_support_in_pred_neighborhood = int(top_support_section in pred_neighborhood)

    top_expl_row = display_section_scores.iloc[0]
    gt_rank = (
        display_section_scores["id_section"].astype(int).tolist().index(true_class) + 1
        if true_class in display_section_scores["id_section"].astype(int).tolist()
        else np.nan
    )
    pred_rank = (
        display_section_scores["id_section"].astype(int).tolist().index(pred_class) + 1
        if pred_class in display_section_scores["id_section"].astype(int).tolist()
        else np.nan
    )

    support_sections = pair_section_df["weighted_positive_thermo_delta"] if not pair_section_df.empty else pd.Series(dtype=float)
    top_support_ratio = (
        top_support_value / total_pair_support if total_pair_support > 0.0 and pd.notna(top_support_value) else np.nan
    )
    broad_support_sections = int((support_sections >= 0.25 * float(support_sections.max())).sum()) if not support_sections.empty and float(support_sections.max()) > 0.0 else 0

    inside_nodes = pair_signals_df[pair_signals_df["touches_pred_neighborhood"]].copy()
    outside_nodes = pair_signals_df[~pair_signals_df["touches_pred_neighborhood"]].copy()
    inside_best = float(inside_nodes["pred_support_score"].max()) if not inside_nodes.empty else np.nan
    outside_best = float(outside_nodes["pred_support_score"].max()) if not outside_nodes.empty else np.nan

    return {
        **record.to_dict(),
        "top_expl_section": int(top_expl_row["id_section"]),
        "top_expl_score": float(top_expl_row["summary_score"]),
        "top_expl_dist_to_true": distance_lookup(distance_df, true_class, int(top_expl_row["id_section"])),
        "gt_rank": gt_rank,
        "pred_rank": pred_rank,
        "pred_neighborhood_mass_ratio": compute_pred_neighborhood_mass_ratio(display_section_scores, pred_neighborhood),
        "total_pair_support": total_pair_support,
        "total_pressure_support": total_pressure_support,
        "total_temperature_support": total_temperature_support,
        "total_temperature_share": safe_ratio(total_temperature_support, total_pair_support),
        "pred_neighborhood_support_ratio": pred_neighborhood_support_ratio,
        "pred_support": pred_support,
        "pred_pressure_support": pred_pressure_support,
        "pred_temperature_support": pred_temperature_support,
        "pred_temperature_share": safe_ratio(pred_temperature_support, pred_support),
        "true_support": true_support,
        "true_pressure_support": true_pressure_support,
        "true_temperature_support": true_temperature_support,
        "true_temperature_share": safe_ratio(true_temperature_support, true_support),
        "competitor_section": competitor_section,
        "competitor_support": competitor_support,
        "competitor_pressure_support": competitor_pressure_support,
        "competitor_temperature_support": competitor_temperature_support,
        "competitor_temperature_share": safe_ratio(competitor_temperature_support, competitor_support),
        "competitor_dist_to_pred": distance_lookup(distance_df, pred_class, competitor_section) if competitor_section is not None else np.nan,
        "pred_minus_competitor_support": pred_support - competitor_support if pd.notna(pred_support) and pd.notna(competitor_support) else np.nan,
        "true_minus_competitor_support": true_support - competitor_support if pd.notna(true_support) and pd.notna(competitor_support) else np.nan,
        "pred_minus_true_support": pred_support - true_support if pd.notna(pred_support) and pd.notna(true_support) else np.nan,
        "top_support_section": top_support_section,
        "top_support_value": top_support_value,
        "top_support_ratio": top_support_ratio,
        "top_support_dist_to_pred": top_support_dist_to_pred,
        "top_support_dist_to_true": top_support_dist_to_true,
        "top_support_in_pred_neighborhood": top_support_in_pred_neighborhood,
        "broad_support_sections_q25": broad_support_sections,
        "best_inside_node_support": inside_best,
        "best_outside_node_support": outside_best,
        "outside_beats_inside_node": (
            int(pd.notna(outside_best) and pd.notna(inside_best) and outside_best > inside_best)
            if pd.notna(outside_best) and pd.notna(inside_best)
            else np.nan
        ),
    }


def summarize_study(example_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CATEGORY_ORDER:
        subset = example_metrics_df[example_metrics_df["requested_category"] == category].copy()
        if subset.empty:
            continue

        rows.append(
            {
                "requested_category": category,
                "num_examples": int(len(subset)),
                "num_fwd": int((subset["direction"] == "fwd").sum()),
                "num_bwd": int((subset["direction"] == "bwd").sum()),
                "mean_pred_confidence": float(subset["pred_confidence"].mean()),
                "mean_pred_margin": float(subset["pred_margin"].mean()),
                "mean_gt_rank": float(subset["gt_rank"].dropna().mean()) if subset["gt_rank"].notna().any() else np.nan,
                "mean_pred_rank": float(subset["pred_rank"].dropna().mean()) if subset["pred_rank"].notna().any() else np.nan,
                "mean_top_expl_dist_to_true": float(subset["top_expl_dist_to_true"].dropna().mean()) if subset["top_expl_dist_to_true"].notna().any() else np.nan,
                "mean_pred_neighborhood_mass_ratio": float(subset["pred_neighborhood_mass_ratio"].dropna().mean()) if subset["pred_neighborhood_mass_ratio"].notna().any() else np.nan,
                "mean_total_pair_support": float(subset["total_pair_support"].mean()),
                "mean_total_pressure_support": float(subset["total_pressure_support"].mean()),
                "mean_total_temperature_support": float(subset["total_temperature_support"].mean()),
                "mean_total_temperature_share": float(subset["total_temperature_share"].dropna().mean()) if subset["total_temperature_share"].notna().any() else np.nan,
                "mean_pred_neighborhood_support_ratio": float(subset["pred_neighborhood_support_ratio"].dropna().mean()) if subset["pred_neighborhood_support_ratio"].notna().any() else np.nan,
                "median_pred_neighborhood_support_ratio": float(subset["pred_neighborhood_support_ratio"].dropna().median()) if subset["pred_neighborhood_support_ratio"].notna().any() else np.nan,
                "mean_pred_support": float(subset["pred_support"].dropna().mean()) if subset["pred_support"].notna().any() else np.nan,
                "mean_pred_pressure_support": float(subset["pred_pressure_support"].dropna().mean()) if subset["pred_pressure_support"].notna().any() else np.nan,
                "mean_pred_temperature_support": float(subset["pred_temperature_support"].dropna().mean()) if subset["pred_temperature_support"].notna().any() else np.nan,
                "mean_pred_temperature_share": float(subset["pred_temperature_share"].dropna().mean()) if subset["pred_temperature_share"].notna().any() else np.nan,
                "mean_true_support": float(subset["true_support"].dropna().mean()) if subset["true_support"].notna().any() else np.nan,
                "mean_true_pressure_support": float(subset["true_pressure_support"].dropna().mean()) if subset["true_pressure_support"].notna().any() else np.nan,
                "mean_true_temperature_support": float(subset["true_temperature_support"].dropna().mean()) if subset["true_temperature_support"].notna().any() else np.nan,
                "mean_true_temperature_share": float(subset["true_temperature_share"].dropna().mean()) if subset["true_temperature_share"].notna().any() else np.nan,
                "mean_competitor_support": float(subset["competitor_support"].dropna().mean()) if subset["competitor_support"].notna().any() else np.nan,
                "mean_competitor_pressure_support": float(subset["competitor_pressure_support"].dropna().mean()) if subset["competitor_pressure_support"].notna().any() else np.nan,
                "mean_competitor_temperature_support": float(subset["competitor_temperature_support"].dropna().mean()) if subset["competitor_temperature_support"].notna().any() else np.nan,
                "mean_competitor_temperature_share": float(subset["competitor_temperature_share"].dropna().mean()) if subset["competitor_temperature_share"].notna().any() else np.nan,
                "median_pred_minus_competitor_support": float(subset["pred_minus_competitor_support"].dropna().median()) if subset["pred_minus_competitor_support"].notna().any() else np.nan,
                "median_true_minus_competitor_support": float(subset["true_minus_competitor_support"].dropna().median()) if subset["true_minus_competitor_support"].notna().any() else np.nan,
                "frac_competitor_beats_pred": float((subset["pred_minus_competitor_support"] < 0).mean()) if subset["pred_minus_competitor_support"].notna().any() else np.nan,
                "frac_competitor_beats_true": float((subset["true_minus_competitor_support"] < 0).mean()) if subset["true_minus_competitor_support"].notna().any() else np.nan,
                "frac_top_support_in_pred_neighborhood": float(subset["top_support_in_pred_neighborhood"].dropna().mean()) if subset["top_support_in_pred_neighborhood"].notna().any() else np.nan,
                "mean_top_support_ratio": float(subset["top_support_ratio"].dropna().mean()) if subset["top_support_ratio"].notna().any() else np.nan,
                "mean_broad_support_sections_q25": float(subset["broad_support_sections_q25"].mean()),
                "frac_outside_beats_inside_node": float(subset["outside_beats_inside_node"].dropna().mean()) if subset["outside_beats_inside_node"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_notable_examples(example_metrics_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for category in CATEGORY_ORDER:
        subset = example_metrics_df[example_metrics_df["requested_category"] == category].copy()
        if subset.empty:
            continue
        if category == "correct":
            subset = subset.sort_values(
                ["pred_neighborhood_support_ratio", "pred_minus_competitor_support", "pred_confidence"],
                ascending=[False, False, False],
            )
        elif category == "no_defect":
            subset = subset.sort_values(
                ["total_pair_support", "top_support_ratio"],
                ascending=[True, False],
            )
        else:
            subset = subset.sort_values(
                ["pred_minus_competitor_support", "pred_neighborhood_support_ratio", "pred_confidence"],
                ascending=[True, True, False],
            )
        frames.append(subset.head(5))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=example_metrics_df.columns)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.resolve())

    cfg = load_cfg(args.exp_dir)
    device = resolve_device(args.device, cfg)
    distance_df = pd.read_csv(args.distances_csv, sep="\t")
    no_defect_class = int(len(distance_df.index))

    pair_dataset, pair_scalers, _, _, test_loader, _ = prepare_data(
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
    model = load_model(cfg, sample_graph, args.exp_dir.resolve(), device)
    explainer_setup = build_explainer(
        model=model,
        epochs=args.epochs,
        node_feature_names=list(cfg["dataset"]["node_attr"]),
        explainer_kind=PAIR_AWARE_EXPLANATION_METHOD,
    )

    prediction_frames = []
    for direction in ["fwd", "bwd"]:
        prediction_frames.append(
            run_inference(
                model=model,
                test_loader=test_loader,
                test_dataset=test_dataset,
                direction=direction,
                distance_df=distance_df,
                no_defect_class=no_defect_class,
                device=device,
            )
        )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df.to_csv(output_dir / "prediction_records.csv", index=False)

    selected_examples = choose_balanced_examples(
        predictions_df,
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )
    selected_examples.to_csv(output_dir / "selected_examples.csv", index=False)

    raw_node_feature_names = list(cfg["dataset"]["node_attr"])
    example_rows = []

    for record in selected_examples.itertuples(index=False):
        record_series = pd.Series(record._asdict())
        direction = str(record.direction)
        direction_idx = DIRECTION_TO_INDEX[direction]
        data = test_dataset[int(record.test_index)][direction_idx].to(device)

        explain_x = explainer_setup.transform_node_inputs(data.x)
        explanation = explainer_setup.explainer(explain_x, data.edge_index, index=None, data=data)
        edge_scores = (
            explanation.edge_mask.detach().cpu().numpy().reshape(-1)
            if explanation.edge_mask is not None
            else np.zeros(data.edge_index.shape[1], dtype=float)
        )
        node_mask_matrix = (
            explanation.node_mask.detach().cpu().numpy()
            if explanation.node_mask is not None
            else np.zeros((data.x.shape[0], len(explainer_setup.node_feature_names)), dtype=float)
        )

        nodes_df, edges_df = build_denorm_tables(
            data=data,
            scalers=pair_scalers[direction_idx],
            cfg=cfg,
            direction=direction,
        )
        nodes_df = augment_node_pair_columns(nodes_df)

        node_table = build_node_table(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_mask_matrix=node_mask_matrix,
            node_feature_names=list(explainer_setup.node_feature_names),
            true_class=int(record.true_class),
            pred_class=int(record.pred_class),
        )
        edge_table = build_edge_table(
            edges_df=edges_df,
            edge_scores=edge_scores,
            true_class=int(record.true_class),
            pred_class=int(record.pred_class),
        )
        display_section_scores = build_display_section_scores(
            build_section_scores(
                node_table=node_table,
                edge_table=edge_table,
                edges_df=edges_df,
                true_class=int(record.true_class),
                pred_class=int(record.pred_class),
            )
        )

        pair_node_table = node_table.head(args.pair_top_nodes).copy()
        pair_signals_df = compute_pair_node_signals(
            model=model,
            data=data,
            node_table=pair_node_table,
            nodes_denorm_df=nodes_df,
            raw_feature_names=raw_node_feature_names,
            target_class=int(record.pred_class),
            pred_class=int(record.pred_class),
            distance_df=distance_df,
            pred_neighborhood_radius=args.pred_neighborhood_radius,
        )
        pair_section_df = build_pair_section_summary(
            pair_signals_df=pair_signals_df,
            section_scores=display_section_scores,
            pred_class=int(record.pred_class),
            true_class=int(record.true_class),
            distance_df=distance_df,
            pred_neighborhood_radius=args.pred_neighborhood_radius,
        )

        example_rows.append(
            build_example_metrics(
                record=record_series,
                display_section_scores=display_section_scores,
                pair_section_df=pair_section_df,
                pair_signals_df=pair_signals_df,
                distance_df=distance_df,
                pred_neighborhood_radius=args.pred_neighborhood_radius,
            )
        )

    example_metrics_df = pd.DataFrame(example_rows)
    summary_df = summarize_study(example_metrics_df)
    notable_df = build_notable_examples(example_metrics_df)

    example_metrics_df.to_csv(output_dir / "example_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "category_summary.csv", index=False)
    notable_df.to_csv(output_dir / "notable_examples.csv", index=False)

    metadata = {
        "exp_dir": str(args.exp_dir.resolve()),
        "output_dir": str(output_dir),
        "examples_per_category": int(args.examples_per_category),
        "epochs": int(args.epochs),
        "pair_top_nodes": int(args.pair_top_nodes),
        "pred_neighborhood_radius": int(args.pred_neighborhood_radius),
        "seed": int(args.seed),
        "device": str(device),
        "explainer_kind": PAIR_AWARE_EXPLANATION_METHOD,
    }
    (output_dir / "study_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved study to: {output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
