#!/usr/bin/env python3
"""
Run end-to-end explain analysis for the thermal network defect model.

Outputs:
1. Batch inference summaries for forward/backward graphs.
2. Four requested result buckets:
   - correct defect localization
   - neighbor miss
   - farther miss
   - no_defect
3. PyG GNNExplainer explanations for representative examples with node-aware masks.
4. Improved graph visualizations based on the project graph renderer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from explain_analysis_utils import (
    EXPLANATION_METHOD_ORDER,
    PAIR_AWARE_EXPLANATION_METHOD,
    PRIMARY_EXPLANATION_METHOD,
    REQUESTED_CATEGORY_ORDER,
    build_explainer,
    build_summary_tables,
    ensure_dir,
    load_cfg,
    load_model,
    prepare_example_explanation,
    resolve_device,
    run_inference,
    select_representative_examples,
)
from explain_visualization import (
    plot_baseline_method_rank_summary,
    plot_analysis_categories,
    plot_baseline_explanation_bundle,
    plot_confusion_matrix,
    plot_distance_histogram,
    plot_example_explanation_bundle,
    plot_feature_importance_summary,
    plot_requested_categories,
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
        help="Section-to-section graph distance matrix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <exp-dir>/explain_analysis",
    )
    parser.add_argument(
        "--direction",
        choices=["fwd", "bwd", "both"],
        default="both",
        help="Which graph directions to analyze",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Default: auto (prefers cfg device if available)",
    )
    parser.add_argument(
        "--explainer-epochs",
        type=int,
        default=100,
        help="Number of GNNExplainer optimization epochs per example",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=1,
        help="Representative examples to explain per requested category",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=12,
        help="How many highest-scoring nodes to keep per explanation",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=12,
        help="How many highest-scoring edges to keep per explanation",
    )
    parser.add_argument(
        "--top-sections",
        type=int,
        default=8,
        help="How many sections to show in the per-example section bar chart",
    )
    parser.add_argument(
        "--top-features-per-node",
        type=int,
        default=4,
        help="How many strongest node features to keep in compact per-example csv",
    )
    parser.add_argument(
        "--top-features-per-edge",
        type=int,
        default=3,
        help="How many strongest edge features to keep in compact per-example csv",
    )
    parser.add_argument(
        "--primary-explainer",
        choices=[PRIMARY_EXPLANATION_METHOD, PAIR_AWARE_EXPLANATION_METHOD],
        default=PRIMARY_EXPLANATION_METHOD,
        help="Which PyG explainer variant to use as the main method",
    )
    return parser.parse_args()


def build_method_order(primary_method: str) -> list[str]:
    ordered = [primary_method]
    ordered.extend(method for method in EXPLANATION_METHOD_ORDER if method != primary_method)
    return ordered


def save_run_metadata(args: argparse.Namespace, cfg: dict, device, output_dir: Path, method_order: list[str]) -> None:
    metadata = {
        "exp_dir": str(args.exp_dir),
        "output_dir": str(output_dir),
        "direction": args.direction,
        "device": str(device),
        "explainer_epochs": args.explainer_epochs,
        "examples_per_category": args.examples_per_category,
        "top_nodes": args.top_nodes,
        "top_edges": args.top_edges,
        "top_sections": args.top_sections,
        "top_features_per_node": args.top_features_per_node,
        "top_features_per_edge": args.top_features_per_edge,
        "dataset_fp": cfg["dataset"]["fp"],
        "model_name": cfg["model"]["name"],
        "node_attr": cfg["dataset"]["node_attr"],
        "edge_attr": cfg["dataset"]["edge_attr"],
        "requested_categories": REQUESTED_CATEGORY_ORDER,
        "explanation_methods": method_order,
        "primary_explanation_method": args.primary_explainer,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def build_method_output_dirs(output_dir: Path, method_order: list[str]) -> tuple[dict[str, Path], Path]:
    methods_root = ensure_dir(output_dir / "methods")
    method_dirs = {
        method: ensure_dir(methods_root / method)
        for method in method_order
    }
    comparison_dir = ensure_dir(output_dir / "method_comparison")
    return method_dirs, comparison_dir


def filter_method_frame(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["method"] == method].copy()


def append_method_context(frame: pd.DataFrame, record) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.assign(
        sample_id=record.sample_id,
        direction=record.direction,
        requested_category=record.requested_category,
        analysis_category=record.analysis_category,
        true_class=int(record.true_class),
        pred_class=int(record.pred_class),
    )


def save_scope_metrics(records_df: pd.DataFrame, scope: str, output_dir: Path, no_defect_class: int) -> None:
    defect_df = records_df[records_df["true_class"] != no_defect_class].copy()
    defect_labels = sorted(
        label
        for label in (set(records_df["true_class"].tolist()) | set(records_df["pred_class"].tolist()))
        if int(label) != no_defect_class
    )

    report = classification_report(
        defect_df["true_class"],
        defect_df["pred_class"],
        labels=defect_labels,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / f"classification_report_{scope}.csv")

    pred_labels = defect_labels.copy()
    if int((defect_df["pred_class"] == no_defect_class).sum()) > 0:
        pred_labels = pred_labels + ["no_defect"]
    pred_for_cm = defect_df["pred_class"].map(
        lambda value: "no_defect" if int(value) == no_defect_class else int(value)
    )
    cm_df = pd.crosstab(
        pd.Categorical(defect_df["true_class"].astype(int), categories=defect_labels, ordered=True),
        pd.Categorical(pred_for_cm, categories=pred_labels, ordered=True),
        dropna=False,
    )
    cm_df.to_csv(output_dir / f"confusion_matrix_{scope}.csv")
    plot_confusion_matrix(
        cm_df,
        output_dir / f"confusion_matrix_{scope}.png",
        f"Confusion matrix ({scope}, defect classes only)",
    )

    summary = {
        "scope": scope,
        "num_samples": int(len(defect_df)),
        "num_samples_total": int(len(records_df)),
        "num_no_defect_samples": int((records_df["true_class"] == no_defect_class).sum()),
        "excluded_class": int(no_defect_class),
        "accuracy": float(accuracy_score(defect_df["true_class"], defect_df["pred_class"])),
        "macro_f1": float(f1_score(defect_df["true_class"], defect_df["pred_class"], labels=defect_labels, average="macro")),
        "weighted_f1": float(f1_score(defect_df["true_class"], defect_df["pred_class"], labels=defect_labels, average="weighted")),
    }
    (output_dir / f"summary_{scope}.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    output_dir = ensure_dir((args.output_dir or exp_dir / "explain_analysis").resolve())
    method_order = build_method_order(args.primary_explainer)
    method_dirs, comparison_dir = build_method_output_dirs(output_dir, method_order)

    cfg = load_cfg(exp_dir)
    device = resolve_device(args.device, cfg)
    save_run_metadata(args, cfg, device, output_dir, method_order)

    distance_df = pd.read_csv(args.distances_csv, sep="\t")
    no_defect_class = int(len(distance_df.index))

    print(f"Using device: {device}")
    print("Loading dataset and restoring test split...")
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
    model = load_model(cfg, sample_graph, exp_dir, device)
    explainer_setup = build_explainer(
        model,
        args.explainer_epochs,
        cfg["dataset"]["node_attr"],
        explainer_kind=args.primary_explainer,
    )

    directions = ["fwd", "bwd"] if args.direction == "both" else [args.direction]

    prediction_frames = []
    for direction in directions:
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

    analysis_summary, requested_summary, per_true_section = build_summary_tables(
        predictions_df,
        no_defect_class=no_defect_class,
    )
    analysis_summary.to_csv(output_dir / "analysis_category_summary.csv", index=False)
    requested_summary.to_csv(output_dir / "requested_category_summary.csv", index=False)
    per_true_section.to_csv(output_dir / "per_true_section_summary.csv", index=False)

    save_scope_metrics(predictions_df, "overall", output_dir, no_defect_class)
    for direction in directions:
        direction_df = predictions_df[predictions_df["direction"] == direction].copy()
        save_scope_metrics(direction_df, direction, output_dir, no_defect_class)

    plot_requested_categories(requested_summary, output_dir / "requested_categories.png")
    plot_analysis_categories(analysis_summary, output_dir / "analysis_categories.png")
    plot_distance_histogram(predictions_df, output_dir / "distance_histogram.png")

    selected_examples = select_representative_examples(
        predictions_df,
        examples_per_category=args.examples_per_category,
    )
    selected_examples.to_csv(output_dir / "selected_examples.csv", index=False)

    node_feature_names = list(explainer_setup.node_feature_names)
    edge_feature_names = cfg["dataset"]["edge_attr"]
    gnn_manifest_rows = []
    node_feature_summary_rows = []
    edge_feature_summary_rows = []
    section_summary_rows = []
    method_manifest_rows = {method: [] for method in method_order}
    baseline_node_summary_rows = {
        method: [] for method in method_order if method != args.primary_explainer
    }
    baseline_edge_summary_rows = {
        method: [] for method in method_order if method != args.primary_explainer
    }
    baseline_section_summary_rows = {method: [] for method in method_order}
    baseline_method_summary_rows = {method: [] for method in method_order}

    for record in tqdm(
        selected_examples.itertuples(index=False),
        total=len(selected_examples),
        desc="Explaining examples",
        unit="example",
    ):
        record_series = pd.Series(record._asdict())
        explanation = prepare_example_explanation(
            record=record_series,
            model=model,
            explainer_setup=explainer_setup,
            test_dataset=test_dataset,
            pair_scalers=pair_scalers,
            cfg=cfg,
            top_nodes=args.top_nodes,
            top_edges=args.top_edges,
            top_features_per_node=args.top_features_per_node,
            top_features_per_edge=args.top_features_per_edge,
            device=device,
            primary_method=args.primary_explainer,
        )

        example_name = (
            f"{record.requested_category or record.analysis_category}_"
            f"{record.direction}_"
            f"t{record.true_class}_p{record.pred_class}_"
            f"idx{record.test_index:04d}"
        )
        method_example_dirs = {
            method: ensure_dir(method_dirs[method] / "examples" / example_name)
            for method in method_order
        }

        gnn_example_dir = method_example_dirs[args.primary_explainer]
        gnn_method_summary = filter_method_frame(
            explanation.baseline_method_summary,
            args.primary_explainer,
        )

        explanation.nodes_df.to_csv(gnn_example_dir / "nodes_denorm.csv", index=False)
        explanation.node_table.to_csv(gnn_example_dir / "node_importance.csv", index=False)
        explanation.edge_table.to_csv(gnn_example_dir / "edge_importance.csv", index=False)
        explanation.section_scores.to_csv(gnn_example_dir / "section_importance.csv", index=False)
        gnn_method_summary.to_csv(gnn_example_dir / "method_summary.csv", index=False)
        explanation.node_feature_df.to_csv(gnn_example_dir / "node_feature_ablation_full.csv", index=False)
        explanation.compact_node_feature_df.to_csv(gnn_example_dir / "node_feature_ablation_top.csv", index=False)
        explanation.edge_feature_df.to_csv(gnn_example_dir / "edge_feature_ablation_full.csv", index=False)
        explanation.compact_edge_feature_df.to_csv(gnn_example_dir / "edge_feature_ablation_top.csv", index=False)

        overview_path = plot_example_explanation_bundle(
            record=record_series,
            nodes_df=explanation.nodes_df,
            edges_df=explanation.edges_df,
            edge_table=explanation.edge_table,
            node_table=explanation.node_table,
            section_scores=explanation.section_scores,
            node_feature_df=explanation.node_feature_df,
            top_edge_table=explanation.top_edge_table,
            top_node_table=explanation.top_node_table,
            node_feature_names=node_feature_names,
            example_dir=gnn_example_dir,
            top_sections=args.top_sections,
            method=args.primary_explainer,
        )

        gnn_manifest_rows.append(
            {
                **record._asdict(),
                "method": args.primary_explainer,
                "example_dir": str(gnn_example_dir),
                "plot_path": str(overview_path),
                "true_section_importance_rank": explanation.true_section_rank,
                "pred_section_importance_rank": explanation.pred_section_rank,
            }
        )
        for method in method_order:
            method_summary = filter_method_frame(explanation.baseline_method_summary, method)
            if method_summary.empty:
                continue
            summary_row = method_summary.iloc[0]
            plot_path = str(overview_path) if method == args.primary_explainer else ""
            if method != args.primary_explainer:
                method_example_dir = method_example_dirs[method]
                method_node_scores = filter_method_frame(explanation.baseline_node_scores, method)
                method_edge_scores = filter_method_frame(explanation.baseline_edge_scores, method)
                method_section_only = filter_method_frame(explanation.baseline_section_scores, method)
                baseline_overview_path = plot_baseline_explanation_bundle(
                    record=record_series,
                    method=method,
                    nodes_df=explanation.nodes_df,
                    edges_df=explanation.edges_df,
                    edge_score_table=method_edge_scores,
                    node_score_table=method_node_scores,
                    section_score_table=method_section_only,
                    example_dir=method_example_dir,
                    top_nodes=args.top_nodes,
                    top_edges=args.top_edges,
                    top_sections=args.top_sections,
                )
                plot_path = str(baseline_overview_path)
            method_manifest_rows[method].append(
                {
                    **record._asdict(),
                    "method": method,
                    "example_dir": str(method_example_dirs[method]),
                    "plot_path": plot_path,
                    "true_section_importance_rank": summary_row.get("true_section_rank"),
                    "pred_section_importance_rank": summary_row.get("pred_section_rank"),
                }
            )

        if not explanation.node_feature_df.empty:
            node_feature_summary_rows.append(
                explanation.node_feature_df.assign(
                    sample_id=record.sample_id,
                    direction=record.direction,
                    requested_category=record.requested_category,
                    analysis_category=record.analysis_category,
                )
            )
        if not explanation.edge_feature_df.empty:
            edge_feature_summary_rows.append(
                explanation.edge_feature_df.assign(
                    sample_id=record.sample_id,
                    direction=record.direction,
                    requested_category=record.requested_category,
                    analysis_category=record.analysis_category,
                )
            )

        section_summary_rows.append(
            explanation.section_scores.assign(
                sample_id=record.sample_id,
                direction=record.direction,
                requested_category=record.requested_category,
                analysis_category=record.analysis_category,
                true_class=int(record.true_class),
                pred_class=int(record.pred_class),
            )
        )
        for method in method_order:
            method_summary = append_method_context(
                filter_method_frame(explanation.baseline_method_summary, method),
                record,
            )
            if not method_summary.empty:
                baseline_method_summary_rows[method].append(method_summary)

            method_section_scores = append_method_context(
                filter_method_frame(
                    explanation.section_scores
                    if method == args.primary_explainer
                    else explanation.baseline_section_scores,
                    method,
                ),
                record,
            )
            if not method_section_scores.empty:
                baseline_section_summary_rows[method].append(method_section_scores)

            if method == args.primary_explainer:
                continue

            method_example_dir = method_example_dirs[method]
            method_node_scores = filter_method_frame(explanation.baseline_node_scores, method)
            method_edge_scores = filter_method_frame(explanation.baseline_edge_scores, method)
            method_section_only = filter_method_frame(explanation.baseline_section_scores, method)

            method_node_scores.to_csv(method_example_dir / "node_scores.csv", index=False)
            method_edge_scores.to_csv(method_example_dir / "edge_scores.csv", index=False)
            method_section_only.to_csv(method_example_dir / "section_scores.csv", index=False)
            method_summary.to_csv(method_example_dir / "method_summary.csv", index=False)

            method_node_summary = append_method_context(method_node_scores, record)
            method_edge_summary = append_method_context(method_edge_scores, record)
            if not method_node_summary.empty:
                baseline_node_summary_rows[method].append(method_node_summary)
            if not method_edge_summary.empty:
                baseline_edge_summary_rows[method].append(method_edge_summary)

    manifest_df = pd.DataFrame(gnn_manifest_rows)
    node_feature_summary_df = (
        pd.concat(node_feature_summary_rows, ignore_index=True) if node_feature_summary_rows else pd.DataFrame()
    )
    edge_feature_summary_df = (
        pd.concat(edge_feature_summary_rows, ignore_index=True) if edge_feature_summary_rows else pd.DataFrame()
    )
    section_summary_df = (
        pd.concat(section_summary_rows, ignore_index=True) if section_summary_rows else pd.DataFrame()
    )
    baseline_section_summary_df = pd.concat(
        [
            pd.concat(rows, ignore_index=True)
            for rows in baseline_section_summary_rows.values()
            if rows
        ],
        ignore_index=True,
    ) if any(baseline_section_summary_rows.values()) else pd.DataFrame()
    baseline_method_summary_df = pd.concat(
        [
            pd.concat(rows, ignore_index=True)
            for rows in baseline_method_summary_rows.values()
            if rows
        ],
        ignore_index=True,
    ) if any(baseline_method_summary_rows.values()) else pd.DataFrame()
    baseline_node_summary_df = {
        method: (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
        for method, rows in baseline_node_summary_rows.items()
    }
    baseline_edge_summary_df = {
        method: (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
        for method, rows in baseline_edge_summary_rows.items()
    }
    combined_feature_summary_frames = []
    if not node_feature_summary_df.empty:
        combined_feature_summary_frames.append(node_feature_summary_df.assign(entity="node"))
    if not edge_feature_summary_df.empty:
        combined_feature_summary_frames.append(edge_feature_summary_df.assign(entity="edge"))
    combined_feature_summary_df = (
        pd.concat(combined_feature_summary_frames, ignore_index=True)
        if combined_feature_summary_frames
        else pd.DataFrame()
    )

    if not manifest_df.empty:
        manifest_df.to_csv(method_dirs[args.primary_explainer] / "explained_examples_manifest.csv", index=False)
    if not combined_feature_summary_df.empty:
        combined_feature_summary_df.to_csv(
            method_dirs[args.primary_explainer] / "explained_feature_summary.csv",
            index=False,
        )
    if not node_feature_summary_df.empty:
        node_feature_summary_df.to_csv(
            method_dirs[args.primary_explainer] / "explained_node_feature_summary.csv",
            index=False,
        )
        plot_feature_importance_summary(
            feature_summary_df=node_feature_summary_df,
            output_path=method_dirs[args.primary_explainer] / "node_feature_importance_summary.png",
            entity_label="Node",
        )
    if not edge_feature_summary_df.empty:
        edge_feature_summary_df.to_csv(
            method_dirs[args.primary_explainer] / "explained_edge_feature_summary.csv",
            index=False,
        )
        plot_feature_importance_summary(
            feature_summary_df=edge_feature_summary_df,
            output_path=method_dirs[args.primary_explainer] / "edge_feature_importance_summary.png",
            entity_label="Edge",
        )
    if not section_summary_df.empty:
        section_summary_df.to_csv(
            method_dirs[args.primary_explainer] / "explained_section_summary.csv",
            index=False,
        )

    for method in method_order:
        method_manifest_df = pd.DataFrame(method_manifest_rows[method])
        if not method_manifest_df.empty:
            method_manifest_df.to_csv(method_dirs[method] / "explained_examples_manifest.csv", index=False)

        method_section_df = (
            pd.concat(baseline_section_summary_rows[method], ignore_index=True)
            if baseline_section_summary_rows[method]
            else pd.DataFrame()
        )
        if not method_section_df.empty:
            output_name = (
                "explained_section_summary.csv"
                if method == args.primary_explainer
                else "explained_section_scores.csv"
            )
            method_section_df.to_csv(method_dirs[method] / output_name, index=False)

        method_summary_df = (
            pd.concat(baseline_method_summary_rows[method], ignore_index=True)
            if baseline_method_summary_rows[method]
            else pd.DataFrame()
        )
        if not method_summary_df.empty:
            method_summary_df.to_csv(method_dirs[method] / "explained_method_summary.csv", index=False)

        if method == args.primary_explainer:
            continue

        if not baseline_node_summary_df[method].empty:
            baseline_node_summary_df[method].to_csv(method_dirs[method] / "explained_node_scores.csv", index=False)
        if not baseline_edge_summary_df[method].empty:
            baseline_edge_summary_df[method].to_csv(method_dirs[method] / "explained_edge_scores.csv", index=False)

    if not baseline_section_summary_df.empty:
        baseline_section_summary_df.to_csv(comparison_dir / "explained_baseline_section_scores.csv", index=False)
    if not baseline_method_summary_df.empty:
        baseline_method_summary_df.to_csv(comparison_dir / "explained_baseline_method_summary.csv", index=False)
        plot_baseline_method_rank_summary(
            method_summary_df=baseline_method_summary_df,
            output_path=comparison_dir / "baseline_method_rank_summary.png",
        )

    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
