#!/usr/bin/env python3
"""
Post-process category-study outputs with defect-strength metadata extracted from
the original edge CSV files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PR_PATTERN = re.compile(r"_pr(\d+)(?:_[fb]wd)?\.csv$")
TUBE_PATTERN = re.compile(r"/tube_(\d+)/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("tmp/explainer_category_study_run1"),
        help="Directory with prediction_records.csv / selected_examples.csv / example_metrics.csv",
    )
    return parser.parse_args()


def canonical_edges_path(edges_fp: str) -> Path:
    path = str(edges_fp)
    path = path.replace("datasets/Termo_model_fwd/", "datasets/Termo_model/")
    path = path.replace("datasets/Termo_model_bwd/", "datasets/Termo_model/")
    path = path.replace("_fwd.csv", ".csv").replace("_bwd.csv", ".csv")
    return Path(path)


def build_defect_meta_reader():
    cache: dict[str, dict[str, object]] = {}

    def read_meta(edges_fp: str) -> dict[str, object]:
        if edges_fp in cache:
            return cache[edges_fp]

        path = canonical_edges_path(edges_fp)
        path_str = str(path)
        pr_match = PR_PATTERN.search(path_str)
        tube_match = TUBE_PATTERN.search(path_str)
        meta: dict[str, object] = {
            "canonical_edges_fp": path_str,
            "pr_level": np.nan,
            "tube_id": np.nan,
            "defect_alpha": np.nan,
            "defect_factor": np.nan,
            "defect_heat": np.nan,
            "is_problem_path": False,
        }
        if pr_match and tube_match:
            pr_level = int(pr_match.group(1))
            tube_id = int(tube_match.group(1))
            edges_df = pd.read_csv(path, sep="\t")
            defect_row = edges_df.loc[edges_df["id"].astype(int) == tube_id].iloc[0]
            meta.update(
                {
                    "pr_level": pr_level,
                    "tube_id": tube_id,
                    "defect_alpha": float(defect_row["alpha"]) if "alpha" in defect_row.index else np.nan,
                    "defect_factor": float(defect_row["moded"]) if "moded" in defect_row.index else np.nan,
                    "defect_heat": float(defect_row["Heat"]) if "Heat" in defect_row.index else np.nan,
                    "is_problem_path": True,
                }
            )

        cache[edges_fp] = meta
        return meta

    return read_meta


def factor_bin(value: float) -> str:
    if pd.isna(value):
        return "no_strength"
    if float(value) < 18.0:
        return "weak(<18x)"
    if float(value) < 30.0:
        return "medium(18-30x)"
    return "strong(>=30x)"


def pr_group(pr_level: float) -> str:
    if pd.isna(pr_level):
        return "none"
    pr_level = int(pr_level)
    if pr_level in {0, 1}:
        return "weak_pr01"
    if pr_level == 2:
        return "mid_pr2"
    if pr_level in {3, 4}:
        return "strong_pr34"
    return "other"


def summarize_problem_categories(prediction_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    problem_df = prediction_df[prediction_df["is_problem_path"]].copy()

    by_pr = (
        problem_df.groupby(["pr_level", "requested_category"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="pr_level", columns="requested_category", values="count")
        .fillna(0)
    )
    for column in ["no_defect", "correct", "neighbor", "farther"]:
        if column not in by_pr.columns:
            by_pr[column] = 0
    by_pr = by_pr[["no_defect", "correct", "neighbor", "farther"]].reset_index()
    by_pr["total"] = by_pr[["no_defect", "correct", "neighbor", "farther"]].sum(axis=1)
    for column in ["no_defect", "correct", "neighbor", "farther"]:
        by_pr[f"rate_{column}"] = by_pr[column] / by_pr["total"]

    by_factor_bin = (
        problem_df.assign(factor_bin=problem_df["defect_factor"].map(factor_bin))
        .groupby(["factor_bin", "requested_category"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="factor_bin", columns="requested_category", values="count")
        .fillna(0)
    )
    for column in ["no_defect", "correct", "neighbor", "farther"]:
        if column not in by_factor_bin.columns:
            by_factor_bin[column] = 0
    by_factor_bin = by_factor_bin[["no_defect", "correct", "neighbor", "farther"]].reset_index()
    by_factor_bin["total"] = by_factor_bin[["no_defect", "correct", "neighbor", "farther"]].sum(axis=1)
    for column in ["no_defect", "correct", "neighbor", "farther"]:
        by_factor_bin[f"rate_{column}"] = by_factor_bin[column] / by_factor_bin["total"]

    pr_factor_stats = (
        problem_df.groupby("pr_level")
        .agg(
            n=("pr_level", "size"),
            mean_factor=("defect_factor", "mean"),
            median_factor=("defect_factor", "median"),
            min_factor=("defect_factor", "min"),
            max_factor=("defect_factor", "max"),
            mean_alpha=("defect_alpha", "mean"),
            median_alpha=("defect_alpha", "median"),
        )
        .reset_index()
    )
    return by_pr, by_factor_bin, pr_factor_stats


def summarize_explainer_strength(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    problem_df = metrics_df[metrics_df["is_problem_path"]].copy()
    problem_df["factor_bin"] = problem_df["defect_factor"].map(factor_bin)
    problem_df["pr_group"] = problem_df["pr_level"].map(pr_group)

    rows = []
    for (category, bin_name), subset in problem_df.groupby(["requested_category", "factor_bin"], sort=False):
        rows.append(
            {
                "requested_category": category,
                "factor_bin": bin_name,
                "n": int(len(subset)),
                "mean_factor": float(subset["defect_factor"].mean()),
                "mean_pred_confidence": float(subset["pred_confidence"].mean()),
                "mean_pred_margin": float(subset["pred_margin"].mean()),
                "mean_gt_rank": float(subset["gt_rank"].dropna().mean()) if subset["gt_rank"].notna().any() else np.nan,
                "mean_pred_rank": float(subset["pred_rank"].dropna().mean()) if subset["pred_rank"].notna().any() else np.nan,
                "mean_pred_neighborhood_support_ratio": float(subset["pred_neighborhood_support_ratio"].dropna().mean()) if subset["pred_neighborhood_support_ratio"].notna().any() else np.nan,
                "median_pred_neighborhood_support_ratio": float(subset["pred_neighborhood_support_ratio"].dropna().median()) if subset["pred_neighborhood_support_ratio"].notna().any() else np.nan,
                "mean_pred_support": float(subset["pred_support"].dropna().mean()) if subset["pred_support"].notna().any() else np.nan,
                "mean_true_support": float(subset["true_support"].dropna().mean()) if subset["true_support"].notna().any() else np.nan,
                "mean_competitor_support": float(subset["competitor_support"].dropna().mean()) if subset["competitor_support"].notna().any() else np.nan,
                "mean_pred_minus_competitor_support": float(subset["pred_minus_competitor_support"].dropna().mean()) if subset["pred_minus_competitor_support"].notna().any() else np.nan,
                "frac_competitor_beats_pred": float((subset["pred_minus_competitor_support"] < 0).mean()) if subset["pred_minus_competitor_support"].notna().any() else np.nan,
                "frac_outside_beats_inside_node": float(subset["outside_beats_inside_node"].dropna().mean()) if subset["outside_beats_inside_node"].notna().any() else np.nan,
                "mean_total_temperature_share": float(subset["total_temperature_share"].dropna().mean()) if subset["total_temperature_share"].notna().any() else np.nan,
            }
        )
    by_factor_bin = pd.DataFrame(rows)

    rows = []
    for (category, group_name), subset in problem_df[problem_df["pr_group"] != "none"].groupby(["requested_category", "pr_group"], sort=False):
        rows.append(
            {
                "requested_category": category,
                "pr_group": group_name,
                "n": int(len(subset)),
                "mean_factor": float(subset["defect_factor"].mean()),
                "mean_pred_confidence": float(subset["pred_confidence"].mean()),
                "mean_pred_margin": float(subset["pred_margin"].mean()),
                "mean_pred_neighborhood_support_ratio": float(subset["pred_neighborhood_support_ratio"].dropna().mean()) if subset["pred_neighborhood_support_ratio"].notna().any() else np.nan,
                "mean_pred_minus_competitor_support": float(subset["pred_minus_competitor_support"].dropna().mean()) if subset["pred_minus_competitor_support"].notna().any() else np.nan,
                "frac_competitor_beats_pred": float((subset["pred_minus_competitor_support"] < 0).mean()) if subset["pred_minus_competitor_support"].notna().any() else np.nan,
                "mean_top_support_dist_to_pred": float(subset["top_support_dist_to_pred"].dropna().mean()) if subset["top_support_dist_to_pred"].notna().any() else np.nan,
            }
        )
    by_pr_group = pd.DataFrame(rows)

    return by_factor_bin, by_pr_group


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()

    prediction_df = pd.read_csv(study_dir / "prediction_records.csv")
    selected_df = pd.read_csv(study_dir / "selected_examples.csv")
    metrics_df = pd.read_csv(study_dir / "example_metrics.csv")

    read_meta = build_defect_meta_reader()
    prediction_meta_df = prediction_df["edges_fp"].apply(read_meta).apply(pd.Series)
    prediction_enriched_df = pd.concat([prediction_df, prediction_meta_df], axis=1)

    selected_meta_df = selected_df["edges_fp"].apply(read_meta).apply(pd.Series)
    selected_enriched_df = pd.concat([selected_df, selected_meta_df], axis=1)

    metrics_enriched_df = metrics_df.merge(
        selected_enriched_df[
            [
                "test_index",
                "direction",
                "pr_level",
                "tube_id",
                "defect_alpha",
                "defect_factor",
                "defect_heat",
                "is_problem_path",
                "canonical_edges_fp",
            ]
        ],
        on=["test_index", "direction"],
        how="left",
    )

    model_by_pr_df, model_by_factor_df, pr_factor_df = summarize_problem_categories(prediction_enriched_df)
    explainer_by_factor_df, explainer_by_pr_group_df = summarize_explainer_strength(metrics_enriched_df)

    prediction_enriched_df.to_csv(study_dir / "prediction_records_with_defect_strength.csv", index=False)
    metrics_enriched_df.to_csv(study_dir / "example_metrics_with_defect_strength.csv", index=False)
    model_by_pr_df.to_csv(study_dir / "model_category_by_pr_strength.csv", index=False)
    model_by_factor_df.to_csv(study_dir / "model_category_by_factor_strength.csv", index=False)
    pr_factor_df.to_csv(study_dir / "problem_pr_factor_mapping.csv", index=False)
    explainer_by_factor_df.to_csv(study_dir / "explainer_strength_summary.csv", index=False)
    explainer_by_pr_group_df.to_csv(study_dir / "explainer_pr_strength_summary.csv", index=False)

    print(f"Saved defect-strength summaries to: {study_dir}")
    print()
    print("Model Category By PR")
    print(model_by_pr_df.to_string(index=False))
    print()
    print("Explainer Strength Summary")
    print(explainer_by_factor_df.to_string(index=False))


if __name__ == "__main__":
    main()
