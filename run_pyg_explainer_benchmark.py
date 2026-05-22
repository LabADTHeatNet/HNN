#!/usr/bin/env python3
"""
Benchmark multiple PyG explainers and explainer-output adapters for the
thermal network defect localization model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.explain import (
    AttentionExplainer,
    Explainer,
    GNNExplainer,
    GraphMaskExplainer,
    PGExplainer,
)
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from tqdm.auto import tqdm

from explain_analysis_utils import (
    DIRECTION_TO_INDEX,
    NODE_DYNAMIC_FEATURES,
    aggregate_scores_by_section,
    build_denorm_tables,
    build_generic_edge_score_table,
    build_generic_node_score_table,
    ensure_dir,
    load_cfg,
    load_model,
    resolve_device,
    run_inference,
)
from explain_visualization import METHOD_DISPLAY_NAMES, plot_baseline_explanation_bundle
from src.datasets import prepare_data, split_dataset


@dataclass(frozen=True)
class ExplainerSpec:
    key: str
    algorithm: str
    output_mode: str
    node_mask_type: str | None
    edge_mask_type: str | None
    explanation_type: str
    return_type: str
    epochs: int
    batchify_data: bool = False
    reference_class: int | None = None


class ModelOutputAdapter(nn.Module):
    """Wrap an explainable model and transform its output for explainer runs."""

    def __init__(
        self,
        model: nn.Module,
        output_mode: str = "logits",
        batchify_data: bool = False,
        reference_class: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.output_mode = output_mode
        self.batchify_data = batchify_data
        self.reference_class = reference_class

    def _format_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "logits":
            return logits
        if self.output_mode == "log_probs":
            return torch.log_softmax(logits, dim=-1)
        if self.output_mode == "margin":
            max_other = []
            for class_idx in range(logits.shape[-1]):
                class_logits = logits[:, class_idx]
                other_logits = torch.cat([logits[:, :class_idx], logits[:, class_idx + 1 :]], dim=-1)
                max_other.append(class_logits - other_logits.max(dim=-1).values)
            return torch.stack(max_other, dim=-1)
        if self.output_mode == "vs_reference_margin":
            if self.reference_class is None:
                raise ValueError("reference_class is required for vs_reference_margin output mode")
            reference_logits = logits[:, self.reference_class].unsqueeze(-1)
            return logits - reference_logits
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def forward(self, x, edge_index, data, edge_attr: torch.Tensor | None = None):
        model_data = data
        if self.batchify_data and not hasattr(data, "batch"):
            model_data = Batch.from_data_list([data])
        logits = self.model(x, edge_index, model_data, edge_attr=edge_attr)
        return self._format_logits(logits)


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
        help="Output directory. Defaults to <exp-dir>/pyg_explainer_benchmark",
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
        help="Torch device. Default: auto",
    )
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=1,
        help="How many examples to benchmark per (direction, category) group",
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=40,
        help="Optimization epochs for GNNExplainer variants",
    )
    parser.add_argument(
        "--graphmask-epochs",
        type=int,
        default=25,
        help="Optimization epochs for GraphMaskExplainer",
    )
    parser.add_argument(
        "--pg-train-epochs",
        type=int,
        default=20,
        help="Training epochs for PGExplainer",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=12,
        help="How many highest-scoring nodes to visualize per example",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=12,
        help="How many highest-scoring edges to visualize per example",
    )
    parser.add_argument(
        "--top-sections",
        type=int,
        default=8,
        help="How many sections to show in per-example section bar charts",
    )
    return parser.parse_args()


def select_benchmark_examples(
    records_df: pd.DataFrame,
    examples_per_group: int,
    no_defect_class: int,
) -> pd.DataFrame:
    selected_frames = []
    categories = ["correct", "neighbor", "farther"]
    directions = ["fwd", "bwd"]
    defect_df = records_df[
        (records_df["true_class"] != no_defect_class)
        & (records_df["requested_category"].isin(categories))
    ].copy()

    for direction in directions:
        for category in categories:
            subset = defect_df[
                (defect_df["direction"] == direction)
                & (defect_df["requested_category"] == category)
            ].copy()
            if subset.empty:
                continue
            if category == "farther":
                subset = subset.sort_values(
                    ["graph_distance", "pred_margin", "pred_confidence"],
                    ascending=[False, False, False],
                )
            else:
                subset = subset.sort_values(
                    ["pred_margin", "pred_confidence", "true_confidence"],
                    ascending=[False, False, False],
                )
            selected = subset.head(examples_per_group).copy()
            selected["selection_group"] = f"{direction}_{category}"
            selected["selection_group_rank"] = np.arange(1, len(selected) + 1, dtype=int)
            selected_frames.append(selected)

    if not selected_frames:
        return pd.DataFrame(columns=records_df.columns)
    return pd.concat(selected_frames, ignore_index=True)


def count_message_passing_layers(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, MessagePassing))


def build_specs(args: argparse.Namespace) -> list[ExplainerSpec]:
    return [
        ExplainerSpec(
            key="notebook_object_log_probs",
            algorithm="gnn",
            output_mode="log_probs",
            node_mask_type="object",
            edge_mask_type="object",
            explanation_type="model",
            return_type="log_probs",
            epochs=args.gnn_epochs,
            batchify_data=True,
        ),
        ExplainerSpec(
            key="notebook_common_attributes_log_probs",
            algorithm="gnn",
            output_mode="log_probs",
            node_mask_type="common_attributes",
            edge_mask_type="object",
            explanation_type="model",
            return_type="log_probs",
            epochs=args.gnn_epochs,
            batchify_data=True,
        ),
        ExplainerSpec(
            key="gnn_attr_raw",
            algorithm="gnn",
            output_mode="logits",
            node_mask_type="attributes",
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="gnn_object_log_probs",
            algorithm="gnn",
            output_mode="log_probs",
            node_mask_type="object",
            edge_mask_type="object",
            explanation_type="model",
            return_type="log_probs",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="gnn_common_attributes_log_probs",
            algorithm="gnn",
            output_mode="log_probs",
            node_mask_type="common_attributes",
            edge_mask_type="object",
            explanation_type="model",
            return_type="log_probs",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="gnn_attr_margin",
            algorithm="gnn",
            output_mode="margin",
            node_mask_type="attributes",
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="gnn_object_margin",
            algorithm="gnn",
            output_mode="margin",
            node_mask_type="object",
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="gnn_attr_vs_no_defect",
            algorithm="gnn",
            output_mode="vs_reference_margin",
            node_mask_type="attributes",
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="gnn_object_vs_no_defect",
            algorithm="gnn",
            output_mode="vs_reference_margin",
            node_mask_type="object",
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=args.gnn_epochs,
        ),
        ExplainerSpec(
            key="attention_explainer",
            algorithm="attention",
            output_mode="logits",
            node_mask_type=None,
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=0,
        ),
        ExplainerSpec(
            key="graphmask_object",
            algorithm="graphmask",
            output_mode="logits",
            node_mask_type="object",
            edge_mask_type="object",
            explanation_type="model",
            return_type="raw",
            epochs=args.graphmask_epochs,
        ),
        ExplainerSpec(
            key="pg_explainer",
            algorithm="pg",
            output_mode="logits",
            node_mask_type=None,
            edge_mask_type="object",
            explanation_type="phenomenon",
            return_type="raw",
            epochs=args.pg_train_epochs,
        ),
    ]


def build_pyg_explainer(
    spec: ExplainerSpec,
    base_model: nn.Module,
    num_message_passing_layers: int,
) -> tuple[Explainer, nn.Module]:
    device = next(base_model.parameters()).device
    wrapped_model = ModelOutputAdapter(
        base_model,
        output_mode=spec.output_mode,
        batchify_data=spec.batchify_data,
        reference_class=spec.reference_class,
    ).to(device)

    if spec.algorithm == "gnn":
        algorithm = GNNExplainer(epochs=spec.epochs, lr=0.01)
    elif spec.algorithm == "attention":
        algorithm = AttentionExplainer(reduce="max")
    elif spec.algorithm == "graphmask":
        algorithm = GraphMaskExplainer(
            num_layers=num_message_passing_layers,
            epochs=spec.epochs,
            lr=0.01,
            log=False,
        )
    elif spec.algorithm == "pg":
        algorithm = PGExplainer(epochs=spec.epochs, lr=0.003)
    else:
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")
    if hasattr(algorithm, "to"):
        algorithm = algorithm.to(device)

    explainer = Explainer(
        model=wrapped_model,
        algorithm=algorithm,
        explanation_type=spec.explanation_type,
        node_mask_type=spec.node_mask_type,
        edge_mask_type=spec.edge_mask_type,
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type=spec.return_type,
        ),
    )
    return explainer, wrapped_model


def train_pg_explainer(
    explainer: Explainer,
    wrapped_model: nn.Module,
    benchmark_examples: pd.DataFrame,
    test_dataset,
    device: torch.device,
) -> None:
    epochs = explainer.algorithm.epochs
    for epoch in tqdm(range(epochs), desc="Training PGExplainer", unit="epoch"):
        epoch_losses = []
        for record in benchmark_examples.itertuples(index=False):
            direction_idx = DIRECTION_TO_INDEX[str(record.direction)]
            data = test_dataset[int(record.test_index)][direction_idx].to(device)
            target = torch.tensor([int(record.pred_class)], device=device)
            loss = explainer.algorithm.train(
                epoch,
                wrapped_model,
                data.x,
                data.edge_index,
                target=target,
                data=data,
            )
            epoch_losses.append(loss)
        if epoch_losses:
            tqdm.write(f"PGExplainer epoch {epoch + 1}/{epochs}: loss={np.mean(epoch_losses):.4f}")


def section_neighborhood(true_class: int, distance_df: pd.DataFrame, radius: int = 1) -> set[int]:
    distances = distance_df.iloc[int(true_class)].astype(int).tolist()
    return {
        int(section_id)
        for section_id, distance in enumerate(distances)
        if distance >= 0 and distance <= radius
    }


def parse_incident_sections(section_text: str) -> list[int]:
    if not isinstance(section_text, str) or not section_text:
        return []
    return [int(value) for value in section_text.split(",") if value]


def extract_global_feature_scores(node_mask: np.ndarray, node_feature_names: list[str]) -> pd.DataFrame:
    scores = node_mask.reshape(-1)
    return pd.DataFrame(
        {
            "feature_name": node_feature_names,
            "score": scores.tolist(),
            "feature_group": [
                "dynamic" if feature_name in NODE_DYNAMIC_FEATURES else "static"
                for feature_name in node_feature_names
            ],
        }
    ).sort_values(["score", "feature_name"], ascending=[False, True]).reset_index(drop=True)


def build_local_score_tables(
    spec: ExplainerSpec,
    explanation,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_feature_names: list[str],
    true_class: int,
    pred_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    edge_scores = (
        explanation.edge_mask.detach().cpu().numpy().reshape(-1)
        if getattr(explanation, "edge_mask", None) is not None
        else np.zeros(len(edges_df), dtype=float)
    )
    edge_table = build_generic_edge_score_table(
        edges_df=edges_df,
        edge_scores=edge_scores,
        method=spec.key,
        true_class=true_class,
        pred_class=pred_class,
    )

    node_scores = np.zeros(len(nodes_df), dtype=float)
    node_table = pd.DataFrame()
    global_feature_df = pd.DataFrame()
    raw_node_mask: np.ndarray | None = None

    if getattr(explanation, "node_mask", None) is not None:
        node_mask = explanation.node_mask.detach().cpu().numpy()
        if node_mask.ndim == 1:
            node_mask = node_mask[:, None]

        if node_mask.shape[0] == len(nodes_df):
            raw_node_mask = node_mask
            if node_mask.shape[1] == len(node_feature_names):
                node_scores = node_mask.max(axis=1)
            else:
                node_scores = node_mask.reshape(len(nodes_df), -1).max(axis=1)

            node_table = build_generic_node_score_table(
                nodes_df=nodes_df,
                edges_df=edges_df,
                node_scores=node_scores,
                method=spec.key,
                true_class=true_class,
                pred_class=pred_class,
            )
        elif node_mask.shape[0] == 1 and node_mask.shape[1] == len(node_feature_names):
            global_feature_df = extract_global_feature_scores(node_mask, node_feature_names)

    section_scores = aggregate_scores_by_section(
        nodes_df=nodes_df,
        edges_df=edges_df,
        node_scores=node_scores,
        edge_scores=edge_scores,
        method=spec.key,
        true_class=true_class,
        pred_class=pred_class,
    )
    return node_table, edge_table, section_scores, global_feature_df, raw_node_mask


def compute_local_metrics(
    record,
    node_table: pd.DataFrame,
    section_scores: pd.DataFrame,
    global_feature_df: pd.DataFrame,
    raw_node_mask: np.ndarray | None,
    node_feature_names: list[str],
    distance_df: pd.DataFrame,
) -> dict[str, float | int | None]:
    score_column = "section_score"
    total_section_score = float(section_scores[score_column].sum()) if not section_scores.empty else 0.0
    if total_section_score <= 0.0:
        total_section_score = 1.0

    ordered_sections = section_scores["id_section"].astype(int).tolist()
    true_class = int(record.true_class)
    pred_class = int(record.pred_class)
    top_section = int(section_scores.iloc[0]["id_section"]) if not section_scores.empty else None
    top_section_distance = (
        int(distance_df.iloc[true_class, top_section])
        if top_section is not None and top_section < len(distance_df.columns)
        else None
    )

    true_rank = ordered_sections.index(true_class) + 1 if true_class in ordered_sections else None
    pred_rank = ordered_sections.index(pred_class) + 1 if pred_class in ordered_sections else None
    true_score = float(
        section_scores.loc[section_scores["id_section"] == true_class, score_column].iloc[0]
    ) if true_class in ordered_sections else 0.0
    near_sections = section_neighborhood(true_class, distance_df, radius=1)
    near_mass_ratio = float(
        section_scores.loc[section_scores["id_section"].isin(near_sections), score_column].sum() / total_section_score
    )

    node_near_mass_ratio = np.nan
    dynamic_feature_share = np.nan
    if not node_table.empty and float(node_table["score"].sum()) > 0.0:
        node_total_score = float(node_table["score"].sum())
        touches_near = node_table["incident_sections"].map(
            lambda value: any(section_id in near_sections for section_id in parse_incident_sections(value))
        )
        node_near_mass_ratio = float(node_table.loc[touches_near, "score"].sum() / node_total_score)
        if (
            raw_node_mask is not None
            and raw_node_mask.ndim == 2
            and raw_node_mask.shape[1] == len(node_feature_names)
            and int(touches_near.sum()) > 0
        ):
            dynamic_indices = [
                feature_idx
                for feature_idx, feature_name in enumerate(node_feature_names)
                if feature_name in NODE_DYNAMIC_FEATURES
            ]
            near_node_indices = node_table.loc[touches_near, "node_idx"].astype(int).tolist()
            near_mask = raw_node_mask[near_node_indices]
            total_feature_score = float(near_mask.sum())
            if total_feature_score > 0.0 and dynamic_indices:
                dynamic_feature_share = float(near_mask[:, dynamic_indices].sum() / total_feature_score)

    if not global_feature_df.empty and float(global_feature_df["score"].sum()) > 0.0:
        dynamic_feature_share = float(
            global_feature_df.loc[global_feature_df["feature_group"] == "dynamic", "score"].sum()
            / global_feature_df["score"].sum()
        )

    return {
        "top_section": top_section,
        "top_section_distance": top_section_distance,
        "true_section_rank": true_rank,
        "pred_section_rank": pred_rank,
        "true_section_score": true_score,
        "true_section_score_ratio": true_score / float(section_scores[score_column].max()) if not section_scores.empty and float(section_scores[score_column].max()) > 0.0 else np.nan,
        "near_section_mass_ratio": near_mass_ratio,
        "node_near_mass_ratio": node_near_mass_ratio,
        "dynamic_feature_share": dynamic_feature_share,
    }


def plot_benchmark_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    plotting_df = summary_df.copy()
    plotting_df["display_name"] = plotting_df["explainer"].map(lambda name: METHOD_DISPLAY_NAMES.get(name, name))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("mean_true_section_rank", "Mean true-section rank", False),
        ("mean_top_section_distance", "Mean top-section distance", False),
        ("mean_near_section_mass_ratio", "Mean mass on true/neighbor sections", True),
        ("mean_dynamic_feature_share", "Mean dynamic-feature share", True),
    ]

    for ax, (column, title, higher_is_better) in zip(axes.flat, metrics, strict=False):
        subset = plotting_df.dropna(subset=[column]).copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        subset = subset.sort_values(column, ascending=not higher_is_better)
        ax.barh(subset["display_name"], subset[column], color="#4c72b0")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=9)
        for idx, row in enumerate(subset.itertuples(index=False)):
            ax.text(float(getattr(row, column)) + 0.01, idx, f"{float(getattr(row, column)):.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    output_dir = ensure_dir((args.output_dir or exp_dir / "pyg_explainer_benchmark").resolve())

    cfg = load_cfg(exp_dir)
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
    base_model = load_model(cfg, sample_graph, exp_dir, device)
    num_message_passing_layers = count_message_passing_layers(base_model)

    directions = ["fwd", "bwd"] if args.direction == "both" else [args.direction]
    prediction_frames = []
    for direction in directions:
        prediction_frames.append(
            run_inference(
                model=base_model,
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

    benchmark_examples = select_benchmark_examples(
        predictions_df,
        examples_per_group=args.examples_per_group,
        no_defect_class=no_defect_class,
    )
    benchmark_examples.to_csv(output_dir / "benchmark_examples.csv", index=False)

    specs = build_specs(args)
    specs = [
        spec
        if spec.output_mode != "vs_reference_margin"
        else ExplainerSpec(
            key=spec.key,
            algorithm=spec.algorithm,
            output_mode=spec.output_mode,
            node_mask_type=spec.node_mask_type,
            edge_mask_type=spec.edge_mask_type,
            explanation_type=spec.explanation_type,
            return_type=spec.return_type,
            epochs=spec.epochs,
            batchify_data=spec.batchify_data,
            reference_class=no_defect_class,
        )
        for spec in specs
    ]
    methods_root = ensure_dir(output_dir / "methods")
    benchmark_rows = []
    manifest_rows = []

    metadata = {
        "exp_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "direction": args.direction,
        "device": str(device),
        "examples_per_group": args.examples_per_group,
        "gnn_epochs": args.gnn_epochs,
        "graphmask_epochs": args.graphmask_epochs,
        "pg_train_epochs": args.pg_train_epochs,
        "num_message_passing_layers": int(num_message_passing_layers),
        "captum_available": False,
        "benchmark_methods": [asdict(spec) for spec in specs],
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2))

    for spec in specs:
        method_dir = ensure_dir(methods_root / spec.key)
        examples_dir = ensure_dir(method_dir / "examples")

        try:
            explainer, wrapped_model = build_pyg_explainer(spec, base_model, num_message_passing_layers)
        except Exception as exc:
            benchmark_rows.append(
                {
                    "explainer": spec.key,
                    "sample_id": "",
                    "direction": "",
                    "requested_category": "",
                    "success": False,
                    "error": f"build_failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        if spec.algorithm == "pg":
            try:
                train_pg_explainer(
                    explainer=explainer,
                    wrapped_model=wrapped_model,
                    benchmark_examples=benchmark_examples,
                    test_dataset=test_dataset,
                    device=device,
                )
            except Exception as exc:
                benchmark_rows.append(
                    {
                        "explainer": spec.key,
                        "sample_id": "",
                        "direction": "",
                        "requested_category": "",
                        "success": False,
                        "error": f"train_failed: {type(exc).__name__}: {exc}",
                    }
                )
                continue

        fail_fast = False
        for record in tqdm(
            benchmark_examples.itertuples(index=False),
            total=len(benchmark_examples),
            desc=f"Benchmark {spec.key}",
            unit="example",
        ):
            if fail_fast:
                benchmark_rows.append(
                    {
                        "explainer": spec.key,
                        "sample_id": record.sample_id,
                        "direction": record.direction,
                        "requested_category": record.requested_category,
                        "success": False,
                        "error": "skipped_after_incompatibility",
                    }
                )
                continue

            record_series = pd.Series(record._asdict())
            direction_idx = DIRECTION_TO_INDEX[str(record.direction)]
            data = test_dataset[int(record.test_index)][direction_idx].to(device)

            try:
                if spec.algorithm == "pg":
                    explanation = explainer(
                        data.x,
                        data.edge_index,
                        target=torch.tensor([int(record.pred_class)], device=device),
                        data=data,
                    )
                else:
                    explanation = explainer(
                        data.x,
                        data.edge_index,
                        index=None,
                        data=data,
                    )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                benchmark_rows.append(
                    {
                        "explainer": spec.key,
                        "sample_id": record.sample_id,
                        "direction": record.direction,
                        "requested_category": record.requested_category,
                        "success": False,
                        "error": error_text,
                    }
                )
                if "attention coefficients" in str(exc).lower():
                    fail_fast = True
                continue

            nodes_df, edges_df = build_denorm_tables(
                data,
                pair_scalers[direction_idx],
                cfg,
                direction=str(record.direction),
            )
            node_table, edge_table, section_scores, global_feature_df, raw_node_mask = build_local_score_tables(
                spec=spec,
                explanation=explanation,
                nodes_df=nodes_df,
                edges_df=edges_df,
                node_feature_names=cfg["dataset"]["node_attr"],
                true_class=int(record.true_class),
                pred_class=int(record.pred_class),
            )
            metrics = compute_local_metrics(
                record=record,
                node_table=node_table,
                section_scores=section_scores,
                global_feature_df=global_feature_df,
                raw_node_mask=raw_node_mask,
                node_feature_names=cfg["dataset"]["node_attr"],
                distance_df=distance_df,
            )

            example_name = (
                f"{record.requested_category}_"
                f"{record.direction}_"
                f"t{record.true_class}_p{record.pred_class}_"
                f"idx{record.test_index:04d}"
            )
            example_dir = ensure_dir(examples_dir / example_name)
            nodes_df.to_csv(example_dir / "nodes_denorm.csv", index=False)
            node_table.to_csv(example_dir / "node_scores.csv", index=False)
            edge_table.to_csv(example_dir / "edge_scores.csv", index=False)
            section_scores.to_csv(example_dir / "section_scores.csv", index=False)
            if not global_feature_df.empty:
                global_feature_df.to_csv(example_dir / "global_feature_scores.csv", index=False)

            plot_path = plot_baseline_explanation_bundle(
                record=record_series,
                method=spec.key,
                nodes_df=nodes_df,
                edges_df=edges_df,
                edge_score_table=edge_table,
                node_score_table=node_table,
                section_score_table=section_scores,
                example_dir=example_dir,
                top_nodes=args.top_nodes,
                top_edges=args.top_edges,
                top_sections=args.top_sections,
            )

            benchmark_rows.append(
                {
                    "explainer": spec.key,
                    "sample_id": record.sample_id,
                    "direction": record.direction,
                    "requested_category": record.requested_category,
                    "selection_group": record.selection_group,
                    "selection_group_rank": int(record.selection_group_rank),
                    "success": True,
                    "error": "",
                    **metrics,
                }
            )
            manifest_rows.append(
                {
                    **record._asdict(),
                    "explainer": spec.key,
                    "example_dir": str(example_dir),
                    "plot_path": str(plot_path),
                    **metrics,
                }
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    benchmark_df.to_csv(output_dir / "explainer_benchmark_examples.csv", index=False)
    if not manifest_df.empty:
        manifest_df.to_csv(output_dir / "explainer_benchmark_manifest.csv", index=False)

    summary_df = (
        benchmark_df[benchmark_df["success"] == True]
        .groupby("explainer")
        .agg(
            num_examples=("sample_id", "count"),
            mean_true_section_rank=("true_section_rank", "mean"),
            median_true_section_rank=("true_section_rank", "median"),
            mean_top_section_distance=("top_section_distance", "mean"),
            mean_near_section_mass_ratio=("near_section_mass_ratio", "mean"),
            mean_node_near_mass_ratio=("node_near_mass_ratio", "mean"),
            mean_dynamic_feature_share=("dynamic_feature_share", "mean"),
            mean_true_section_score_ratio=("true_section_score_ratio", "mean"),
        )
        .reset_index()
    )

    failure_df = (
        benchmark_df[benchmark_df["success"] == False]
        .groupby("explainer")
        .agg(
            num_failures=("error", "count"),
            errors=("error", lambda values: " | ".join(sorted(set(values))[:3])),
        )
        .reset_index()
    )
    if not failure_df.empty:
        summary_df = summary_df.merge(failure_df, on="explainer", how="outer")
    summary_df = summary_df.sort_values(
        ["mean_true_section_rank", "mean_top_section_distance", "mean_near_section_mass_ratio"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    summary_df.to_csv(output_dir / "explainer_benchmark_summary.csv", index=False)
    plot_benchmark_summary(summary_df, output_dir / "explainer_benchmark_summary.png")

    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    (output_dir / "best_explainer.json").write_text(json.dumps(best_row, indent=2, default=str))

    print(f"Saved PyG explainer benchmark to: {output_dir}")


if __name__ == "__main__":
    main()
