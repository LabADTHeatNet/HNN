from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm.auto import tqdm

from src.datasets import add_sections, data_to_tables


DIRECTION_TO_INDEX = {"fwd": 0, "bwd": 1}
REQUESTED_CATEGORY_ORDER = ["correct", "neighbor", "farther", "no_defect"]
ANALYSIS_CATEGORY_ORDER = [
    "correct_defect",
    "neighbor_error",
    "far_error",
    "missed_defect_no_prediction",
    "correct_no_defect",
    "false_positive_no_defect",
]
PRIMARY_EXPLANATION_METHOD = "gnn_object_margin"
PAIR_AWARE_EXPLANATION_METHOD = "gnn_pair_ideal_delta_margin"
EXPLANATION_METHOD_ORDER = [
    PRIMARY_EXPLANATION_METHOD,
    PAIR_AWARE_EXPLANATION_METHOD,
    "activation_attention",
    "grad_x_input",
    "section_occlusion",
]
NODE_DYNAMIC_FEATURES = {"P", "Temp", "P_ideal", "Temp_ideal"}
PAIR_DYNAMIC_FEATURES = {"delta_P", "delta_T"}
FWD_SECTION_IDS = np.array(
    [
        0, 0, 41, 24, 0, 13, 18, 18, 18, 30, 31, 10, 11, 11, 20, 21, 21,
        1, 1, 0, 0, 11, 9, 9, 10, 5, 3, 3, 3, 3, 3, 3, 3, 37,
        8, 9, 14, 25, 25, 25, 32, 19, 11, 6, 2, 15, 15, 16, 4, 15, 16,
        14, 25, 25, 35, 10, 19, 18, 32, 36, 36, 36, 36, 36, 38, 8, 9, 14,
        35, 36, 32, 32, 32, 18, 11, 6, 2, 4, 15, 16, 16, 14, 25, 35, 10,
        10, 19, 18, 32, 32, 32, 36, 36, 42, 39, 29, 17, 12, 33, 34, 22, 23,
        7, 40, 28, 27, 26,
    ],
    dtype=int,
)
BWD_SECTION_IDS = np.array(
    [
        0, 41, 24, 0, 13, 18, 18, 18, 30, 31, 10, 11, 11, 20, 21, 21, 1,
        1, 0, 0, 11, 9, 9, 10, 5, 3, 3, 3, 3, 3, 3, 3, 37, 8,
        9, 14, 25, 25, 25, 32, 19, 11, 6, 2, 15, 15, 16, 4, 15, 16, 14,
        25, 25, 35, 10, 19, 18, 32, 36, 36, 36, 36, 36, 38, 8, 9, 14, 35,
        36, 32, 32, 32, 18, 11, 6, 2, 4, 15, 16, 16, 14, 25, 35, 10, 10,
        19, 18, 32, 32, 32, 36, 36, 0, 42, 39, 29, 17, 12, 33, 34, 22, 23,
        7, 40, 28, 27, 26,
    ],
    dtype=int,
)


@dataclass
class PredictionRecord:
    test_index: int
    dataset_index: int
    direction: str
    sample_id: str
    tout: int
    nodes_fp: str
    edges_fp: str
    true_class: int
    pred_class: int
    pred_confidence: float
    true_confidence: float
    pred_margin: float
    graph_distance: int | None
    requested_category: str | None
    analysis_category: str


@dataclass
class ExampleExplanation:
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    node_table: pd.DataFrame
    top_node_table: pd.DataFrame
    node_feature_df: pd.DataFrame
    compact_node_feature_df: pd.DataFrame
    edge_table: pd.DataFrame
    top_edge_table: pd.DataFrame
    edge_feature_df: pd.DataFrame
    compact_edge_feature_df: pd.DataFrame
    section_scores: pd.DataFrame
    baseline_node_scores: pd.DataFrame
    baseline_edge_scores: pd.DataFrame
    baseline_section_scores: pd.DataFrame
    baseline_method_summary: pd.DataFrame
    true_section_rank: int | None
    pred_section_rank: int | None


@dataclass(frozen=True)
class ExplainerSetup:
    method_key: str
    explainer: Explainer
    node_feature_names: list[str]
    transform_node_inputs: Callable[[torch.Tensor], torch.Tensor]
    inverse_node_inputs: Callable[[torch.Tensor], torch.Tensor]


class ExplainableEdgeClassifierAdapter(nn.Module):
    """Adapter around the trained model with a stable forward API for PyG Explainer."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _prepare_graph(data, edge_attr: torch.Tensor | None = None):
        data = data.clone()
        data.global_attrs = data.global_attrs.reshape(-1, data.global_attrs.shape[-1])
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if getattr(data, "batch", None) is None and "batch" in data:
            del data.batch
        return data

    def forward(self, x, edge_index, data, edge_attr: torch.Tensor | None = None):
        graph_state = self.extract_graph_state(x, edge_index, data, edge_attr=edge_attr)
        return graph_state["logits"]

    def extract_graph_state(
        self,
        x,
        edge_index,
        data,
        edge_attr: torch.Tensor | None = None,
        retain_grads: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        data = self._prepare_graph(data, edge_attr=edge_attr)
        gp = data.global_attrs
        batch = getattr(data, "batch", None)

        if batch is None:
            gp_per_node = gp.expand(x.shape[0], -1)
        else:
            gp_per_node = gp[batch]

        x_with_global = torch.cat([x, gp_per_node], dim=-1)
        node_features = self.model.node_encoder(x_with_global, edge_index)

        src, dst = edge_index
        if batch is None:
            edge_batch = None
            gp_per_edge = gp.expand(src.shape[0], -1)
        else:
            edge_batch = batch[src]
            gp_per_edge = gp[edge_batch]

        edge_init_feat = torch.cat(
            [node_features[src], node_features[dst], data.edge_attr, gp_per_edge],
            dim=-1,
        )
        edge_feat = self.model.edge_init(edge_init_feat)

        for layer in self.model.edge_update_layers:
            edge_feat = layer(edge_feat, edge_index, num_nodes=node_features.shape[0])

        fused_edge_feat = torch.cat([edge_feat, edge_init_feat], dim=-1)
        attention_scores = self.model.attention_weights(fused_edge_feat)
        attention_weights = torch.softmax(attention_scores, dim=0)
        global_repr = self.model._fused_pool(edge_feat, edge_init_feat, edge_batch)
        logits = self.model.classifier(global_repr)

        if retain_grads:
            node_features.retain_grad()
            edge_init_feat.retain_grad()
            edge_feat.retain_grad()
            fused_edge_feat.retain_grad()

        return {
            "data": data,
            "node_features": node_features,
            "edge_init_feat": edge_init_feat,
            "edge_feat": edge_feat,
            "fused_edge_feat": fused_edge_feat,
            "attention_scores": attention_scores,
            "attention_weights": attention_weights,
            "global_repr": global_repr,
            "logits": logits,
            "edge_batch": edge_batch,
        }


class PyGExplainOutputAdapter(nn.Module):
    """Wrap the explainable model and expose a PyG-friendly output space."""

    def __init__(
        self,
        model: ExplainableEdgeClassifierAdapter,
        node_feature_names: list[str] | None = None,
        output_mode: str = "margin",
        batchify_data: bool = False,
        reference_class: int | None = None,
        explain_space: str = "raw",
    ):
        super().__init__()
        self.model = model
        self.original_node_feature_names = list(node_feature_names or [])
        self.output_mode = output_mode
        self.batchify_data = batchify_data
        self.reference_class = reference_class
        self.explain_space = explain_space
        (
            self.explain_node_feature_names,
            self.transform_node_inputs,
            self.inverse_node_inputs,
        ) = self._build_input_transforms()

    def _build_input_transforms(
        self,
    ) -> tuple[list[str], Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        if self.explain_space == "raw":
            return (
                list(self.original_node_feature_names),
                lambda x: x,
                lambda x: x,
            )

        if self.explain_space != "pair_ideal_delta":
            raise ValueError(f"Unsupported explain_space: {self.explain_space}")

        node_feature_names = list(self.original_node_feature_names)
        required = ["P", "Temp", "P_ideal", "Temp_ideal"]
        missing = [name for name in required if name not in node_feature_names]
        if missing:
            raise ValueError(f"Missing required pair features: {missing}")

        p_idx = node_feature_names.index("P")
        t_idx = node_feature_names.index("Temp")
        p_ideal_idx = node_feature_names.index("P_ideal")
        t_ideal_idx = node_feature_names.index("Temp_ideal")
        excluded = {"P", "Temp", "P_ideal", "Temp_ideal"}
        static_indices = [idx for idx, name in enumerate(node_feature_names) if name not in excluded]
        explain_feature_names = [
            *[node_feature_names[idx] for idx in static_indices],
            "P_ideal",
            "Temp_ideal",
            "delta_P",
            "delta_T",
        ]

        def transform_node_inputs(x: torch.Tensor) -> torch.Tensor:
            parts = []
            if static_indices:
                parts.append(x[:, static_indices])
            p_ideal = x[:, p_ideal_idx].unsqueeze(-1)
            t_ideal = x[:, t_ideal_idx].unsqueeze(-1)
            delta_p = (x[:, p_idx] - x[:, p_ideal_idx]).unsqueeze(-1)
            delta_t = (x[:, t_idx] - x[:, t_ideal_idx]).unsqueeze(-1)
            parts.extend([p_ideal, t_ideal, delta_p, delta_t])
            return torch.cat(parts, dim=-1)

        def inverse_node_inputs(x_pair: torch.Tensor) -> torch.Tensor:
            restored = x_pair.new_zeros((x_pair.shape[0], len(node_feature_names)))
            cursor = 0
            if static_indices:
                static_width = len(static_indices)
                restored[:, static_indices] = x_pair[:, :static_width]
                cursor = static_width
            p_ideal = x_pair[:, cursor]
            t_ideal = x_pair[:, cursor + 1]
            delta_p = x_pair[:, cursor + 2]
            delta_t = x_pair[:, cursor + 3]
            restored[:, p_ideal_idx] = p_ideal
            restored[:, t_ideal_idx] = t_ideal
            restored[:, p_idx] = p_ideal + delta_p
            restored[:, t_idx] = t_ideal + delta_t
            return restored

        return explain_feature_names, transform_node_inputs, inverse_node_inputs

    def _format_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "logits":
            return logits
        if self.output_mode == "log_probs":
            return torch.log_softmax(logits, dim=-1)
        if self.output_mode == "margin":
            margins = []
            for class_idx in range(logits.shape[-1]):
                class_logits = logits[:, class_idx]
                other_logits = torch.cat([logits[:, :class_idx], logits[:, class_idx + 1 :]], dim=-1)
                margins.append(class_logits - other_logits.max(dim=-1).values)
            return torch.stack(margins, dim=-1)
        if self.output_mode == "vs_reference_margin":
            if self.reference_class is None:
                raise ValueError("reference_class is required for vs_reference_margin")
            return logits - logits[:, self.reference_class].unsqueeze(-1)
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def forward(self, x, edge_index, data, edge_attr: torch.Tensor | None = None):
        model_data = data
        if self.batchify_data and not hasattr(data, "batch"):
            model_data = Batch.from_data_list([data])
        model_x = self.inverse_node_inputs(x)
        logits = self.model(model_x, edge_index, model_data, edge_attr=edge_attr)
        return self._format_logits(logits)


def resolve_device(device_arg: str, cfg: dict) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if cfg["utils"].get("device") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_id_from_path(path_like: str | Path) -> str:
    path = Path(path_like)
    parts = path.parts
    start = next((idx for idx, part in enumerate(parts) if part.startswith("Tout_")), None)
    if start is None:
        return path.with_suffix("").as_posix()
    return Path(*parts[start:]).with_suffix("").as_posix()


def get_tout_from_path(path_like: str | Path) -> int:
    path = Path(path_like)
    for part in path.parts:
        if part.startswith("Tout_"):
            return int(part.replace("Tout_", ""))
    raise ValueError(f"Cannot recover Tout from path: {path}")


def compute_graph_distance(
    true_class: int,
    pred_class: int,
    distance_df: pd.DataFrame,
    no_defect_class: int,
) -> int | None:
    if true_class == no_defect_class or pred_class == no_defect_class:
        return None
    if true_class >= len(distance_df.index) or pred_class >= len(distance_df.columns):
        return None
    value = int(distance_df.iloc[true_class, pred_class])
    return None if value < 0 else value


def categorize_prediction(
    true_class: int,
    pred_class: int,
    graph_distance: int | None,
    no_defect_class: int,
) -> tuple[str | None, str]:
    if true_class == no_defect_class:
        if pred_class == no_defect_class:
            return "no_defect", "correct_no_defect"
        return None, "false_positive_no_defect"

    if pred_class == true_class:
        return "correct", "correct_defect"

    if pred_class == no_defect_class:
        return None, "missed_defect_no_prediction"

    if graph_distance == 1:
        return "neighbor", "neighbor_error"

    return "farther", "far_error"


def load_cfg(exp_dir: Path) -> dict:
    cfg = json.loads((exp_dir / "params.json").read_text())
    cfg["dataset"]["load"] = True
    return cfg


def load_model(cfg: dict, sample_graph, exp_dir: Path, device: torch.device) -> ExplainableEdgeClassifierAdapter:
    model_module = importlib.import_module(f"src.models.{cfg['model']['name']}")
    model_class = getattr(model_module, cfg["model"]["name"])
    base_model = model_class(
        in_node_dim=sample_graph.x.shape[1],
        in_edge_dim=sample_graph.edge_attr.shape[1],
        **cfg["model"]["kwargs"],
    )
    state = torch.load(exp_dir / "best_model.pth", map_location="cpu", weights_only=True)
    base_model.load_state_dict(state)
    base_model = base_model.to(device)
    base_model.eval()
    return ExplainableEdgeClassifierAdapter(base_model).to(device)


def build_explainer(
    model: ExplainableEdgeClassifierAdapter,
    epochs: int,
    node_feature_names: list[str],
    explainer_kind: str = PRIMARY_EXPLANATION_METHOD,
) -> ExplainerSetup:
    if explainer_kind == PRIMARY_EXPLANATION_METHOD:
        wrapped_model = PyGExplainOutputAdapter(
            model=model,
            node_feature_names=node_feature_names,
            output_mode="margin",
            batchify_data=False,
            explain_space="raw",
        ).to(next(model.parameters()).device)
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=epochs, lr=0.01),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )
        return ExplainerSetup(
            method_key=explainer_kind,
            explainer=explainer,
            node_feature_names=list(node_feature_names),
            transform_node_inputs=wrapped_model.transform_node_inputs,
            inverse_node_inputs=wrapped_model.inverse_node_inputs,
        )

    if explainer_kind == PAIR_AWARE_EXPLANATION_METHOD:
        wrapped_model = PyGExplainOutputAdapter(
            model=model,
            node_feature_names=node_feature_names,
            output_mode="margin",
            batchify_data=False,
            explain_space="pair_ideal_delta",
        ).to(next(model.parameters()).device)
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=epochs, lr=0.01),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )
        return ExplainerSetup(
            method_key=explainer_kind,
            explainer=explainer,
            node_feature_names=list(wrapped_model.explain_node_feature_names),
            transform_node_inputs=wrapped_model.transform_node_inputs,
            inverse_node_inputs=wrapped_model.inverse_node_inputs,
        )

    raise ValueError(f"Unsupported explainer kind: {explainer_kind}")


def run_inference(
    model: nn.Module,
    test_loader,
    test_dataset,
    direction: str,
    distance_df: pd.DataFrame,
    no_defect_class: int,
    device: torch.device,
) -> pd.DataFrame:
    direction_idx = DIRECTION_TO_INDEX[direction]
    dataset_indices = list(getattr(test_dataset, "indices", range(len(test_dataset))))
    cursor = 0
    records: list[PredictionRecord] = []

    with torch.no_grad():
        for pair_batch in tqdm(test_loader, desc=f"Inference {direction}", unit="batch"):
            batch = pair_batch[direction_idx].to(device)
            logits = model(batch.x, batch.edge_index, batch)
            probs = logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)
            top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=-1).values
            graphs = batch.to_data_list()
            true_values = batch.edge_label.view(-1)

            for local_idx, graph in enumerate(graphs):
                dataset_index = int(dataset_indices[cursor])
                test_index = cursor
                cursor += 1

                true_class = int(true_values[local_idx].item())
                pred_class = int(pred[local_idx].item())
                graph_distance = compute_graph_distance(
                    true_class=true_class,
                    pred_class=pred_class,
                    distance_df=distance_df,
                    no_defect_class=no_defect_class,
                )
                requested_category, analysis_category = categorize_prediction(
                    true_class=true_class,
                    pred_class=pred_class,
                    graph_distance=graph_distance,
                    no_defect_class=no_defect_class,
                )
                margin = float(
                    top2[local_idx, 0].item() - top2[local_idx, 1].item()
                    if top2.shape[1] > 1
                    else top2[local_idx, 0].item()
                )

                records.append(
                    PredictionRecord(
                        test_index=test_index,
                        dataset_index=dataset_index,
                        direction=direction,
                        sample_id=sample_id_from_path(graph.edges_fp),
                        tout=get_tout_from_path(graph.edges_fp),
                        nodes_fp=str(graph.nodes_fp),
                        edges_fp=str(graph.edges_fp),
                        true_class=true_class,
                        pred_class=pred_class,
                        pred_confidence=float(probs[local_idx, pred_class].item()),
                        true_confidence=float(probs[local_idx, true_class].item()),
                        pred_margin=margin,
                        graph_distance=graph_distance,
                        requested_category=requested_category,
                        analysis_category=analysis_category,
                    )
                )

    if cursor != len(dataset_indices):
        raise RuntimeError(f"{direction}: consumed {cursor} graphs, expected {len(dataset_indices)}")

    return pd.DataFrame(asdict(record) for record in records)


def normalize_boolean_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(float) > 0.5
    return df


def build_denorm_tables(data, scalers: dict, cfg: dict, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scalers = dict(scalers)
    scalers.setdefault("edge_label_scaler", None)
    nodes_df, edges_df = data_to_tables(
        data.clone().cpu(),
        node_attr=cfg["dataset"]["node_attr"],
        edge_attr=cfg["dataset"]["edge_attr"],
        edge_label=cfg["dataset"]["edge_label"],
        scalers=scalers,
        edge_label_pred=None,
    )
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    nodes_df["id"] = nodes_df["id"].round().astype(int)
    edges_df["id_in"] = edges_df["id_in"].round().astype(int)
    edges_df["id_out"] = edges_df["id_out"].round().astype(int)
    if "graph_label" in edges_df.columns:
        edges_df["graph_label"] = edges_df["graph_label"].round().astype(int)
    nodes_df = normalize_boolean_columns(nodes_df, ["types_def", "types_usr", "types_src"])
    edges_df = normalize_boolean_columns(edges_df, ["Vid_fwd", "Vid_bwd", "Vid_usr"])

    predefined_sections = FWD_SECTION_IDS if direction == "fwd" else BWD_SECTION_IDS
    if len(edges_df) == len(predefined_sections):
        edges_df["id_section"] = predefined_sections
    else:
        edges_df = add_sections(nodes_df.copy(), edges_df.copy())
        edges_df["id_section"] = edges_df["id_section"].astype(int)

    edges_df.insert(0, "edge_idx", np.arange(len(edges_df), dtype=int))
    return nodes_df, edges_df


def build_node_section_map(edges_df: pd.DataFrame) -> dict[int, list[int]]:
    node_sections: dict[int, set[int]] = {}
    for row in edges_df.itertuples(index=False):
        section_id = int(row.id_section)
        src = int(row.id_in)
        dst = int(row.id_out)
        node_sections.setdefault(src, set()).add(section_id)
        node_sections.setdefault(dst, set()).add(section_id)
    return {
        node_id: sorted(section_ids)
        for node_id, section_ids in node_sections.items()
    }


def format_section_ids(section_ids: Iterable[int]) -> str:
    section_ids = sorted(int(section_id) for section_id in section_ids)
    return ",".join(str(section_id) for section_id in section_ids)


def classify_node_feature(feature_name: str) -> str:
    if feature_name in PAIR_DYNAMIC_FEATURES:
        return "dynamic"
    return "dynamic" if feature_name in NODE_DYNAMIC_FEATURES else "static"


def build_node_table(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_mask_matrix: np.ndarray,
    node_feature_names: list[str],
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    if node_mask_matrix.ndim == 1:
        node_mask_matrix = node_mask_matrix[:, None]

    if node_mask_matrix.shape[0] != len(nodes_df):
        raise RuntimeError(
            f"Node mask rows ({node_mask_matrix.shape[0]}) do not match nodes ({len(nodes_df)})"
        )

    node_sections = build_node_section_map(edges_df)
    node_table = nodes_df.copy().reset_index(drop=True)
    node_table["node_idx"] = np.arange(len(node_table), dtype=int)
    node_table["importance_sum"] = node_mask_matrix.sum(axis=1)
    node_table["importance_mean"] = node_mask_matrix.mean(axis=1)
    node_table["importance_max"] = node_mask_matrix.max(axis=1)
    if node_mask_matrix.shape[1] == len(node_feature_names):
        node_table["top_feature_idx"] = node_mask_matrix.argmax(axis=1)
        node_table["top_feature_name"] = node_table["top_feature_idx"].map(
            lambda idx: node_feature_names[int(idx)]
        )
        node_table["top_feature_score"] = node_table["importance_max"]
        node_table["top_feature_group"] = node_table["top_feature_name"].map(classify_node_feature)
    else:
        node_table["top_feature_idx"] = -1
        node_table["top_feature_name"] = "object_mask"
        node_table["top_feature_score"] = node_table["importance_max"]
        node_table["top_feature_group"] = "object"
    node_table["incident_sections"] = node_table["id"].map(
        lambda node_id: format_section_ids(node_sections.get(int(node_id), []))
    )
    node_table["num_sections"] = node_table["id"].map(
        lambda node_id: len(node_sections.get(int(node_id), []))
    )
    node_table["touches_true_section"] = node_table["id"].map(
        lambda node_id: true_class in node_sections.get(int(node_id), [])
    )
    node_table["touches_pred_section"] = node_table["id"].map(
        lambda node_id: pred_class in node_sections.get(int(node_id), [])
    )
    node_table = node_table.sort_values(
        ["importance_max", "importance_sum", "top_feature_score", "node_idx"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    node_table["importance_rank"] = np.arange(1, len(node_table) + 1, dtype=int)
    return node_table


def build_edge_table(
    edges_df: pd.DataFrame,
    edge_scores: np.ndarray,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    edge_scores = np.asarray(edge_scores, dtype=float).reshape(-1)
    if edge_scores.shape[0] != len(edges_df):
        raise RuntimeError(
            f"Edge mask size ({edge_scores.shape[0]}) does not match edges ({len(edges_df)})"
        )

    edge_table = edges_df.copy()
    edge_table["importance"] = edge_scores
    edge_table["is_true_section"] = edge_table["id_section"] == int(true_class)
    edge_table["is_pred_section"] = edge_table["id_section"] == int(pred_class)
    edge_table = edge_table.sort_values(["importance", "edge_idx"], ascending=[False, True]).reset_index(drop=True)
    edge_table["importance_rank"] = np.arange(1, len(edge_table) + 1, dtype=int)
    return edge_table


def build_section_scores(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    edges_df: pd.DataFrame,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    node_sections = build_node_section_map(edges_df)
    node_section_rows = []
    for row in node_table.itertuples(index=False):
        for section_id in node_sections.get(int(row.id), []):
            node_section_rows.append(
                {
                    "id_section": int(section_id),
                    "id": int(row.id),
                    "importance_sum": float(row.importance_sum),
                    "importance_mean": float(row.importance_mean),
                    "importance_max": float(row.importance_max),
                }
            )

    if node_section_rows:
        node_section_df = pd.DataFrame(node_section_rows)
        node_summary = (
            node_section_df.groupby("id_section")
            .agg(
                node_importance_sum=("importance_sum", "sum"),
                node_importance_mean=("importance_mean", "mean"),
                node_importance_max=("importance_max", "max"),
                num_nodes=("id", "nunique"),
            )
            .reset_index()
        )
    else:
        node_summary = pd.DataFrame(
            columns=[
                "id_section",
                "node_importance_sum",
                "node_importance_mean",
                "node_importance_max",
                "num_nodes",
            ]
        )

    if edge_table.empty:
        edge_summary = pd.DataFrame(
            columns=[
                "id_section",
                "edge_importance_sum",
                "edge_importance_mean",
                "edge_importance_max",
                "num_edges",
            ]
        )
    else:
        edge_summary = (
            edge_table.groupby("id_section")
            .agg(
                edge_importance_sum=("importance", "sum"),
                edge_importance_mean=("importance", "mean"),
                edge_importance_max=("importance", "max"),
                num_edges=("edge_idx", "count"),
            )
            .reset_index()
        )

    all_sections = sorted(set(edges_df["id_section"].astype(int).tolist()))
    section_scores = pd.DataFrame({"id_section": all_sections})
    section_scores = section_scores.merge(node_summary, on="id_section", how="left")
    section_scores = section_scores.merge(edge_summary, on="id_section", how="left")

    numeric_defaults = {
        "node_importance_sum": 0.0,
        "node_importance_mean": 0.0,
        "node_importance_max": 0.0,
        "num_nodes": 0,
        "edge_importance_sum": 0.0,
        "edge_importance_mean": 0.0,
        "edge_importance_max": 0.0,
        "num_edges": 0,
    }
    for column, default_value in numeric_defaults.items():
        section_scores[column] = section_scores[column].fillna(default_value)

    section_scores["section_rank_score"] = np.where(
        section_scores["node_importance_max"] > 0.0,
        section_scores["node_importance_max"],
        section_scores["edge_importance_max"],
    )
    section_scores["is_true_section"] = section_scores["id_section"] == int(true_class)
    section_scores["is_pred_section"] = section_scores["id_section"] == int(pred_class)
    section_scores = section_scores.sort_values(
        [
            "section_rank_score",
            "node_importance_sum",
            "edge_importance_max",
            "edge_importance_sum",
            "id_section",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return section_scores


def build_generic_node_score_table(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_scores: np.ndarray,
    method: str,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    node_scores = np.asarray(node_scores, dtype=float).reshape(-1)
    if node_scores.shape[0] != len(nodes_df):
        raise RuntimeError(
            f"{method}: node score size ({node_scores.shape[0]}) does not match nodes ({len(nodes_df)})"
        )

    node_sections = build_node_section_map(edges_df)
    node_table = nodes_df.copy().reset_index(drop=True)
    node_table["node_idx"] = np.arange(len(node_table), dtype=int)
    node_table["method"] = method
    node_table["score"] = node_scores
    node_table["incident_sections"] = node_table["id"].map(
        lambda node_id: format_section_ids(node_sections.get(int(node_id), []))
    )
    node_table["touches_true_section"] = node_table["id"].map(
        lambda node_id: true_class in node_sections.get(int(node_id), [])
    )
    node_table["touches_pred_section"] = node_table["id"].map(
        lambda node_id: pred_class in node_sections.get(int(node_id), [])
    )
    node_table = node_table.sort_values(["score", "node_idx"], ascending=[False, True]).reset_index(drop=True)
    node_table["rank"] = np.arange(1, len(node_table) + 1, dtype=int)
    return node_table


def build_generic_edge_score_table(
    edges_df: pd.DataFrame,
    edge_scores: np.ndarray,
    method: str,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    edge_scores = np.asarray(edge_scores, dtype=float).reshape(-1)
    if edge_scores.shape[0] != len(edges_df):
        raise RuntimeError(
            f"{method}: edge score size ({edge_scores.shape[0]}) does not match edges ({len(edges_df)})"
        )

    edge_table = edges_df.copy()
    edge_table["method"] = method
    edge_table["score"] = edge_scores
    edge_table["is_true_section"] = edge_table["id_section"] == int(true_class)
    edge_table["is_pred_section"] = edge_table["id_section"] == int(pred_class)
    edge_table = edge_table.sort_values(["score", "edge_idx"], ascending=[False, True]).reset_index(drop=True)
    edge_table["rank"] = np.arange(1, len(edge_table) + 1, dtype=int)
    return edge_table


def aggregate_scores_by_section(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
    method: str,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    node_section_map = build_node_section_map(edges_df)
    node_id_to_idx = {
        int(node_id): int(idx)
        for idx, node_id in enumerate(nodes_df["id"].astype(int).tolist())
    }

    node_rows = []
    for node_id, section_ids in node_section_map.items():
        node_idx = node_id_to_idx.get(int(node_id))
        if node_idx is None:
            continue
        for section_id in section_ids:
            node_rows.append(
                {
                    "id_section": int(section_id),
                    "score": float(node_scores[node_idx]),
                    "node_id": int(node_id),
                }
            )

    if node_rows:
        node_summary = (
            pd.DataFrame(node_rows)
            .groupby("id_section")
            .agg(
                node_score_max=("score", "max"),
                node_score_sum=("score", "sum"),
                num_nodes=("node_id", "nunique"),
            )
            .reset_index()
        )
    else:
        node_summary = pd.DataFrame(columns=["id_section", "node_score_max", "node_score_sum", "num_nodes"])

    edge_summary = (
        pd.DataFrame(
            {
                "id_section": edges_df["id_section"].astype(int).tolist(),
                "score": np.asarray(edge_scores, dtype=float).tolist(),
                "edge_idx": edges_df["edge_idx"].astype(int).tolist(),
            }
        )
        .groupby("id_section")
        .agg(
            edge_score_max=("score", "max"),
            edge_score_sum=("score", "sum"),
            num_edges=("edge_idx", "count"),
        )
        .reset_index()
    )

    all_sections = sorted(set(edges_df["id_section"].astype(int).tolist()))
    section_scores = pd.DataFrame({"id_section": all_sections})
    section_scores = section_scores.merge(node_summary, on="id_section", how="left")
    section_scores = section_scores.merge(edge_summary, on="id_section", how="left")

    for column in ["node_score_max", "node_score_sum", "edge_score_max", "edge_score_sum"]:
        section_scores[column] = section_scores[column].fillna(0.0)
    for column in ["num_nodes", "num_edges"]:
        section_scores[column] = section_scores[column].fillna(0).astype(int)

    section_scores["section_score"] = np.maximum(
        section_scores["node_score_max"],
        section_scores["edge_score_max"],
    )
    section_scores["method"] = method
    section_scores["is_true_section"] = section_scores["id_section"] == int(true_class)
    section_scores["is_pred_section"] = section_scores["id_section"] == int(pred_class)
    section_scores = section_scores.sort_values(
        ["section_score", "node_score_sum", "edge_score_sum", "id_section"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    section_scores["rank"] = np.arange(1, len(section_scores) + 1, dtype=int)
    return section_scores


def expand_section_scores_to_node_scores(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    section_scores_df: pd.DataFrame,
) -> np.ndarray:
    section_score_map = {
        int(row.id_section): float(row.section_score)
        for row in section_scores_df.itertuples(index=False)
    }
    node_section_map = build_node_section_map(edges_df)
    node_scores = np.zeros(len(nodes_df), dtype=float)
    for node_idx, node_id in enumerate(nodes_df["id"].astype(int).tolist()):
        incident_sections = node_section_map.get(int(node_id), [])
        if not incident_sections:
            continue
        node_scores[node_idx] = max(section_score_map.get(int(section_id), 0.0) for section_id in incident_sections)
    return node_scores


def expand_section_scores_to_edge_scores(
    edges_df: pd.DataFrame,
    section_scores_df: pd.DataFrame,
) -> np.ndarray:
    section_score_map = {
        int(row.id_section): float(row.section_score)
        for row in section_scores_df.itertuples(index=False)
    }
    return np.asarray(
        [
            section_score_map.get(int(section_id), 0.0)
            for section_id in edges_df["id_section"].astype(int).tolist()
        ],
        dtype=float,
    )


def build_method_summary(
    section_scores_df: pd.DataFrame,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    rows = []
    for method in section_scores_df["method"].drop_duplicates().tolist():
        subset = section_scores_df[section_scores_df["method"] == method].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["section_score", "rank"], ascending=[False, True]).reset_index(drop=True)
        ordered_sections = subset["id_section"].astype(int).tolist()
        true_rank = ordered_sections.index(true_class) + 1 if true_class in ordered_sections else None
        pred_rank = ordered_sections.index(pred_class) + 1 if pred_class in ordered_sections else None
        true_score_series = subset.loc[subset["id_section"] == true_class, "section_score"]
        pred_score_series = subset.loc[subset["id_section"] == pred_class, "section_score"]
        rows.append(
            {
                "method": method,
                "top_section": int(subset.iloc[0]["id_section"]),
                "top_score": float(subset.iloc[0]["section_score"]),
                "true_section_rank": true_rank,
                "pred_section_rank": pred_rank,
                "true_section_score": float(true_score_series.iloc[0]) if not true_score_series.empty else np.nan,
                "pred_section_score": float(pred_score_series.iloc[0]) if not pred_score_series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_activation_attention_scores(
    model: ExplainableEdgeClassifierAdapter,
    data,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        state = model.extract_graph_state(data.x, data.edge_index, data)
    node_scores = state["node_features"].abs().sum(dim=-1).detach().cpu().numpy()
    edge_scores = (
        state["attention_weights"].reshape(-1).abs()
        * state["fused_edge_feat"].abs().sum(dim=-1)
    ).detach().cpu().numpy()
    return node_scores, edge_scores


def compute_grad_x_input_scores(
    model: ExplainableEdgeClassifierAdapter,
    data,
    target_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    x = data.x.detach().clone().requires_grad_(True)
    edge_attr = data.edge_attr.detach().clone().requires_grad_(True)
    logits = model(x, data.edge_index, data, edge_attr=edge_attr)
    logits[0, target_class].backward()

    node_scores = (x.grad * x).abs().sum(dim=-1).detach().cpu().numpy()
    edge_scores = (edge_attr.grad * edge_attr).abs().sum(dim=-1).detach().cpu().numpy()
    return node_scores, edge_scores


def compute_section_occlusion_scores(
    model: ExplainableEdgeClassifierAdapter,
    data,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    target_class: int,
    true_class: int,
    pred_class: int,
) -> pd.DataFrame:
    with torch.no_grad():
        base_logits = model(data.x, data.edge_index, data)
    base_logit = float(base_logits[0, target_class].item())

    base_x = data.x.detach().clone()
    base_edge_attr = data.edge_attr.detach().clone()
    node_section_map = build_node_section_map(edges_df)
    node_id_to_idx = {
        int(node_id): int(idx)
        for idx, node_id in enumerate(nodes_df["id"].astype(int).tolist())
    }

    rows = []
    for section_id in sorted(set(edges_df["id_section"].astype(int).tolist())):
        node_indices = sorted(
            node_id_to_idx[node_id]
            for node_id, section_ids in node_section_map.items()
            if int(section_id) in section_ids and node_id in node_id_to_idx
        )
        edge_indices = edges_df.loc[edges_df["id_section"] == int(section_id), "edge_idx"].astype(int).tolist()

        perturbed_x = base_x.clone()
        perturbed_edge_attr = base_edge_attr.clone()
        if node_indices:
            perturbed_x[node_indices] = 0.0
        if edge_indices:
            perturbed_edge_attr[edge_indices] = 0.0

        with torch.no_grad():
            perturbed_logits = model(perturbed_x, data.edge_index, data, edge_attr=perturbed_edge_attr)
        delta_logit = base_logit - float(perturbed_logits[0, target_class].item())
        rows.append(
            {
                "id_section": int(section_id),
                "method": "section_occlusion",
                "section_score": max(0.0, delta_logit),
                "delta_logit": delta_logit,
                "num_nodes": int(len(node_indices)),
                "num_edges": int(len(edge_indices)),
                "is_true_section": int(section_id) == int(true_class),
                "is_pred_section": int(section_id) == int(pred_class),
            }
        )

    section_scores = pd.DataFrame(rows).sort_values(
        ["section_score", "delta_logit", "id_section"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    section_scores["rank"] = np.arange(1, len(section_scores) + 1, dtype=int)
    return section_scores


def compute_baseline_method_scores(
    model: ExplainableEdgeClassifierAdapter,
    data,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    gnn_section_scores: pd.DataFrame,
    true_class: int,
    pred_class: int,
    primary_method: str = PRIMARY_EXPLANATION_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_frames = []
    edge_frames = []
    section_frames = []

    activation_node_scores, activation_edge_scores = compute_activation_attention_scores(model, data)
    node_frames.append(
        build_generic_node_score_table(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_scores=activation_node_scores,
            method="activation_attention",
            true_class=true_class,
            pred_class=pred_class,
        )
    )
    edge_frames.append(
        build_generic_edge_score_table(
            edges_df=edges_df,
            edge_scores=activation_edge_scores,
            method="activation_attention",
            true_class=true_class,
            pred_class=pred_class,
        )
    )
    section_frames.append(
        aggregate_scores_by_section(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_scores=activation_node_scores,
            edge_scores=activation_edge_scores,
            method="activation_attention",
            true_class=true_class,
            pred_class=pred_class,
        )
    )

    grad_node_scores, grad_edge_scores = compute_grad_x_input_scores(model, data, target_class=pred_class)
    node_frames.append(
        build_generic_node_score_table(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_scores=grad_node_scores,
            method="grad_x_input",
            true_class=true_class,
            pred_class=pred_class,
        )
    )
    edge_frames.append(
        build_generic_edge_score_table(
            edges_df=edges_df,
            edge_scores=grad_edge_scores,
            method="grad_x_input",
            true_class=true_class,
            pred_class=pred_class,
        )
    )
    section_frames.append(
        aggregate_scores_by_section(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_scores=grad_node_scores,
            edge_scores=grad_edge_scores,
            method="grad_x_input",
            true_class=true_class,
            pred_class=pred_class,
        )
    )

    section_occlusion_scores = compute_section_occlusion_scores(
        model=model,
        data=data,
        nodes_df=nodes_df,
        edges_df=edges_df,
        target_class=pred_class,
        true_class=true_class,
        pred_class=pred_class,
    )
    section_frames.append(section_occlusion_scores)
    section_occlusion_node_scores = expand_section_scores_to_node_scores(
        nodes_df=nodes_df,
        edges_df=edges_df,
        section_scores_df=section_occlusion_scores,
    )
    section_occlusion_edge_scores = expand_section_scores_to_edge_scores(
        edges_df=edges_df,
        section_scores_df=section_occlusion_scores,
    )
    node_frames.append(
        build_generic_node_score_table(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_scores=section_occlusion_node_scores,
            method="section_occlusion",
            true_class=true_class,
            pred_class=pred_class,
        )
    )
    edge_frames.append(
        build_generic_edge_score_table(
            edges_df=edges_df,
            edge_scores=section_occlusion_edge_scores,
            method="section_occlusion",
            true_class=true_class,
            pred_class=pred_class,
        )
    )

    gnn_section_frame = gnn_section_scores.copy()
    gnn_section_frame["method"] = primary_method
    gnn_section_frame["section_score"] = gnn_section_frame["section_rank_score"].astype(float)
    gnn_section_frame["rank"] = np.arange(1, len(gnn_section_frame) + 1, dtype=int)
    section_frames.append(
        gnn_section_frame[
            [
                "method",
                "id_section",
                "section_score",
                "rank",
                "is_true_section",
                "is_pred_section",
                "node_importance_max",
                "edge_importance_max",
            ]
        ].copy()
    )

    baseline_node_scores = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    baseline_edge_scores = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    baseline_section_scores = pd.concat(section_frames, ignore_index=True) if section_frames else pd.DataFrame()
    baseline_method_summary = build_method_summary(
        baseline_section_scores,
        true_class=true_class,
        pred_class=pred_class,
    )
    return baseline_node_scores, baseline_edge_scores, baseline_section_scores, baseline_method_summary


def compute_edge_feature_ablation(
    model: nn.Module,
    data,
    target_class: int,
    edge_indices: list[int],
    feature_names: list[str],
) -> pd.DataFrame:
    edge_indices = [int(idx) for idx in edge_indices]
    if not edge_indices:
        return pd.DataFrame(
            columns=[
                "edge_idx",
                "feature_idx",
                "feature_name",
                "feature_value_norm",
                "delta_logit",
                "abs_delta_logit",
            ]
        )

    with torch.no_grad():
        base_logits = model(data.x, data.edge_index, data)
    base_logit = float(base_logits[0, target_class].item())
    base_edge_attr = data.edge_attr.detach().clone()

    rows = []
    for edge_idx in edge_indices:
        for feature_idx, feature_name in enumerate(feature_names):
            perturbed_edge_attr = base_edge_attr.clone()
            feature_value = float(perturbed_edge_attr[edge_idx, feature_idx].item())
            perturbed_edge_attr[edge_idx, feature_idx] = 0.0
            with torch.no_grad():
                perturbed_logits = model(data.x, data.edge_index, data, edge_attr=perturbed_edge_attr)
            delta_logit = base_logit - float(perturbed_logits[0, target_class].item())
            rows.append(
                {
                    "edge_idx": edge_idx,
                    "feature_idx": feature_idx,
                    "feature_name": feature_name,
                    "feature_value_norm": feature_value,
                    "delta_logit": delta_logit,
                    "abs_delta_logit": abs(delta_logit),
                }
            )

    return pd.DataFrame(rows)


def compute_node_feature_ablation(
    model: nn.Module,
    data,
    target_class: int,
    node_indices: list[int],
    feature_names: list[str],
    transform_node_inputs: Callable[[torch.Tensor], torch.Tensor] | None = None,
    inverse_node_inputs: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> pd.DataFrame:
    node_indices = [int(idx) for idx in node_indices]
    if not node_indices:
        return pd.DataFrame(
            columns=[
                "node_idx",
                "feature_idx",
                "feature_name",
                "feature_group",
                "feature_value_norm",
                "delta_logit",
                "abs_delta_logit",
            ]
        )

    with torch.no_grad():
        base_logits = model(data.x, data.edge_index, data)
    base_logit = float(base_logits[0, target_class].item())
    base_x = data.x.detach().clone()
    base_explain_x = transform_node_inputs(base_x) if transform_node_inputs is not None else None

    rows = []
    for node_idx in node_indices:
        for feature_idx, feature_name in enumerate(feature_names):
            if base_explain_x is not None and inverse_node_inputs is not None:
                perturbed_explain_x = base_explain_x.clone()
                feature_value = float(perturbed_explain_x[node_idx, feature_idx].item())
                perturbed_explain_x[node_idx, feature_idx] = 0.0
                perturbed_x = inverse_node_inputs(perturbed_explain_x)
            else:
                perturbed_x = base_x.clone()
                feature_value = float(perturbed_x[node_idx, feature_idx].item())
                perturbed_x[node_idx, feature_idx] = 0.0
            with torch.no_grad():
                perturbed_logits = model(perturbed_x, data.edge_index, data)
            delta_logit = base_logit - float(perturbed_logits[0, target_class].item())
            rows.append(
                {
                    "node_idx": node_idx,
                    "feature_idx": feature_idx,
                    "feature_name": feature_name,
                    "feature_group": classify_node_feature(feature_name),
                    "feature_value_norm": feature_value,
                    "delta_logit": delta_logit,
                    "abs_delta_logit": abs(delta_logit),
                }
            )

    return pd.DataFrame(rows)


def build_summary_tables(
    records_df: pd.DataFrame,
    no_defect_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_summary = (
        records_df.groupby(["direction", "analysis_category"])
        .size()
        .rename("count")
        .reset_index()
    )

    requested_summary = (
        records_df[records_df["requested_category"].notna()]
        .groupby(["direction", "requested_category"])
        .size()
        .rename("count")
        .reset_index()
    )

    per_true_section = (
        records_df[records_df["true_class"] != no_defect_class]
        .groupby(["direction", "true_class", "analysis_category"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["direction", "true_class", "count"], ascending=[True, True, False])
    )

    return analysis_summary, requested_summary, per_true_section


def select_representative_examples(records_df: pd.DataFrame, examples_per_category: int) -> pd.DataFrame:
    selected_frames = []
    for category in REQUESTED_CATEGORY_ORDER:
        subset = records_df[records_df["requested_category"] == category].copy()
        if subset.empty:
            continue
        if category == "farther":
            subset = subset.sort_values(
                ["graph_distance", "pred_margin", "pred_confidence"],
                ascending=[False, False, False],
            )
        elif category == "no_defect":
            subset = subset.sort_values(
                ["pred_margin", "pred_confidence"],
                ascending=[False, False],
            )
        else:
            subset = subset.sort_values(
                ["pred_margin", "pred_confidence", "true_confidence"],
                ascending=[False, False, False],
            )
        selected_frames.append(subset.head(examples_per_category))

    if not selected_frames:
        return pd.DataFrame(columns=records_df.columns)

    selected = pd.concat(selected_frames, ignore_index=True)
    selected["selection_rank"] = selected.groupby("requested_category").cumcount() + 1
    return selected


def compute_section_ranks(
    section_scores: pd.DataFrame,
    true_class: int,
    pred_class: int,
) -> tuple[int | None, int | None]:
    ordered_sections = section_scores["id_section"].astype(int).tolist()
    true_rank = ordered_sections.index(true_class) + 1 if true_class in ordered_sections else None
    pred_rank = ordered_sections.index(pred_class) + 1 if pred_class in ordered_sections else None
    return true_rank, pred_rank


def prepare_example_explanation(
    record: pd.Series,
    model: nn.Module,
    explainer_setup: ExplainerSetup,
    test_dataset,
    pair_scalers,
    cfg: dict,
    top_nodes: int,
    top_edges: int,
    top_features_per_node: int,
    top_features_per_edge: int,
    device: torch.device,
    primary_method: str = PRIMARY_EXPLANATION_METHOD,
) -> ExampleExplanation:
    direction_idx = DIRECTION_TO_INDEX[str(record.direction)]
    data = test_dataset[int(record.test_index)][direction_idx].to(device)

    explain_x = explainer_setup.transform_node_inputs(data.x)
    explanation = explainer_setup.explainer(explain_x, data.edge_index, index=None, data=data)
    edge_scores = explanation.edge_mask.detach().cpu().numpy() if explanation.edge_mask is not None else np.zeros(
        data.edge_index.shape[1],
        dtype=float,
    )
    node_mask_matrix = explanation.node_mask.detach().cpu().numpy() if explanation.node_mask is not None else np.zeros(
        (data.x.shape[0], data.x.shape[1]),
        dtype=float,
    )

    nodes_df, edges_df = build_denorm_tables(
        data,
        pair_scalers[direction_idx],
        cfg,
        direction=str(record.direction),
    )
    edges_df = edges_df.copy()
    edges_df["graph_label"] = int(record.true_class)
    edges_df["graph_label_pred"] = int(record.pred_class)

    node_feature_names = list(explainer_setup.node_feature_names)
    edge_feature_names = cfg["dataset"]["edge_attr"]
    node_table = build_node_table(
        nodes_df=nodes_df,
        edges_df=edges_df,
        node_mask_matrix=node_mask_matrix,
        node_feature_names=node_feature_names,
        true_class=int(record.true_class),
        pred_class=int(record.pred_class),
    )
    top_node_table = node_table.head(top_nodes).copy()

    edge_table = build_edge_table(
        edges_df=edges_df,
        edge_scores=edge_scores,
        true_class=int(record.true_class),
        pred_class=int(record.pred_class),
    )
    top_edge_table = edge_table.head(top_edges).copy()

    node_feature_df = compute_node_feature_ablation(
        model=model,
        data=data,
        target_class=int(record.pred_class),
        node_indices=top_node_table["node_idx"].tolist(),
        feature_names=node_feature_names,
        transform_node_inputs=explainer_setup.transform_node_inputs,
        inverse_node_inputs=explainer_setup.inverse_node_inputs,
    )
    if not node_feature_df.empty:
        node_feature_df = node_feature_df.merge(
            top_node_table[["node_idx", "id", "incident_sections", "top_feature_name", "top_feature_group"]],
            on="node_idx",
            how="left",
            suffixes=("", "_node"),
        )
        compact_node_feature_df = (
            node_feature_df.sort_values(["node_idx", "abs_delta_logit"], ascending=[True, False])
            .groupby("node_idx")
            .head(top_features_per_node)
            .reset_index(drop=True)
        )
    else:
        compact_node_feature_df = node_feature_df.copy()

    edge_feature_df = compute_edge_feature_ablation(
        model=model,
        data=data,
        target_class=int(record.pred_class),
        edge_indices=top_edge_table["edge_idx"].tolist(),
        feature_names=edge_feature_names,
    )
    if not edge_feature_df.empty:
        edge_feature_df = edge_feature_df.merge(
            top_edge_table[["edge_idx", "id_section"]],
            on="edge_idx",
            how="left",
        )
        compact_edge_feature_df = (
            edge_feature_df.sort_values(["edge_idx", "abs_delta_logit"], ascending=[True, False])
            .groupby("edge_idx")
            .head(top_features_per_edge)
            .reset_index(drop=True)
        )
    else:
        compact_edge_feature_df = edge_feature_df.copy()

    section_scores = build_section_scores(
        node_table=node_table,
        edge_table=edge_table,
        edges_df=edges_df,
        true_class=int(record.true_class),
        pred_class=int(record.pred_class),
    )
    baseline_node_scores, baseline_edge_scores, baseline_section_scores, baseline_method_summary = (
        compute_baseline_method_scores(
            model=model,
            data=data,
            nodes_df=nodes_df,
            edges_df=edges_df,
            gnn_section_scores=section_scores,
            true_class=int(record.true_class),
            pred_class=int(record.pred_class),
            primary_method=primary_method,
        )
    )

    true_rank, pred_rank = compute_section_ranks(
        section_scores=section_scores,
        true_class=int(record.true_class),
        pred_class=int(record.pred_class),
    )

    return ExampleExplanation(
        nodes_df=nodes_df,
        edges_df=edges_df,
        node_table=node_table,
        top_node_table=top_node_table,
        node_feature_df=node_feature_df,
        compact_node_feature_df=compact_node_feature_df,
        edge_table=edge_table,
        top_edge_table=top_edge_table,
        edge_feature_df=edge_feature_df,
        compact_edge_feature_df=compact_edge_feature_df,
        section_scores=section_scores,
        baseline_node_scores=baseline_node_scores,
        baseline_edge_scores=baseline_edge_scores,
        baseline_section_scores=baseline_section_scores,
        baseline_method_summary=baseline_method_summary,
        true_section_rank=true_rank,
        pred_section_rank=pred_rank,
    )
