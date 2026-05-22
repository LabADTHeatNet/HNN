from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import isomorphism as iso

from .article_style import PALETTE, apply_article_style, apply_standard_axis_style, save_figure
from .distance_confusion import _iter_sample_pairs, _read_label_pair, load_targets_predictions
from .network_examples import load_graph_tables


@dataclass(frozen=True)
class EdgePosition:
    saved_edge_idx: int
    offset_from_saved_start: float


@dataclass
class CompressedRawEdge:
    start_node: int
    end_node: int
    length: float
    diameter: float
    is_consumer: int
    raw_mid_offsets: dict[int, float]


@dataclass
class MetricDistanceContext:
    raw_edge_positions: dict[int, EdgePosition]
    section_to_rep_edge: dict[int, int]
    node_distance_lookup: dict[int, dict[int, float]]
    saved_edge_geometry: dict[int, tuple[int, int, float]]
    section_ids: list[int]


def _saved_to_raw_paths(
    saved_nodes_path: Path,
    saved_edges_path: Path,
    results_dir: Path,
    direction: str,
    full_graph_root: Path,
) -> tuple[Path, Path]:
    saved_nodes_rel = saved_nodes_path.relative_to(results_dir)
    saved_edges_rel = saved_edges_path.relative_to(results_dir)
    raw_nodes_path = full_graph_root / saved_nodes_rel
    raw_edges_path = full_graph_root / saved_edges_rel
    raw_nodes_path = raw_nodes_path.with_name(raw_nodes_path.name.replace(f"_{direction}", ""))
    raw_edges_path = raw_edges_path.with_name(raw_edges_path.name.replace(f"_{direction}", ""))
    return raw_nodes_path, raw_edges_path


def _full_direction_mask(edges_df: pd.DataFrame, direction: str) -> pd.Series:
    if direction == "fwd":
        return edges_df["Vid"].isin([0, 2])
    if direction == "bwd":
        return edges_df["Vid"].isin([1, 2])
    raise ValueError(f"Unsupported direction: {direction}")


def _node_type_saved(row: pd.Series) -> int:
    if float(row["types_def"]) > 0.5:
        return 0
    if float(row["types_usr"]) > 0.5:
        return 1
    return 2


def _edge_attr_key(length: float, diameter: float, is_consumer: int) -> tuple[float, float, int]:
    return (round(float(length), 3), round(float(diameter), 3), int(is_consumer))


def _build_target_attr_counter(saved_edges_df: pd.DataFrame) -> dict[tuple[float, float, int], int]:
    counter: dict[tuple[float, float, int], int] = {}
    for _, row in saved_edges_df.iterrows():
        key = _edge_attr_key(row["l"], row["d"], int(round(float(row["Vid_usr"]))))
        counter[key] = counter.get(key, 0) + 1
    return counter


def _counter_distance(
    left: dict[tuple[float, float, int], int],
    right: dict[tuple[float, float, int], int],
) -> int:
    keys = set(left) | set(right)
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys)


def _build_edge_attr_counter(edges: list[CompressedRawEdge]) -> dict[tuple[float, float, int], int]:
    counter: dict[tuple[float, float, int], int] = {}
    for edge in edges:
        key = _edge_attr_key(edge.length, edge.diameter, edge.is_consumer)
        counter[key] = counter.get(key, 0) + 1
    return counter


def _maybe_build_merged_edge(
    left_edge: CompressedRawEdge,
    right_edge: CompressedRawEdge,
    node_id: int,
) -> CompressedRawEdge | None:
    if left_edge.is_consumer or right_edge.is_consumer:
        return None
    if left_edge.end_node == node_id and right_edge.start_node == node_id:
        first_edge, second_edge = left_edge, right_edge
    elif right_edge.end_node == node_id and left_edge.start_node == node_id:
        first_edge, second_edge = right_edge, left_edge
    else:
        return None

    merged_offsets = dict(first_edge.raw_mid_offsets)
    offset_base = first_edge.length
    for raw_edge_id, raw_offset in second_edge.raw_mid_offsets.items():
        merged_offsets[raw_edge_id] = offset_base + raw_offset

    return CompressedRawEdge(
        start_node=first_edge.start_node,
        end_node=second_edge.end_node,
        length=first_edge.length + second_edge.length,
        diameter=max(first_edge.diameter, second_edge.diameter),
        is_consumer=0,
        raw_mid_offsets=merged_offsets,
    )


def _compress_raw_direction_edges(
    raw_nodes_df: pd.DataFrame,
    raw_edges_df: pd.DataFrame,
    direction: str,
    target_edge_count: int | None = None,
    target_attr_counter: dict[tuple[float, float, int], int] | None = None,
) -> tuple[pd.DataFrame, list[CompressedRawEdge]]:
    filtered_edges_df = raw_edges_df.loc[_full_direction_mask(raw_edges_df, direction)].copy().reset_index(drop=True)

    compressed_edges: list[CompressedRawEdge] = []
    for row in filtered_edges_df.itertuples(index=False):
        edge_id = int(row.id)
        edge_length = float(row.l)
        compressed_edges.append(
            CompressedRawEdge(
                start_node=int(row.id_in),
                end_node=int(row.id_out),
                length=edge_length,
                diameter=float(row.d),
                is_consumer=int(row.Vid == 2),
                raw_mid_offsets={edge_id: edge_length / 2.0},
            )
        )

    changed = True
    while changed and target_edge_count is not None and len(compressed_edges) > target_edge_count:
        changed = False
        incident: dict[int, list[int]] = {}
        for edge_idx, edge in enumerate(compressed_edges):
            incident.setdefault(edge.start_node, []).append(edge_idx)
            incident.setdefault(edge.end_node, []).append(edge_idx)

        current_counter = _build_edge_attr_counter(compressed_edges)
        best_choice: tuple[int, list[int], CompressedRawEdge] | None = None
        best_score: int | None = None

        for node_id, edge_indices in incident.items():
            if len(edge_indices) != 2:
                continue

            left_idx, right_idx = edge_indices
            left_edge = compressed_edges[left_idx]
            right_edge = compressed_edges[right_idx]
            merged_edge = _maybe_build_merged_edge(left_edge, right_edge, node_id)
            if merged_edge is None:
                continue

            if target_attr_counter is not None:
                candidate_counter = dict(current_counter)
                for edge in (left_edge, right_edge):
                    key = _edge_attr_key(edge.length, edge.diameter, edge.is_consumer)
                    candidate_counter[key] -= 1
                    if candidate_counter[key] == 0:
                        del candidate_counter[key]
                merged_key = _edge_attr_key(merged_edge.length, merged_edge.diameter, merged_edge.is_consumer)
                candidate_counter[merged_key] = candidate_counter.get(merged_key, 0) + 1
                score = _counter_distance(candidate_counter, target_attr_counter)
            else:
                score = 0

            if best_score is None or score < best_score:
                best_score = score
                best_choice = (node_id, edge_indices, merged_edge)

        if best_choice is not None:
            _, edge_indices, merged_edge = best_choice
            for edge_idx in sorted(edge_indices, reverse=True):
                compressed_edges.pop(edge_idx)
            compressed_edges.append(merged_edge)
            changed = True

    active_node_ids = sorted({edge.start_node for edge in compressed_edges} | {edge.end_node for edge in compressed_edges})
    compressed_nodes_df = raw_nodes_df.loc[raw_nodes_df["id"].isin(active_node_ids)].copy()
    return compressed_nodes_df, compressed_edges


def _build_reference_mapping(
    saved_nodes_df: pd.DataFrame,
    saved_edges_df: pd.DataFrame,
    raw_nodes_df: pd.DataFrame,
    raw_edges_df: pd.DataFrame,
    direction: str,
) -> dict[int, EdgePosition]:
    target_attr_counter = _build_target_attr_counter(saved_edges_df)
    compressed_nodes_df, compressed_raw_edges = _compress_raw_direction_edges(
        raw_nodes_df,
        raw_edges_df,
        direction,
        target_edge_count=len(saved_edges_df),
        target_attr_counter=target_attr_counter,
    )

    raw_graph = nx.Graph()
    for row in compressed_nodes_df.itertuples(index=False):
        raw_graph.add_node(int(row.id), node_type=int(row.types))
    for edge_idx, edge in enumerate(compressed_raw_edges):
        raw_graph.add_edge(
            edge.start_node,
            edge.end_node,
            edge_idx=edge_idx,
            length=edge.length,
            diameter=edge.diameter,
            is_consumer=edge.is_consumer,
        )

    saved_graph = nx.Graph()
    for row in saved_nodes_df.itertuples(index=False):
        saved_graph.add_node(int(row.id), node_type=_node_type_saved(pd.Series(row._asdict())))
    for saved_edge_idx, row in saved_edges_df.iterrows():
        saved_graph.add_edge(
            int(row["id_in"]),
            int(row["id_out"]),
            edge_idx=int(saved_edge_idx),
            length=float(row["l"]),
            diameter=float(row["d"]),
            is_consumer=int(round(float(row["Vid_usr"]))),
        )

    matcher = iso.GraphMatcher(
        raw_graph,
        saved_graph,
        node_match=lambda left, right: left["node_type"] == right["node_type"],
        edge_match=lambda left, right: (
            left["is_consumer"] == right["is_consumer"]
            and abs(left["length"] - right["length"]) < 0.02
            and abs(left["diameter"] - right["diameter"]) < 1e-4
        ),
    )
    if not matcher.is_isomorphic():
        raise RuntimeError(f"Could not match raw and saved {direction} graphs for metric distance scatter")

    node_mapping = matcher.mapping
    raw_edge_positions: dict[int, EdgePosition] = {}
    for compressed_edge in compressed_raw_edges:
        saved_u = node_mapping[compressed_edge.start_node]
        saved_v = node_mapping[compressed_edge.end_node]
        mask = (
            ((saved_edges_df["id_in"] == saved_u) & (saved_edges_df["id_out"] == saved_v))
            | ((saved_edges_df["id_in"] == saved_v) & (saved_edges_df["id_out"] == saved_u))
        )
        mask &= saved_edges_df["Vid_usr"].round().astype(int) == compressed_edge.is_consumer
        mask &= saved_edges_df["d"].sub(compressed_edge.diameter).abs() < 1e-4
        mask &= saved_edges_df["l"].sub(compressed_edge.length).abs() < 0.02
        candidates = saved_edges_df.index[mask].tolist()
        if len(candidates) != 1:
            raise RuntimeError(
                "Ambiguous compressed-raw -> saved edge match "
                f"for {direction} edge ({compressed_edge.start_node}, {compressed_edge.end_node}): {candidates}"
            )

        saved_edge_idx = int(candidates[0])
        saved_row = saved_edges_df.loc[saved_edge_idx]
        saved_length = float(saved_row["l"])
        saved_forward = int(saved_row["id_in"]) == saved_u and int(saved_row["id_out"]) == saved_v

        for raw_edge_id, raw_offset in compressed_edge.raw_mid_offsets.items():
            offset_from_saved_start = raw_offset if saved_forward else saved_length - raw_offset
            raw_edge_positions[int(raw_edge_id)] = EdgePosition(
                saved_edge_idx=saved_edge_idx,
                offset_from_saved_start=float(offset_from_saved_start),
            )
    return raw_edge_positions


def _build_saved_edge_geometry(edges_df: pd.DataFrame) -> dict[int, tuple[int, int, float]]:
    return {
        int(edge_idx): (int(row["id_in"]), int(row["id_out"]), float(row["l"]))
        for edge_idx, row in edges_df.iterrows()
    }


def _build_node_distance_lookup(edges_df: pd.DataFrame) -> dict[int, dict[int, float]]:
    graph = nx.Graph()
    for _, row in edges_df.iterrows():
        graph.add_edge(int(row["id_in"]), int(row["id_out"]), weight=float(row["l"]))
    return {
        int(node_id): {int(other_id): float(dist) for other_id, dist in dists.items()}
        for node_id, dists in nx.all_pairs_dijkstra_path_length(graph, weight="weight")
    }


def _distance_between_positions(
    left: EdgePosition,
    right: EdgePosition,
    node_distance_lookup: dict[int, dict[int, float]],
    saved_edge_geometry: dict[int, tuple[int, int, float]],
) -> float:
    left_u, left_v, left_length = saved_edge_geometry[left.saved_edge_idx]
    right_u, right_v, right_length = saved_edge_geometry[right.saved_edge_idx]
    candidates = []

    if left.saved_edge_idx == right.saved_edge_idx:
        candidates.append(abs(left.offset_from_saved_start - right.offset_from_saved_start))

    left_options = (
        (left_u, left.offset_from_saved_start),
        (left_v, left_length - left.offset_from_saved_start),
    )
    right_options = (
        (right_u, right.offset_from_saved_start),
        (right_v, right_length - right.offset_from_saved_start),
    )
    for left_node, left_distance in left_options:
        for right_node, right_distance in right_options:
            node_distance = node_distance_lookup[left_node][right_node]
            candidates.append(left_distance + node_distance + right_distance)
    return float(min(candidates))


def _select_section_rep_edges(
    edges_df: pd.DataFrame,
    node_distance_lookup: dict[int, dict[int, float]],
    saved_edge_geometry: dict[int, tuple[int, int, float]],
) -> dict[int, int]:
    edge_midpoints = {
        edge_idx: EdgePosition(saved_edge_idx=int(edge_idx), offset_from_saved_start=edge_length / 2.0)
        for edge_idx, (_, _, edge_length) in saved_edge_geometry.items()
    }
    section_to_rep_edge: dict[int, int] = {}
    for section_id, group in edges_df.groupby("id_section"):
        group_edge_indices = group.index.tolist()
        section_to_rep_edge[int(section_id)] = min(
            group_edge_indices,
            key=lambda edge_idx: (
                max(
                    _distance_between_positions(
                        edge_midpoints[edge_idx],
                        edge_midpoints[other_edge_idx],
                        node_distance_lookup,
                        saved_edge_geometry,
                    )
                    for other_edge_idx in group_edge_indices
                ),
                sum(
                    _distance_between_positions(
                        edge_midpoints[edge_idx],
                        edge_midpoints[other_edge_idx],
                        node_distance_lookup,
                        saved_edge_geometry,
                    )
                    for other_edge_idx in group_edge_indices
                ),
                edge_idx,
            ),
        )
    return section_to_rep_edge


def build_metric_distance_context(
    results_dir: Path,
    full_graph_root: Path,
    direction: str,
) -> MetricDistanceContext:
    for saved_nodes_path, saved_edges_path in _iter_sample_pairs(results_dir):
        raw_nodes_path, raw_edges_path = _saved_to_raw_paths(
            saved_nodes_path=saved_nodes_path,
            saved_edges_path=saved_edges_path,
            results_dir=results_dir,
            direction=direction,
            full_graph_root=full_graph_root,
        )
        if not raw_nodes_path.exists() or not raw_edges_path.exists():
            continue

        saved_nodes_df, saved_edges_df = load_graph_tables(saved_nodes_path, saved_edges_path, direction=direction)
        raw_nodes_df = pd.read_csv(raw_nodes_path, sep="\t")
        raw_edges_df = pd.read_csv(raw_edges_path, sep="\t")
        raw_edge_positions = _build_reference_mapping(
            saved_nodes_df=saved_nodes_df,
            saved_edges_df=saved_edges_df,
            raw_nodes_df=raw_nodes_df,
            raw_edges_df=raw_edges_df,
            direction=direction,
        )
        saved_edge_geometry = _build_saved_edge_geometry(saved_edges_df)
        node_distance_lookup = _build_node_distance_lookup(saved_edges_df)
        section_to_rep_edge = _select_section_rep_edges(
            saved_edges_df,
            node_distance_lookup=node_distance_lookup,
            saved_edge_geometry=saved_edge_geometry,
        )
        return MetricDistanceContext(
            raw_edge_positions=raw_edge_positions,
            section_to_rep_edge=section_to_rep_edge,
            node_distance_lookup=node_distance_lookup,
            saved_edge_geometry=saved_edge_geometry,
            section_ids=sorted(section_to_rep_edge),
        )
    raise RuntimeError(f"Could not find a reference sample to build metric distance context under {results_dir}")


def _parse_problem_tube_id(edges_path: Path, results_dir: Path) -> int | None:
    rel = edges_path.relative_to(results_dir)
    if len(rel.parts) < 4:
        return None
    if rel.parts[1] != "problem":
        return None
    tube_part = rel.parts[2]
    if not tube_part.startswith("tube_"):
        return None
    return int(tube_part.split("_", 1)[1])


def compute_metric_distance_points(
    results_dir: Path,
    full_graph_root: Path,
    direction: str,
    exclude_classes: list[int] | None = None,
    auto_exclude_max_label: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    targets, predictions = load_targets_predictions(results_dir)
    unique_classes = sorted(set(targets) | set(predictions))
    excluded = set(int(class_id) for class_id in (exclude_classes or []))
    if auto_exclude_max_label and unique_classes:
        excluded.add(int(max(unique_classes)))

    context = build_metric_distance_context(results_dir, full_graph_root, direction)
    rows: list[dict[str, object]] = []
    for _, edges_path in _iter_sample_pairs(results_dir):
        true_class, pred_class = _read_label_pair(edges_path)
        if true_class in excluded or pred_class in excluded:
            continue

        problem_tube_id = _parse_problem_tube_id(edges_path, results_dir)
        if problem_tube_id is None:
            continue

        defect_position = context.raw_edge_positions.get(problem_tube_id)
        pred_rep_edge_idx = context.section_to_rep_edge.get(pred_class)
        if defect_position is None or pred_rep_edge_idx is None:
            continue

        _, _, pred_edge_length = context.saved_edge_geometry[pred_rep_edge_idx]
        pred_position = EdgePosition(
            saved_edge_idx=pred_rep_edge_idx,
            offset_from_saved_start=pred_edge_length / 2.0,
        )
        distance_m = _distance_between_positions(
            pred_position,
            defect_position,
            node_distance_lookup=context.node_distance_lookup,
            saved_edge_geometry=context.saved_edge_geometry,
        )
        rows.append(
            {
                "sample_id": str(edges_path.relative_to(results_dir)),
                "true_class": int(true_class),
                "pred_class": int(pred_class),
                "defect_tube_id": int(problem_tube_id),
                "distance_m": float(distance_m),
                "is_correct": bool(true_class == pred_class),
            }
        )

    points_df = pd.DataFrame(rows)
    classes = [class_id for class_id in unique_classes if class_id not in excluded]
    return points_df, classes


def plot_metric_distance_scatter(
    points_df: pd.DataFrame,
    classes: list[int],
    output_path: Path,
    dpi: int = 220,
    style: dict | None = None,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 10.5)))

    if "figsize" in style:
        fig_size = tuple(style["figsize"])
    else:
        fig_width = max(8.0, 0.30 * len(classes) + 2.8)
        fig_size = (fig_width, 4.8)

    fig, ax = plt.subplots(figsize=fig_size)
    apply_standard_axis_style(ax)
    ax.grid(axis="x", visible=False)

    class_to_x = {class_id: idx for idx, class_id in enumerate(classes)}
    jitter_width = float(style.get("jitter_width", 0.28))
    scatter_size = float(style.get("point_size", 18.0))
    point_alpha = float(style.get("point_alpha", 0.55))
    correct_point_size = float(style.get("correct_point_size", scatter_size))
    incorrect_point_size = float(style.get("incorrect_point_size", scatter_size * 1.35))
    incorrect_point_alpha = float(style.get("incorrect_point_alpha", min(1.0, point_alpha + 0.25)))
    incorrect_marker = style.get("incorrect_marker", "x")
    incorrect_linewidth = float(style.get("incorrect_linewidth", 1.1))

    if not points_df.empty:
        working_df = points_df.loc[points_df["pred_class"].isin(classes)].copy()
        working_df["x_base"] = working_df["pred_class"].map(class_to_x).astype(float)
        x_positions = np.empty(len(working_df), dtype=float)
        for _, group in working_df.groupby("pred_class", sort=True):
            if len(group) == 1:
                jitter = np.array([0.0], dtype=float)
            else:
                jitter = np.linspace(-jitter_width, jitter_width, len(group), dtype=float)
            for row_idx, jitter_value in zip(group.index, jitter):
                x_positions[working_df.index.get_loc(row_idx)] = working_df.at[row_idx, "x_base"] + jitter_value
        working_df["x_pos"] = x_positions

        correct_mask = working_df["is_correct"].astype(bool)
        ax.scatter(
            working_df.loc[correct_mask, "x_pos"],
            working_df.loc[correct_mask, "distance_m"],
            s=correct_point_size,
            c=style.get("correct_color", "#2F855A"),
            alpha=point_alpha,
            edgecolors="none",
            label=style.get("correct_label", "Correct"),
            zorder=2,
        )
        ax.scatter(
            working_df.loc[~correct_mask, "x_pos"],
            working_df.loc[~correct_mask, "distance_m"],
            s=incorrect_point_size,
            c=style.get("incorrect_color", "#D1495B"),
            alpha=incorrect_point_alpha,
            marker=incorrect_marker,
            linewidths=incorrect_linewidth,
            label=style.get("incorrect_label", "Incorrect"),
            zorder=4,
        )

        mean_df = (
            working_df.groupby("pred_class", as_index=False)["distance_m"]
            .mean()
            .sort_values("pred_class")
        )
        mean_x = mean_df["pred_class"].map(class_to_x).to_numpy(dtype=float)
        mean_y = mean_df["distance_m"].to_numpy(dtype=float)
        if bool(style.get("draw_mean_line", True)):
            mean_bin_half_width = float(style.get("mean_bin_half_width", 0.5))
            ax.hlines(
                mean_y,
                mean_x - mean_bin_half_width,
                mean_x + mean_bin_half_width,
                colors=style.get("mean_color", PALETTE["axis"]),
                linewidth=float(style.get("mean_line_width", 1.8)),
                alpha=float(style.get("mean_line_alpha", 0.9)),
                label=style.get("mean_label", "Mean"),
                zorder=5,
            )

    ax.set_xlabel(style.get("xlabel", "Predicted section"))
    ax.set_ylabel(style.get("ylabel", "Distance to defect pipe midpoint, m"))
    title = style.get("title")
    if title:
        ax.set_title(title)

    x_ticks = np.arange(len(classes), dtype=float)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(class_id) for class_id in classes], fontsize=float(style.get("xtick_font_size", 8.0)))
    if style.get("xtick_rotation", 0):
        plt.setp(ax.get_xticklabels(), rotation=float(style["xtick_rotation"]), ha=style.get("xtick_ha", "center"))
    ax.tick_params(axis="y", labelsize=float(style.get("ytick_font_size", 9.0)))

    ax.set_xlim(-0.75, len(classes) - 0.25)
    if "min_y" in style or "max_y" in style:
        lower = float(style.get("min_y", 0.0))
        if "max_y" in style:
            upper = float(style["max_y"])
        else:
            max_val = float(points_df["distance_m"].max()) if not points_df.empty else 1.0
            upper = max_val * 1.05 if max_val > 0 else 1.0
        ax.set_ylim(lower, upper)

    ax.legend(
        loc=style.get("legend_loc", "upper right"),
        fontsize=float(style.get("legend_font_size", style.get("base_font_size", 10.5) - 0.5)),
        ncol=int(style.get("legend_ncol", 1)),
    )
    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
