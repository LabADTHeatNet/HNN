from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from .article_style import PALETTE, apply_article_style, save_figure


# Matches the section ids used during training preprocessing for split subnet graphs.
FWD_SECTION_IDS = np.array(
    [
        0,
        0,
        41,
        24,
        0,
        13,
        18,
        18,
        18,
        30,
        31,
        10,
        11,
        11,
        20,
        21,
        21,
        1,
        1,
        0,
        0,
        11,
        9,
        9,
        10,
        5,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        37,
        8,
        9,
        14,
        25,
        25,
        25,
        32,
        19,
        11,
        6,
        2,
        15,
        15,
        16,
        4,
        15,
        16,
        14,
        25,
        25,
        35,
        10,
        19,
        18,
        32,
        36,
        36,
        36,
        36,
        36,
        38,
        8,
        9,
        14,
        35,
        36,
        32,
        32,
        32,
        18,
        11,
        6,
        2,
        4,
        15,
        16,
        16,
        14,
        25,
        35,
        10,
        10,
        19,
        18,
        32,
        32,
        32,
        36,
        36,
        42,
        39,
        29,
        17,
        12,
        33,
        34,
        22,
        23,
        7,
        40,
        28,
        27,
        26,
    ],
    dtype=int,
)

BWD_SECTION_IDS = np.array(
    [
        0,
        41,
        24,
        0,
        13,
        18,
        18,
        18,
        30,
        31,
        10,
        11,
        11,
        20,
        21,
        21,
        1,
        1,
        0,
        0,
        11,
        9,
        9,
        10,
        5,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        37,
        8,
        9,
        14,
        25,
        25,
        25,
        32,
        19,
        11,
        6,
        2,
        15,
        15,
        16,
        4,
        15,
        16,
        14,
        25,
        25,
        35,
        10,
        19,
        18,
        32,
        36,
        36,
        36,
        36,
        36,
        38,
        8,
        9,
        14,
        35,
        36,
        32,
        32,
        32,
        18,
        11,
        6,
        2,
        4,
        15,
        16,
        16,
        14,
        25,
        35,
        10,
        10,
        19,
        18,
        32,
        32,
        32,
        36,
        36,
        0,
        42,
        39,
        29,
        17,
        12,
        33,
        34,
        22,
        23,
        7,
        40,
        28,
        27,
        26,
    ],
    dtype=int,
)

DEFAULT_NODE_STYLE_MAP = {
    "source": {"marker": "^", "color": PALETTE["source_node"], "size": 150, "label": "Source"},
    "sink": {"marker": "v", "color": PALETTE["sink_node"], "size": 150, "label": "Sink"},
    "consumer": {"marker": "s", "color": PALETTE["consumer_node"], "size": 115, "label": "Consumer"},
    "junction": {"marker": "D", "color": PALETTE["junction_node"], "size": 36, "label": "Junction"},
    "intermediate": {"marker": "o", "color": PALETTE["intermediate_node"], "size": 12, "label": "Intermediate"},
}

NODE_DRAW_ORDER = ["intermediate", "junction", "consumer", "sink", "source"]
NODE_ZORDER = {
    "intermediate": 3.0,
    "junction": 4.0,
    "consumer": 5.0,
    "sink": 6.0,
    "source": 7.0,
}
PIPE_ZORDER = {
    "bwd": 1.0,
    "fwd": 2.0,
}


def _merge_node_style(style: dict | None) -> dict[str, dict]:
    style = style or {}
    overrides = style.get("node_styles", {})
    merged: dict[str, dict] = {}
    for key, base in DEFAULT_NODE_STYLE_MAP.items():
        item = dict(base)
        item.update(overrides.get(key, {}))
        merged[key] = item
    return merged


def _positions_from_nodes(nodes_df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        int(row.id): (float(row.pos_x), float(row.pos_y))
        for row in nodes_df.itertuples(index=False)
    }


def _get_node_degrees(edges_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    node_deg_out: dict[int, int] = {}
    node_deg_in: dict[int, int] = {}
    for row in edges_df.itertuples(index=False):
        node_deg_out[int(row.id_in)] = node_deg_out.get(int(row.id_in), 0) + 1
        node_deg_in[int(row.id_out)] = node_deg_in.get(int(row.id_out), 0) + 1
    return node_deg_out, node_deg_in


def _compute_generic_sections(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    edges_df = edges_df.copy()
    edges_df["id_section"] = -1

    node_deg_out, node_deg_in = _get_node_degrees(edges_df)
    node_ids = set(int(node_id) for node_id in nodes_df["id"].tolist())
    start_vertices = sorted(node_ids.difference(node_deg_in.keys()))
    if not start_vertices:
        start_vertices = [min(node_ids)]

    start_edges = [edges_df.loc[edges_df["id_in"] == vertex_id] for vertex_id in start_vertices]
    if all(subset.empty for subset in start_edges):
        start_edges = [edges_df.loc[[edges_df.index[0]]]]

    num_sources = max(len(start_edges), 1)
    next_section_ids = [idx for idx in range(num_sources)]
    edge_queue: deque[tuple[int, int]] = deque()
    visited_ids: set[int] = set()

    def queue_edge(edge_idx: int, section_id: int) -> None:
        edge_queue.append((edge_idx, section_id))
        visited_ids.add(edge_idx)

    for source_idx, subset in enumerate(start_edges):
        for row in subset.itertuples():
            section_id = source_idx + num_sources if float(getattr(row, "Vid_usr", 0.0)) > 0.5 else source_idx
            queue_edge(int(row.Index), section_id)

    while edge_queue:
        edge_idx, section_id = edge_queue.popleft()
        row = edges_df.loc[edge_idx]
        id_out = int(row["id_out"])
        edges_df.at[edge_idx, "id_section"] = section_id
        next_edges = edges_df.loc[edges_df["id_in"] == id_out]
        deg_sum = node_deg_out.get(id_out, 0) + node_deg_in.get(id_out, 0)
        for next_row in next_edges.itertuples():
            if int(next_row.Index) in visited_ids:
                continue
            if float(getattr(next_row, "Vid_usr", 0.0)) > 0.5 or deg_sum > 2:
                base_idx = section_id % num_sources
                next_section_ids[base_idx] += num_sources
                next_section = next_section_ids[base_idx]
            else:
                next_section = section_id
            queue_edge(int(next_row.Index), next_section)

    current_max = int(edges_df["id_section"].max())
    for edge_idx in edges_df.index[edges_df["id_section"] == -1]:
        current_max += 1
        edges_df.at[edge_idx, "id_section"] = current_max

    unique_sections = sorted(int(section_id) for section_id in edges_df["id_section"].unique())
    mapping = {old_section: new_section for new_section, old_section in enumerate(unique_sections)}
    edges_df["id_section"] = edges_df["id_section"].map(mapping).astype(int)
    return edges_df


def _assign_sections(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, direction: str | None) -> pd.DataFrame:
    edges_df = edges_df.copy()
    if "id_section" in edges_df.columns:
        edges_df["id_section"] = edges_df["id_section"].astype(int)
        return edges_df

    if direction == "fwd" and len(edges_df) == len(FWD_SECTION_IDS):
        edges_df["id_section"] = FWD_SECTION_IDS
        return edges_df
    if direction == "bwd" and len(edges_df) == len(BWD_SECTION_IDS):
        edges_df["id_section"] = BWD_SECTION_IDS
        return edges_df
    return _compute_generic_sections(nodes_df, edges_df)


def load_graph_tables(nodes_path: Path, edges_path: Path, direction: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path).copy()
    edges_df = pd.read_csv(edges_path).copy()

    if "id" not in nodes_df.columns:
        nodes_df["id"] = np.arange(len(nodes_df))
    nodes_df["id"] = nodes_df["id"].astype(int)
    edges_df["id_in"] = edges_df["id_in"].astype(int)
    edges_df["id_out"] = edges_df["id_out"].astype(int)
    edges_df = _assign_sections(nodes_df, edges_df, direction)
    return nodes_df, edges_df


def _merge_node_tables(fwd_nodes_df: pd.DataFrame, bwd_nodes_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["id", "pos_x", "pos_y", "types_def", "types_usr", "types_src"]
    fwd_subset = fwd_nodes_df[merge_cols].copy()
    bwd_subset = bwd_nodes_df[merge_cols].copy()
    merged = fwd_subset.merge(
        bwd_subset,
        on="id",
        how="outer",
        suffixes=("_fwd", "_bwd"),
    )
    combined = pd.DataFrame()
    combined["id"] = merged["id"].astype(int)
    for col in ("pos_x", "pos_y"):
        combined[col] = merged[f"{col}_fwd"].fillna(merged[f"{col}_bwd"])
    for col in ("types_def", "types_usr", "types_src"):
        combined[col] = merged[f"{col}_fwd"].fillna(0.0).combine(
            merged[f"{col}_bwd"].fillna(0.0),
            max,
        )
    return combined


def categorize_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict[int, str]:
    node_ids = nodes_df["id"].astype(int).tolist()
    node_deg_out, node_deg_in = _get_node_degrees(edges_df)

    consumer_nodes: set[int] = set()
    for row in edges_df.itertuples(index=False):
        if float(getattr(row, "Vid_usr", 0.0)) > 0.5:
            consumer_nodes.add(int(row.id_in))
            consumer_nodes.add(int(row.id_out))

    # In the saved subnet csv tables the physical source/sink meaning is opposite
    # to the historical internal names of these one-hot columns.
    source_nodes = set(nodes_df.loc[nodes_df["types_usr"] > 0.5, "id"].astype(int))
    sink_nodes = set(nodes_df.loc[nodes_df["types_src"] > 0.5, "id"].astype(int))
    junction_nodes = {
        node_id for node_id in node_ids if node_deg_out.get(node_id, 0) + node_deg_in.get(node_id, 0) > 2
    }

    categories: dict[int, str] = {}
    for node_id in node_ids:
        if node_id in source_nodes:
            categories[node_id] = "source"
        elif node_id in sink_nodes:
            categories[node_id] = "sink"
        elif node_id in consumer_nodes:
            categories[node_id] = "consumer"
        elif node_id in junction_nodes:
            categories[node_id] = "junction"
        else:
            categories[node_id] = "intermediate"
    return categories


def compute_segment_centroids(
    edges_df: pd.DataFrame,
    positions: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    accum: dict[int, dict[str, float]] = {}
    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        pos_in = positions.get(src)
        pos_out = positions.get(dst)
        if pos_in is None or pos_out is None:
            continue
        seg_id = int(row.id_section)
        weight = max(float(getattr(row, "l", 1.0)), 1.0)
        mid_x = (pos_in[0] + pos_out[0]) / 2.0
        mid_y = (pos_in[1] + pos_out[1]) / 2.0
        entry = accum.setdefault(seg_id, {"x": 0.0, "y": 0.0, "w": 0.0})
        entry["x"] += mid_x * weight
        entry["y"] += mid_y * weight
        entry["w"] += weight
    return {
        seg_id: (info["x"] / info["w"], info["y"] / info["w"])
        for seg_id, info in accum.items()
        if info["w"] > 0.0
    }


def _compute_section_layout_info(
    edges_df: pd.DataFrame,
    positions: dict[int, tuple[float, float]],
) -> dict[int, dict[str, float | int | bool | tuple[float, float]]]:
    accum: dict[int, dict[str, float | int | bool | tuple[float, float]]] = {}
    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        pos_in = positions.get(src)
        pos_out = positions.get(dst)
        if pos_in is None or pos_out is None:
            continue

        seg_id = int(row.id_section)
        length = max(float(getattr(row, "l", 1.0)), 1.0)
        is_consumer = float(getattr(row, "Vid_usr", 0.0)) > 0.5
        mid_x = (pos_in[0] + pos_out[0]) / 2.0
        mid_y = (pos_in[1] + pos_out[1]) / 2.0
        vec_x = pos_out[0] - pos_in[0]
        vec_y = pos_out[1] - pos_in[1]

        entry = accum.setdefault(
            seg_id,
            {
                "x": 0.0,
                "y": 0.0,
                "w": 0.0,
                "edge_count": 0,
                "consumer_edge_count": 0,
                "total_length": 0.0,
                "non_consumer_length": 0.0,
                "anchor_x": mid_x,
                "anchor_y": mid_y,
                "anchor_vec_x": vec_x,
                "anchor_vec_y": vec_y,
                "anchor_len": -1.0,
                "anchor_is_consumer": True,
            },
        )
        entry["x"] += mid_x * length
        entry["y"] += mid_y * length
        entry["w"] += length
        entry["edge_count"] += 1
        entry["total_length"] += length
        if is_consumer:
            entry["consumer_edge_count"] += 1
        else:
            entry["non_consumer_length"] += length

        current_anchor_len = float(entry["anchor_len"])
        use_as_anchor = False
        if not is_consumer and bool(entry["anchor_is_consumer"]):
            use_as_anchor = True
        elif is_consumer == bool(entry["anchor_is_consumer"]) and length > current_anchor_len:
            use_as_anchor = True
        elif not is_consumer and not bool(entry["anchor_is_consumer"]) and length > current_anchor_len:
            use_as_anchor = True

        if use_as_anchor:
            entry["anchor_x"] = mid_x
            entry["anchor_y"] = mid_y
            entry["anchor_vec_x"] = vec_x
            entry["anchor_vec_y"] = vec_y
            entry["anchor_len"] = length
            entry["anchor_is_consumer"] = is_consumer

    result: dict[int, dict[str, float | int | bool | tuple[float, float]]] = {}
    for seg_id, entry in accum.items():
        weight = float(entry["w"])
        if weight <= 0.0:
            continue
        edge_count = int(entry["edge_count"])
        consumer_edge_count = int(entry["consumer_edge_count"])
        result[seg_id] = {
            "centroid": (float(entry["x"]) / weight, float(entry["y"]) / weight),
            "anchor": (float(entry["anchor_x"]), float(entry["anchor_y"])),
            "anchor_vector": (float(entry["anchor_vec_x"]), float(entry["anchor_vec_y"])),
            "edge_count": edge_count,
            "consumer_edge_count": consumer_edge_count,
            "total_length": float(entry["total_length"]),
            "non_consumer_length": float(entry["non_consumer_length"]),
            "is_consumer_only": consumer_edge_count == edge_count,
        }
    return result


def _normalize_display_vector(
    ax: plt.Axes,
    anchor: tuple[float, float],
    vector: tuple[float, float],
) -> np.ndarray:
    start = ax.transData.transform(anchor)
    end = ax.transData.transform((anchor[0] + vector[0], anchor[1] + vector[1]))
    display_vec = np.asarray(end - start, dtype=float)
    norm = float(np.linalg.norm(display_vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return display_vec / norm


def _pixels_to_points(fig: plt.Figure, value: float) -> float:
    return float(value) * 72.0 / float(fig.dpi)


def _candidate_label_offsets(
    ax: plt.Axes,
    anchor: tuple[float, float],
    anchor_vector: tuple[float, float],
    is_consumer_only: bool,
    style: dict,
) -> list[tuple[float, float]]:
    tangent = _normalize_display_vector(ax, anchor, anchor_vector)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    directions = [
        normal,
        -normal,
        normal + 0.55 * tangent,
        normal - 0.55 * tangent,
        -normal + 0.55 * tangent,
        -normal - 0.55 * tangent,
        tangent,
        -tangent,
    ]

    base_offset_px = float(style.get("section_label_base_offset_px", 18.0))
    consumer_scale = float(style.get("section_label_consumer_offset_scale", 1.35))
    scales = style.get("section_label_distance_scales", [1.0, 1.35, 1.75, 2.2, 2.8])
    if not isinstance(scales, list) or not scales:
        scales = [1.0, 1.35, 1.75, 2.2, 2.8]
    base_scale = consumer_scale if is_consumer_only else 1.0

    offsets: list[tuple[float, float]] = []
    for scale in scales:
        radius_px = base_offset_px * base_scale * float(scale)
        for direction in directions:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            vec_px = direction / norm * radius_px
            offsets.append((_pixels_to_points(ax.figure, vec_px[0]), _pixels_to_points(ax.figure, vec_px[1])))
    offsets.append((0.0, 0.0))
    return offsets


def _bbox_overlap_area(left: Bbox, right: Bbox) -> float:
    dx = min(left.x1, right.x1) - max(left.x0, right.x0)
    dy = min(left.y1, right.y1) - max(left.y0, right.y0)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(dx * dy)


def _bbox_outside_area(inner: Bbox, outer: Bbox) -> float:
    width = max(0.0, float(inner.x1 - inner.x0))
    height = max(0.0, float(inner.y1 - inner.y0))
    area = width * height
    inside_width = max(0.0, min(inner.x1, outer.x1) - max(inner.x0, outer.x0))
    inside_height = max(0.0, min(inner.y1, outer.y1) - max(inner.y0, outer.y0))
    return float(max(0.0, area - inside_width * inside_height))


def _annotation_alignment(dx_points: float, dy_points: float) -> tuple[str, str]:
    ha = "center"
    va = "center"
    if dx_points > 6.0:
        ha = "left"
    elif dx_points < -6.0:
        ha = "right"
    if dy_points > 6.0:
        va = "bottom"
    elif dy_points < -6.0:
        va = "top"
    return ha, va


def _place_section_annotation(
    ax: plt.Axes,
    seg_id: int,
    info: dict[str, float | int | bool | tuple[float, float]],
    color: str,
    font_size: float,
    occupied_bboxes: list[Bbox],
    style: dict,
    renderer,
) -> Bbox | None:
    anchor = info["anchor"]
    assert isinstance(anchor, tuple)
    anchor_vector = info["anchor_vector"]
    assert isinstance(anchor_vector, tuple)
    is_consumer_only = bool(info["is_consumer_only"])
    offsets = _candidate_label_offsets(ax, anchor, anchor_vector, is_consumer_only, style)
    axes_bbox = ax.get_window_extent(renderer=renderer)
    consumer_font_scale = float(style.get("section_label_consumer_font_scale", 0.88))
    current_font_size = font_size * (consumer_font_scale if is_consumer_only else 1.0)

    best: tuple[float, float, float, float] | None = None
    best_annotation = None
    expand_x = float(style.get("section_label_bbox_expand_x", 1.16))
    expand_y = float(style.get("section_label_bbox_expand_y", 1.22))

    for dx_points, dy_points in offsets:
        ha, va = _annotation_alignment(dx_points, dy_points)
        annotation = ax.annotate(
            str(int(seg_id)),
            xy=anchor,
            xytext=(dx_points, dy_points),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=current_font_size,
            color=color,
            zorder=6,
            annotation_clip=False,
            bbox={
                "boxstyle": f"round,pad={float(style.get('section_label_box_pad', 0.16))}",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": float(style.get("section_label_box_alpha", 0.84)),
            },
            arrowprops={
                "arrowstyle": "-",
                "color": style.get("section_label_arrow_color", color),
                "linewidth": float(style.get("section_label_arrow_line_width", 0.7)),
                "alpha": float(style.get("section_label_arrow_alpha", 0.65)),
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            } if abs(dx_points) + abs(dy_points) > 1.0 else None,
        )
        bbox = annotation.get_window_extent(renderer=renderer).expanded(expand_x, expand_y)
        overlap_penalty = sum(_bbox_overlap_area(bbox, other) for other in occupied_bboxes)
        outside_penalty = _bbox_outside_area(bbox, axes_bbox)
        distance_penalty = abs(dx_points) + abs(dy_points)
        consumer_penalty = 80.0 if is_consumer_only else 0.0
        score = overlap_penalty * 10.0 + outside_penalty * 3.0 + distance_penalty + consumer_penalty

        if best is None or score < best[0]:
            if best_annotation is not None:
                best_annotation.remove()
            best = (score, dx_points, dy_points, current_font_size)
            best_annotation = annotation
        else:
            annotation.remove()

    if best_annotation is None:
        return None

    best_annotation.set_path_effects(
        [path_effects.Stroke(linewidth=2.0, foreground="white"), path_effects.Normal()]
    )
    return best_annotation.get_window_extent(renderer=renderer).expanded(expand_x, expand_y)


def _combine_position_bounds(*position_maps: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    merged: dict[int, tuple[float, float]] = {}
    offset = 0
    for position_map in position_maps:
        for _, point in position_map.items():
            merged[offset] = point
            offset += 1
    return merged


def _compute_bounds_from_positions(positions: dict[int, tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = np.array([point[0] for point in positions.values()], dtype=float)
    ys = np.array([point[1] for point in positions.values()], dtype=float)
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _resolve_fixed_bounds(bounds: dict | None) -> tuple[float, float, float, float] | None:
    if not bounds:
        return None
    required_keys = ("min_x", "max_x", "min_y", "max_y")
    if not all(key in bounds for key in required_keys):
        raise ValueError(f"fixed_bounds must contain keys {required_keys}, got: {sorted(bounds.keys())}")
    return (
        float(bounds["min_x"]),
        float(bounds["max_x"]),
        float(bounds["min_y"]),
        float(bounds["max_y"]),
    )


def _make_axis(figsize: tuple[float, float], base_font_size: float) -> tuple[plt.Figure, plt.Axes]:
    apply_article_style(base_font_size=base_font_size)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _set_graph_limits(
    ax: plt.Axes,
    positions: dict[int, tuple[float, float]],
    pad_ratio: float = 0.06,
    min_pad_x: float = 30.0,
    min_pad_y: float = 25.0,
    invert_y: bool = False,
    fixed_bounds: dict | None = None,
) -> None:
    bounds = _resolve_fixed_bounds(fixed_bounds)
    if bounds is None:
        min_x, max_x, min_y, max_y = _compute_bounds_from_positions(positions)
    else:
        min_x, max_x, min_y, max_y = bounds
    pad_x = max((max_x - min_x) * pad_ratio, min_pad_x)
    pad_y = max((max_y - min_y) * pad_ratio, min_pad_y)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    if invert_y:
        ax.invert_yaxis()


def _draw_nodes(ax: plt.Axes, nodes_df: pd.DataFrame, categories: dict[int, str], node_style_map: dict[str, dict]) -> None:
    for category_name in NODE_DRAW_ORDER:
        style = node_style_map[category_name]
        subset = nodes_df.loc[nodes_df["id"].map(categories.get) == category_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["pos_x"],
            subset["pos_y"],
            s=style["size"],
            marker=style["marker"],
            c=style["color"],
            edgecolors="black",
            linewidths=0.6,
            zorder=NODE_ZORDER[category_name],
        )


def _draw_edge_layer(
    ax: plt.Axes,
    edges_df: pd.DataFrame,
    positions: dict[int, tuple[float, float]],
    pipe_color: str,
    consumer_color: str,
    pipe_line_width: float,
    consumer_line_width: float,
    mode_column: str,
    draw_consumers: bool,
    z_pipe: int,
    z_consumer: int,
) -> None:
    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        is_consumer = float(getattr(row, "Vid_usr", 0.0)) > 0.5
        is_mode = float(getattr(row, mode_column, 0.0)) > 0.5
        if is_consumer:
            if not draw_consumers:
                continue
            ax.plot(
                [positions[src][0], positions[dst][0]],
                [positions[src][1], positions[dst][1]],
                color=consumer_color,
                linewidth=consumer_line_width,
                solid_capstyle="round",
                zorder=z_consumer,
            )
        elif is_mode:
            ax.plot(
                [positions[src][0], positions[dst][0]],
                [positions[src][1], positions[dst][1]],
                color=pipe_color,
                linewidth=pipe_line_width,
                solid_capstyle="round",
                zorder=z_pipe,
            )


def _draw_section_labels(
    ax: plt.Axes,
    edges_df: pd.DataFrame,
    positions: dict[int, tuple[float, float]],
    color: str,
    font_size: float,
    style: dict | None = None,
) -> None:
    style = style or {}
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    section_info = _compute_section_layout_info(edges_df, positions)
    ordered_sections = sorted(
        section_info.items(),
        key=lambda item: (
            0 if bool(item[1]["is_consumer_only"]) else 1,
            float(item[1]["non_consumer_length"]),
            float(item[1]["total_length"]),
            int(item[1]["edge_count"]),
        ),
        reverse=True,
    )
    occupied_bboxes: list[Bbox] = []
    for seg_id, info in ordered_sections:
        bbox = _place_section_annotation(
            ax=ax,
            seg_id=seg_id,
            info=info,
            color=color,
            font_size=font_size,
            occupied_bboxes=occupied_bboxes,
            style=style,
            renderer=renderer,
        )
        if bbox is not None:
            occupied_bboxes.append(bbox)


def _subnet_legend_handles(direction: str, style: dict | None = None) -> list[Line2D]:
    style = style or {}
    node_style_map = _merge_node_style(style)
    pipe_label = style.get("legend_label_supply", "Supply pipes") if direction == "fwd" else style.get("legend_label_return", "Return pipes")
    pipe_color = style.get("pipe_color_fwd", PALETTE["supply"]) if direction == "fwd" else style.get("pipe_color_bwd", PALETTE["return"])
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="",
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markersize=max(5.0, np.sqrt(style["size"])),
            label=style["label"],
        )
        for style in node_style_map.values()
    ]
    handles.append(Line2D([0], [0], color=pipe_color, linewidth=3.0, label=pipe_label))
    return handles


def _combined_legend_handles(style: dict | None = None) -> list[Line2D]:
    style = style or {}
    node_style_map = _merge_node_style(style)
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="",
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markersize=max(5.0, np.sqrt(style["size"])),
            label=style["label"],
        )
        for style in node_style_map.values()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=style.get("pipe_color_fwd", PALETTE["supply"]),
                linewidth=3.0,
                label=style.get("legend_label_supply", "Supply pipes"),
            ),
            Line2D(
                [0],
                [0],
                color=style.get("pipe_color_bwd", PALETTE["return"]),
                linewidth=3.0,
                label=style.get("legend_label_return", "Return pipes"),
            ),
        ]
    )
    return handles


def _draw_network_legend(fig: plt.Figure, handles: list[Line2D], style: dict, default_ncol: int) -> None:
    legend_ncol = len(handles) if bool(style.get("legend_single_row", True)) else int(style.get("legend_ncol", default_ncol))
    fig.legend(
        handles=handles,
        loc=style.get("legend_loc", "lower left"),
        ncol=legend_ncol,
        bbox_to_anchor=tuple(style.get("legend_bbox_to_anchor", [0.018, 0.018])),
        bbox_transform=fig.transFigure,
        fontsize=float(style.get("legend_font_size", style.get("base_font_size", 10.5))),
        columnspacing=float(style.get("legend_columnspacing", 1.4)),
        handletextpad=float(style.get("legend_handletextpad", 0.6)),
    )


def _finalize_network_figure(
    fig: plt.Figure,
    legend_handles: list[Line2D],
    output_path: Path,
    dpi: int,
    style: dict,
    default_legend_ncol: int,
) -> None:
    margins = style.get("figure_margins", {})
    fig.subplots_adjust(
        left=float(margins.get("left", 0.015)),
        right=float(margins.get("right", 0.985)),
        top=float(margins.get("top", 0.985)),
        bottom=float(margins.get("bottom", 0.14)),
    )
    _draw_network_legend(fig, legend_handles, style, default_legend_ncol)
    save_figure(fig, output_path, dpi=dpi, tight=False)


def _normalize_section_label_mode(show_section_labels: bool | str, direction: str | None = None) -> tuple[bool, bool]:
    if isinstance(show_section_labels, bool):
        if direction is None:
            return show_section_labels, show_section_labels
        return (
            show_section_labels and direction == "fwd",
            show_section_labels and direction == "bwd",
        )

    mode = str(show_section_labels).strip().lower()
    if mode in {"", "false", "none", "off", "no"}:
        return False, False
    if mode in {"true", "both", "all", "on"}:
        if direction is None:
            return True, True
        return direction == "fwd", direction == "bwd"
    if mode == "fwd":
        if direction is None:
            return True, False
        return direction == "fwd", False
    if mode == "bwd":
        if direction is None:
            return False, True
        return False, direction == "bwd"
    raise ValueError(
        "show_section_labels must be one of false/true/'none'/'both'/'fwd'/'bwd', "
        f"got: {show_section_labels!r}"
    )


def render_subnet_figure(
    nodes_path: Path,
    edges_path: Path,
    direction: str,
    output_path: Path,
    dpi: int = 220,
    show_section_labels: bool | str = True,
    highlight_graph_label: bool = False,
    style: dict | None = None,
) -> None:
    style = style or {}
    nodes_df, edges_df = load_graph_tables(nodes_path, edges_path, direction=direction)
    categories = categorize_nodes(nodes_df, edges_df)
    node_style_map = _merge_node_style(style)
    positions = _positions_from_nodes(nodes_df)

    pipe_color = style.get("pipe_color_fwd", PALETTE["supply"]) if direction == "fwd" else style.get("pipe_color_bwd", PALETTE["return"])
    pipe_zorder = PIPE_ZORDER["fwd"] if direction == "fwd" else PIPE_ZORDER["bwd"]
    highlight_section = None
    if highlight_graph_label and "graph_label" in edges_df.columns:
        candidate = int(round(float(edges_df["graph_label"].iloc[0])))
        if candidate in set(edges_df["id_section"].astype(int).tolist()):
            highlight_section = candidate

    fig, ax = _make_axis(tuple(style.get("figsize", [10.5, 6.8])), float(style.get("base_font_size", 10.5)))
    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        section_id = int(row.id_section)
        color = style.get("highlight_color", PALETTE["highlight"]) if highlight_section == section_id else pipe_color
        linewidth = float(style.get("highlight_line_width", 3.6)) if highlight_section == section_id else float(style.get("pipe_line_width", 2.6))
        ax.plot(
            [positions[src][0], positions[dst][0]],
            [positions[src][1], positions[dst][1]],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=pipe_zorder,
        )

    _draw_nodes(ax, nodes_df, categories, node_style_map)
    show_fwd_labels, show_bwd_labels = _normalize_section_label_mode(show_section_labels, direction=direction)
    if show_fwd_labels or show_bwd_labels:
        label_color = style.get("highlight_color", PALETTE["highlight"]) if highlight_section is not None else style.get("section_label_color", PALETTE["axis"])
        _draw_section_labels(
            ax,
            edges_df,
            positions,
            label_color,
            font_size=float(style.get("section_label_font_size", 7.2)),
            style=style,
        )

    _set_graph_limits(
        ax,
        positions,
        pad_ratio=float(style.get("pad_ratio", 0.06)),
        min_pad_x=float(style.get("min_pad_x", 30.0)),
        min_pad_y=float(style.get("min_pad_y", 25.0)),
        invert_y=bool(style.get("invert_y", False)),
        fixed_bounds=style.get("fixed_bounds"),
    )
    legend_handles = _subnet_legend_handles(direction, style)
    _finalize_network_figure(
        fig=fig,
        legend_handles=legend_handles,
        output_path=output_path,
        dpi=dpi,
        style=style,
        default_legend_ncol=3,
    )


def render_combined_network_figure(
    fwd_nodes_path: Path,
    fwd_edges_path: Path,
    bwd_nodes_path: Path,
    bwd_edges_path: Path,
    output_path: Path,
    dpi: int = 220,
    show_section_labels: bool | str = False,
    style: dict | None = None,
) -> None:
    style = style or {}
    fwd_nodes_df, fwd_edges_df = load_graph_tables(fwd_nodes_path, fwd_edges_path, direction="fwd")
    bwd_nodes_df, bwd_edges_df = load_graph_tables(bwd_nodes_path, bwd_edges_path, direction="bwd")
    node_style_map = _merge_node_style(style)
    fwd_positions = _positions_from_nodes(fwd_nodes_df)
    bwd_positions = _positions_from_nodes(bwd_nodes_df)
    fwd_categories = categorize_nodes(fwd_nodes_df, fwd_edges_df)
    bwd_categories = categorize_nodes(bwd_nodes_df, bwd_edges_df)
    bounds_positions = _combine_position_bounds(fwd_positions, bwd_positions)

    fig, ax = _make_axis(tuple(style.get("figsize", [13.5, 8.8])), float(style.get("base_font_size", 10.5)))
    draw_consumer_edges_on = str(style.get("draw_consumer_edges_on", "both")).lower()
    draw_fwd_consumers = draw_consumer_edges_on in {"both", "fwd"}
    draw_bwd_consumers = draw_consumer_edges_on in {"both", "bwd"}

    _draw_edge_layer(
        ax=ax,
        edges_df=bwd_edges_df,
        positions=bwd_positions,
        pipe_color=style.get("pipe_color_bwd", PALETTE["return"]),
        consumer_color=style.get("pipe_color_consumer", PALETTE["consumer_edge"]),
        pipe_line_width=float(style.get("pipe_line_width", 2.6)),
        consumer_line_width=float(style.get("consumer_line_width", style.get("pipe_line_width", 2.6) + 0.2)),
        mode_column="Vid_bwd",
        draw_consumers=draw_bwd_consumers,
        z_pipe=PIPE_ZORDER["bwd"],
        z_consumer=PIPE_ZORDER["bwd"],
    )
    _draw_edge_layer(
        ax=ax,
        edges_df=fwd_edges_df,
        positions=fwd_positions,
        pipe_color=style.get("pipe_color_fwd", PALETTE["supply"]),
        consumer_color=style.get("pipe_color_consumer", PALETTE["consumer_edge"]),
        pipe_line_width=float(style.get("pipe_line_width", 2.6)),
        consumer_line_width=float(style.get("consumer_line_width", style.get("pipe_line_width", 2.6) + 0.2)),
        mode_column="Vid_fwd",
        draw_consumers=draw_fwd_consumers,
        z_pipe=PIPE_ZORDER["fwd"],
        z_consumer=PIPE_ZORDER["fwd"],
    )

    _draw_nodes(ax, bwd_nodes_df, bwd_categories, node_style_map)
    _draw_nodes(ax, fwd_nodes_df, fwd_categories, node_style_map)
    show_fwd_labels, show_bwd_labels = _normalize_section_label_mode(show_section_labels)
    if show_fwd_labels:
        _draw_section_labels(
            ax,
            fwd_edges_df,
            fwd_positions,
            style.get("pipe_color_fwd", PALETTE["supply"]),
            font_size=float(style.get("section_label_font_size", 7.2)),
            style=style,
        )
    if show_bwd_labels:
        _draw_section_labels(
            ax,
            bwd_edges_df,
            bwd_positions,
            style.get("pipe_color_bwd", PALETTE["return"]),
            font_size=float(style.get("section_label_font_size", 7.2)),
            style=style,
        )

    _set_graph_limits(
        ax,
        bounds_positions,
        pad_ratio=float(style.get("pad_ratio", 0.06)),
        min_pad_x=float(style.get("min_pad_x", 30.0)),
        min_pad_y=float(style.get("min_pad_y", 25.0)),
        invert_y=bool(style.get("invert_y", False)),
        fixed_bounds=style.get("fixed_bounds"),
    )
    legend_handles = _combined_legend_handles(style)
    _finalize_network_figure(
        fig=fig,
        legend_handles=legend_handles,
        output_path=output_path,
        dpi=dpi,
        style=style,
        default_legend_ncol=4,
    )
