from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from explain_analysis_utils import (
    ANALYSIS_CATEGORY_ORDER,
    EXPLANATION_METHOD_ORDER,
    PRIMARY_EXPLANATION_METHOD,
    REQUESTED_CATEGORY_ORDER,
)


CATEGORY_STYLES = {
    "consumer": {"color": (220, 220, 220), "size": 8, "shape": "square", "label": "Consumer"},
    "source": {"color": (220, 220, 220), "size": 8, "shape": "triangle_up", "label": "Source"},
    "sink": {"color": (220, 220, 220), "size": 8, "shape": "triangle_down", "label": "Sink"},
    "junction": {"color": (220, 220, 220), "size": 5, "shape": "diamond", "label": "Junction"},
    "intermediate": {"color": (220, 220, 220), "size": 2, "shape": "circle", "label": "Intermediate"},
}
COLOR_PIPE_FWD = (150, 150, 150)
COLOR_PIPE_BWD = (150, 150, 150)
COLOR_PIPE_OTHER = (160, 160, 160)
COLOR_DEFECT_MATCH = (44, 160, 44)
COLOR_DEFECT_TRUE = (220, 50, 47)
COLOR_DEFECT_PRED = (255, 140, 0)
EDGE_WIDTH_DEFAULT = 5
EDGE_WIDTH_HIGHLIGHT = 8
PANEL_BG = "white"
METHOD_DISPLAY_NAMES = {
    "gnn_explainer": "PyG GNNExplainer",
    "gnn_object_margin": "GNNExplainer | object/margin",
    "gnn_pair_ideal_delta_margin": "GNNExplainer | pair(ideal, delta)/margin",
    "gnn_attr_raw": "GNNExplainer | attributes/raw",
    "gnn_attr_vs_no_defect": "GNNExplainer | attributes/vs_no_defect",
    "gnn_object_log_probs": "GNNExplainer | object/log_probs",
    "gnn_object_vs_no_defect": "GNNExplainer | object/vs_no_defect",
    "gnn_common_attributes_log_probs": "GNNExplainer | common_attributes/log_probs",
    "notebook_object_log_probs": "Notebook GNNExplainer | object/log_probs",
    "notebook_common_attributes_log_probs": "Notebook GNNExplainer | common_attributes/log_probs",
    "activation_attention": "Activation + attention",
    "attention_explainer": "AttentionExplainer",
    "graphmask_object": "GraphMask | object",
    "grad_x_input": "Grad x Input",
    "pg_explainer": "PGExplainer",
    "section_occlusion": "Section occlusion",
}


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _boxes_overlap(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box1
    x2, y2, x3, y3 = box2
    return not (x1 < x2 or x3 < x0 or y1 < y2 or y3 < y0)


def _candidate_offsets(step: int = 12, rings: int = 6) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    for ring in range(1, rings + 1):
        radius = step * ring
        for dx in range(-radius, radius + 1, step):
            for dy in range(-radius, radius + 1, step):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                offsets.append((dx, dy))
    return offsets


def _find_label_position(
    center: tuple[int, int],
    size: tuple[int, int],
    existing_boxes: list[tuple[float, float, float, float]],
    bounds: tuple[int, int],
    margin: int = 3,
) -> tuple[int, int, tuple[float, float, float, float]]:
    cx, cy = center
    width, height = size
    canvas_w, canvas_h = bounds

    for dx, dy in _candidate_offsets():
        x0 = cx - width / 2 + dx
        y0 = cy - height / 2 + dy
        x1 = x0 + width
        y1 = y0 + height

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(canvas_w, x1)
        y1 = min(canvas_h, y1)

        if x1 - x0 < width:
            x0 = max(0, x1 - width)
        if y1 - y0 < height:
            y0 = max(0, y1 - height)
        x1 = x0 + width
        y1 = y0 + height

        candidate_box = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        if any(_boxes_overlap(candidate_box, box) for box in existing_boxes):
            continue
        return int(x0), int(y0), candidate_box

    x0 = max(0, cx - width / 2)
    y0 = max(0, cy - height / 2)
    x1 = min(canvas_w, x0 + width)
    y1 = min(canvas_h, y0 + height)
    candidate_box = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    return int(x0), int(y0), candidate_box


def _draw_node_symbol(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    style: dict,
    fill_color: tuple[int, int, int] | str | None = None,
    outline_color: tuple[int, int, int] | str = "black",
) -> None:
    x, y = center
    size = style["size"]
    color = fill_color if fill_color is not None else style["color"]
    outline = outline_color
    shape = style["shape"]

    if shape == "circle":
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=outline)
    elif shape == "square":
        draw.rectangle([x - size, y - size, x + size, y + size], fill=color, outline=outline)
    elif shape == "triangle_up":
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        draw.polygon(points, fill=color, outline=outline)
    elif shape == "triangle_down":
        points = [(x - size, y - size), (x + size, y - size), (x, y + size)]
        draw.polygon(points, fill=color, outline=outline)
    elif shape == "diamond":
        points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        draw.polygon(points, fill=color, outline=outline)
    else:
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=outline)


def _interpolate_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    factor: float,
) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    return tuple(int(round(s + (e - s) * factor)) for s, e in zip(start, end))


def _score_to_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    mid_color = (255, 215, 0)
    low_color = (220, 20, 60)
    high_color = (34, 139, 34)
    if score <= 0.5:
        factor = 0.0 if score == 0.0 else score / 0.5
        return _interpolate_color(low_color, mid_color, factor)
    return _interpolate_color(mid_color, high_color, (score - 0.5) / 0.5 if score < 1.0 else 1.0)


def _importance_to_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    rgba = matplotlib.colormaps["seismic"](score)
    return tuple(int(round(channel * 255)) for channel in rgba[:3])


def _importance_to_mpl_color(score: float) -> tuple[float, float, float]:
    score = max(0.0, min(1.0, float(score)))
    rgba = matplotlib.colormaps["seismic"](score)
    return tuple(float(channel) for channel in rgba[:3])


def _normalize_map(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}
    max_value = max(float(value) for value in score_map.values())
    if max_value <= 0.0:
        return {key: 0.0 for key in score_map}
    return {key: float(value) / max_value for key, value in score_map.items()}


def get_node_degrees(edges_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    node_deg_out: dict[int, int] = {}
    node_deg_in: dict[int, int] = {}
    for row in edges_df.itertuples(index=False):
        node_deg_out[int(row.id_in)] = node_deg_out.get(int(row.id_in), 0) + 1
        node_deg_in[int(row.id_out)] = node_deg_in.get(int(row.id_out), 0) + 1
    return node_deg_out, node_deg_in


def categorize_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict[int, str]:
    node_ids = nodes_df["id"].astype(int).tolist()
    node_deg_out, node_deg_in = get_node_degrees(edges_df)

    consumer_nodes: set[int] = set()
    for row in edges_df.itertuples(index=False):
        if float(getattr(row, "Vid_usr", 0.0)) > 0.5:
            consumer_nodes.add(int(row.id_in))
            consumer_nodes.add(int(row.id_out))

    source_nodes = set(nodes_df.loc[nodes_df["types_src"] > 0.5, "id"].astype(int))
    sink_nodes = set(nodes_df.loc[nodes_df["types_usr"] > 0.5, "id"].astype(int))
    junction_nodes = {
        node_id for node_id in node_ids
        if node_deg_out.get(node_id, 0) + node_deg_in.get(node_id, 0) > 2
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
    fallback_points: dict[int, list[tuple[float, float]]] = {}

    for row in edges_df.itertuples():
        seg = int(row.id_section)
        src = int(row.id_in)
        dst = int(row.id_out)
        pos_in = positions.get(src)
        pos_out = positions.get(dst)
        if pos_in is None or pos_out is None:
            continue
        weight = float(getattr(row, "l", 1.0))
        if weight <= 0:
            weight = 1.0
        mid_x = (pos_in[0] + pos_out[0]) / 2
        mid_y = (pos_in[1] + pos_out[1]) / 2
        entry = accum.setdefault(seg, {"w": 0.0, "x": 0.0, "y": 0.0})
        entry["w"] += weight
        entry["x"] += mid_x * weight
        entry["y"] += mid_y * weight
        fallback_points.setdefault(seg, []).append((mid_x, mid_y))

    centroids: dict[int, tuple[float, float]] = {}
    for seg, info in accum.items():
        if info["w"] > 0:
            centroids[seg] = (info["x"] / info["w"], info["y"] / info["w"])
        elif fallback_points.get(seg):
            pts = fallback_points[seg]
            centroids[seg] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    return centroids


def _prepare_graph_projection(
    nodes_df: pd.DataFrame,
    width: int = 1400,
    height: int = 900,
    margin_left: int = 70,
    margin_right: int = 70,
    margin_top: int = 70,
    margin_bottom: int = 70,
):
    positions = {
        int(row.id): (float(row.pos_x), float(row.pos_y))
        for row in nodes_df.itertuples(index=False)
    }
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)
    scale_x = (width - margin_left - margin_right) / range_x
    scale_y = (height - margin_top - margin_bottom) / range_y

    def project(pt_x: float, pt_y: float) -> tuple[int, int]:
        px = margin_left + (pt_x - min_x) * scale_x
        py = height - margin_bottom - (pt_y - min_y) * scale_y
        return int(px), int(py)

    return positions, project, width, height


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_color: tuple[int, int, int] | str,
    fill_color: tuple[int, int, int] | str = "white",
    outline_color: tuple[int, int, int] | str = "black",
) -> tuple[float, float, float, float]:
    x0, y0 = position
    tw, th = _measure_text(draw, text, font)
    padding_x = 4
    padding_y = 2
    box = (x0 - padding_x, y0 - padding_y, x0 + tw + padding_x, y0 + th + padding_y)
    draw.rectangle(box, fill=fill_color, outline=outline_color)
    draw.text((x0, y0), text, fill=text_color, font=font)
    return box


def _base_edge_color(row) -> tuple[int, int, int]:
    vid_fwd = float(getattr(row, "Vid_fwd", 0.0))
    vid_bwd = float(getattr(row, "Vid_bwd", 0.0))
    if vid_fwd > 0.5:
        return COLOR_PIPE_FWD
    if vid_bwd > 0.5:
        return COLOR_PIPE_BWD
    return COLOR_PIPE_OTHER


def _segment_role_color(
    seg_id: int,
    true_label: int,
    pred_label: int,
    has_true_segment: bool,
    has_pred_segment: bool,
) -> tuple[int, int, int] | None:
    if has_true_segment and has_pred_segment and true_label == pred_label and seg_id == pred_label:
        return COLOR_DEFECT_MATCH
    if has_true_segment and seg_id == true_label:
        return COLOR_DEFECT_TRUE
    if has_pred_segment and seg_id == pred_label:
        return COLOR_DEFECT_PRED
    return None


def _render_prediction_graph(
    record: pd.Series,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    output_path: Path,
) -> None:
    positions, project, width, height = _prepare_graph_projection(nodes_df)
    node_categories = categorize_nodes(nodes_df, edges_df)
    centroids = compute_segment_centroids(edges_df, positions)
    font = ImageFont.load_default()
    draw_width, draw_height = width, height
    img = Image.new("RGB", (draw_width, draw_height), PANEL_BG)
    draw = ImageDraw.Draw(img)

    title = (
        f"{record['requested_category'] or record['analysis_category']} | {record['direction']} | "
        f"true={record['true_class']} pred={record['pred_class']} | "
        f"dist={record['graph_distance'] if pd.notna(record['graph_distance']) else 'n/a'}"
    )
    draw.text((20, 20), title, fill="black", font=font)

    true_label = int(record["true_class"])
    pred_label = int(record["pred_class"])
    has_true_segment = true_label in set(edges_df["id_section"].astype(int).tolist())
    has_pred_segment = pred_label in set(edges_df["id_section"].astype(int).tolist())
    highlight_same = has_true_segment and has_pred_segment and true_label == pred_label

    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        p1 = project(*positions[src])
        p2 = project(*positions[dst])
        seg = int(row.id_section)
        color = _base_edge_color(row)
        width_line = EDGE_WIDTH_DEFAULT

        if highlight_same and seg == true_label:
            color = COLOR_DEFECT_MATCH
            width_line = EDGE_WIDTH_HIGHLIGHT
        else:
            if has_true_segment and seg == true_label:
                color = COLOR_DEFECT_TRUE
                width_line = EDGE_WIDTH_HIGHLIGHT
            if has_pred_segment and seg == pred_label:
                color = COLOR_DEFECT_PRED
                width_line = EDGE_WIDTH_HIGHLIGHT

        draw.line([p1, p2], fill=color, width=width_line)

    for node_id, pos in positions.items():
        px, py = project(*pos)
        style = CATEGORY_STYLES.get(node_categories.get(node_id, "intermediate"), CATEGORY_STYLES["intermediate"])
        _draw_node_symbol(draw, (px, py), style)

    label_boxes: list[tuple[float, float, float, float]] = []
    for seg_id, coord in centroids.items():
        px, py = project(*coord)
        label_text = str(seg_id)
        if highlight_same and seg_id == true_label:
            label_color = COLOR_DEFECT_MATCH
        elif has_true_segment and seg_id == true_label:
            label_color = COLOR_DEFECT_TRUE
        elif has_pred_segment and seg_id == pred_label:
            label_color = COLOR_DEFECT_PRED
        else:
            label_color = (0, 0, 0)
        tw, th = _measure_text(draw, label_text, font)
        x0, y0, box = _find_label_position((px, py), (tw + 8, th + 4), label_boxes, (draw_width, draw_height))
        box = _draw_text_box(draw, (x0 + 4, y0 + 2), label_text, font, label_color, "white", label_color)
        label_boxes.append(box)

    legend_order = ["source", "sink", "consumer", "junction", "intermediate"]
    legend_start_x = 40
    legend_start_y = draw_height - 120
    column_width = 150
    for idx, key in enumerate(legend_order):
        style = CATEGORY_STYLES[key]
        col = idx % 3
        row = idx // 3
        cx = legend_start_x + col * column_width
        cy = legend_start_y + row * 45
        _draw_node_symbol(draw, (cx, cy), style)
        draw.text((cx + style["size"] + 8, cy - 7), style["label"], fill="black", font=font)

    edge_legend_items: list[tuple[str, tuple[int, int, int], int]] = [
        ("Forward pipes", COLOR_PIPE_FWD, EDGE_WIDTH_DEFAULT),
        ("Backward pipes", COLOR_PIPE_BWD, EDGE_WIDTH_DEFAULT),
        ("True defect section", COLOR_DEFECT_TRUE, EDGE_WIDTH_HIGHLIGHT),
        ("Predicted defect section", COLOR_DEFECT_PRED, EDGE_WIDTH_HIGHLIGHT),
    ]
    if highlight_same:
        edge_legend_items[2] = ("Matched defect section", COLOR_DEFECT_MATCH, EDGE_WIDTH_HIGHLIGHT)
        edge_legend_items = edge_legend_items[:3]
    if not has_true_segment and not has_pred_segment:
        edge_legend_items = edge_legend_items[:2]

    base_y = draw_height - 175
    base_x = 40
    for label_text, color, line_width in edge_legend_items:
        draw.line([base_x, base_y, base_x + 50, base_y], fill=color, width=line_width)
        draw.text((base_x + 60, base_y - 7), label_text, fill="black", font=font)
        base_y -= 28

    img.save(output_path)


def _render_prediction_explain_graph(
    record: pd.Series,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    edge_table: pd.DataFrame,
    node_table: pd.DataFrame,
    top_node_table: pd.DataFrame,
    output_path: Path,
    title_prefix: str = "Prediction + node explain",
) -> None:
    positions, project, width, height = _prepare_graph_projection(
        nodes_df,
        width=1550,
        height=980,
        margin_left=80,
        margin_right=180,
        margin_top=90,
        margin_bottom=230,
    )
    node_categories = categorize_nodes(nodes_df, edges_df)
    centroids = compute_segment_centroids(edges_df, positions)
    title_font = _load_font(22, bold=True)
    label_font = _load_font(16, bold=True)
    legend_font = _load_font(17)
    rank_font = _load_font(15, bold=True)
    draw_width, draw_height = width, height
    img = Image.new("RGB", (draw_width, draw_height), PANEL_BG)
    draw = ImageDraw.Draw(img)

    title = (
        f"{title_prefix} | {record['requested_category'] or record['analysis_category']} | "
        f"{record['direction']} | true={record['true_class']} pred={record['pred_class']} | "
        f"dist={record['graph_distance'] if pd.notna(record['graph_distance']) else 'n/a'}"
    )
    draw.text((24, 24), title, fill="black", font=title_font)

    true_label = int(record["true_class"])
    pred_label = int(record["pred_class"])
    section_ids = set(edges_df["id_section"].astype(int).tolist())
    has_true_segment = true_label in section_ids
    has_pred_segment = pred_label in section_ids
    highlight_same = has_true_segment and has_pred_segment and true_label == pred_label

    node_score_map = {
        int(row.id): float(row.importance_max)
        for row in node_table.itertuples(index=False)
    }
    edge_score_map = {
        int(row.edge_idx): float(row.importance)
        for row in edge_table.itertuples(index=False)
    }
    normalized_node_scores = _normalize_map(node_score_map)
    normalized_edge_scores = _normalize_map(edge_score_map)

    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        p1 = project(*positions[src])
        p2 = project(*positions[dst])
        seg = int(row.id_section)
        edge_idx = int(row.edge_idx)
        score = normalized_edge_scores.get(edge_idx, 0.0)
        highlight_color = _segment_role_color(seg, true_label, pred_label, has_true_segment, has_pred_segment)
        if highlight_color is not None:
            highlight_width = EDGE_WIDTH_HIGHLIGHT + 4 if highlight_same else EDGE_WIDTH_HIGHLIGHT + 2
            draw.line([p1, p2], fill=highlight_color, width=highlight_width)
        importance_color = _importance_to_color(score)
        width_line = max(2, int(round(2 + 7 * score)))
        draw.line([p1, p2], fill=importance_color, width=width_line)

    for node_id, pos in positions.items():
        px, py = project(*pos)
        style = CATEGORY_STYLES.get(node_categories.get(node_id, "intermediate"), CATEGORY_STYLES["intermediate"])
        node_color = _importance_to_color(normalized_node_scores.get(node_id, 0.0))
        _draw_node_symbol(draw, (px, py), style, fill_color=node_color, outline_color="black")

    label_boxes: list[tuple[float, float, float, float]] = []
    for seg_id, coord in centroids.items():
        px, py = project(*coord)
        label_text = str(seg_id)
        label_color = _segment_role_color(seg_id, true_label, pred_label, has_true_segment, has_pred_segment) or (0, 0, 0)
        tw, th = _measure_text(draw, label_text, label_font)
        x0, y0, box = _find_label_position((px, py), (tw + 8, th + 4), label_boxes, (draw_width, draw_height))
        box = _draw_text_box(draw, (x0 + 4, y0 + 2), label_text, label_font, label_color, "white", label_color)
        label_boxes.append(box)

    for rank, row in enumerate(top_node_table.itertuples(index=False), start=1):
        node_id = int(row.id)
        if node_id not in positions:
            continue
        x_proj, y_proj = project(*positions[node_id])
        rank_text = f"n{rank}"
        tw, th = _measure_text(draw, rank_text, rank_font)
        x0, y0, box = _find_label_position(
            (x_proj, y_proj - 18),
            (tw + 8, th + 4),
            label_boxes,
            (draw_width, draw_height),
        )
        box = _draw_text_box(
            draw,
            (x0 + 4, y0 + 2),
            rank_text,
            rank_font,
            text_color="black",
            fill_color="white",
            outline_color=_importance_to_color(0.92),
        )
        label_boxes.append(box)

    legend_order = ["source", "sink", "consumer", "junction", "intermediate"]
    legend_start_x = 60
    legend_start_y = draw_height - 150
    column_width = 220
    for idx, key in enumerate(legend_order):
        style = CATEGORY_STYLES[key]
        col = idx % 3
        row = idx // 3
        cx = legend_start_x + col * column_width
        cy = legend_start_y + row * 45
        _draw_node_symbol(draw, (cx, cy), style, fill_color=(215, 215, 215), outline_color="black")
        draw.text((cx + style["size"] + 10, cy - 10), style["label"], fill="black", font=legend_font)

    edge_legend_items: list[tuple[str, tuple[int, int, int], int]] = [
        ("Correct prediction", COLOR_DEFECT_MATCH, EDGE_WIDTH_HIGHLIGHT),
        ("Predicted defect (wrong)", COLOR_DEFECT_PRED, EDGE_WIDTH_HIGHLIGHT),
        ("Ground truth defect", COLOR_DEFECT_TRUE, EDGE_WIDTH_HIGHLIGHT),
        ("Low importance", _importance_to_color(0.05), 5),
        ("High importance", _importance_to_color(0.95), 8),
    ]
    if highlight_same:
        edge_legend_items = [edge_legend_items[0], edge_legend_items[3], edge_legend_items[4]]
    elif not has_true_segment and not has_pred_segment:
        edge_legend_items = edge_legend_items[3:]

    base_y = draw_height - 210
    base_x = 760
    for label_text, color, line_width in edge_legend_items:
        draw.line([base_x, base_y, base_x + 50, base_y], fill=color, width=line_width)
        draw.text((base_x + 68, base_y - 11), label_text, fill="black", font=legend_font)
        base_y += 32

    bar_height = 200
    bar_width = 24
    bar_x0 = draw_width - 110
    bar_y0 = 120
    for idx in range(bar_height):
        value = 1.0 - idx / max(1, bar_height - 1)
        color = _importance_to_color(value)
        draw.line([bar_x0, bar_y0 + idx, bar_x0 + bar_width, bar_y0 + idx], fill=color)
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_width, bar_y0 + bar_height], outline="black")
    draw.text((bar_x0 - 10, bar_y0 - 30), "Importance", fill="black", font=legend_font)
    draw.text((bar_x0 + bar_width + 10, bar_y0 - 6), "high", fill="black", font=legend_font)
    draw.text((bar_x0 + bar_width + 10, bar_y0 + bar_height - 6), "low", fill="black", font=legend_font)
    draw.text((bar_x0 - 32, bar_y0 + bar_height + 18), "n1..nk = top nodes", fill="black", font=legend_font)

    img.save(output_path)


def plot_edge_importance_bar(
    record: pd.Series,
    top_edge_table: pd.DataFrame,
    output_path: Path,
    score_column: str = "importance",
    title: str = "Top explained pipes",
    xlabel: str = "Edge importance",
) -> None:
    if top_edge_table.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.text(0.5, 0.5, "No edge importance data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    shown_edges = top_edge_table.copy()
    shown_edges = shown_edges.sort_values([score_column, "edge_idx"], ascending=[True, True])
    max_importance = float(shown_edges[score_column].max()) if not shown_edges.empty else 1.0
    if max_importance <= 0.0:
        max_importance = 1.0

    labels = [
        f"e{int(row.edge_idx)} | sec {int(row.id_section)} | {int(row.id_in)}->{int(row.id_out)}"
        for row in shown_edges.itertuples(index=False)
    ]
    colors = [
        _importance_to_mpl_color(float(getattr(row, score_column)) / max_importance)
        for row in shown_edges.itertuples(index=False)
    ]
    edge_colors = []
    true_label = int(record["true_class"])
    pred_label = int(record["pred_class"])
    for row in shown_edges.itertuples(index=False):
        seg_id = int(row.id_section)
        if seg_id == true_label == pred_label:
            edge_colors.append("#2ca02c")
        elif seg_id == pred_label:
            edge_colors.append("#ff8c00")
        elif seg_id == true_label:
            edge_colors.append("#dc322f")
        else:
            edge_colors.append("#555555")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.barh(labels, shown_edges[score_column], color=colors, edgecolor=edge_colors, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=10)

    correct_patch = Patch(facecolor="white", edgecolor="#2ca02c", linewidth=2)
    pred_patch = Patch(facecolor="white", edgecolor="#ff8c00", linewidth=2)
    true_patch = Patch(facecolor="white", edgecolor="#dc322f", linewidth=2)
    ax.legend(
        [correct_patch, pred_patch, true_patch],
        ["Predicted correctly", "Predicted segment", "Ground truth segment"],
        fontsize=9,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_node_importance_bar(
    record: pd.Series,
    top_node_table: pd.DataFrame,
    output_path: Path,
    score_column: str = "importance_max",
    title: str = "Top explained nodes",
    xlabel: str = "Node importance (max feature mask)",
    label_suffix_column: str = "top_feature_name",
    label_suffix_fallback: str = "score",
) -> None:
    if top_node_table.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No node importance data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    shown_nodes = top_node_table.copy()
    secondary_sort = "importance_sum" if "importance_sum" in shown_nodes.columns else score_column
    shown_nodes = shown_nodes.sort_values([score_column, secondary_sort], ascending=[True, True])
    labels = [
        f"n{int(row.id)} | sec {row.incident_sections or '-'} | "
        f"{getattr(row, label_suffix_column, label_suffix_fallback) or label_suffix_fallback}"
        for row in shown_nodes.itertuples(index=False)
    ]
    max_importance = float(shown_nodes[score_column].max()) if not shown_nodes.empty else 1.0
    if max_importance <= 0.0:
        max_importance = 1.0
    colors = [
        _importance_to_mpl_color(float(getattr(row, score_column)) / max_importance)
        for row in shown_nodes.itertuples(index=False)
    ]
    edge_colors = []
    for row in shown_nodes.itertuples(index=False):
        if bool(row.touches_true_section) and bool(row.touches_pred_section):
            edge_colors.append("#2ca02c")
        elif bool(row.touches_true_section):
            edge_colors.append("#dc322f")
        elif bool(row.touches_pred_section):
            edge_colors.append("#ff8c00")
        else:
            edge_colors.append("#555555")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.barh(labels, shown_nodes[score_column], color=colors, edgecolor=edge_colors, linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=10)

    correct_patch = Patch(facecolor="white", edgecolor="#2ca02c", linewidth=2)
    true_patch = Patch(facecolor="white", edgecolor="#dc322f", linewidth=2)
    pred_patch = Patch(facecolor="white", edgecolor="#ff8c00", linewidth=2)
    ax.legend(
        [correct_patch, true_patch, pred_patch],
        ["Touches matched segment", "Touches ground truth segment", "Touches predicted segment"],
        fontsize=9,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_section_importance_bar(
    record: pd.Series,
    top_section_table: pd.DataFrame,
    output_path: Path,
    score_column: str = "section_score",
    title: str = "Top explained sections",
    xlabel: str = "Section score",
) -> None:
    if top_section_table.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No section score data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    shown_sections = top_section_table.copy()
    sort_tail = "id_section" if "id_section" in shown_sections.columns else score_column
    shown_sections = shown_sections.sort_values([score_column, sort_tail], ascending=[True, True])
    labels = [f"sec {int(row.id_section)}" for row in shown_sections.itertuples(index=False)]
    max_score = float(shown_sections[score_column].max()) if not shown_sections.empty else 1.0
    if max_score <= 0.0:
        max_score = 1.0
    colors = [
        _importance_to_mpl_color(float(getattr(row, score_column)) / max_score)
        for row in shown_sections.itertuples(index=False)
    ]
    edge_colors = []
    true_label = int(record["true_class"])
    pred_label = int(record["pred_class"])
    for row in shown_sections.itertuples(index=False):
        section_id = int(row.id_section)
        if section_id == true_label == pred_label:
            edge_colors.append("#2ca02c")
        elif section_id == pred_label:
            edge_colors.append("#ff8c00")
        elif section_id == true_label:
            edge_colors.append("#dc322f")
        else:
            edge_colors.append("#555555")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.barh(labels, shown_sections[score_column], color=colors, edgecolor=edge_colors, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=10)

    correct_patch = Patch(facecolor="white", edgecolor="#2ca02c", linewidth=2)
    pred_patch = Patch(facecolor="white", edgecolor="#ff8c00", linewidth=2)
    true_patch = Patch(facecolor="white", edgecolor="#dc322f", linewidth=2)
    ax.legend(
        [correct_patch, pred_patch, true_patch],
        ["Predicted correctly", "Predicted segment", "Ground truth segment"],
        fontsize=9,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_node_feature_ablation_heatmap(
    feature_df: pd.DataFrame,
    top_node_table: pd.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> None:
    if feature_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No node feature ablation data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    ordered_nodes = top_node_table["node_idx"].tolist()
    node_labels = []
    for node_idx in ordered_nodes:
        row = top_node_table.loc[top_node_table["node_idx"] == node_idx].iloc[0]
        sections = row["incident_sections"] if row["incident_sections"] else "-"
        node_labels.append(f"n{int(row['id'])} | sec {sections}")

    heatmap_df = (
        feature_df.pivot(index="node_idx", columns="feature_name", values="delta_logit")
        .reindex(index=ordered_nodes, columns=feature_names)
        .fillna(0.0)
    )
    vmax = float(np.abs(heatmap_df.to_numpy()).max()) if not heatmap_df.empty else 1.0
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(11, 5.8))
    image = ax.imshow(heatmap_df.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(node_labels)))
    ax.set_yticklabels(node_labels)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_title("Node feature ablation on top explained nodes")
    ax.tick_params(axis="both", labelsize=10)
    for row_idx in range(heatmap_df.shape[0]):
        for col_idx in range(heatmap_df.shape[1]):
            value = float(heatmap_df.iloc[row_idx, col_idx])
            if abs(value) < 1e-6:
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > vmax * 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _fit_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.contain(image, size)


def combine_custom_panels(
    panels: Sequence[tuple[str, Path]],
    output_path: Path,
) -> None:
    canvas_size = (2500, 1700)
    canvas = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(20, bold=True)

    top_size = (1200, 760)
    bottom_size = (1200, 760)
    layout = [
        ((40, 70), top_size),
        ((1260, 70), top_size),
        ((40, 880), bottom_size),
        ((1260, 880), bottom_size),
    ]
    for (title, path), (origin, size) in zip(panels, layout, strict=False):
        draw.text((origin[0], origin[1] - 24), title, fill="black", font=font)
        image = _fit_panel(path, size)
        canvas.paste(image, origin)

    canvas.save(output_path)


def combine_example_panels(
    combined_graph_path: Path,
    edge_bar_path: Path,
    node_bar_path: Path,
    node_feature_heatmap_path: Path,
    output_path: Path,
) -> None:
    combine_custom_panels(
        panels=[
            ("Prediction + Node Explain", combined_graph_path),
            ("Top Pipes", edge_bar_path),
            ("Top Nodes", node_bar_path),
            ("Node Feature Ablation", node_feature_heatmap_path),
        ],
        output_path=output_path,
    )


def _prepare_edge_plot_table(
    edge_table: pd.DataFrame,
    score_column: str = "score",
) -> pd.DataFrame:
    if edge_table.empty:
        return edge_table.copy()
    prepared = edge_table.copy()
    if score_column != "importance" or "importance" not in prepared.columns:
        prepared["importance"] = prepared[score_column].astype(float)
    prepared = prepared.sort_values(["importance", "edge_idx"], ascending=[False, True]).reset_index(drop=True)
    return prepared


def _prepare_node_plot_table(
    node_table: pd.DataFrame,
    score_column: str = "score",
) -> pd.DataFrame:
    if node_table.empty:
        return node_table.copy()
    prepared = node_table.copy()
    if score_column != "importance_max" or "importance_max" not in prepared.columns:
        prepared["importance_max"] = prepared[score_column].astype(float)
    if "importance_sum" not in prepared.columns:
        prepared["importance_sum"] = prepared["importance_max"]
    prepared = prepared.sort_values(["importance_max", "importance_sum", "node_idx"], ascending=[False, False, True]).reset_index(drop=True)
    return prepared


def plot_example_explanation_bundle(
    record: pd.Series,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    edge_table: pd.DataFrame,
    node_table: pd.DataFrame,
    section_scores: pd.DataFrame,
    node_feature_df: pd.DataFrame,
    top_edge_table: pd.DataFrame,
    top_node_table: pd.DataFrame,
    node_feature_names: list[str],
    example_dir: Path,
    top_sections: int,
    method: str = PRIMARY_EXPLANATION_METHOD,
) -> Path:
    method_label = METHOD_DISPLAY_NAMES.get(method, method)
    combined_graph_path = example_dir / "prediction_explain_graph.png"
    edge_bar_path = example_dir / "edge_importance_bar.png"
    node_bar_path = example_dir / "node_importance_bar.png"
    node_feature_heatmap_path = example_dir / "node_feature_ablation_heatmap.png"
    overview_path = example_dir / "explanation_overview.png"

    _render_prediction_explain_graph(
        record=record,
        nodes_df=nodes_df,
        edges_df=edges_df,
        edge_table=edge_table,
        node_table=node_table,
        top_node_table=top_node_table,
        output_path=combined_graph_path,
        title_prefix=f"Prediction + {method_label}",
    )
    plot_edge_importance_bar(
        record,
        top_edge_table,
        edge_bar_path,
        title=f"Top pipes | {method_label}",
        xlabel=f"{method_label} edge score",
    )
    plot_node_importance_bar(
        record,
        top_node_table,
        node_bar_path,
        title=f"Top nodes | {method_label}",
        xlabel="Node importance",
    )
    plot_node_feature_ablation_heatmap(
        feature_df=node_feature_df,
        top_node_table=top_node_table,
        feature_names=node_feature_names,
        output_path=node_feature_heatmap_path,
    )
    combine_example_panels(
        combined_graph_path=combined_graph_path,
        edge_bar_path=edge_bar_path,
        node_bar_path=node_bar_path,
        node_feature_heatmap_path=node_feature_heatmap_path,
        output_path=overview_path,
    )
    return overview_path


def plot_baseline_explanation_bundle(
    record: pd.Series,
    method: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    edge_score_table: pd.DataFrame,
    node_score_table: pd.DataFrame,
    section_score_table: pd.DataFrame,
    example_dir: Path,
    top_nodes: int,
    top_edges: int,
    top_sections: int,
) -> Path:
    method_label = METHOD_DISPLAY_NAMES.get(method, method)
    plot_edge_table = _prepare_edge_plot_table(edge_score_table, score_column="score")
    plot_node_table = _prepare_node_plot_table(node_score_table, score_column="score")
    top_edge_table = plot_edge_table.head(top_edges).copy()
    top_node_table = plot_node_table.head(top_nodes).copy()
    top_section_table = (
        section_score_table.sort_values(["section_score", "rank", "id_section"], ascending=[False, True, True])
        .head(top_sections)
        .copy()
        if not section_score_table.empty
        else section_score_table.copy()
    )

    combined_graph_path = example_dir / "prediction_explain_graph.png"
    edge_bar_path = example_dir / "edge_importance_bar.png"
    node_bar_path = example_dir / "node_importance_bar.png"
    section_bar_path = example_dir / "section_importance_bar.png"
    overview_path = example_dir / "explanation_overview.png"

    _render_prediction_explain_graph(
        record=record,
        nodes_df=nodes_df,
        edges_df=edges_df,
        edge_table=plot_edge_table,
        node_table=plot_node_table,
        top_node_table=top_node_table,
        output_path=combined_graph_path,
        title_prefix=f"Prediction + {method_label}",
    )
    plot_edge_importance_bar(
        record=record,
        top_edge_table=top_edge_table,
        output_path=edge_bar_path,
        score_column="importance",
        title=f"Top pipes | {method_label}",
        xlabel=f"{method_label} edge score",
    )
    plot_node_importance_bar(
        record=record,
        top_node_table=top_node_table,
        output_path=node_bar_path,
        score_column="importance_max",
        title=f"Top nodes | {method_label}",
        xlabel=f"{method_label} node score",
        label_suffix_fallback=method_label,
    )
    plot_section_importance_bar(
        record=record,
        top_section_table=top_section_table,
        output_path=section_bar_path,
        score_column="section_score",
        title=f"Top sections | {method_label}",
        xlabel=f"{method_label} section score",
    )
    combine_custom_panels(
        panels=[
            (f"Prediction + {method_label}", combined_graph_path),
            ("Top Pipes", edge_bar_path),
            ("Top Nodes", node_bar_path),
            ("Top Sections", section_bar_path),
        ],
        output_path=overview_path,
    )
    return overview_path


def plot_confusion_matrix(cm_df: pd.DataFrame, output_path: Path, title: str) -> None:
    row_labels = cm_df.index.tolist()
    col_labels = cm_df.columns.tolist()
    cm = cm_df.values
    fig_size = max(10.0, min(20.0, 0.35 * max(len(row_labels), len(col_labels))))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(cm, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90 if len(col_labels) > 20 else 0)
    ax.set_yticklabels(row_labels)

    if max(len(row_labels), len(col_labels)) <= 18:
        threshold = cm.max() * 0.5 if cm.size else 0.0
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                value = int(cm[row_idx, col_idx])
                if not value:
                    continue
                color = "white" if value > threshold else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_requested_categories(requested_summary: pd.DataFrame, output_path: Path) -> None:
    directions = sorted(requested_summary["direction"].unique().tolist())
    if not directions:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(directions))
    width = 0.18
    colors = {
        "correct": "#2ca02c",
        "neighbor": "#ff7f0e",
        "farther": "#d62728",
        "no_defect": "#1f77b4",
    }

    center = (len(REQUESTED_CATEGORY_ORDER) - 1) / 2
    for offset_idx, category in enumerate(REQUESTED_CATEGORY_ORDER):
        subset = requested_summary[requested_summary["requested_category"] == category]
        values = [
            int(subset.loc[subset["direction"] == direction, "count"].sum())
            for direction in directions
        ]
        ax.bar(
            x + (offset_idx - center) * width,
            values,
            width=width,
            label=category,
            color=colors[category],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.set_ylabel("Count")
    ax.set_title("Requested categories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_analysis_categories(analysis_summary: pd.DataFrame, output_path: Path) -> None:
    directions = sorted(analysis_summary["direction"].unique().tolist())
    if not directions:
        return

    pivot = (
        analysis_summary.pivot(index="direction", columns="analysis_category", values="count")
        .reindex(index=directions, columns=ANALYSIS_CATEGORY_ORDER, fill_value=0)
    )
    colors = {
        "correct_defect": "#2ca02c",
        "neighbor_error": "#ffbb78",
        "far_error": "#d62728",
        "missed_defect_no_prediction": "#8c564b",
        "correct_no_defect": "#1f77b4",
        "false_positive_no_defect": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(pivot.index), dtype=float)
    for category in ANALYSIS_CATEGORY_ORDER:
        values = pivot[category].to_numpy(dtype=float)
        ax.bar(pivot.index.tolist(), values, bottom=bottoms, label=category, color=colors[category])
        bottoms += values

    ax.set_ylabel("Count")
    ax.set_title("All analysis categories")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distance_histogram(records_df: pd.DataFrame, output_path: Path) -> None:
    subset = records_df[records_df["graph_distance"].notna()].copy()
    if subset.empty:
        return

    distance_counts = (
        subset.groupby(["direction", "graph_distance"])
        .size()
        .rename("count")
        .reset_index()
    )
    directions = sorted(distance_counts["direction"].unique().tolist())
    all_distances = sorted(set(distance_counts["graph_distance"].astype(int).unique().tolist()) | {0})

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35 if len(directions) > 1 else 0.6
    x = np.arange(len(all_distances))
    for direction_idx, direction in enumerate(directions):
        direction_counts = distance_counts[distance_counts["direction"] == direction]
        values = [
            int(direction_counts.loc[direction_counts["graph_distance"] == distance, "count"].sum())
            for distance in all_distances
        ]
        offset = (direction_idx - (len(directions) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=direction)

    ax.set_xticks(x)
    ax.set_xticklabels(all_distances)
    ax.set_xlabel("Graph distance between true and predicted sections")
    ax.set_ylabel("Count")
    ax.set_title("Distance distribution for defect localization (0 = correct segment)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance_summary(
    feature_summary_df: pd.DataFrame,
    output_path: Path,
    entity_label: str = "Edge",
) -> None:
    if feature_summary_df.empty:
        return

    grouped = (
        feature_summary_df.groupby(["requested_category", "feature_name"])["abs_delta_logit"]
        .mean()
        .rename("mean_abs_delta_logit")
        .reset_index()
    )
    categories = [c for c in REQUESTED_CATEGORY_ORDER if c in grouped["requested_category"].unique()]
    feature_names = grouped["feature_name"].drop_duplicates().tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(feature_names))
    width = 0.18
    center = (len(categories) - 1) / 2
    for idx, category in enumerate(categories):
        subset = grouped[grouped["requested_category"] == category]
        values = [
            float(subset.loc[subset["feature_name"] == feature_name, "mean_abs_delta_logit"].mean())
            if not subset.loc[subset["feature_name"] == feature_name, "mean_abs_delta_logit"].empty
            else 0.0
            for feature_name in feature_names
        ]
        ax.bar(x + (idx - center) * width, values, width=width, label=category)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Mean |delta logit|")
    ax.set_title(f"Aggregated {entity_label.lower()}-feature influence on explained examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_baseline_method_rank_summary(method_summary_df: pd.DataFrame, output_path: Path) -> None:
    if method_summary_df.empty:
        return

    extra_methods = [
        method
        for method in method_summary_df["method"].drop_duplicates().tolist()
        if method not in EXPLANATION_METHOD_ORDER
    ]
    method_order = list(EXPLANATION_METHOD_ORDER) + extra_methods
    grouped = (
        method_summary_df.groupby("method")
        .agg(
            mean_true_rank=("true_section_rank", "mean"),
            mean_pred_rank=("pred_section_rank", "mean"),
            num_examples=("sample_id", "nunique"),
        )
        .reindex(method_order)
        .dropna(how="all")
        .reset_index()
    )
    if grouped.empty:
        return

    x = np.arange(len(grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, grouped["mean_true_rank"], width=width, label="Mean true-section rank", color="#dc322f")
    ax.bar(x + width / 2, grouped["mean_pred_rank"], width=width, label="Mean predicted-section rank", color="#ff8c00")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_DISPLAY_NAMES.get(method, method) for method in grouped["method"]],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_title("Comparison of simple explanation methods on selected examples")
    ax.legend()

    for idx, row in grouped.iterrows():
        ax.text(idx, max(row["mean_true_rank"], row["mean_pred_rank"]) + 0.2, f"n={int(row['num_examples'])}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
