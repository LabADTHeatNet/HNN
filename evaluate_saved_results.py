#!/usr/bin/env python3
"""
Utility script that reads saved inference results and recomputes
classification metrics together with a small visualisation pack.

The script expects the same directory layout that `test_exp` creates:
<results_dir>/
    Tout_*/*/*.csv
where both node and edge csv files are stored. Only the csv files are
required. Model inference is not executed – all numbers are taken from
the saved csv tables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SampleResult:
    """Container with paths and label information for one graph."""

    sample_id: str
    tout: str
    nodes_path: Path
    edges_path: Path
    true_class: int
    pred_class: int

    @property
    def is_correct(self) -> bool:
        return self.true_class == self.pred_class


def _iter_sample_pairs(results_dir: Path) -> Iterable[tuple[Path, Path]]:
    """Yield pairs of nodes/edges csv paths found recursively."""
    for nodes_path in sorted(results_dir.rglob("*nodes*.csv")):
        edges_path = nodes_path.with_name(nodes_path.name.replace("nodes", "tubes"))
        if edges_path.exists():
            yield nodes_path, edges_path


def _read_label_pair(edges_path: Path) -> tuple[int, int]:
    """Extract (true, pred) labels from a edges csv file."""
    df = pd.read_csv(edges_path, usecols=["graph_label", "graph_label_pred"], nrows=1)
    if df.empty:
        raise ValueError(f"{edges_path} is empty – cannot read labels.")
    true_val = df.loc[0, "graph_label"]
    pred_val = df.loc[0, "graph_label_pred"]
    return int(round(true_val)), int(round(pred_val))


def collect_samples(results_dir: Path) -> List[SampleResult]:
    """Collect all sample pairs together with their labels."""
    samples: List[SampleResult] = []
    for nodes_path, edges_path in _iter_sample_pairs(results_dir):
        true_class, pred_class = _read_label_pair(edges_path)
        rel_path = edges_path.relative_to(results_dir)
        sample_id = rel_path.with_suffix("").as_posix()
        tout = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
        samples.append(
            SampleResult(
                sample_id=sample_id,
                tout=tout,
                nodes_path=nodes_path,
                edges_path=edges_path,
                true_class=true_class,
                pred_class=pred_class,
            )
        )
    if not samples:
        raise RuntimeError(f"No csv pairs found under {results_dir}")
    return samples

def create_distance_confusion_matrix(all_targets, all_predictions, sections_distances_df, max_distance=None):
    """
    Создает матрицу, где по вертикали - истинные классы, по горизонтали - расстояния,
    а в ячейках - количество предсказаний на таком расстоянии от истинного дефекта.
    
    Parameters:
    all_targets: список истинных классов
    all_predictions: список предсказанных классов  
    sections_distances_df: DataFrame с матрицей расстояний между секциями
    max_distance: максимальное расстояние для отображения (если None - берется максимум из матрицы)
    """

    distance_matrix = sections_distances_df.values
    

    unique_classes = sorted(set(all_targets) | set(all_predictions))
    no_defect = max(unique_classes)
    unique_classes.remove(no_defect)
    
    if max_distance is None:
        max_distance = int(np.nanmax(distance_matrix[distance_matrix != -1]))

    distance_confusion = np.zeros((len(unique_classes), max_distance + 1), dtype=int)
    
    for true_class, pred_class in zip(all_targets, all_predictions):
        if true_class == no_defect or pred_class == no_defect:
            continue
            

        distance = int(distance_matrix[true_class, pred_class])
        
        if distance != -1 and distance <= max_distance:
            class_idx = unique_classes.index(true_class)
            distance_confusion[class_idx, distance] += 1
    distance_confusion_norm = distance_confusion.astype(float)
    row_sums = distance_confusion_norm.sum(axis=1)

    row_sums[row_sums == 0] = 1
    distance_confusion_norm = distance_confusion_norm / row_sums[:, np.newaxis]
    return distance_confusion_norm, unique_classes, list(range(max_distance + 1))

def compute_metrics(samples: Sequence[SampleResult], sections_distances_df : pd.DataFrame) -> dict:
    """Compute confusion matrix and derived metrics."""
    y_true = np.array([s.true_class for s in samples])
    y_pred = np.array([s.pred_class for s in samples])

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    total_samples = cm.sum()
    accuracy = float(np.trace(cm) / total_samples) if total_samples else 0.0

    precision_list = []
    recall_list = []
    f1_per_class = []
    support = []

    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        row_sum = cm[idx, :].sum()
        col_sum = cm[:, idx].sum()
        fp = col_sum - tp
        fn = row_sum - tp

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        precision_list.append(float(prec))
        recall_list.append(float(rec))
        f1_per_class.append(float(f1_val))
        support.append(int(row_sum))

    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class else 0.0
    support_sum = sum(support)
    if support_sum:
        weighted_f1 = float(
            sum(f * s for f, s in zip(f1_per_class, support)) / support_sum
        )
    else:
        weighted_f1 = 0.0

    per_class_accuracy = {}
    for idx, label in enumerate(labels):
        total = cm[idx, :].sum()
        per_class_accuracy[label] = float(cm[idx, idx] / total) if total else None

    report = {
        str(label): {
            "precision": precision_list[idx],
            "recall": recall_list[idx],
            "f1-score": f1_per_class[idx],
            "support": support[idx],
        }
        for idx, label in enumerate(labels)
    }
    report["accuracy"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1-score": accuracy,
        "support": int(total_samples),
    }
    report["macro avg"] = {
        "precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "recall": float(np.mean(recall_list)) if recall_list else 0.0,
        "f1-score": macro_f1,
        "support": int(total_samples),
    }
    report["weighted avg"] = {
        "precision": float(
            sum(p * s for p, s in zip(precision_list, support)) / support_sum
        )
        if support_sum
        else 0.0,
        "recall": float(
            sum(r * s for r, s in zip(recall_list, support)) / support_sum
        )
        if support_sum
        else 0.0,
        "f1-score": weighted_f1,
        "support": int(total_samples),
    }

    errors_df = (
        pd.DataFrame({"true_class": y_true, "pred_class": y_pred, "id": [s.sample_id for s in samples]})
        .query("true_class != pred_class")
        .groupby(["true_class", "pred_class"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    distance_metrics = {}
    if sections_distances_df is not None:
        distance_confusion, distance_classes, distances = create_distance_confusion_matrix(
            y_true, y_pred, sections_distances_df
        )
        distance_metrics = {
            "distance_confusion_matrix": distance_confusion,
            "distance_classes": distance_classes,
            "distances": distances
        }

    return {
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": precision_list,
        "recall": recall_list,
        "f1_per_class": f1_per_class,
        "support": support,
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
        "errors_table": errors_df,
        "distance_metrics": distance_metrics if distance_metrics is not None else None,
    }


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(metrics: dict, output_dir: Path) -> None:
    """Persist metrics to disk in a couple of convenient formats."""
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]
    distance_metrics = metrics["distance_metrics"]
    summary = {
        "num_samples": int(cm.sum()),
        "num_classes": len(labels),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")

    errors_df: pd.DataFrame = metrics["errors_table"]
    if not errors_df.empty:
        errors_df.to_csv(output_dir / "top_error_transitions.csv", index=False)

    per_class_df = pd.DataFrame(
        {
            "class_id": labels,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_per_class"],
            "support": metrics["support"],
            "accuracy": [metrics["per_class_accuracy"][c] for c in labels],
        }
    )
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    if distance_metrics is not None:
        _plot_distance_confusion_matrix(distance_metrics["distance_classes"],
                                        distance_metrics["distances"],
                                        distance_metrics["distance_confusion_matrix"], 
                                        path = output_dir / "distance_confusion_matrix_from_results.png")
    _plot_confusion_matrix(labels, cm, output_dir / "confusion_matrix_from_results.png")
    _plot_class_distribution(
        metrics["support"],
        metrics["confusion_matrix"],
        labels,
        output_dir / "class_distribution_from_results.png",
    )


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _boxes_overlap(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box1
    x2, y2, x3, y3 = box2
    return not (x1 < x2 or x3 < x0 or y1 < y2 or y3 < y0)


def _candidate_offsets(step: int = 12, rings: int = 6) -> List[tuple[int, int]]:
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
    existing_boxes: List[tuple[float, float, float, float]],
    bounds: tuple[int, int],
    margin: int = 2,
) -> tuple[int, int, tuple[float, float, float, float]]:
    cx, cy = center
    width, height = size
    canvas_w, canvas_h = bounds
    offsets = _candidate_offsets()

    for dx, dy in offsets:
        x0 = cx - width / 2 + dx
        y0 = cy - height / 2 + dy
        x1 = x0 + width
        y1 = y0 + height

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(canvas_w, x1)
        y1 = min(canvas_h, y1)

        # Adjust if clamped changed size
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

    # Fallback: use original position even if it overlaps
    x0 = max(0, cx - width / 2)
    y0 = max(0, cy - height / 2)
    x1 = min(canvas_w, x0 + width)
    y1 = min(canvas_h, y0 + height)
    candidate_box = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    return int(x0), int(y0), candidate_box

def _plot_distance_confusion_matrix(distance_classes: Sequence[int], distances: Sequence[int], 
                                  distance_confusion: np.ndarray, path: Path) -> None:
    """Draw a distance confusion matrix heatmap using Pillow."""
    n_classes = len(distance_classes)
    n_distances = len(distances)
    if n_classes == 0 or n_distances == 0:
        return

    cell_width = 40
    cell_height = 30
    left_margin = 100
    top_margin = 80
    bottom_margin = 120
    right_margin = 40
    width = left_margin + n_distances * cell_width + right_margin
    height = top_margin + n_classes * cell_height + bottom_margin
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    max_val = distance_confusion.max() if distance_confusion.size else 1
    max_val = max(max_val, 1)

    # Draw cells
    for i in range(n_classes):
        for j in range(n_distances):
            val = distance_confusion[i, j]
            intensity = int(255 - min(200, (val / max_val) * 200))
            color = (intensity, intensity, 255)
            x0 = left_margin + j * cell_width
            y0 = top_margin + i * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="gray")
            if val:
                text = "{:.2f}".format(float(val))
                tw, th = _measure_text(draw, text, font)
                draw.text(
                    (x0 + (cell_width - tw) / 2, y0 + (cell_height - th) / 2),
                    text,
                    fill="black",
                    font=font,
                )

    # Y-axis labels (true classes)
    for idx, class_label in enumerate(distance_classes):
        text = str(class_label)
        tw, th = _measure_text(draw, text, font)
        x = left_margin - tw - 10
        y = top_margin + idx * cell_height + (cell_height - th) / 2
        draw.text((x, y), text, fill="black", font=font)

    # X-axis labels (distances)
    for idx, distance in enumerate(distances):
        text = str(distance)
        tw, th = _measure_text(draw, text, font)
        x = left_margin + idx * cell_width + (cell_width - tw) / 2
        y = top_margin + n_classes * cell_height + 10
        draw.text((x, y), text, fill="black", font=font)

    # Title
    title = "Distance Confusion Matrix"
    tw, th = _measure_text(draw, title, font)
    draw.text(((width - tw) / 2, 20), title, fill="black", font=font)

    # Axis descriptions
    draw.text(
        (left_margin, height - bottom_margin + 60),
        "Distance to true defect",
        fill="black",
        font=font,
    )
    draw.text(
        (20, top_margin + (n_classes * cell_height) / 2),
        "True class",
        fill="black",
        font=font,
    )

    # Legend for values
    legend_x = width - 120
    legend_y = height - bottom_margin + 30
    legend_text = f"Max: {int(max_val)}"
    tw, th = _measure_text(draw, legend_text, font)
    draw.text((legend_x, legend_y), legend_text, fill="black", font=font)

    img.save(path)

def _plot_confusion_matrix(labels: Sequence[int], cm: np.ndarray, path: Path) -> None:
    """Draw a confusion matrix heatmap using Pillow."""
    n = len(labels)
    if n == 0:
        return

    cell = 40
    left_margin = 100
    top_margin = 80
    bottom_margin = 120
    right_margin = 40
    width = left_margin + n * cell + right_margin
    height = top_margin + n * cell + bottom_margin
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    max_val = cm.max() if cm.size else 1
    max_val = max(max_val, 1)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            intensity = int(255 - min(200, (val / max_val) * 200))
            color = (intensity, intensity, 255)
            x0 = left_margin + j * cell
            y0 = top_margin + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="gray")
            if val:
                text = str(int(val))
                tw, th = _measure_text(draw, text, font)
                draw.text(
                    (x0 + (cell - tw) / 2, y0 + (cell - th) / 2),
                    text,
                    fill="black",
                    font=font,
                )

    # Axis labels
    for idx, label in enumerate(labels):
        text = str(label)
        tw, th = _measure_text(draw, text, font)
        # x-axis
        x = left_margin + idx * cell + (cell - tw) / 2
        y = top_margin + n * cell + 10
        draw.text((x, y), text, fill="black", font=font)
        # y-axis
        x = left_margin - tw - 10
        y = top_margin + idx * cell + (cell - th) / 2
        draw.text((x, y), text, fill="black", font=font)

    title = "Confusion matrix (saved results)"
    tw, th = _measure_text(draw, title, font)
    draw.text(((width - tw) / 2, 20), title, fill="black", font=font)
    draw.text(
        (left_margin, height - bottom_margin + 60),
        "Predicted class",
        fill="black",
        font=font,
    )
    draw.text(
        (20, top_margin + (n * cell) / 2),
        "True class",
        fill="black",
        font=font,
    )

    img.save(path)


def _plot_class_distribution(
    support: Sequence[float],
    cm: np.ndarray,
    labels: Sequence[int],
    path: Path,
) -> None:
    """Plot true vs predicted histogram using Pillow."""
    n = len(labels)
    if n == 0:
        return

    true_counts = support
    pred_counts = cm.sum(axis=0)

    chart_height = 300
    top_margin = 70
    left_margin = 100
    bottom_margin = 140
    right_margin = 40
    bar_width = 10
    gap = 6
    group_width = 2 * bar_width + gap

    width = left_margin + n * group_width + right_margin
    height = top_margin + chart_height + bottom_margin
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    max_count = max(list(true_counts) + list(pred_counts)) if len(true_counts) else 1
    max_count = max(max_count, 1)

    baseline = top_margin + chart_height
    for idx, label in enumerate(labels):
        gx = left_margin + idx * group_width

        true_val = true_counts[idx] if idx < len(true_counts) else 0
        predicted_val = pred_counts[idx] if idx < len(pred_counts) else 0

        true_height = int(chart_height * (true_val / max_count))
        pred_height = int(chart_height * (predicted_val / max_count))

        # True bar
        draw.rectangle(
            [gx, baseline - true_height, gx + bar_width, baseline],
            fill=(70, 130, 180),
            outline="black",
        )

        # Pred bar
        draw.rectangle(
            [gx + bar_width + 2, baseline - pred_height, gx + 2 * bar_width + 2, baseline],
            fill=(220, 20, 60),
            outline="black",
        )

        text = str(label)
        tw, th = _measure_text(draw, text, font)
        draw.text(
            (gx + (group_width - gap - tw) / 2, baseline + 10),
            text,
            fill="black",
            font=font,
        )

    # Axes
    draw.line([left_margin, top_margin, left_margin, baseline], fill="black")
    draw.line([left_margin, baseline, width - right_margin, baseline], fill="black")

    # Legends and titles
    title = "Class distribution (true vs predicted)"
    tw, th = _measure_text(draw, title, font)
    draw.text(((width - tw) / 2, 20), title, fill="black", font=font)

    legend_y = height - bottom_margin + 60
    draw.rectangle([left_margin, legend_y, left_margin + 15, legend_y + 15], fill=(70, 130, 180))
    draw.text((left_margin + 20, legend_y), "True", fill="black", font=font)
    draw.rectangle([left_margin + 100, legend_y, left_margin + 115, legend_y + 15], fill=(220, 20, 60))
    draw.text((left_margin + 120, legend_y), "Predicted", fill="black", font=font)

    img.save(path)


def get_node_degrees(edges_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Return dictionaries with outgoing and incoming degrees per node."""
    node_deg_out: dict[int, int] = {}
    node_deg_in: dict[int, int] = {}
    for row in edges_df.itertuples(index=False):
        node_deg_out[int(row.id_in)] = node_deg_out.get(int(row.id_in), 0) + 1
        node_deg_in[int(row.id_out)] = node_deg_in.get(int(row.id_out), 0) + 1
    return node_deg_out, node_deg_in


def compute_id_sections(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate add_sections logic to obtain id_section labels without sklearn."""
    edges_df = edges_df.copy()
    edges_df["id_section"] = -1

    node_deg_out, node_deg_in = get_node_degrees(edges_df)
    node_ids = set(int(n) for n in nodes_df["id"].tolist())
    start_vertices = sorted(node_ids.difference(node_deg_in.keys()))

    if not start_vertices:
        # Fallback: choose the smallest node id as a start point
        if not node_ids:
            edges_df["id_section"] = 0
            return edges_df
        start_vertices = [min(node_ids)]

    start_edges = [edges_df.loc[edges_df["id_in"] == v] for v in start_vertices]
    if all(df.empty for df in start_edges):
        # Fallback to the very first edge
        first_idx = edges_df.index[0]
        start_edges = [edges_df.loc[[first_idx]]]

    num_sources = max(len(start_edges), 1)
    next_section_ids = [i for i in range(num_sources)]

    edge_queue: deque[tuple[int, int, int, int]] = deque()
    visited_ids: set[int] = set()

    def queue_edge(edge_idx: int, id_section: int) -> None:
        row = edges_df.loc[edge_idx]
        edge_queue.append((edge_idx, int(row["id_in"]), int(row["id_out"]), id_section))
        visited_ids.add(edge_idx)

    for i, edge_group in enumerate(start_edges):
        for edge_t in edge_group.itertuples():
            vid_usr = float(getattr(edge_t, "Vid_usr", 0.0))
            id_section = i + num_sources if vid_usr > 0.5 else i
            queue_edge(edge_t.Index, id_section)

    while edge_queue:
        edge_idx, id_in, id_out, id_section = edge_queue.popleft()
        edges_df.at[edge_idx, "id_section"] = id_section

        deg_sum = node_deg_in.get(id_out, 0) + node_deg_out.get(id_out, 0)
        next_edges = edges_df.loc[edges_df["id_in"] == id_out]
        for next_edge in next_edges.itertuples():
            if next_edge.Index in visited_ids:
                continue
            vid_usr = float(getattr(next_edge, "Vid_usr", 0.0))
            if vid_usr > 0.5 or deg_sum > 2:
                base_idx = id_section % num_sources if num_sources else 0
                while base_idx >= len(next_section_ids):
                    next_section_ids.append(len(next_section_ids))
                next_section_ids[base_idx] += num_sources
                next_id_section = next_section_ids[base_idx]
            else:
                next_id_section = id_section
            queue_edge(next_edge.Index, next_id_section)

    # Assign standalone sections for any remaining unvisited edges
    if (edges_df["id_section"] == -1).any():
        current_max = edges_df["id_section"].max()
        for edge_idx in edges_df.index[edges_df["id_section"] == -1]:
            current_max += 1
            edges_df.at[edge_idx, "id_section"] = current_max

    unique_sections = sorted(edges_df["id_section"].unique())
    mapping = {old: new for new, old in enumerate(unique_sections)}
    edges_df["id_section"] = edges_df["id_section"].map(mapping).astype(int)
    return edges_df

CATEGORY_STYLES = {
    "consumer": {"color": (255, 127, 14), "size": 12, "shape": "square", "label": "Consumer"},
    "source": {"color": (214, 39, 40), "size": 12, "shape": "triangle_up", "label": "Source"},
    "sink": {"color": (44, 160, 44), "size": 12, "shape": "triangle_down", "label": "Sink"},
    "junction": {"color": (148, 103, 189), "size": 8, "shape": "diamond", "label": "Junction"},
    "intermediate": {"color": (44, 160, 44), "size": 2, "shape": "circle", "label": "Intermediate"},  # (31, 119, 180)
}


def categorize_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict[int, str]:
    """Assign each node to one of the requested categories."""
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


def compute_segment_centroids(edges_df: pd.DataFrame, positions: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """Return weighted centroid for each segment."""
    accum: dict[int, dict[str, float]] = {}
    fallback_points: dict[int, List[tuple[float, float]]] = {}

    for row in edges_df.itertuples():
        seg = int(row.id_section)
        src = int(row.id_in)
        dst = int(row.id_out)
        pos_in = positions.get(src)
        pos_out = positions.get(dst)
        if pos_in is None or pos_out is None:
            continue
        try:
            weight = float(getattr(row, "l", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
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
        else:
            pts = fallback_points.get(seg)
            if pts:
                centroids[seg] = (
                    sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts),
                )
    return centroids


def _draw_node_symbol(draw: ImageDraw.ImageDraw, center: tuple[int, int], style: dict) -> None:
    x, y = center
    size = style["size"]
    color = style["color"]
    outline = "black"
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
    """Map score in [0, 1] to gradient color (red -> yellow -> green)."""
    score = 0.0 if score is None else float(score)
    score = max(0.0, min(1.0, score))
    mid_color = (255, 215, 0)  # gold
    low_color = (220, 20, 60)  # crimson
    high_color = (34, 139, 34)  # forest green
    if score <= 0.5:
        return _interpolate_color(low_color, mid_color, score / 0.5 if score else 0.0)
    return _interpolate_color(mid_color, high_color, (score - 0.5) / 0.5 if score < 1.0 else 1.0)


def pick_random_samples(
    samples: Sequence[SampleResult], num_samples: int, seed: int
) -> List[SampleResult]:
    rng = random.Random(seed)
    num = min(num_samples, len(samples))
    return rng.sample(list(samples), k=num)


def plot_graph(
    sample: SampleResult,
    output_dir: Path,
    segment_score_map: dict[int, float] | None = None,
    score_label: str = "F1-score",
) -> None:
    """Draw a graph using node positions stored in the csv tables."""
    nodes_df = pd.read_csv(sample.nodes_path)
    edges_df = pd.read_csv(sample.edges_path)

    if "id" not in nodes_df.columns:
        nodes_df["id"] = range(len(nodes_df))

    nodes_df["id"] = nodes_df["id"].astype(int)
    edges_df["id_in"] = edges_df["id_in"].astype(int)
    edges_df["id_out"] = edges_df["id_out"].astype(int)

    edges_df = compute_id_sections(nodes_df, edges_df)
    node_categories = categorize_nodes(nodes_df, edges_df)

    positions = {
        int(row.id): (float(row.pos_x), float(row.pos_y))
        for row in nodes_df.itertuples(index=False)
    }
    if not positions:
        return

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width, height = 1200, 800
    margin = 60
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)
    scale_x = (width - 2 * margin) / range_x
    scale_y = (height - 2 * margin) / range_y

    def project(pt_x: float, pt_y: float) -> tuple[int, int]:
        px = margin + (pt_x - min_x) * scale_x
        py = height - margin - (pt_y - min_y) * scale_y
        return int(px), int(py)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    title = (
        f"{sample.sample_id} | true={sample.true_class} | "
        f"pred={sample.pred_class} | {'OK' if sample.is_correct else 'MISMATCH'}"
    )
    draw.text((20, 20), title, fill="black", font=font)

    true_label = int(round(float(edges_df["graph_label"].iloc[0])))
    pred_label = int(round(float(edges_df["graph_label_pred"].iloc[0])))
    highlight_same = true_label == pred_label

    COLOR_PIPE_FWD = (220, 50, 50)  # Vid == 0
    COLOR_PIPE_BWD = (50, 90, 200)  # Vid == 1
    COLOR_PIPE_OTHER = (120, 120, 120)
    COLOR_DEFECT_MATCH = (44, 160, 44)
    COLOR_DEFECT_TRUE = (255, 140, 0)
    COLOR_DEFECT_PRED = (0, 0, 0)
    EDGE_WIDTH_DEFAULT = 5
    EDGE_WIDTH_HIGHLIGHT = 8

    edge_records: List[dict[str, object]] = []

    # Draw edges with highlighting of defect segments
    for row in edges_df.itertuples():
        src = int(row.id_in)
        dst = int(row.id_out)
        pos_src = positions.get(src)
        pos_dst = positions.get(dst)
        if pos_src is None or pos_dst is None:
            continue
        p1 = project(*pos_src)
        p2 = project(*pos_dst)

        vid_fwd = float(getattr(row, "Vid_fwd", 0.0))
        vid_bwd = float(getattr(row, "Vid_bwd", 0.0))
        if vid_fwd > 0.5:
            base_color = COLOR_PIPE_FWD
        elif vid_bwd > 0.5:
            base_color = COLOR_PIPE_BWD
        else:
            base_color = COLOR_PIPE_OTHER

        color = base_color
        width_line = EDGE_WIDTH_DEFAULT

        seg = int(row.id_section)
        if highlight_same:
            if seg == true_label:
                color = COLOR_DEFECT_MATCH
                width_line = EDGE_WIDTH_HIGHLIGHT
        else:
            if seg == true_label:
                color = COLOR_DEFECT_TRUE
                width_line = EDGE_WIDTH_HIGHLIGHT
            if seg == pred_label:
                color = COLOR_DEFECT_PRED
                width_line = EDGE_WIDTH_HIGHLIGHT

        edge_records.append(
            {
                "points": (p1, p2),
                "color": color,
                "width": int(width_line),
                "segment": seg,
            }
        )

    for edge_info in edge_records:
        draw.line(
            [edge_info["points"][0], edge_info["points"][1]],
            fill=edge_info["color"],
            width=int(edge_info["width"]),
        )

    # Draw nodes with category-specific symbols
    for node_id, pos in positions.items():
        px, py = project(*pos)
        style = CATEGORY_STYLES.get(node_categories.get(node_id, "intermediate"), CATEGORY_STYLES["intermediate"])
        _draw_node_symbol(draw, (px, py), style)

    # Segment labels
    centroids = compute_segment_centroids(edges_df, positions)
    label_boxes: List[tuple[float, float, float, float]] = []
    for seg_id, coord in centroids.items():
        px, py = project(*coord)
        px = int(px)
        py = int(py)
        label_text = f"{seg_id}"
        if highlight_same and seg_id == true_label:
            label_color = COLOR_DEFECT_MATCH
        else:
            if seg_id == true_label:
                label_color = COLOR_DEFECT_TRUE
            elif seg_id == pred_label:
                label_color = COLOR_DEFECT_PRED
            else:
                label_color = (0, 0, 0)

        tw, th = _measure_text(draw, label_text, font)
        x0, y0, box = _find_label_position(
            (px, py),
            (tw, th),
            label_boxes,
            (width, height),
        )
        label_boxes.append(box)
        draw.text((x0, y0), label_text, fill=label_color, font=font)

    # Node legend
    legend_order = ["source", "sink", "consumer", "junction", "intermediate"]
    legend_start_x = 40
    legend_start_y = height - 120
    column_width = 150
    for idx, key in enumerate(legend_order):
        style = CATEGORY_STYLES[key]
        col = idx % 3
        row = idx // 3
        cx = legend_start_x + col * column_width
        cy = legend_start_y + row * 45
        _draw_node_symbol(draw, (cx, cy), style)
        draw.text((cx + style["size"] + 8, cy - 7), style["label"], fill="black", font=font)

    # Edge legend
    edge_legend_items: List[tuple[str, tuple[int, int, int], int]] = [
        ("Forward pipes", COLOR_PIPE_FWD, EDGE_WIDTH_DEFAULT),
        ("Backward pipes", COLOR_PIPE_BWD, EDGE_WIDTH_DEFAULT),
    ]

    base_y = height - 150
    base_x = 40
    for label_text, color, line_width in edge_legend_items:
        draw.line([base_x, base_y, base_x + 50, base_y], fill=color, width=line_width)
        draw.text((base_x + 60, base_y - 7), label_text, fill="black", font=font)
        base_y -= 30

    # draw.text((40, base_y), "Толстая линия — сегмент с дефектом (зелёный=совпало, оранжевый/чёрный — True/Pred)", fill="black", font=font)

    safe_id = sample.sample_id.replace("/", "__")
    img.save(output_dir / f"{safe_id}.png")

    if segment_score_map:
        _draw_segment_scores_graph(
            sample=sample,
            output_dir=output_dir,
            node_categories=node_categories,
            positions=positions,
            project=project,
            edge_records=edge_records,
            centroids=centroids,
            segment_scores=segment_score_map,
            width=width,
            height=height,
            font=font,
            score_label=score_label,
        )


def _draw_segment_scores_graph(
    sample: SampleResult,
    output_dir: Path,
    node_categories: dict[int, str],
    positions: dict[int, tuple[float, float]],
    project: Callable[[float, float], tuple[int, int]],
    edge_records: Sequence[dict[str, object]],
    centroids: dict[int, tuple[float, float]],
    segment_scores: dict[int, float],
    width: int,
    height: int,
    font: ImageFont.ImageFont,
    score_label: str,
) -> None:
    """Draw additional graph view with edges coloured by segment scores."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title = f"{sample.sample_id} | Segment {score_label}"
    draw.text((20, 20), title, fill="black", font=font)
    subtitle = f"Edge colour encodes {score_label} in [0, 1]"
    draw.text((20, 38), subtitle, fill="black", font=font)

    missing_color = (160, 160, 160)

    for edge_info in edge_records:
        seg_id = int(edge_info["segment"])
        points = edge_info["points"]
        score = segment_scores.get(seg_id)
        color = _score_to_color(score) if score is not None else missing_color
        width_line = max(4, int(edge_info["width"]))
        draw.line([points[0], points[1]], fill=color, width=width_line)

    for node_id, pos in positions.items():
        px, py = project(*pos)
        style = CATEGORY_STYLES.get(
            node_categories.get(node_id, "intermediate"),
            CATEGORY_STYLES["intermediate"],
        )
        _draw_node_symbol(draw, (px, py), style)

    label_boxes: List[tuple[float, float, float, float]] = []
    for seg_id, coord in centroids.items():
        px, py = project(*coord)
        px = int(px)
        py = int(py)
        score = segment_scores.get(int(seg_id))
        if score is None:
            label_text = f"{seg_id}: n/a"
            label_color = missing_color
        else:
            label_text = f"{seg_id}: {score:.2f}"
            label_color = _score_to_color(score)
        tw, th = _measure_text(draw, label_text, font)
        x0, y0, box = _find_label_position(
            (px, py),
            (tw, th),
            label_boxes,
            (width, height),
        )
        label_boxes.append(box)
        draw.text((x0, y0), label_text, fill=label_color, font=font)

    bar_height = 220
    bar_width = 24
    bar_x0 = 80
    bar_y0 = 120
    for idx in range(bar_height):
        value = 1.0 - idx / max(1, bar_height - 1)
        color = _score_to_color(value)
        draw.line([bar_x0, bar_y0 + idx, bar_x0 + bar_width, bar_y0 + idx], fill=color)
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_width, bar_y0 + bar_height], outline="black")
    draw.text((bar_x0 + bar_width + 8, bar_y0 - 6), "1.0", fill="black", font=font)
    draw.text((bar_x0 + bar_width + 8, bar_y0 + bar_height - 6), "0.0", fill="black", font=font)

    label_text = score_label
    tw, th = _measure_text(draw, label_text, font)
    draw.text((bar_x0 - max(0, (tw - bar_width) / 2), bar_y0 - 24), label_text, fill="black", font=font)

    note = "Gray edges: score unavailable"
    draw.text((20, height - 40), note, fill="black", font=font)

    safe_id = sample.sample_id.replace("/", "__")
    suffix = score_label.lower().replace(" ", "_").replace("-", "_")
    img.save(output_dir / f"{safe_id}__segment_{suffix}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute metrics and draw graphs from saved inference results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("out_Termo_Ablation_heads/16_heads/results/bwd"),
        help="Directory with saved csv outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store metrics and plots. Defaults to <results-dir>/analysis_from_saved.",
    )
    parser.add_argument(
        "--sections-distances-csv",
        type=Path,
        default="sections_distances.csv",
        help="Path to sections_distances.csv file for distance-based analysis.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=200,
        help="Number of random graphs to visualise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sample selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "analysis_from_saved"
    _ensure_output_dir(output_dir)

    samples = collect_samples(results_dir)
    sections_distances_df = None
    if args.sections_distances_csv and args.sections_distances_csv.exists():
        sections_distances_df = pd.read_csv(args.sections_distances_csv, sep='\t')
    metrics = compute_metrics(samples, sections_distances_df)
    save_metrics(metrics, output_dir)

    labels = metrics.get("labels", [])
    f1_scores = metrics.get("f1_per_class", [])
    segment_score_map = {
        int(label): float(f1_scores[idx])
        for idx, label in enumerate(labels)
        if idx < len(f1_scores)
    }

    pick_samples_num = args.num_graphs
    if pick_samples_num is None:
        pick_samples_num = len(samples)
    selected = pick_random_samples(samples, pick_samples_num, args.seed)
    graph_dir = _ensure_output_dir(output_dir / "sample_graphs")
    for idx, sample in enumerate(selected):
        if idx == 0 and segment_score_map:
            plot_graph(
                sample,
                graph_dir,
                segment_score_map=segment_score_map,
                score_label="F1-score",
            )
        else:
            plot_graph(sample, graph_dir)

    print("Metrics recomputed from saved results:")
    print(json.dumps(
        {
            "num_samples": len(samples),
            "num_classes": metrics["confusion_matrix"].shape[0],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
        indent=2,
    ))
    if metrics["errors_table"].empty:
        print("No misclassifications were found.")
    else:
        print("Top error transitions:")
        print(metrics["errors_table"].head(10).to_string(index=False))
    print(f"Artifacts stored in: {output_dir}")


if __name__ == "__main__":
    main()
