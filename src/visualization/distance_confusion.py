from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .article_style import PALETTE, apply_article_style, save_figure


def _iter_sample_pairs(results_dir: Path):
    for nodes_path in sorted(results_dir.rglob("*nodes*.csv")):
        edges_path = nodes_path.with_name(nodes_path.name.replace("nodes", "tubes"))
        if edges_path.exists():
            yield nodes_path, edges_path


def _read_label_pair(edges_path: Path) -> tuple[int, int]:
    labels_df = pd.read_csv(edges_path, usecols=["graph_label", "graph_label_pred"], nrows=1)
    if labels_df.empty:
        raise ValueError(f"Missing graph labels in {edges_path}")
    return (
        int(round(float(labels_df.loc[0, "graph_label"]))),
        int(round(float(labels_df.loc[0, "graph_label_pred"]))),
    )


def load_targets_predictions(results_dir: Path) -> tuple[list[int], list[int]]:
    targets: list[int] = []
    predictions: list[int] = []
    for _, edges_path in _iter_sample_pairs(results_dir):
        true_class, pred_class = _read_label_pair(edges_path)
        targets.append(true_class)
        predictions.append(pred_class)
    if not targets:
        raise RuntimeError(f"No saved csv pairs were found under {results_dir}")
    return targets, predictions


def compute_distance_confusion(
    results_dir: Path,
    sections_distances_csv: Path,
    max_distance: int | None = None,
    exclude_classes: list[int] | None = None,
    auto_exclude_max_label: bool = True,
) -> tuple[np.ndarray, list[int], list[int]]:
    targets, predictions = load_targets_predictions(results_dir)

    distance_df = pd.read_csv(sections_distances_csv, sep="\t")
    distance_matrix = distance_df.to_numpy()
    unique_classes = sorted(set(targets) | set(predictions))
    excluded = set(int(class_id) for class_id in (exclude_classes or []))
    if auto_exclude_max_label and unique_classes:
        excluded.add(int(max(unique_classes)))
    fault_classes = [class_id for class_id in unique_classes if class_id not in excluded]
    class_to_row = {class_id: row_idx for row_idx, class_id in enumerate(fault_classes)}

    if max_distance is None:
        valid_distances = distance_matrix[distance_matrix >= 0]
        max_distance = int(valid_distances.max()) if valid_distances.size else 0

    confusion = np.zeros((len(fault_classes), max_distance + 1), dtype=float)
    for true_class, pred_class in zip(targets, predictions):
        if true_class in excluded or pred_class in excluded:
            continue
        distance = int(distance_matrix[int(true_class), int(pred_class)])
        if 0 <= distance <= max_distance and true_class in class_to_row:
            confusion[class_to_row[true_class], distance] += 1.0

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    normalized = confusion / row_sums
    return normalized, fault_classes, list(range(max_distance + 1))


def plot_distance_confusion(
    matrix: np.ndarray,
    fault_classes: list[int],
    distances: list[int],
    output_path: Path,
    dpi: int = 220,
    annotation_threshold: float = 0.01,
    style: dict | None = None,
    transpose: bool = False,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 10.0)))
    cmap_colors = style.get("cmap_colors", [PALETTE["heat_low"], PALETTE["heat_mid"], PALETTE["heat_high"]])
    cmap = mcolors.LinearSegmentedColormap.from_list(str(style.get("cmap_name", "article_heatmap")), cmap_colors)

    matrix_to_plot = matrix.T if transpose else matrix
    x_labels = [str(class_id) for class_id in fault_classes] if transpose else [str(distance) for distance in distances]
    y_labels = [str(distance) for distance in distances] if transpose else [str(class_id) for class_id in fault_classes]
    if transpose:
        xlabel = style.get("xlabel", style.get("xlabel_transposed", "True section"))
        ylabel = style.get("ylabel", style.get("ylabel_transposed", "Topological distance to true section"))
        title = style.get("title", style.get("title_transposed"))
    else:
        xlabel = style.get("xlabel", "Topological distance to true section")
        ylabel = style.get("ylabel", "True section")
        title = style.get("title")

    if "figsize" in style:
        fig_size = tuple(style["figsize_transposed"] if transpose and "figsize_transposed" in style else style["figsize"])
    else:
        fig_height = max(7.5, 0.16 * len(y_labels) + 1.8)
        fig_width = max(4.8, 0.45 * len(x_labels) + 2.4)
        fig_size = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=fig_size)
    vmax = float(matrix_to_plot.max()) if matrix_to_plot.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    image = ax.imshow(
        matrix_to_plot,
        aspect=style.get("aspect", "auto"),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        origin="lower" if transpose else "upper",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=float(style.get("xtick_font_size", 8)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=float(style.get("ytick_font_size", 7)))

    if style.get("xtick_rotation", 0):
        plt.setp(ax.get_xticklabels(), rotation=float(style["xtick_rotation"]), ha=style.get("xtick_ha", "center"))

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color=style.get("grid_color", PALETTE["grid"]), linewidth=float(style.get("grid_linewidth", 0.65)))
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(matrix_to_plot.shape[0]):
        for col_idx in range(matrix_to_plot.shape[1]):
            value = float(matrix_to_plot[row_idx, col_idx])
            if value < annotation_threshold:
                continue
            text_color = style.get("annotation_text_color_high", "white") if value > vmax * 0.6 else style.get("annotation_text_color_low", PALETTE["text"])
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=float(style.get("annotation_font_size", 6.2)),
                color=text_color,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=float(style.get("colorbar_fraction", 0.046)),
        pad=float(style.get("colorbar_pad", 0.03)),
    )
    colorbar.set_label(style.get("colorbar_label", "Share of predictions"), rotation=float(style.get("colorbar_rotation", 90)))
    colorbar.ax.tick_params(labelsize=float(style.get("colorbar_tick_font_size", 9)))
    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
