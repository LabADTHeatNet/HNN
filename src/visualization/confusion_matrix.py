from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .article_style import PALETTE, apply_article_style, save_figure
from .distance_confusion import load_targets_predictions


def compute_confusion_matrix(
    results_dir: Path,
    exclude_classes: list[int] | None = None,
    auto_exclude_max_label: bool = True,
    normalize_rows: bool = False,
) -> tuple[np.ndarray, list[int]]:
    targets, predictions = load_targets_predictions(results_dir)
    unique_classes = sorted(set(targets) | set(predictions))

    excluded = set(int(class_id) for class_id in (exclude_classes or []))
    if auto_exclude_max_label and unique_classes:
        excluded.add(int(max(unique_classes)))

    labels = [class_id for class_id in unique_classes if class_id not in excluded]
    label_to_idx = {class_id: idx for idx, class_id in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)

    for true_class, pred_class in zip(targets, predictions):
        if true_class in excluded or pred_class in excluded:
            continue
        matrix[label_to_idx[int(true_class)], label_to_idx[int(pred_class)]] += 1.0

    if normalize_rows:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        matrix = matrix / row_sums

    return matrix, labels


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[int],
    output_path: Path,
    dpi: int = 220,
    style: dict | None = None,
    normalize_rows: bool = False,
    annotation_threshold: float = 0.5,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 10.0)))
    cmap_colors = style.get("cmap_colors", [PALETTE["heat_low"], PALETTE["heat_mid"], PALETTE["heat_high"]])
    cmap = mcolors.LinearSegmentedColormap.from_list(str(style.get("cmap_name", "article_confusion")), cmap_colors)

    if "figsize" in style:
        fig_size = tuple(style["figsize"])
    else:
        fig_side = max(7.0, 0.24 * len(labels) + 2.8)
        fig_size = (fig_side, fig_side)

    fig, ax = plt.subplots(figsize=fig_size)
    vmax = float(matrix.max()) if matrix.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    image = ax.imshow(
        matrix,
        aspect=style.get("aspect", "equal"),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        origin="upper",
    )
    ax.set_xlabel(style.get("xlabel", "Predicted section"))
    ax.set_ylabel(style.get("ylabel", "True section"))
    title = style.get("title")
    if title:
        ax.set_title(title)

    tick_positions = np.arange(len(labels))
    tick_labels = [str(label) for label in labels]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=float(style.get("xtick_font_size", 8.0)))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=float(style.get("ytick_font_size", 8.0)))
    if style.get("xtick_rotation", 0):
        plt.setp(ax.get_xticklabels(), rotation=float(style["xtick_rotation"]), ha=style.get("xtick_ha", "center"))

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=style.get("grid_color", PALETTE["grid"]), linewidth=float(style.get("grid_linewidth", 0.65)))
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            if value < annotation_threshold:
                continue
            text_color = style.get("annotation_text_color_high", "white") if value > vmax * 0.6 else style.get("annotation_text_color_low", PALETTE["text"])
            text = f"{value:.2f}" if normalize_rows else str(int(round(value)))
            ax.text(
                col_idx,
                row_idx,
                text,
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
    default_colorbar_label = "Share of predictions" if normalize_rows else "Number of samples"
    colorbar.set_label(style.get("colorbar_label", default_colorbar_label), rotation=float(style.get("colorbar_rotation", 90)))
    colorbar.ax.tick_params(labelsize=float(style.get("colorbar_tick_font_size", 9.0)))

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
