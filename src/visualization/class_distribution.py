from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .article_style import PALETTE, apply_article_style, apply_standard_axis_style, save_figure
from .distance_confusion import load_targets_predictions


def compute_class_distribution(
    results_dir: Path,
    exclude_classes: list[int] | None = None,
    auto_exclude_max_label: bool = True,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    targets, predictions = load_targets_predictions(results_dir)
    unique_classes = sorted(set(targets) | set(predictions))

    excluded = set(int(class_id) for class_id in (exclude_classes or []))
    if auto_exclude_max_label and unique_classes:
        excluded.add(int(max(unique_classes)))

    labels = [class_id for class_id in unique_classes if class_id not in excluded]
    label_to_idx = {class_id: idx for idx, class_id in enumerate(labels)}
    true_counts = np.zeros(len(labels), dtype=int)
    pred_counts = np.zeros(len(labels), dtype=int)

    for true_class, pred_class in zip(targets, predictions):
        if true_class in excluded or pred_class in excluded:
            continue
        true_counts[label_to_idx[int(true_class)]] += 1
        pred_counts[label_to_idx[int(pred_class)]] += 1

    return labels, true_counts, pred_counts


def compute_class_distribution_split(
    results_dir: Path,
    exclude_classes: list[int] | None = None,
    auto_exclude_max_label: bool = True,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    targets, predictions = load_targets_predictions(results_dir)
    unique_classes = sorted(set(targets) | set(predictions))

    excluded = set(int(class_id) for class_id in (exclude_classes or []))
    if auto_exclude_max_label and unique_classes:
        excluded.add(int(max(unique_classes)))

    labels = [class_id for class_id in unique_classes if class_id not in excluded]
    label_to_idx = {class_id: idx for idx, class_id in enumerate(labels)}
    true_counts = np.zeros(len(labels), dtype=int)
    pred_right_counts = np.zeros(len(labels), dtype=int)
    pred_wrong_counts = np.zeros(len(labels), dtype=int)

    for true_class, pred_class in zip(targets, predictions):
        if true_class in excluded or pred_class in excluded:
            continue
        true_counts[label_to_idx[int(true_class)]] += 1
        if int(true_class) == int(pred_class):
            pred_right_counts[label_to_idx[int(pred_class)]] += 1
        else:
            pred_wrong_counts[label_to_idx[int(pred_class)]] += 1

    return labels, true_counts, pred_right_counts, pred_wrong_counts


def plot_class_distribution(
    labels: list[int],
    true_counts: np.ndarray,
    pred_counts: np.ndarray,
    output_path: Path,
    dpi: int = 220,
    style: dict | None = None,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 10.5)))

    if "figsize" in style:
        fig_size = tuple(style["figsize"])
    else:
        fig_width = max(7.2, 0.28 * len(labels) + 2.8)
        fig_size = (fig_width, 4.6)

    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(labels), dtype=float)
    bar_width = float(style.get("bar_width", 0.42))

    ax.bar(
        x - bar_width / 2.0,
        true_counts,
        width=bar_width,
        color=style.get("true_color", "#B8C2CC"),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("true_label", "True"),
        zorder=2,
    )
    ax.bar(
        x + bar_width / 2.0,
        pred_counts,
        width=bar_width,
        color=style.get("pred_color", PALETTE["heat_high"]),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("pred_label", "Predicted"),
        zorder=2,
    )

    ax.set_xlabel(style.get("xlabel", "Class"))
    ax.set_ylabel(style.get("ylabel", "Number of samples"))
    title = style.get("title")
    if title:
        ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], fontsize=float(style.get("xtick_font_size", 8.5)))
    if style.get("xtick_rotation", 0):
        plt.setp(ax.get_xticklabels(), rotation=float(style["xtick_rotation"]), ha=style.get("xtick_ha", "right"))
    ax.tick_params(axis="y", labelsize=float(style.get("ytick_font_size", 9.0)))
    ax.set_xlim(-0.75, len(labels) - 0.25)

    if "min_y" in style or "max_y" in style:
        max_count = float(max(int(true_counts.max()) if true_counts.size else 0, int(pred_counts.max()) if pred_counts.size else 0))
        lower = float(style.get("min_y", 0.0))
        upper = float(style.get("max_y", max_count * 1.08 if max_count > 0.0 else 1.0))
        ax.set_ylim(lower, upper)

    apply_standard_axis_style(ax)
    ax.legend(
        loc=style.get("legend_loc", "upper right"),
        fontsize=float(style.get("legend_font_size", style.get("base_font_size", 10.5) - 0.5)),
        ncol=int(style.get("legend_ncol", 1)),
    )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)


def plot_class_distribution_split(
    labels: list[int],
    true_counts: np.ndarray,
    pred_right_counts: np.ndarray,
    pred_wrong_counts: np.ndarray,
    output_path: Path,
    dpi: int = 220,
    style: dict | None = None,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 10.5)))

    if "figsize" in style:
        fig_size = tuple(style["figsize"])
    else:
        fig_width = max(7.6, 0.32 * len(labels) + 3.0)
        fig_size = (fig_width, 4.8)

    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(labels), dtype=float)
    bar_width = float(style.get("bar_width", 0.28))

    ax.bar(
        x - bar_width,
        true_counts,
        width=bar_width,
        color=style.get("true_color", "#B8C2CC"),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("true_label", "True"),
        zorder=2,
    )
    ax.bar(
        x,
        pred_right_counts,
        width=bar_width,
        color=style.get("pred_right_color", "#2F855A"),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("pred_right_label", "Predicted Right"),
        zorder=2,
    )
    ax.bar(
        x + bar_width,
        pred_wrong_counts,
        width=bar_width,
        color=style.get("pred_wrong_color", "#D1495B"),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("pred_wrong_label", "Predicted Wrong"),
        zorder=2,
    )

    ax.set_xlabel(style.get("xlabel", "Class"))
    ax.set_ylabel(style.get("ylabel", "Number of samples"))
    title = style.get("title")
    if title:
        ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], fontsize=float(style.get("xtick_font_size", 8.5)))
    if style.get("xtick_rotation", 0):
        plt.setp(ax.get_xticklabels(), rotation=float(style["xtick_rotation"]), ha=style.get("xtick_ha", "right"))
    ax.tick_params(axis="y", labelsize=float(style.get("ytick_font_size", 9.0)))
    ax.set_xlim(-0.75, len(labels) - 0.25)

    if "min_y" in style or "max_y" in style:
        max_count = float(
            max(
                int(true_counts.max()) if true_counts.size else 0,
                int(pred_right_counts.max()) if pred_right_counts.size else 0,
                int(pred_wrong_counts.max()) if pred_wrong_counts.size else 0,
            )
        )
        lower = float(style.get("min_y", 0.0))
        upper = float(style.get("max_y", max_count * 1.08 if max_count > 0.0 else 1.0))
        ax.set_ylim(lower, upper)

    apply_standard_axis_style(ax)
    ax.legend(
        loc=style.get("legend_loc", "upper right"),
        fontsize=float(style.get("legend_font_size", style.get("base_font_size", 10.5) - 0.5)),
        ncol=int(style.get("legend_ncol", 1)),
    )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
