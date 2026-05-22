from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .article_style import PALETTE, apply_article_style, apply_standard_axis_style, save_figure
from .network_examples import load_graph_tables


def compute_segment_structure(
    nodes_path: Path,
    edges_path: Path,
    direction: str,
    labels: list[int] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    _, edges_df = load_graph_tables(nodes_path, edges_path, direction=direction)
    grouped = edges_df.groupby("id_section", sort=True)

    if labels is None:
        labels = sorted(int(section_id) for section_id in edges_df["id_section"].astype(int).unique())

    pipe_counts = np.array(
        [int(len(grouped.get_group(label))) if label in grouped.groups else 0 for label in labels],
        dtype=float,
    )
    total_lengths = np.array(
        [float(grouped.get_group(label)["l"].sum()) if label in grouped.groups else 0.0 for label in labels],
        dtype=float,
    )
    return labels, pipe_counts, total_lengths


def plot_segment_structure_histogram(
    labels: list[int],
    pipe_counts: np.ndarray,
    total_lengths: np.ndarray,
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
        fig_size = (fig_width, 4.8)

    fig, ax_left = plt.subplots(figsize=fig_size)
    ax_right = ax_left.twinx()

    x = np.arange(len(labels), dtype=float)
    bar_width = float(style.get("bar_width", 0.38))

    left_bars = ax_left.bar(
        x - bar_width / 2.0,
        pipe_counts,
        width=bar_width,
        color=style.get("count_color", "#AAB6C3"),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("count_label", "Number of pipes"),
        zorder=2,
    )
    right_bars = ax_right.bar(
        x + bar_width / 2.0,
        total_lengths,
        width=bar_width,
        color=style.get("length_color", "#C9B89B"),
        edgecolor=style.get("bar_edge_color", PALETTE["axis"]),
        linewidth=float(style.get("bar_edge_width", 0.6)),
        label=style.get("length_label", "Total length"),
        zorder=2,
    )

    ax_left.set_xlabel(style.get("xlabel", "Section"))
    ax_left.set_ylabel(style.get("count_ylabel", "Number of pipes"))
    ax_right.set_ylabel(style.get("length_ylabel", "Total length, m"))
    title = style.get("title")
    if title:
        ax_left.set_title(title)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([str(label) for label in labels], fontsize=float(style.get("xtick_font_size", 8.0)))
    if style.get("xtick_rotation", 0):
        plt.setp(ax_left.get_xticklabels(), rotation=float(style["xtick_rotation"]), ha=style.get("xtick_ha", "center"))
    ax_left.tick_params(axis="y", labelsize=float(style.get("ytick_font_size", 9.0)))
    ax_right.tick_params(axis="y", labelsize=float(style.get("ytick_font_size", 9.0)))
    ax_left.set_xlim(-0.75, len(labels) - 0.25)

    if "min_y_left" in style or "max_y_left" in style:
        lower = float(style.get("min_y_left", 0.0))
        upper = float(style.get("max_y_left", float(pipe_counts.max()) * 1.08 if pipe_counts.size else 1.0))
        ax_left.set_ylim(lower, upper)
    if "min_y_right" in style or "max_y_right" in style:
        lower = float(style.get("min_y_right", 0.0))
        upper = float(style.get("max_y_right", float(total_lengths.max()) * 1.08 if total_lengths.size else 1.0))
        ax_right.set_ylim(lower, upper)

    apply_standard_axis_style(ax_left)
    ax_right.grid(False)
    ax_right.spines["right"].set_color(PALETTE["axis"])
    ax_right.spines["top"].set_visible(False)
    ax_right.yaxis.label.set_color(PALETTE["text"])
    ax_right.tick_params(axis="y", colors=PALETTE["text"])

    handles = [left_bars, right_bars]
    labels_legend = [style.get("count_label", "Number of pipes"), style.get("length_label", "Total length")]
    ax_left.legend(
        handles,
        labels_legend,
        loc=style.get("legend_loc", "upper right"),
        fontsize=float(style.get("legend_font_size", style.get("base_font_size", 10.5) - 0.5)),
        ncol=int(style.get("legend_ncol", 2)),
    )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
