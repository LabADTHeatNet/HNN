from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = (Path.cwd() / ".tmp" / "matplotlib").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker


PALETTE = {
    "supply": "#D1495B",
    "return": "#2B59C3",
    "consumer_edge": "#2F855A",
    "consumer_node": "#F28E2B",
    "source_node": "#C62828",
    "sink_node": "#2E7D32",
    "junction_node": "#8E6BBE",
    "intermediate_node": "#3A7D44",
    "grid": "#D7DCE5",
    "axis": "#273142",
    "text": "#1F2933",
    "highlight": "#2F855A",
    "heat_low": "#FFFFFF",
    "heat_mid": "#C9D8FF",
    "heat_high": "#2B59C3",
}


def apply_article_style(base_font_size: float = 11.0) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 1.0,
            "axes.labelsize": base_font_size,
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "text.color": PALETTE["text"],
            "legend.frameon": False,
            "legend.fontsize": base_font_size - 1.0,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.9,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 220, tight: bool = True) -> None:
    ensure_parent(output_path)
    save_kwargs = {"dpi": dpi}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def apply_standard_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])


def format_accuracy_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
