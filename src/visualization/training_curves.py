from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import numpy as np

from .article_style import PALETTE, apply_article_style, apply_standard_axis_style, format_accuracy_axis, save_figure


def load_scalar_series(event_file: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(f"Scalar tag '{tag}' was not found in {event_file}. Available: {available}")

    events = accumulator.Scalars(tag)
    epochs = np.array([event.step + 1 for event in events], dtype=float)
    values = np.array([event.value for event in events], dtype=float)
    return epochs, values


def _resolve_axis_limits(
    style: dict,
    default_limits: tuple[float, float],
    min_key: str,
    max_key: str,
    legacy_key: str,
) -> tuple[float, float]:
    if min_key in style or max_key in style:
        lower = float(style.get(min_key, default_limits[0]))
        upper = float(style.get(max_key, default_limits[1]))
        return lower, upper
    if legacy_key in style:
        lower, upper = style[legacy_key]
        return float(lower), float(upper)
    return default_limits


def _resolve_best_point(values: np.ndarray, accuracy_axis: bool) -> tuple[int, str]:
    if accuracy_axis:
        return int(np.argmax(values)), "Best"
    return int(np.argmin(values)), "Min"


def _apply_curve_axes(
    ax: plt.Axes,
    epochs: np.ndarray,
    all_values: list[np.ndarray],
    ylabel: str,
    accuracy_axis: bool,
    style: dict,
) -> None:
    ax.set_xlabel(style.get("xlabel", "Epoch"))
    ax.set_ylabel(ylabel)
    title = style.get("title")
    if title:
        ax.set_title(title)

    default_xlim = (float(epochs.min()), float(epochs.max()))
    ax.set_xlim(*_resolve_axis_limits(style, default_xlim, "min_x", "max_x", "xlim"))
    apply_standard_axis_style(ax)

    if accuracy_axis:
        all_concat = np.concatenate(all_values) if all_values else np.array([0.0, 1.0], dtype=float)
        lower = max(0.0, float(all_concat.min()) - 0.03)
        upper = min(1.0, float(all_concat.max()) + 0.03)
        ax.set_ylim(*_resolve_axis_limits(style, (lower, upper), "min_y", "max_y", "ylim"))
        format_accuracy_axis(ax)
    else:
        default_ylim = (float(min(values.min() for values in all_values)), float(max(values.max() for values in all_values)))
        ax.set_ylim(*_resolve_axis_limits(style, default_ylim, "min_y", "max_y", "ylim"))


def plot_scalar_curve(
    event_file: Path,
    tag: str,
    output_path: Path,
    ylabel: str,
    color: str,
    dpi: int = 220,
    accuracy_axis: bool = False,
    annotate_best: bool = False,
    style: dict | None = None,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 11.0)))
    epochs, values = load_scalar_series(event_file, tag)

    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", [6.0, 4.0])))
    ax.plot(
        epochs,
        values,
        color=color,
        linewidth=float(style.get("line_width", 2.6)),
    )
    _apply_curve_axes(ax, epochs, [values], ylabel, accuracy_axis, style)

    if annotate_best and len(values) > 0:
        best_idx, default_best_label = _resolve_best_point(values, accuracy_axis)
        best_label = style.get("best_label", default_best_label)
        best_epoch = int(epochs[best_idx])
        best_value = float(values[best_idx])
        ax.scatter(
            [best_epoch],
            [best_value],
            s=float(style.get("best_marker_size", 32)),
            color=style.get("best_marker_color", PALETTE["axis"]),
            zorder=3,
        )
        ax.annotate(
            style.get("best_text_template", "{label}: {value:.3f} @ {epoch}").format(
                label=best_label,
                value=best_value,
                epoch=best_epoch,
            ),
            xy=(best_epoch, best_value),
            xytext=tuple(style.get("best_text_offset", [8, 10])),
            textcoords="offset points",
            fontsize=float(style.get("best_text_font_size", 9)),
            color=style.get("best_text_color", PALETTE["axis"]),
        )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)


def plot_scalar_curves(
    event_file: Path,
    series: list[dict],
    output_path: Path,
    ylabel: str,
    dpi: int = 220,
    accuracy_axis: bool = False,
    annotate_best: bool = False,
    style: dict | None = None,
) -> None:
    style = style or {}
    apply_article_style(base_font_size=float(style.get("base_font_size", 11.0)))
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", [6.0, 4.0])))

    epochs_reference: np.ndarray | None = None
    all_values: list[np.ndarray] = []
    legend_enabled = False

    for series_spec in series:
        epochs, values = load_scalar_series(event_file, series_spec["tag"])
        if epochs_reference is None:
            epochs_reference = epochs
        elif len(epochs_reference) != len(epochs) or not np.array_equal(epochs_reference, epochs):
            raise ValueError("All combined training curves must use the same epoch steps")

        all_values.append(values)
        color = series_spec.get("color", style.get("color", PALETTE["axis"]))
        label = series_spec.get("label")
        if label:
            legend_enabled = True
        ax.plot(
            epochs,
            values,
            color=color,
            linewidth=float(series_spec.get("line_width", style.get("line_width", 2.6))),
            label=label,
        )

        if annotate_best and len(values) > 0:
            best_idx, default_best_label = _resolve_best_point(values, accuracy_axis)
            best_epoch = int(epochs[best_idx])
            best_value = float(values[best_idx])
            ax.scatter(
                [best_epoch],
                [best_value],
                s=float(series_spec.get("best_marker_size", style.get("best_marker_size", 32))),
                color=series_spec.get("best_marker_color", color),
                zorder=3,
            )
            ax.annotate(
                series_spec.get(
                    "best_text_template",
                    style.get("best_text_template", "{series}: {label} {value:.3f} @ {epoch}"),
                ).format(
                    series=label or series_spec["tag"],
                    label=series_spec.get("best_label", style.get("best_label", default_best_label)),
                    value=best_value,
                    epoch=best_epoch,
                ),
                xy=(best_epoch, best_value),
                xytext=tuple(series_spec.get("best_text_offset", style.get("best_text_offset", [8, 10]))),
                textcoords="offset points",
                fontsize=float(series_spec.get("best_text_font_size", style.get("best_text_font_size", 9))),
                color=series_spec.get("best_text_color", style.get("best_text_color", color)),
            )

    if epochs_reference is None:
        raise ValueError("Combined training curve requires at least one series")

    _apply_curve_axes(ax, epochs_reference, all_values, ylabel, accuracy_axis, style)

    if legend_enabled:
        ax.legend(
            loc=style.get("legend_loc", "best"),
            fontsize=float(style.get("legend_font_size", style.get("base_font_size", 11.0) - 1.0)),
            ncol=int(style.get("legend_ncol", 1)),
        )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
