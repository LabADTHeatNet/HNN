#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

_MPL_CONFIG_DIR = (Path.cwd() / ".tmp" / "matplotlib").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

from src.visualization.class_distribution import (
    compute_class_distribution,
    compute_class_distribution_split,
    plot_class_distribution,
    plot_class_distribution_split,
)
from src.visualization.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
from src.visualization.distance_confusion import compute_distance_confusion, plot_distance_confusion
from src.visualization.metric_distance_scatter import compute_metric_distance_points, plot_metric_distance_scatter
from src.visualization.network_examples import render_combined_network_figure, render_subnet_figure
from src.visualization.segment_structure_histogram import compute_segment_structure, plot_segment_structure_histogram
from src.visualization.training_curves import plot_scalar_curve, plot_scalar_curves


TRAINING_STYLE_KEYS = {
    "base_font_size",
    "best_label",
    "best_marker_color",
    "best_marker_size",
    "best_text_color",
    "best_text_font_size",
    "best_text_offset",
    "best_text_template",
    "color",
    "figsize",
    "legend_font_size",
    "legend_loc",
    "legend_ncol",
    "line_width",
    "max_x",
    "max_y",
    "min_x",
    "min_y",
    "title",
    "xlabel",
    "xlim",
    "ylim",
}

TRAINING_SERIES_KEYS = {
    "best_label",
    "best_marker_color",
    "best_marker_size",
    "best_text_color",
    "best_text_font_size",
    "best_text_offset",
    "best_text_template",
    "color",
    "label",
    "line_width",
    "tag",
}

DISTANCE_CONFUSION_STYLE_KEYS = {
    "annotation_font_size",
    "annotation_text_color_high",
    "annotation_text_color_low",
    "aspect",
    "base_font_size",
    "cmap_colors",
    "cmap_name",
    "colorbar_fraction",
    "colorbar_label",
    "colorbar_pad",
    "colorbar_rotation",
    "colorbar_tick_font_size",
    "figsize",
    "figsize_transposed",
    "grid_color",
    "grid_linewidth",
    "title",
    "title_transposed",
    "xlabel",
    "xlabel_transposed",
    "xtick_font_size",
    "xtick_ha",
    "xtick_rotation",
    "ylabel",
    "ylabel_transposed",
    "ytick_font_size",
}

CONFUSION_MATRIX_STYLE_KEYS = {
    "annotation_font_size",
    "annotation_text_color_high",
    "annotation_text_color_low",
    "aspect",
    "base_font_size",
    "cmap_colors",
    "cmap_name",
    "colorbar_fraction",
    "colorbar_label",
    "colorbar_pad",
    "colorbar_rotation",
    "colorbar_tick_font_size",
    "figsize",
    "grid_color",
    "grid_linewidth",
    "title",
    "xlabel",
    "xtick_font_size",
    "xtick_ha",
    "xtick_rotation",
    "ylabel",
    "ytick_font_size",
}

NETWORK_NODE_STYLE_KEYS = {"color", "label", "marker", "size"}
NETWORK_NODE_KEYS = {"consumer", "intermediate", "junction", "sink", "source"}
NETWORK_STYLE_KEYS = {
    "base_font_size",
    "consumer_line_width",
    "draw_consumer_edges_on",
    "figsize",
    "figure_margins",
    "fixed_bounds",
    "highlight_color",
    "highlight_line_width",
    "invert_y",
    "legend_bbox_to_anchor",
    "legend_columnspacing",
    "legend_font_size",
    "legend_handletextpad",
    "legend_label_return",
    "legend_label_supply",
    "legend_loc",
    "legend_ncol",
    "legend_single_row",
    "min_pad_x",
    "min_pad_y",
    "node_styles",
    "pad_ratio",
    "pipe_color_bwd",
    "pipe_color_consumer",
    "pipe_color_fwd",
    "pipe_line_width",
    "section_label_color",
    "section_label_font_size",
    "use_shared_bounds",
}
NETWORK_FIGURE_MARGIN_KEYS = {"bottom", "left", "right", "top"}

CLASS_DISTRIBUTION_STYLE_KEYS = {
    "bar_edge_color",
    "bar_edge_width",
    "bar_width",
    "base_font_size",
    "figsize",
    "legend_font_size",
    "legend_loc",
    "legend_ncol",
    "max_y",
    "min_y",
    "pred_color",
    "pred_label",
    "pred_right_color",
    "pred_right_label",
    "pred_wrong_color",
    "pred_wrong_label",
    "title",
    "true_color",
    "true_label",
    "xlabel",
    "xtick_font_size",
    "xtick_ha",
    "xtick_rotation",
    "ylabel",
    "ytick_font_size",
}

METRIC_DISTANCE_SCATTER_STYLE_KEYS = {
    "base_font_size",
    "correct_color",
    "correct_label",
    "draw_mean_line",
    "figsize",
    "incorrect_color",
    "incorrect_label",
    "incorrect_linewidth",
    "incorrect_marker",
    "incorrect_point_alpha",
    "incorrect_point_size",
    "jitter_width",
    "legend_font_size",
    "legend_loc",
    "legend_ncol",
    "max_y",
    "mean_bin_half_width",
    "mean_color",
    "mean_label",
    "mean_line_alpha",
    "mean_line_color",
    "mean_line_width",
    "min_y",
    "correct_point_size",
    "point_alpha",
    "point_size",
    "title",
    "xlabel",
    "xtick_font_size",
    "xtick_ha",
    "xtick_rotation",
    "ylabel",
    "ytick_font_size",
}

SEGMENT_STRUCTURE_STYLE_KEYS = {
    "bar_edge_color",
    "bar_edge_width",
    "bar_width",
    "base_font_size",
    "count_color",
    "count_label",
    "count_ylabel",
    "figsize",
    "legend_font_size",
    "legend_loc",
    "legend_ncol",
    "length_color",
    "length_label",
    "length_ylabel",
    "max_y_left",
    "max_y_right",
    "min_y_left",
    "min_y_right",
    "title",
    "xlabel",
    "xtick_font_size",
    "xtick_ha",
    "xtick_rotation",
    "ytick_font_size",
}


def _deep_merge_dicts(base: dict | None, override: dict | None) -> dict:
    result = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _format_unknown_keys(path: str, unknown_keys: Iterable[str]) -> str:
    keys_text = ", ".join(sorted(unknown_keys))
    return f"Unused/unknown style keys at {path}: {keys_text}"


def _validate_style_dict(style: dict | None, allowed_keys: set[str], path: str, warnings: list[str]) -> None:
    if not isinstance(style, dict):
        return
    unknown = set(style.keys()) - allowed_keys
    if unknown:
        warnings.append(_format_unknown_keys(path, unknown))


def _validate_network_style(style: dict | None, path: str, warnings: list[str]) -> None:
    if not isinstance(style, dict):
        return
    _validate_style_dict(style, NETWORK_STYLE_KEYS, path, warnings)
    node_styles = style.get("node_styles")
    if isinstance(node_styles, dict):
        unknown_nodes = set(node_styles.keys()) - NETWORK_NODE_KEYS
        if unknown_nodes:
            warnings.append(_format_unknown_keys(f"{path}.node_styles", unknown_nodes))
        for node_key, node_style in node_styles.items():
            if node_key not in NETWORK_NODE_KEYS or not isinstance(node_style, dict):
                continue
            _validate_style_dict(node_style, NETWORK_NODE_STYLE_KEYS, f"{path}.node_styles.{node_key}", warnings)
    figure_margins = style.get("figure_margins")
    if isinstance(figure_margins, dict):
        _validate_style_dict(figure_margins, NETWORK_FIGURE_MARGIN_KEYS, f"{path}.figure_margins", warnings)
    fixed_bounds = style.get("fixed_bounds")
    if isinstance(fixed_bounds, dict):
        _validate_style_dict(fixed_bounds, {"min_x", "max_x", "min_y", "max_y"}, f"{path}.fixed_bounds", warnings)


def _validate_visualization_config(config: dict) -> list[str]:
    warnings: list[str] = []

    training_config = config.get("training_curves", {})
    _validate_style_dict(training_config.get("style"), TRAINING_STYLE_KEYS, "training_curves.style", warnings)
    for idx, figure in enumerate(training_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_style_dict(figure.get("style"), TRAINING_STYLE_KEYS, f"training_curves.figures[{figure_name}].style", warnings)
        for series_idx, series in enumerate(figure.get("series", [])):
            _validate_style_dict(
                series,
                TRAINING_SERIES_KEYS,
                f"training_curves.figures[{figure_name}].series[{series_idx}]",
                warnings,
            )

    distance_config = config.get("distance_confusion", {})
    _validate_style_dict(distance_config.get("style"), DISTANCE_CONFUSION_STYLE_KEYS, "distance_confusion.style", warnings)
    _validate_style_dict(
        distance_config.get("transposed_style"),
        DISTANCE_CONFUSION_STYLE_KEYS,
        "distance_confusion.transposed_style",
        warnings,
    )
    for idx, figure in enumerate(distance_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_style_dict(figure.get("style"), DISTANCE_CONFUSION_STYLE_KEYS, f"distance_confusion.figures[{figure_name}].style", warnings)
        _validate_style_dict(
            figure.get("transposed_style"),
            DISTANCE_CONFUSION_STYLE_KEYS,
            f"distance_confusion.figures[{figure_name}].transposed_style",
            warnings,
        )

    confusion_config = config.get("confusion_matrix", {})
    _validate_style_dict(confusion_config.get("style"), CONFUSION_MATRIX_STYLE_KEYS, "confusion_matrix.style", warnings)
    for idx, figure in enumerate(confusion_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_style_dict(
            figure.get("style"),
            CONFUSION_MATRIX_STYLE_KEYS,
            f"confusion_matrix.figures[{figure_name}].style",
            warnings,
        )

    network_config = config.get("network_examples", {})
    _validate_network_style(network_config.get("style"), "network_examples.style", warnings)
    for idx, figure in enumerate(network_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_network_style(figure.get("style"), f"network_examples.figures[{figure_name}].style", warnings)

    class_dist_config = config.get("class_distribution", {})
    _validate_style_dict(class_dist_config.get("style"), CLASS_DISTRIBUTION_STYLE_KEYS, "class_distribution.style", warnings)
    for idx, figure in enumerate(class_dist_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_style_dict(
            figure.get("style"),
            CLASS_DISTRIBUTION_STYLE_KEYS,
            f"class_distribution.figures[{figure_name}].style",
            warnings,
        )

    metric_dist_config = config.get("metric_distance_scatter", {})
    _validate_style_dict(
        metric_dist_config.get("style"),
        METRIC_DISTANCE_SCATTER_STYLE_KEYS,
        "metric_distance_scatter.style",
        warnings,
    )
    for idx, figure in enumerate(metric_dist_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_style_dict(
            figure.get("style"),
            METRIC_DISTANCE_SCATTER_STYLE_KEYS,
            f"metric_distance_scatter.figures[{figure_name}].style",
            warnings,
        )

    segment_structure_config = config.get("segment_structure_histogram", {})
    _validate_style_dict(
        segment_structure_config.get("style"),
        SEGMENT_STRUCTURE_STYLE_KEYS,
        "segment_structure_histogram.style",
        warnings,
    )
    for idx, figure in enumerate(segment_structure_config.get("figures", [])):
        figure_name = figure.get("file", f"figure[{idx}]")
        _validate_style_dict(
            figure.get("style"),
            SEGMENT_STRUCTURE_STYLE_KEYS,
            f"segment_structure_histogram.figures[{figure_name}].style",
            warnings,
        )

    return warnings


def _compute_network_shared_bounds(config_dir: Path, config: dict) -> dict[str, float] | None:
    common_style = config.get("style", {})
    if not bool(common_style.get("use_shared_bounds", True)):
        return None
    if common_style.get("fixed_bounds"):
        return common_style["fixed_bounds"]

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for spec in config.get("figures", []):
        node_keys: list[str]
        if spec["kind"] == "subnet":
            node_keys = ["nodes_csv"]
        elif spec["kind"] == "combined":
            node_keys = ["fwd_nodes_csv", "bwd_nodes_csv"]
        else:
            raise ValueError(f"Unsupported network figure kind: {spec['kind']}")

        for key in node_keys:
            nodes_path = _resolve_path(config_dir, spec[key])
            nodes_df = pd.read_csv(nodes_path, usecols=["pos_x", "pos_y"])
            min_x = min(min_x, float(nodes_df["pos_x"].min()))
            max_x = max(max_x, float(nodes_df["pos_x"].max()))
            min_y = min(min_y, float(nodes_df["pos_y"].min()))
            max_y = max(max_y, float(nodes_df["pos_y"].max()))

    if min_x == float("inf"):
        return None
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}


def _render_network_figures(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    common_style = config.get("style", {})
    shared_bounds = _compute_network_shared_bounds(config_dir, config)
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        kind = spec["kind"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        if shared_bounds is not None and "fixed_bounds" not in figure_style:
            figure_style["fixed_bounds"] = dict(shared_bounds)
        figure_dpi = int(spec.get("dpi", dpi))
        if kind == "subnet":
            render_subnet_figure(
                nodes_path=_resolve_path(config_dir, spec["nodes_csv"]),
                edges_path=_resolve_path(config_dir, spec["edges_csv"]),
                direction=spec["direction"],
                output_path=output_path,
                dpi=figure_dpi,
                show_section_labels=spec.get("show_section_labels", True),
                highlight_graph_label=bool(spec.get("highlight_graph_label", False)),
                style=figure_style,
            )
        elif kind == "combined":
            render_combined_network_figure(
                fwd_nodes_path=_resolve_path(config_dir, spec["fwd_nodes_csv"]),
                fwd_edges_path=_resolve_path(config_dir, spec["fwd_edges_csv"]),
                bwd_nodes_path=_resolve_path(config_dir, spec["bwd_nodes_csv"]),
                bwd_edges_path=_resolve_path(config_dir, spec["bwd_edges_csv"]),
                output_path=output_path,
                dpi=figure_dpi,
                show_section_labels=spec.get("show_section_labels", False),
                style=figure_style,
            )
        else:
            raise ValueError(f"Unsupported network figure kind: {kind}")
        generated.append(output_path)
    return generated


def _render_training_curves(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    event_file = _resolve_path(config_dir, config["event_file"])
    common_style = config.get("style", {})
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        figure_dpi = int(spec.get("dpi", dpi))
        if "series" in spec:
            plot_scalar_curves(
                event_file=event_file,
                series=spec["series"],
                output_path=output_path,
                ylabel=spec.get("ylabel", figure_style.get("ylabel", "")),
                dpi=figure_dpi,
                accuracy_axis=bool(spec.get("accuracy_axis", False)),
                annotate_best=bool(spec.get("annotate_best", False)),
                style=figure_style,
            )
        else:
            plot_scalar_curve(
                event_file=event_file,
                tag=spec["tag"],
                output_path=output_path,
                ylabel=spec.get("ylabel", figure_style.get("ylabel", "")),
                color=spec.get("color", figure_style.get("color")),
                dpi=figure_dpi,
                accuracy_axis=bool(spec.get("accuracy_axis", False)),
                annotate_best=bool(spec.get("annotate_best", False)),
                style=figure_style,
            )
        generated.append(output_path)
    return generated


def _render_distance_confusions(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    sections_distances_csv = _resolve_path(config_dir, config["sections_distances_csv"])
    common_style = config.get("style", {})
    common_transposed_style = config.get("transposed_style", {})
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        figure_dpi = int(spec.get("dpi", dpi))
        matrix, fault_classes, distances = compute_distance_confusion(
            results_dir=_resolve_path(config_dir, spec["results_dir"]),
            sections_distances_csv=sections_distances_csv,
            max_distance=spec.get("max_distance"),
            exclude_classes=spec.get("exclude_classes"),
            auto_exclude_max_label=bool(spec.get("auto_exclude_max_label", True)),
        )
        plot_distance_confusion(
            matrix=matrix,
            fault_classes=fault_classes,
            distances=distances,
            output_path=output_path,
            dpi=figure_dpi,
            annotation_threshold=float(spec.get("annotation_threshold", 0.01)),
            style=figure_style,
            transpose=False,
        )
        generated.append(output_path)
        if bool(spec.get("save_transposed", False)):
            transposed_name = spec.get("transposed_file")
            if transposed_name is None:
                transposed_name = f"{Path(spec['file']).stem}_transposed{Path(spec['file']).suffix}"
            transposed_output_path = output_dir / transposed_name
            transposed_style = _deep_merge_dicts(
                figure_style,
                _deep_merge_dicts(common_transposed_style, spec.get("transposed_style", {})),
            )
            plot_distance_confusion(
                matrix=matrix,
                fault_classes=fault_classes,
                distances=distances,
                output_path=transposed_output_path,
                dpi=figure_dpi,
                annotation_threshold=float(spec.get("annotation_threshold", 0.01)),
                style=transposed_style,
                transpose=True,
            )
            generated.append(transposed_output_path)
    return generated


def _render_confusion_matrices(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    common_style = config.get("style", {})
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        figure_dpi = int(spec.get("dpi", dpi))
        matrix, labels = compute_confusion_matrix(
            results_dir=_resolve_path(config_dir, spec["results_dir"]),
            exclude_classes=spec.get("exclude_classes"),
            auto_exclude_max_label=bool(spec.get("auto_exclude_max_label", True)),
            normalize_rows=bool(spec.get("normalize_rows", False)),
        )
        plot_confusion_matrix(
            matrix=matrix,
            labels=labels,
            output_path=output_path,
            dpi=figure_dpi,
            style=figure_style,
            normalize_rows=bool(spec.get("normalize_rows", False)),
            annotation_threshold=float(spec.get("annotation_threshold", 0.5)),
        )
        generated.append(output_path)
    return generated


def _render_class_distributions(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    common_style = config.get("style", {})
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        figure_dpi = int(spec.get("dpi", dpi))
        if bool(spec.get("split_predicted", False)):
            labels, true_counts, pred_right_counts, pred_wrong_counts = compute_class_distribution_split(
                results_dir=_resolve_path(config_dir, spec["results_dir"]),
                exclude_classes=spec.get("exclude_classes"),
                auto_exclude_max_label=bool(spec.get("auto_exclude_max_label", True)),
            )
            plot_class_distribution_split(
                labels=labels,
                true_counts=true_counts,
                pred_right_counts=pred_right_counts,
                pred_wrong_counts=pred_wrong_counts,
                output_path=output_path,
                dpi=figure_dpi,
                style=figure_style,
            )
        else:
            labels, true_counts, pred_counts = compute_class_distribution(
                results_dir=_resolve_path(config_dir, spec["results_dir"]),
                exclude_classes=spec.get("exclude_classes"),
                auto_exclude_max_label=bool(spec.get("auto_exclude_max_label", True)),
            )
            plot_class_distribution(
                labels=labels,
                true_counts=true_counts,
                pred_counts=pred_counts,
                output_path=output_path,
                dpi=figure_dpi,
                style=figure_style,
            )
        generated.append(output_path)
    return generated


def _render_segment_structure_histograms(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    common_style = config.get("style", {})
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        figure_dpi = int(spec.get("dpi", dpi))
        labels: list[int] | None = None
        if spec.get("results_dir"):
            labels, _, _ = compute_class_distribution(
                results_dir=_resolve_path(config_dir, spec["results_dir"]),
                exclude_classes=spec.get("exclude_classes"),
                auto_exclude_max_label=bool(spec.get("auto_exclude_max_label", True)),
            )
        labels, pipe_counts, total_lengths = compute_segment_structure(
            nodes_path=_resolve_path(config_dir, spec["nodes_csv"]),
            edges_path=_resolve_path(config_dir, spec["edges_csv"]),
            direction=spec["direction"],
            labels=labels,
        )
        plot_segment_structure_histogram(
            labels=labels,
            pipe_counts=pipe_counts,
            total_lengths=total_lengths,
            output_path=output_path,
            dpi=figure_dpi,
            style=figure_style,
        )
        generated.append(output_path)
    return generated


def _render_metric_distance_scatters(config_dir: Path, output_dir: Path, config: dict, dpi: int) -> list[Path]:
    generated: list[Path] = []
    common_style = config.get("style", {})
    full_graph_root = _resolve_path(config_dir, config["full_graph_root"])
    for spec in config.get("figures", []):
        output_path = output_dir / spec["file"]
        figure_style = _deep_merge_dicts(common_style, spec.get("style", {}))
        figure_dpi = int(spec.get("dpi", dpi))
        points_df, classes = compute_metric_distance_points(
            results_dir=_resolve_path(config_dir, spec["results_dir"]),
            full_graph_root=full_graph_root,
            direction=spec["direction"],
            exclude_classes=spec.get("exclude_classes"),
            auto_exclude_max_label=bool(spec.get("auto_exclude_max_label", True)),
        )
        plot_metric_distance_scatter(
            points_df=points_df,
            classes=classes,
            output_path=output_path,
            dpi=figure_dpi,
            style=figure_style,
        )
        generated.append(output_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare publication visualizations from saved artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("prepare_visualziations.json"),
        help="Path to JSON configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = _load_config(config_path)
    validation_warnings = _validate_visualization_config(config)
    for warning in validation_warnings:
        print(f"[config warning] {warning}")
    dpi = int(config.get("style", {}).get("dpi", 220))
    output_dir = _resolve_path(config_dir, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    generated_paths.extend(_render_network_figures(config_dir, output_dir, config.get("network_examples", {}), dpi))
    generated_paths.extend(_render_training_curves(config_dir, output_dir, config.get("training_curves", {}), dpi))
    generated_paths.extend(_render_distance_confusions(config_dir, output_dir, config.get("distance_confusion", {}), dpi))
    generated_paths.extend(_render_confusion_matrices(config_dir, output_dir, config.get("confusion_matrix", {}), dpi))
    generated_paths.extend(_render_class_distributions(config_dir, output_dir, config.get("class_distribution", {}), dpi))
    generated_paths.extend(_render_segment_structure_histograms(config_dir, output_dir, config.get("segment_structure_histogram", {}), dpi))
    generated_paths.extend(_render_metric_distance_scatters(config_dir, output_dir, config.get("metric_distance_scatter", {}), dpi))

    print(f"Generated {len(generated_paths)} figures into {output_dir}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
