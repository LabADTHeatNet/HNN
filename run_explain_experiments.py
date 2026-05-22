#!/usr/bin/env python3
"""
Run the selected explain-analysis pipeline and generate a markdown report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from explain_analysis_utils import PRIMARY_EXPLANATION_METHOD
from explain_visualization import METHOD_DISPLAY_NAMES


METHOD_NOTES = {
    "notebook_object_log_probs": {
        "idea": "Подход из explain.ipynb: GNNExplainer, object-mask, log_probs, wrapper через Batch.from_data_list([data]).",
        "assessment": "Работает стабильно, но локализует хуже лучшего margin-варианта.",
    },
    "notebook_common_attributes_log_probs": {
        "idea": "Подход из explain.ipynb с общей feature-mask для всех узлов.",
        "assessment": "Слишком глобальный, хуже локализует конкретный сегмент.",
    },
    "gnn_attr_raw": {
        "idea": "GNNExplainer с feature-mask по узлам и raw logits.",
        "assessment": "Не дает явного выигрыша по динамическим признакам; локализация слабее.",
    },
    "gnn_object_log_probs": {
        "idea": "GNNExplainer с object-mask по узлам и log_probs.",
        "assessment": "Чище, чем feature-mask, но хуже margin-адаптера.",
    },
    "gnn_common_attributes_log_probs": {
        "idea": "GNNExplainer с общей feature-mask по всем узлам.",
        "assessment": "Маска получается слишком усредненной и плохо указывает на локальный дефект.",
    },
    "gnn_attr_margin": {
        "idea": "GNNExplainer с feature-mask и margin между классом и лучшей альтернативой.",
        "assessment": "Margin помогает, но feature-mask все равно остается размытой.",
    },
    "gnn_object_margin": {
        "idea": "GNNExplainer с object-mask и margin между выбранным классом и лучшей альтернативой.",
        "assessment": "Лучший общий баланс локализации и стабильности; выбран как основной.",
    },
    "gnn_attr_vs_no_defect": {
        "idea": "GNNExplainer с feature-mask и контрастом к классу no_defect.",
        "assessment": "Интересный контрастный вариант, но слабее по общей локализации.",
    },
    "gnn_object_vs_no_defect": {
        "idea": "GNNExplainer с object-mask и контрастом к классу no_defect.",
        "assessment": "Сильный запасной вариант, но в среднем хуже по distance, чем margin.",
    },
    "attention_explainer": {
        "idea": "AttentionExplainer по attention-коэффициентам message-passing слоев.",
        "assessment": "Видит только GAT-часть node encoder и не покрывает custom edge-attention блок модели.",
    },
    "graphmask_object": {
        "idea": "GraphMaskExplainer с object-mask.",
        "assessment": "Существенно тяжелее и не дает лучшей локализации.",
    },
    "pg_explainer": {
        "idea": "PGExplainer после дообучения на выбранных примерах.",
        "assessment": "Иногда хорошо находит окрестность, но маски получаются слишком диффузными.",
    },
    "activation_attention": {
        "idea": "Простой baseline по внутренним активациям и attention-весам.",
        "assessment": "Быстрый sanity-check, но это не полноценный PyG explainer.",
    },
    "grad_x_input": {
        "idea": "Baseline по |grad * input| на узлах и ребрах.",
        "assessment": "Показывает чувствительность, но часто шумный и нестабильный.",
    },
    "section_occlusion": {
        "idea": "Baseline с занулением целых сегментов и измерением падения logit.",
        "assessment": "Полезен как причинный sanity-check, но грубый и дорогой.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("out_Termo_Ablation_heads/8_heads"),
        help="Experiment directory with model checkpoint",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=None,
        help="Where to store the selected-method explain analysis",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Directory with the PyG explainer benchmark results",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("explain_experiments.md"),
        help="Where to write the markdown report",
    )
    parser.add_argument(
        "--direction",
        choices=["fwd", "bwd", "both"],
        default="both",
        help="Which graph directions to analyze",
    )
    parser.add_argument(
        "--explainer-epochs",
        type=int,
        default=100,
        help="Optimization epochs for the selected PyG explainer",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=3,
        help="How many representative examples to explain for each requested category",
    )
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument("--top-edges", type=int, default=12)
    parser.add_argument("--top-sections", type=int, default=8)
    parser.add_argument("--top-features-per-node", type=int, default=4)
    parser.add_argument("--top-features-per-edge", type=int, default=3)
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only rebuild the report from existing outputs",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_selected_analysis(args: argparse.Namespace, analysis_output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "run_explain_analysis.py",
        "--output-dir",
        str(analysis_output_dir),
        "--direction",
        args.direction,
        "--explainer-epochs",
        str(args.explainer_epochs),
        "--examples-per-category",
        str(args.examples_per_category),
        "--top-nodes",
        str(args.top_nodes),
        "--top-edges",
        str(args.top_edges),
        "--top-sections",
        str(args.top_sections),
        "--top-features-per-node",
        str(args.top_features_per_node),
        "--top-features-per-edge",
        str(args.top_features_per_edge),
    ]
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_primary_method_rank_histogram(method_summary_df: pd.DataFrame, output_path: Path) -> None:
    filtered = method_summary_df.dropna(subset=["true_section_rank"]).copy()
    if filtered.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(filtered["true_section_rank"], bins=min(10, len(filtered)), color="#4c72b0", edgecolor="white")
    ax.set_title("Best method: distribution of true-section ranks")
    ax.set_xlabel("True-section rank (lower is better)")
    ax.set_ylabel("Number of explained examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_primary_method_rank_by_category(method_summary_df: pd.DataFrame, output_path: Path) -> None:
    if method_summary_df.empty or "requested_category" not in method_summary_df.columns:
        return
    filtered = method_summary_df.dropna(subset=["true_section_rank", "pred_section_rank"]).copy()
    if filtered.empty:
        return
    grouped = (
        filtered.groupby("requested_category")
        .agg(
            mean_true_section_rank=("true_section_rank", "mean"),
            mean_pred_section_rank=("pred_section_rank", "mean"),
            num_examples=("sample_id", "nunique"),
        )
        .reset_index()
    )
    if grouped.empty:
        return
    x = range(len(grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar([idx - width / 2 for idx in x], grouped["mean_true_section_rank"], width=width, color="#dc322f", label="True-section rank")
    ax.bar([idx + width / 2 for idx in x], grouped["mean_pred_section_rank"], width=width, color="#ff8c00", label="Pred-section rank")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["requested_category"])
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_title("Best method: rank by example category")
    ax.legend()
    for idx, row in grouped.iterrows():
        ax.text(idx, max(row["mean_true_section_rank"], row["mean_pred_section_rank"]) + 0.15, f"n={int(row['num_examples'])}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _relative(path: Path, start: Path) -> str:
    del start
    return str(path.resolve())


def build_method_table(benchmark_summary_df: pd.DataFrame) -> str:
    if benchmark_summary_df.empty:
        return "_Benchmark summary was not found._"

    rows = [
        "| Method | What was tried | Mean true rank | Mean top distance | Short conclusion |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in benchmark_summary_df.itertuples(index=False):
        key = str(row.explainer)
        note = METHOD_NOTES.get(key, {})
        idea = note.get("idea", key)
        assessment = note.get("assessment", "")
        rows.append(
            "| "
            + " | ".join(
                [
                    METHOD_DISPLAY_NAMES.get(key, key).replace("|", "\\|"),
                    idea.replace("|", "\\|"),
                    f"{float(row.mean_true_section_rank):.2f}" if pd.notna(row.mean_true_section_rank) else "-",
                    f"{float(row.mean_top_section_distance):.2f}" if pd.notna(row.mean_top_section_distance) else "-",
                    assessment.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def build_category_rank_table(category_rank_df: pd.DataFrame) -> str:
    if category_rank_df.empty:
        return "_No category rank table._"
    rows = [
        "| Category | Mean true rank | Mean pred rank | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in category_rank_df.itertuples(index=False):
        rows.append(
            f"| {row.requested_category} | {float(row.mean_true_section_rank):.2f} | "
            f"{float(row.mean_pred_section_rank):.2f} | {int(row.num_examples)} |"
        )
    return "\n".join(rows)


def build_example_list(manifest_df: pd.DataFrame, analysis_output_dir: Path) -> str:
    if manifest_df.empty:
        return "_Representative examples were not generated._"

    lines = []
    for category, subset in manifest_df.groupby("requested_category", sort=False):
        lines.append(f"**{category}**")
        for row in subset.head(2).itertuples(index=False):
            example_dir = Path(row.example_dir)
            image_path = example_dir / "explanation_overview.png"
            rel = _relative(image_path, Path.cwd())
            lines.append(
                f"- `true={int(row.true_class)}`, `pred={int(row.pred_class)}`, `dir={row.direction}`, "
                f"`rank_true={row.true_section_importance_rank}`: [{example_dir.name}]({rel})"
            )
    return "\n".join(lines)


def build_feature_takeaways(node_feature_summary_df: pd.DataFrame) -> tuple[str, str]:
    if node_feature_summary_df.empty:
        return "No node feature ablation summary available.", "No dynamic/static comparison available."

    feature_rank = (
        node_feature_summary_df.groupby("feature_name")["abs_delta_logit"]
        .mean()
        .sort_values(ascending=False)
        .head(6)
    )
    feature_text = ", ".join(
        f"`{feature}` ({value:.3f})"
        for feature, value in feature_rank.items()
    )

    group_rank = (
        node_feature_summary_df.groupby("feature_group")["abs_delta_logit"]
        .mean()
        .sort_values(ascending=False)
    )
    group_text = ", ".join(
        f"`{group}` ({value:.3f})"
        for group, value in group_rank.items()
    )
    return feature_text, group_text


def build_report(
    analysis_output_dir: Path,
    benchmark_dir: Path,
    report_path: Path,
) -> None:
    report_assets_dir = ensure_dir(analysis_output_dir / "report_assets")

    summary_overall = read_json(analysis_output_dir / "summary_overall.json")
    selected_examples_df = read_csv(analysis_output_dir / "selected_examples.csv")
    requested_summary_df = read_csv(analysis_output_dir / "requested_category_summary.csv")
    analysis_summary_df = read_csv(analysis_output_dir / "analysis_category_summary.csv")

    primary_dir = analysis_output_dir / "methods" / PRIMARY_EXPLANATION_METHOD
    primary_manifest_df = read_csv(primary_dir / "explained_examples_manifest.csv")
    primary_method_summary_df = read_csv(primary_dir / "explained_method_summary.csv")
    node_feature_summary_df = read_csv(primary_dir / "explained_node_feature_summary.csv")

    benchmark_summary_path = benchmark_dir / "explainer_benchmark_summary.csv"
    benchmark_examples_path = benchmark_dir / "explainer_benchmark_examples.csv"
    benchmark_summary_df = read_csv(benchmark_summary_path) if benchmark_summary_path.exists() else pd.DataFrame()
    benchmark_examples_df = read_csv(benchmark_examples_path) if benchmark_examples_path.exists() else pd.DataFrame()

    plot_primary_method_rank_histogram(
        primary_method_summary_df,
        report_assets_dir / "primary_true_rank_histogram.png",
    )
    plot_primary_method_rank_by_category(
        primary_method_summary_df,
        report_assets_dir / "primary_rank_by_category.png",
    )

    top_features_text, feature_group_text = build_feature_takeaways(node_feature_summary_df)

    category_rank_df = (
        primary_method_summary_df.dropna(subset=["true_section_rank", "pred_section_rank"])
        .groupby("requested_category")
        .agg(
            mean_true_section_rank=("true_section_rank", "mean"),
            mean_pred_section_rank=("pred_section_rank", "mean"),
            num_examples=("sample_id", "nunique"),
        )
        .reset_index()
    )
    category_rank_table = build_category_rank_table(category_rank_df)
    category_rank_map = {
        str(row.requested_category): float(row.mean_true_section_rank)
        for row in category_rank_df.itertuples(index=False)
    }

    method_table = build_method_table(benchmark_summary_df.sort_values("mean_true_section_rank"))
    chosen_row = (
        benchmark_summary_df.loc[benchmark_summary_df["explainer"] == PRIMARY_EXPLANATION_METHOD].iloc[0]
        if not benchmark_summary_df.empty and PRIMARY_EXPLANATION_METHOD in benchmark_summary_df["explainer"].tolist()
        else None
    )
    runner_up_row = benchmark_summary_df.iloc[1] if len(benchmark_summary_df) > 1 else None

    requested_counts = requested_summary_df.groupby("requested_category")["count"].sum().to_dict()
    analysis_counts = analysis_summary_df.groupby("analysis_category")["count"].sum().to_dict()

    report = f"""# Explain Experiments

## Goal

Цель экспериментов: понять, почему модель выбирает конкретный сегмент как дефектный, и можно ли построить объяснение в форме:

> модель приняла решение по этим узлам и этим данным, потому что именно они толкают логит выбранного сегмента вверх, а альтернативные сегменты проигрывают.

Для этого были разделены две задачи:

1. найти метод, который лучше локализует область графа, действительно связанную с дефектом;
2. после локализации понять, какие признаки в этих узлах сильнее всего влияют на решение.

## Что было выбрано как основной explain-подход

Основным выбран метод **{METHOD_DISPLAY_NAMES.get(PRIMARY_EXPLANATION_METHOD, PRIMARY_EXPLANATION_METHOD)}**.

Идея метода:

- PyG `GNNExplainer` строит маску по **узлам** и **ребрам**;
- на выход модели подается не raw-logit и не log-probability, а **margin**:
  логит выбранного класса минус максимальный логит среди остальных классов;
- это заставляет explainer искать не просто сильный сигнал, а именно те части графа, которые делают выбранный сегмент лучше ближайшего конкурента.

Практически это оказалось лучше, чем feature-mask по узлам: сначала берется **топология важной области** через `object-mask`, затем по top-узлам делается `node_feature_ablation`, чтобы понять, какие именно признаки в них действительно двигают предсказание.

## Как выбирался лучший метод

Сначала был выполнен отдельный benchmark PyG explainers и adapter-вариантов на одинаковом наборе из 12 дефектных примеров:

- `correct`, `neighbor`, `farther`
- для `fwd` и `bwd`
- по 2 примера на группу

Критерии выбора:

1. `mean_true_section_rank`  
   Чем ниже, тем лучше: истинный сегмент должен находиться как можно выше в ранжировании explainer.
2. `mean_top_section_distance`  
   Чем ниже, тем лучше: если explainer ошибся, важно, чтобы его top-segment был хотя бы рядом с истинным.
3. `mean_near_section_mass_ratio`  
   Чем выше, тем лучше: доля важности должна попадать в истинный сегмент и его соседей.
4. Визуальная sanity-проверка  
   Маска не должна быть слишком диффузной или полностью глобальной.

По этим критериям лучший результат дал **{METHOD_DISPLAY_NAMES.get(PRIMARY_EXPLANATION_METHOD, PRIMARY_EXPLANATION_METHOD)}**:

"""

    if chosen_row is not None:
        report += (
            f"- `mean_true_section_rank = {float(chosen_row['mean_true_section_rank']):.2f}`\n"
            f"- `mean_top_section_distance = {float(chosen_row['mean_top_section_distance']):.2f}`\n"
            f"- `mean_near_section_mass_ratio = {float(chosen_row['mean_near_section_mass_ratio']):.4f}`\n"
        )
    if runner_up_row is not None:
        report += (
            f"- ближайший конкурент: **{METHOD_DISPLAY_NAMES.get(str(runner_up_row['explainer']), str(runner_up_row['explainer']))}** "
            f"с `mean_true_section_rank = {float(runner_up_row['mean_true_section_rank']):.2f}` и "
            f"`mean_top_section_distance = {float(runner_up_row['mean_top_section_distance']):.2f}`\n"
        )

    report += f"""

Сводная таблица benchmark-методов:

{method_table}

Сводный график benchmark:

- [explainer_benchmark_summary.png]({_relative(benchmark_dir / "explainer_benchmark_summary.png", Path.cwd())})

## Какие методы были попробованы и почему часть из них не подошла

Короткий вывод по группам методов:

- `object-mask` оказался лучше `feature-mask`, потому что сначала нужно локализовать **где** модель смотрит в графе, а не сразу пытаться объяснять **какой из 9 node-features** важен.
- `margin` оказался лучше `log_probs` и `raw logits`, потому что он объясняет не абсолютную уверенность, а **почему выбран именно этот сегмент, а не ближайший конкурент**.
- `common_attributes` оказался слишком глобальным: он хуже локализует конкретный дефектный сегмент.
- `AttentionExplainer` ограничен тем, что видит только attention-coefficients `GATv2Conv` в node encoder и не объясняет custom edge-attention блок модели.
- `PGExplainer` иногда хорошо выделяет окрестность, но объяснение становится слишком размазанным по подграфу.

## Большой прогон выбранного метода

Большой прогон сделан скриптом `run_explain_analysis.py` с выбранным методом по умолчанию, а orchestration и сбор отчета делает `run_explain_experiments.py`.

Общие classification-метрики по defect-классам:

- accuracy: `{summary_overall['accuracy']:.4f}`
- macro-F1: `{summary_overall['macro_f1']:.4f}`
- weighted-F1: `{summary_overall['weighted_f1']:.4f}`
- defect samples: `{summary_overall['num_samples']}`
- no-defect samples (исключены из сегментной статистики): `{summary_overall['num_no_defect_samples']}`
- explained representative examples: `{len(selected_examples_df)}`

Распределение примеров по запрошенным категориям:

- `correct`: `{int(requested_counts.get('correct', 0))}`
- `neighbor`: `{int(requested_counts.get('neighbor', 0))}`
- `farther`: `{int(requested_counts.get('farther', 0))}`
- `no_defect`: `{int(requested_counts.get('no_defect', 0))}`

Дополнительная аналитика по типам ошибок:

- `correct_defect`: `{int(analysis_counts.get('correct_defect', 0))}`
- `neighbor_error`: `{int(analysis_counts.get('neighbor_error', 0))}`
- `far_error`: `{int(analysis_counts.get('far_error', 0))}`
- `missed_defect_no_prediction`: `{int(analysis_counts.get('missed_defect_no_prediction', 0))}`
- `correct_no_defect`: `{int(analysis_counts.get('correct_no_defect', 0))}`
- `false_positive_no_defect`: `{int(analysis_counts.get('false_positive_no_defect', 0))}`

Общие графики:

- [requested_categories.png]({_relative(analysis_output_dir / "requested_categories.png", Path.cwd())})
- [analysis_categories.png]({_relative(analysis_output_dir / "analysis_categories.png", Path.cwd())})
- [distance_histogram.png]({_relative(analysis_output_dir / "distance_histogram.png", Path.cwd())})
- [baseline_method_rank_summary.png]({_relative(analysis_output_dir / "method_comparison" / "baseline_method_rank_summary.png", Path.cwd())})

## Как выбранный метод объясняет решение

Главная идея интерпретации здесь такая:

1. `GNNExplainer | object/margin` показывает **какие узлы и трубы поднимают margin выбранного сегмента**.
2. После этого по top-узлам делается `node_feature_ablation`, чтобы проверить, какие признаки в этих узлах реально двигают логит вниз при занулении.
3. В итоге объяснение получается двухступенчатым:
   сначала **где** модель увидела сигнал, потом **какие данные в этой зоне** повлияли на ответ.

Сводка по rank выбранного метода на объясняемых примерах:

{category_rank_table}

Наблюдение по этим примерам:

- `neighbor` объясняется лучше всего: средний rank истинного сегмента `{category_rank_map.get('neighbor', float('nan')):.2f}`
- `farther` заметно сложнее: `{category_rank_map.get('farther', float('nan')):.2f}`
- `correct` оказался не самым простым случаем для explain: `{category_rank_map.get('correct', float('nan')):.2f}`

Это важный результат сам по себе: explainer может хорошо объяснять **локальную ошибку модели**, но хуже разбирать даже корректное предсказание, если решение основано на более распределенном глобальном сигнале.

Дополнительные summary-графики по выбранному методу:

- [primary_true_rank_histogram.png]({_relative(report_assets_dir / "primary_true_rank_histogram.png", Path.cwd())})
- [primary_rank_by_category.png]({_relative(report_assets_dir / "primary_rank_by_category.png", Path.cwd())})
- [node_feature_importance_summary.png]({_relative(primary_dir / "node_feature_importance_summary.png", Path.cwd())})
- [edge_feature_importance_summary.png]({_relative(primary_dir / "edge_feature_importance_summary.png", Path.cwd())})

По `node_feature_ablation` наиболее сильные признаки на выбранных top-узлах в среднем:

- {top_features_text}

Если агрегировать по группам, получается:

- {feature_group_text}

Это означает, что после локализации узлов уже можно обсуждать не только “какой участок графа важен”, но и “какие измерения внутри этой области двигают решение”.

## Что лучший метод описывает хорошо

- Хорошо показывает **локальную область графа**, в которой модель ищет дефект.
- Лучше других PyG-вариантов ранжирует истинный сегмент и чаще оставляет top importance рядом с ним.
- Лучше соответствует инженерной интерпретации “модель смотрит на близкие к дефекту узлы и связанные с ними трубы”.
- В комбинации с `node_feature_ablation` позволяет перейти от геометрии графа к признакам: температура, давление и связанные с ними отклонения на важных узлах.
- На практике особенно полезен на `neighbor`-ошибках: там explainer часто поднимает истинный сегмент или его ближайшую окрестность почти в самый верх ранга.

## Где он ошибается и почему это может происходить

Выбранный метод все еще ошибается в нескольких типичных случаях:

- Если модель сама перепутала соседние ветви, explainer обычно честно объясняет **ошибочное решение модели**, а не истинный дефект.
- В симметричных или гидравлически похожих частях сети модель может опираться на близкий по поведению сегмент; explainer в этом случае тоже остается в неправильной локальной области.
- Глобальный pooling по ребрам в архитектуре модели сжимает информацию в один graph-level логит, поэтому часть объяснения становится менее локальной, чем хотелось бы.
- Custom edge-attention блок и последующий attention-pooling делают архитектуру сильной как классификатор, но не максимально прозрачной для off-the-shelf explainers.
- На части `correct`-примеров explainer все еще дает высокий rank истинного сегмента; это похоже на признак того, что модель принимала решение по более распределенному паттерну, а не по компактному локальному подграфу.

Практический смысл этого ограничения:

> объяснение здесь отвечает на вопрос “что толкало модель к этому ответу”, а не на вопрос “какой физически истинный участок сети виноват”.

## Показательные примеры

Ниже несколько representative examples, по которым удобно смотреть совпадение prediction, explain и feature ablation:

{build_example_list(primary_manifest_df, analysis_output_dir)}

## Есть ли шанс получить explainer, который будет объяснять решение заметно лучше

Шанс есть, но, вероятно, уже не за счет простого переключения между стандартными PyG explainers.

Что показали текущие эксперименты:

- стандартные методы PyG уже были перебраны в нескольких режимах;
- лучший вариант нашелся не из “другого explainer-класса”, а из **правильной адаптации выхода модели** (`margin`) и правильного типа маски (`object`);
- это означает, что главная проблема была не только в алгоритме explain, а в постановке explain-задачи для конкретной модели.

Что стоит пробовать дальше:

1. **Model-specific causal explainer** поверх текущей архитектуры.  
   Самый перспективный вариант: объяснять не только входные узлы/ребра, но и внутренние `fused_edge_feat`, `attention_weights` и pooling.
2. **Counterfactual / perturbation-based subgraph search**.  
   Например, искать минимальный подграф или минимальный набор узлов, удаление которых сильнее всего рушит margin выбранного сегмента.
3. **Captum / Integrated Gradients / LayerConductance** после установки `captum`.  
   Это может дать более аккуратную feature-level attribution для узлов, чем стандартный GNNExplainer feature-mask.
4. **Изменение самой модели в сторону explainability**.  
   Если нужен действительно сильный инженерный explain, возможно стоит делать не graph-level class prediction сразу, а сначала section-level scoring / anomaly map, а потом уже выбирать сегмент.

Итоговый вывод:

- **для текущей модели лучший рабочий explain-пайплайн уже собран**:  
  `GNNExplainer | object/margin` + `node_feature_ablation` по top-узлам;
- **очень сильный explainer** все еще возможен, но, скорее всего, это будет уже **model-specific** решение, а не просто еще один стандартный класс из PyG.
"""

    report_path.write_text(report)
    (analysis_output_dir / "explain_experiments.md").write_text(report)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    analysis_output_dir = (
        args.analysis_output_dir.resolve()
        if args.analysis_output_dir is not None
        else (exp_dir / "explain_experiments" / "analysis").resolve()
    )
    benchmark_dir = (
        args.benchmark_dir.resolve()
        if args.benchmark_dir is not None
        else (exp_dir / "pyg_explainer_benchmark").resolve()
    )
    report_path = args.report_path.resolve()

    if not args.skip_analysis:
        ensure_dir(analysis_output_dir.parent)
        run_selected_analysis(args, analysis_output_dir)

    build_report(
        analysis_output_dir=analysis_output_dir,
        benchmark_dir=benchmark_dir,
        report_path=report_path,
    )
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
