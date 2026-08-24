#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("datasets/Termo_model")
OUTPUT_DIR = Path("reports")
CASES = [
    {"requested_tout": -35, "source_tout": -34},
    {"requested_tout": 10, "source_tout": 10},
]
NODE_GROUPS = [
    ("forward", "Прямой трубопровод без потребителей"),
    ("return", "Обратный трубопровод без потребителей"),
    ("consumer_inlet", "Потребители (вход)"),
    ("consumer_outlet", "Потребители (выход)"),
]


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _statistics(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="raise")
    minimum = float(values.min())
    maximum = float(values.max())
    mean = float(values.mean())
    delta = maximum - minimum
    return {
        "Минимум, °C": minimum,
        "Максимум, °C": maximum,
        "Среднее, °C": mean,
        "Std, °C": float(values.std(ddof=1)),
        "Дельта, °C": delta,
        "Дельта, %": delta / mean * 100.0,
    }


def _consumer_temperature_differences(
    nodes_df: pd.DataFrame,
    tubes_df: pd.DataFrame,
) -> pd.Series:
    node_temperatures = nodes_df.set_index("id")["Temp"]
    consumer_edges = tubes_df.loc[tubes_df["Vid"] == 2, ["id_in", "id_out"]]
    inlet = consumer_edges["id_in"].map(node_temperatures)
    outlet = consumer_edges["id_out"].map(node_temperatures)
    if inlet.isna().any() or outlet.isna().any():
        raise ValueError("Consumer edge references a node without a temperature")
    return (inlet - outlet).abs()


def _node_temperatures_by_group(
    nodes_df: pd.DataFrame,
    tubes_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    node_temperatures = nodes_df.set_index("id")["Temp"]
    forward_node_ids = set(
        tubes_df.loc[tubes_df["Vid"] == 0, ["id_in", "id_out"]].to_numpy().ravel()
    )
    return_node_ids = set(
        tubes_df.loc[tubes_df["Vid"] == 1, ["id_in", "id_out"]].to_numpy().ravel()
    )
    consumer_edges = tubes_df.loc[tubes_df["Vid"] == 2, ["id_in", "id_out"]]
    consumer_inlet_node_ids = set(consumer_edges["id_in"])
    consumer_outlet_node_ids = set(consumer_edges["id_out"])
    consumer_node_ids = consumer_inlet_node_ids | consumer_outlet_node_ids
    group_node_ids = {
        "forward": forward_node_ids - consumer_node_ids,
        "return": return_node_ids - consumer_node_ids,
        "consumer_inlet": consumer_inlet_node_ids,
        "consumer_outlet": consumer_outlet_node_ids,
    }
    group_keys = list(group_node_ids)
    if any(
        group_node_ids[left] & group_node_ids[right]
        for left_idx, left in enumerate(group_keys)
        for right in group_keys[left_idx + 1 :]
    ):
        raise ValueError("Temperature node groups must not overlap")

    return {
        vid: node_temperatures.loc[sorted(node_ids)].loc[lambda values: values > 0.0]
        for vid, node_ids in group_node_ids.items()
    }


def compute_table() -> pd.DataFrame:
    rows: list[dict] = []
    for case in CASES:
        source_dir = DATASET_ROOT / f"Tout_{case['source_tout']}" / "good"
        nodes_path = source_dir / "Thermo_model_db_n_nodes_ideal.csv"
        tubes_path = source_dir / "Thermo_model_db_n_tubes_ideal.csv"
        nodes_df = _read_table(nodes_path)
        tubes_df = _read_table(tubes_path)

        common = {
            "Запрошенная температура воздуха, °C": case["requested_tout"],
            "Использованный Tout": case["source_tout"],
        }
        node_temperature_groups = _node_temperatures_by_group(nodes_df, tubes_df)
        for group_key, location in NODE_GROUPS:
            values = node_temperature_groups[group_key]
            rows.append(
                {
                    **common,
                    "Место": location,
                    "Показатель": "Температура узла",
                    **_statistics(values),
                }
            )

        consumption = _consumer_temperature_differences(nodes_df, tubes_df)
        rows.append(
            {
                **common,
                "Место": "Потребители",
                "Показатель": "Уровень потребления (|T узла 1 - T узла 2|)",
                **_statistics(consumption),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    result = compute_table()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "ideal_temperature_statistics.csv"
    xlsx_path = OUTPUT_DIR / "ideal_temperature_statistics.xlsx"
    markdown_path = OUTPUT_DIR / "ideal_temperature_statistics.md"
    result.to_csv(csv_path, index=False, float_format="%.6f")
    result.to_excel(xlsx_path, index=False, float_format="%.6f")
    markdown_path.write_text(
        "# Статистика температур ideal-примеров\n\n"
        "Для запрошенной температуры -35 °C использован ближайший доступный "
        "набор `Tout_-34`, поскольку каталога `Tout_-35` в датасете нет.\n\n"
        "Статистика температуры рассчитана по узлам. Из прямого и обратного "
        "трубопроводов исключены consumer-трубы (`Vid=2`) и оба их конечных узла. "
        "Узлы `id_in` consumer-труб учитываются как входы потребителей, а `id_out` "
        "как выходы. Таким образом, один узел не учитывается одновременно в "
        "нескольких группах. Служебные узлы с нулевой температурой исключены "
        "из статистики. Процентная дельта рассчитана как "
        "`(максимум - минимум) / среднее * 100%`. Стандартное отклонение "
        "рассчитано как выборочное (`ddof=1`).\n\n"
        + result.to_markdown(index=False, floatfmt=".3f")
        + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {xlsx_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
