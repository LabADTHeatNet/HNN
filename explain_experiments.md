# Explain Experiments

## Goal

Цель экспериментов: понять, почему модель выбирает конкретный сегмент как дефектный, и можно ли построить объяснение в форме:

> модель приняла решение по этим узлам и этим данным, потому что именно они толкают логит выбранного сегмента вверх, а альтернативные сегменты проигрывают.

Для этого были разделены две задачи:

1. найти метод, который лучше локализует область графа, действительно связанную с дефектом;
2. после локализации понять, какие признаки в этих узлах сильнее всего влияют на решение.

## Что было выбрано как основной explain-подход

Основным выбран метод **GNNExplainer | object/margin**.

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

По этим критериям лучший результат дал **GNNExplainer | object/margin**:

- `mean_true_section_rank = 12.17`
- `mean_top_section_distance = 3.00`
- `mean_near_section_mass_ratio = 0.0839`
- ближайший конкурент: **GNNExplainer | object/vs_no_defect** с `mean_true_section_rank = 12.33` и `mean_top_section_distance = 3.92`


Сводная таблица benchmark-методов:

| Method | What was tried | Mean true rank | Mean top distance | Short conclusion |
| --- | --- | ---: | ---: | --- |
| GNNExplainer \| object/margin | GNNExplainer с object-mask и margin между выбранным классом и лучшей альтернативой. | 12.17 | 3.00 | Лучший общий баланс локализации и стабильности; выбран как основной. |
| GNNExplainer \| object/vs_no_defect | GNNExplainer с object-mask и контрастом к классу no_defect. | 12.33 | 3.92 | Сильный запасной вариант, но в среднем хуже по distance, чем margin. |
| PGExplainer | PGExplainer после дообучения на выбранных примерах. | 15.25 | 3.42 | Иногда хорошо находит окрестность, но маски получаются слишком диффузными. |
| GNNExplainer \| attributes/vs_no_defect | GNNExplainer с feature-mask и контрастом к классу no_defect. | 15.25 | 5.17 | Интересный контрастный вариант, но слабее по общей локализации. |
| gnn_attr_margin | GNNExplainer с feature-mask и margin между классом и лучшей альтернативой. | 16.08 | 4.25 | Margin помогает, но feature-mask все равно остается размытой. |
| Notebook GNNExplainer \| object/log_probs | Подход из explain.ipynb: GNNExplainer, object-mask, log_probs, wrapper через Batch.from_data_list([data]). | 16.25 | 5.75 | Работает стабильно, но локализует хуже лучшего margin-варианта. |
| GraphMask \| object | GraphMaskExplainer с object-mask. | 16.83 | 4.17 | Существенно тяжелее и не дает лучшей локализации. |
| GNNExplainer \| object/log_probs | GNNExplainer с object-mask по узлам и log_probs. | 18.25 | 4.67 | Чище, чем feature-mask, но хуже margin-адаптера. |
| GNNExplainer \| common_attributes/log_probs | GNNExplainer с общей feature-mask по всем узлам. | 18.42 | 5.08 | Маска получается слишком усредненной и плохо указывает на локальный дефект. |
| AttentionExplainer | AttentionExplainer по attention-коэффициентам message-passing слоев. | 19.50 | 3.17 | Видит только GAT-часть node encoder и не покрывает custom edge-attention блок модели. |
| GNNExplainer \| attributes/raw | GNNExplainer с feature-mask по узлам и raw logits. | 20.92 | 5.92 | Не дает явного выигрыша по динамическим признакам; локализация слабее. |
| Notebook GNNExplainer \| common_attributes/log_probs | Подход из explain.ipynb с общей feature-mask для всех узлов. | 22.08 | 5.00 | Слишком глобальный, хуже локализует конкретный сегмент. |

Сводный график benchmark:

- [explainer_benchmark_summary.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/pyg_explainer_benchmark/explainer_benchmark_summary.png)

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

- accuracy: `0.8744`
- macro-F1: `0.9070`
- weighted-F1: `0.8846`
- defect samples: `6290`
- no-defect samples (исключены из сегментной статистики): `6470`
- explained representative examples: `12`

Распределение примеров по запрошенным категориям:

- `correct`: `5500`
- `neighbor`: `284`
- `farther`: `351`
- `no_defect`: `6433`

Дополнительная аналитика по типам ошибок:

- `correct_defect`: `5500`
- `neighbor_error`: `284`
- `far_error`: `351`
- `missed_defect_no_prediction`: `155`
- `correct_no_defect`: `6433`
- `false_positive_no_defect`: `37`

Общие графики:

- [requested_categories.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/requested_categories.png)
- [analysis_categories.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/analysis_categories.png)
- [distance_histogram.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/distance_histogram.png)
- [baseline_method_rank_summary.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/method_comparison/baseline_method_rank_summary.png)

## Как выбранный метод объясняет решение

Главная идея интерпретации здесь такая:

1. `GNNExplainer | object/margin` показывает **какие узлы и трубы поднимают margin выбранного сегмента**.
2. После этого по top-узлам делается `node_feature_ablation`, чтобы проверить, какие признаки в этих узлах реально двигают логит вниз при занулении.
3. В итоге объяснение получается двухступенчатым:
   сначала **где** модель увидела сигнал, потом **какие данные в этой зоне** повлияли на ответ.

Сводка по rank выбранного метода на объясняемых примерах:

| Category | Mean true rank | Mean pred rank | n |
| --- | ---: | ---: | ---: |
| correct | 25.33 | 25.33 | 3 |
| farther | 13.33 | 11.00 | 3 |
| neighbor | 3.33 | 7.00 | 3 |

Наблюдение по этим примерам:

- `neighbor` объясняется лучше всего: средний rank истинного сегмента `3.33`
- `farther` заметно сложнее: `13.33`
- `correct` оказался не самым простым случаем для explain: `25.33`

Это важный результат сам по себе: explainer может хорошо объяснять **локальную ошибку модели**, но хуже разбирать даже корректное предсказание, если решение основано на более распределенном глобальном сигнале.

Дополнительные summary-графики по выбранному методу:

- [primary_true_rank_histogram.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/report_assets/primary_true_rank_histogram.png)
- [primary_rank_by_category.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/report_assets/primary_rank_by_category.png)
- [node_feature_importance_summary.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/node_feature_importance_summary.png)
- [edge_feature_importance_summary.png](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/edge_feature_importance_summary.png)

По `node_feature_ablation` наиболее сильные признаки на выбранных top-узлах в среднем:

- `Temp` (5.908), `P` (4.616), `types_def` (4.572), `pos_x` (4.568), `P_ideal` (4.517), `Temp_ideal` (4.433)

Если агрегировать по группам, получается:

- `dynamic` (4.868), `static` (4.402)

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

**correct**
- `true=16`, `pred=16`, `dir=bwd`, `rank_true=2.0`: [correct_bwd_t16_p16_idx1973](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/correct_bwd_t16_p16_idx1973/explanation_overview.png)
- `true=11`, `pred=11`, `dir=fwd`, `rank_true=34.0`: [correct_fwd_t11_p11_idx4603](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/correct_fwd_t11_p11_idx4603/explanation_overview.png)
**neighbor**
- `true=35`, `pred=25`, `dir=fwd`, `rank_true=1.0`: [neighbor_fwd_t35_p25_idx2031](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/neighbor_fwd_t35_p25_idx2031/explanation_overview.png)
- `true=19`, `pred=10`, `dir=bwd`, `rank_true=1.0`: [neighbor_bwd_t19_p10_idx1742](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/neighbor_bwd_t19_p10_idx1742/explanation_overview.png)
**farther**
- `true=2`, `pred=8`, `dir=bwd`, `rank_true=8.0`: [farther_bwd_t2_p8_idx2234](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/farther_bwd_t2_p8_idx2234/explanation_overview.png)
- `true=3`, `pred=8`, `dir=bwd`, `rank_true=19.0`: [farther_bwd_t3_p8_idx5303](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/farther_bwd_t3_p8_idx5303/explanation_overview.png)
**no_defect**
- `true=43`, `pred=43`, `dir=fwd`, `rank_true=nan`: [no_defect_fwd_t43_p43_idx0639](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/no_defect_fwd_t43_p43_idx0639/explanation_overview.png)
- `true=43`, `pred=43`, `dir=fwd`, `rank_true=nan`: [no_defect_fwd_t43_p43_idx2738](/home/ivan/python/heatnet/out_Termo_Ablation_heads/8_heads/explain_experiments/analysis/methods/gnn_object_margin/examples/no_defect_fwd_t43_p43_idx2738/explanation_overview.png)

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
