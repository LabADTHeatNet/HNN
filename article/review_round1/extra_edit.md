# Extra edits before final submission

Ниже список правок/проверок, которые стоит сделать перед отправкой revised manuscript и ответов рецензентам.

## Критично проверить

1. **Metric-distance figures**  
   В статью добавлены `Figures/fwd_result_metric_distance_scatter.png` и `Figures/bwd_result_metric_distance_scatter.png` как новый рисунок с ошибкой в метрах. У исходных PNG слегка обрезана левая подпись оси Y. Если есть доступ к скрипту генерации, лучше перегенерировать с увеличенным `left`/`bbox_inches='tight'`, чтобы подпись читалась полностью.

2. **Диапазоны труб и скоростей для Reviewer 3, comment 2**  
   В статье сейчас добавлено объяснение quasi-stationary модели и указано, что transport delays / pipe-wall inertia не моделируются. Рецензент также просил range pipe dimensions / flow velocities / residence times. Если эти числа есть в исходных таблицах, добавить в Section 2.3 короткую фразу вида:  
   `Pipe lengths range from ... to ... m, diameters from ... to ... mm, and the simulated steady-state flow velocities range from ... to ... m/s, corresponding to residence times from ... to ... s/min.`  
   Это усилит ответ и снимет возможный повторный вопрос.

3. **Boundary-condition values для Reviewer 2**  
   Сейчас описана структура граничных условий: supply flow rate and temperature, fixed return pressure, outdoor-temperature curve from -34 to +10 C. Если можно раскрыть численные диапазоны supply temperature, mass flow rate и return pressure, лучше добавить их в Section 2.3. Если раскрывать нельзя, в ответе Reviewer 2 оставить текущую формулировку и при необходимости добавить, что точные эксплуатационные значения не раскрываются из-за ограничений данных.

4. **Стандарты в bibliography**  
   Я добавил BibTeX entries для GOST R 55596-2013, GOST 30732-2006, EN 13941-1:2019+A1:2021 и EN 253:2019+A1:2023. Перед отправкой желательно сверить официальные английские названия стандартов с тем форматом, который требует журнал/издатель.

5. **Цвет правок**  
   В manuscript правки помечены `\textcolor{blue}{...}`. В шаблоне ответа рецензентам написано про red revisions, но в нашем пакете статья уже размечена синим. Если редакция требует именно red, нужно массово заменить `blue` на `red` в финальной версии и в ответах заменить "marked in blue" на "marked in red".

## Желательно, если останется время

1. **Sensor-placement sensitivity**  
   Reviewer 1 и Reviewer 3 спрашивают про чувствительность к размещению сенсоров. Сейчас мы честно указали выбранные 30 узлов и вынесли sensitivity study в future work. Сильнее было бы добавить небольшой эксперимент на 15/30/45 сенсорах или на нескольких random sensor layouts, но это не обязательно, если времени нет.

2. **Multiple random seeds**  
   В статье добавлена 3-fold cross-validation: 0.913±0.011. Если есть результаты по нескольким random seeds, стоит добавить одну фразу или маленькую таблицу. Если нет, текущий ответ через k-fold выглядит приемлемо.

3. **Figure 2 legend**  
   Проверить визуально, что обновленный `dhn_example_new.png` не оставляет неоднозначности "customer" vs "heat sink". В тексте consumer уже объяснен как узел/станция теплопотребления и consumer-modeling edge.

4. **Reference numbering after rebuild**  
   После добавления стандартов и Adam reference номера литературы изменились. В ответах я не указываю конкретные номера, только разделы/таблицы/рисунки, поэтому это безопасно. Но если будете вручную править ответы под шаблон MDPI, не вставляйте старые номера ref. 34/37 без повторной проверки PDF.
