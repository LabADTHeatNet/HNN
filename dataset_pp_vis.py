import networkx as nx                                  # Импорт библиотеки для работы с графами
from torch_geometric.utils import to_networkx         # Функция для преобразования объекта Data в граф NetworkX

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.io as pio
import plotly.graph_objects as go
import pandapower.plotting as plot

# pio.renderers.default = "browser"
# gui = 'QtAgg'
# matplotlib.use(gui, force=True)


##########################
# Функция визуализации sample (Data)
##########################


def visualize_sample(sample, title="Graph Visualization",
                     nodes_categories=None, edges_categories=None,
                     figsize=(8, 6),
                     nodes_size=300):
    """
    Визуализирует объект sample (типа Data) с кодированием:
      - Узлы с нагрузкой (load_count > 0) окрашиваются красным, остальные – синим.
      - Узлы с генератором (gen_count > 0) отображаются в виде треугольника, остальные – в виде круга.
      - Рёбра раскрашиваются в зависимости от типа ребра (edge_type) с использованием переданных категорий.
      - На рёбрах отображаются числовые значения из edges_label.

    Важно: функция предполагает, что sample.x уже денормализован и имеет как минимум следующие столбцы:
      [pos_x, pos_y, vn_kv, in_service, vm_pu, va_degree, p_load_mw, q_load_mvar, load_count, p_gen_mw, q_gen_mvar, gen_count, ...]
    """

    # Преобразуем объект sample в неориентированный граф NetworkX
    G = to_networkx(sample, to_undirected=True)

    # Определяем позиции узлов: используем первые два столбца sample.x как координаты (pos_x, pos_y)
    if sample.x is not None and sample.x.size(1) >= 2:
        pos = {i: (sample.x[i, 0].item(), sample.x[i, 1].item())
               for i in range(sample.x.size(0))}
    else:
        # Если координаты не заданы, используем алгоритм расстановки узлов
        pos = nx.spring_layout(G)

    num_nodes = sample.x.size(0)  # Общее количество узлов
    # Инициализируем списки для хранения цветов и маркеров узлов
    nodes_colors = []   # Цвет каждого узла
    nodes_markers = []  # Форма маркера для каждого узла
    for i in range(num_nodes):
        # Извлекаем значение load_count (индекс 8) и gen_count (индекс 11), если они доступны
        load = sample.x[i, 8].item() if sample.x.size(1) > 8 else 0
        gen = sample.x[i, 11].item() if sample.x.size(1) > 11 else 0
        # Определяем цвет узла: красный, если load_count > 0, иначе синий
        color = 'red' if load > 1e-1 else 'blue'
        # Определяем форму маркера: треугольник, если gen_count > 0, иначе круг
        marker = '^' if gen > 1e-1 else 'o'
        nodes_colors.append(color)
        nodes_markers.append(marker)

    # Группируем узлы по форме маркера, так как nx.draw_networkx_nodes позволяет задать маркер за один вызов
    marker_groups = {}
    for i, marker in enumerate(nodes_markers):
        marker_groups.setdefault(marker, []).append(i)
    # Формируем для каждой группы список соответствующих цветов
    group_colors = {marker: [nodes_colors[i] for i in indices]
                    for marker, indices in marker_groups.items()}

    # Обработка ребер: извлечение категориальных признаков и меток ребер
    edges_cats_np = sample.edges_cats.numpy()            # Массив one-hot представлений для типов ребер (размер: num_edges x num_edges_categories)
    edges_label_np = sample.edges_label.numpy().flatten()  # Массив меток ребер (например, значение r_ohm_per_km)
    edges_index_np = sample.edges_index.numpy()            # Массив индексов ребер (размер: 2 x num_edges)
    num_edges = edges_index_np.shape[1]                    # Общее количество ребер
    for i in range(num_edges):
        u = int(edges_index_np[0, i])
        v = int(edges_index_np[1, i])
        # Определяем индекс категории ребра по максимальному значению в one-hot векторе
        type_index = edges_cats_np[i].argmax()
        # Если переданы категории ребер, используем их, иначе выводим индекс категории как строку
        if edges_categories is not None:
            edges_type_label = edges_categories[type_index]
        else:
            edges_type_label = str(type_index)
        # Если ребро уже существует в графе, добавляем к нему тип и метку
        if G.has_edges(u, v):
            G[u][v]['edge_type'] = edges_type_label
            G[u][v]['edges_label'] = edges_label_np[i]

    # Определяем цветовую карту для ребер на основе их типов
    edges_types = [G[u][v]['edge_type'] for u, v in G.edges]
    unique_edges_types = sorted(set(edges_types))
    cmap_edges = cm.get_cmap('Set1', len(unique_edges_types))
    edges_color_map = {cat: cmap_edges(i) for i, cat in enumerate(unique_edges_types)}
    # Назначаем цвет каждому ребру согласно его типу
    edges_colors = [edges_color_map[G[u][v]['edge_type']] for u, v in G.edges]

    # Формируем словарь меток для ребер с форматированием значения r_ohm_per_km до 2 знаков после запятой
    edges_labels = {(u, v): f"{G[u][v]['r_ohm_per_km']:.2f}" for u, v in G.edges}

    plt.figure(figsize=figsize)  # Создаем фигуру matplotlib с заданным размером
    # Отрисовываем узлы для каждой группы по маркеру и цветам
    for marker, nodeslist in marker_groups.items():
        colors = group_colors[marker]
        nx.draw_networkx_nodes(G, pos, nodelist=nodeslist, node_color=colors,
                               node_shape=marker, node_size=nodes_size)
    nx.draw_networkx_edges(G, pos, edge_color=edges_colors)  # Отрисовка ребер с назначенными цветами
    nx.draw_networkx_labels(G, pos, font_size=10)              # Добавление подписей к узлам
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color="red")  # Подписи ребер
    plt.title(title)              # Задаем заголовок графика
    plt.axis("off")               # Отключаем оси
    plt.show()                    # Отображаем график

##########################
# Функция визуализации через pandapower с использованием Plotly
##########################


def visualize_data_as_net_plotly(net, figsize=(8, 6), fontsize=16, title="Pandapower Network Visualization"):
    """
    Преобразует pandapower-сеть (net) в интерактивную визуализацию с использованием Plotly.
    Особенности визуализации:
      - Линии из net.line окрашиваются по полю moded:
            moded == 1 → синий, moded == 2 → оранжевый, иначе серый.
      - Трансформаторы (net.trafo) отображаются пунктиром и окрашиваются в пурпурный.
      - Для каждого узла выводится подпись в формате 'id: voltage' (id – номер шины, voltage – напряжение).
      - Узлы окрашиваются по значению напряжения (vm_pu) с использованием цветовой шкалы Viridis.
      - Форма маркера для узла определяется:
            * Если шина имеет запись в net.ext_grid, используется квадрат.
            * Если шина имеет запись в net.gen, используется ромб.
            * В противном случае – круг.
      - Если шина имеет нагрузку (net.load), маркер отображается с более толстой окантовкой.
    """
    # Извлекаем список шин и их геоданные
    buses = net.bus.index.tolist()
    bus_geodata = net.bus_geodata
    # Определяем координаты узлов и их напряжения
    nodes_x = [bus_geodata.loc[bus, 'x'] for bus in buses]
    nodes_y = [bus_geodata.loc[bus, 'y'] for bus in buses]
    voltages = [net.res_bus.loc[bus, 'vm_pu'] for bus in buses]
    # Формируем подписи для узлов: номер шины и напряжение
    nodes_labels = [f"{bus+1:2d}. V: {voltage:.2e}" for bus, voltage in zip(buses, voltages)]

    # Определяем для каждого узла символ маркера и толщину контура на основе наличия ext_grid, gen и load
    marker_symbols = []
    marker_line_widths = []
    for bus in buses:
        # Проверка наличия внешней сети (ext_grid), генератора (gen) и нагрузки (load) на шине
        has_ext = (net.ext_grid[net.ext_grid.bus == bus].shape[0] > 0) if hasattr(net, "ext_grid") and not net.ext_grid.empty else False
        has_gen = (net.gen[net.gen.bus == bus].shape[0] > 0) if hasattr(net, "gen") and not net.gen.empty else False
        has_load = (net.load[net.load.bus == bus].shape[0] > 0) if hasattr(net, "load") and not net.load.empty else False
        # Назначаем символ: квадрат для ext_grid, ромб для gen, иначе круг
        if has_ext:
            symbol = "square"
        elif has_gen:
            symbol = "diamond"
        else:
            symbol = "circle"
        marker_symbols.append(symbol)
        # Толщина контура: толще (6) если есть нагрузка, иначе стандартно (2)
        lw = 6 if has_load else 2
        marker_line_widths.append(lw)

    # Группируем узлы по комбинации (символ маркера, толщина линии) для оптимальной отрисовки
    nodes_groups = {}
    for i, bus in enumerate(buses):
        key = (marker_symbols[i], marker_line_widths[i])
        if key not in nodes_groups:
            nodes_groups[key] = {"x": [], "y": [], "volt": [], "label": []}
        nodes_groups[key]["x"].append(nodes_x[i])
        nodes_groups[key]["y"].append(nodes_y[i])
        nodes_groups[key]["volt"].append(voltages[i])
        nodes_groups[key]["label"].append(nodes_labels[i])

    # Определяем карту цветов для линий на основе поля moded
    group_color_map_line = {0: "gray", 1: "blue", 2: "orange"}

    # Обработка ребер из net.line
    edge_groups_line = {}
    edge_annotations = []
    for idx, row in net.line.iterrows():
        from_bus = row['from_bus']
        to_bus = row['to_bus']
        # Определяем значение модификации (moded) для линии, если отсутствует – считаем равным 0
        try:
            moded = int(float(row.get('moded', 0)))
        except Exception:
            moded = 0
        key = ("line", moded)
        # Проверяем наличие геоданных для обеих шин ребра
        if from_bus in bus_geodata.index and to_bus in bus_geodata.index:
            x0, y0 = bus_geodata.loc[from_bus, ['x', 'y']]
            x1, y1 = bus_geodata.loc[to_bus, ['x', 'y']]
            if key not in edge_groups_line:
                edge_groups_line[key] = {"x": [], "y": []}
            # Добавляем координаты для отрисовки линии (используем None для разрыва между линиями)
            edge_groups_line[key]["x"].extend([x0, x1, None])
            edge_groups_line[key]["y"].extend([y0, y1, None])
            # Вычисляем среднюю точку ребра для размещения аннотации
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            # Формируем текстовую метку для ребра, используя значение r_ohm_per_km
            e_label = f'r [ohm/km]: {row.get("r_ohm_per_km", None):.2e}'
            # Если линия модифицирована (moded != 0), добавляем также идеальное значение сопротивления
            if moded != 0:
                e_label += f' | {row.get("r_ohm_per_km_ideal", None):.2e}'
            if e_label is not None:
                # Добавляем аннотацию для ребра в список
                edge_annotations.append(dict(
                    x=mid_x,
                    y=mid_y,
                    text=e_label,
                    showarrow=True,
                    font=dict(color=group_color_map_line.get(moded, "gray"), size=fontsize),
                    xanchor="center",
                    yanchor="middle"
                ))

    # Обработка ребер типа "trafo" (трансформаторов) из net.trafo
    edge_groups_trafo = {}
    for idx, row in net.trafo.iterrows():
        from_bus = row['hv_bus']
        to_bus = row['lv_bus']
        key = ("trafo", None)
        if from_bus in bus_geodata.index and to_bus in bus_geodata.index:
            x0, y0 = bus_geodata.loc[from_bus, ['x', 'y']]
            x1, y1 = bus_geodata.loc[to_bus, ['x', 'y']]
            if key not in edge_groups_trafo:
                edge_groups_trafo[key] = {"x": [], "y": []}
            edge_groups_trafo[key]["x"].extend([x0, x1, None])
            edge_groups_trafo[key]["y"].extend([y0, y1, None])
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            # Формируем аннотацию для трансформатора (например, загрузка в процентах)
            e_label = f'load: {net.res_trafo.iloc[idx]["loading_percent"]:.1f}%'
            if e_label is not None:
                edge_annotations.append(dict(
                    x=mid_x,
                    y=mid_y,
                    text=e_label,
                    showarrow=True,
                    font=dict(color="purple", size=fontsize),
                    xanchor="center",
                    yanchor="middle"
                ))

    # Объединяем группы ребер: для линий и трансформаторов формируем Scatter-трейсы Plotly
    edge_traces = []
    marker = dict(
        symbol="arrow",
        size=fontsize,
        standoff=fontsize,
        angleref="previous",
    )
    # Трейсы для линий: задаем режим отображения "lines+markers" с цветом, зависящим от moded
    edge_traces += [
        go.Scatter(
            x=coords["x"],
            y=coords["y"],
            mode="lines+markers",
            marker=marker,
            line=dict(width=2, color=group_color_map_line.get(moded, "gray"), dash="solid"),
            hoverinfo="none"
        )
        for (_, moded), coords in edge_groups_line.items()
    ]

    # Трейсы для трансформаторов: отображаются пунктирной линией в пурпурном цвете
    edge_traces += [
        go.Scatter(
            x=coords["x"],
            y=coords["y"],
            mode="lines+markers",
            marker=marker,
            line=dict(width=2, color="purple", dash="dash"),
            hoverinfo="none"
        )
        for (_, _), coords in edge_groups_trafo.items()
    ]

    # Создаем Scatter-трейсы для узлов, группируя их по (символ, толщина контура)
    nodes_traces = []
    first_node_trace = True  # Флаг для добавления colorbar только для первого трейса
    for (symbol, lw), group in nodes_groups.items():
        trace = go.Scatter(
            x=group["x"],
            y=group["y"],
            mode="markers+text",
            text=group["label"],
            textposition="top center",
            marker=dict(
                size=20,
                symbol=symbol,
                color=group["volt"],
                colorscale="Viridis",
                colorbar=dict(title="Voltage (p.u.)") if first_node_trace else None,
                showscale=True if first_node_trace else False,
                line=dict(width=lw, color="black")
            ),
            hoverinfo="text"
        )
        first_node_trace = False
        nodes_traces.append(trace)

    # Формируем итоговую фигуру Plotly, объединяя трейсы ребер и узлов, а также добавляя аннотации
    fig = go.Figure(
        data=edge_traces + nodes_traces,
        layout=go.Layout(
            title=title,
            font=dict(size=fontsize),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=edge_annotations
        )
    )

    fig.show()  # Отображаем интерактивную фигуру Plotly

##########################
# Функция визуализации через pandapower с использованием matplotlib
##########################


def visualize_data_as_net(net, figsize=(8, 6), title="Pandapower Network Visualization"):
    """
    Преобразует объект Data (sample) в pandapower-сеть и визуализирует её с помощью plot.simple_plot.
    Особенности визуализации:
      - На графике линии отображаются с числовыми значениями (edges_label).
      - Узлы подписываются в формате 'id: voltage' (id – номер шины, voltage – напряжение).
      - Узлы окрашиваются по значению напряжения (vm_pu) с добавлением цветовой шкалы.
      - Рёбра с модификацией (moded == 1 или 2) выделяются синим или оранжевым соответственно.
    Важно: Функция предполагает, что sample.x уже денормализован и имеет необходимые столбцы.
    """
    # (При необходимости можно выполнить расчет потоков для обновления net.res_bus)
    # pp.runpp(net)

    # Создаем фигуру и оси для matplotlib
    fig, ax = plt.subplots(figsize=figsize)

    # Отрисовка базовой электросети с использованием pandapower (без отображения размеров шин)
    plot.simple_plot(net, ax=ax,
                     bus_size=0,
                     scale_size=True,
                     respect_switches=True,
                     plot_loads=True,
                     plot_sgens=True,
                     show_plot=False)

    # Цветовое кодирование узлов по напряжению: получаем напряжения с шин и нормализуем их для цветовой карты
    voltages = net.res_bus.loc[net.bus.index, 'vm_pu'].values.astype(float)
    norm = plt.Normalize(vmin=voltages.min(), vmax=voltages.max())
    cmap = cm.get_cmap('viridis')
    xs = net.bus_geodata.loc[net.bus.index, 'x'].values
    ys = net.bus_geodata.loc[net.bus.index, 'y'].values
    # Отрисовываем узлы с использованием scatter и добавляем цветовую шкалу
    sc = ax.scatter(xs, ys, c=voltages, cmap=cmap, norm=norm, s=100, zorder=10, edgecolor='k')

    # Перебираем линии сети для их перекраски в зависимости от модификации (moded)
    for idx, row in net.line.iterrows():
        from_bus = row['from_bus']
        to_bus = row['to_bus']
        if from_bus in net.bus_geodata.index and to_bus in net.bus_geodata.index:
            x1, y1 = net.bus_geodata.loc[from_bus, ['x', 'y']]
            x2, y2 = net.bus_geodata.loc[to_bus, ['x', 'y']]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Определяем цвет линии на основе поля moded
            moded = row.get('moded', 0)
            try:
                moded = int(float(moded))
            except Exception:
                moded = 0
            if moded == 1:
                edge_color = "blue"
            elif moded == 2:
                edge_color = "orange"
            else:
                edge_color = "gray"
            # Отрисовываем линию с заданными координатами и цветом
            ax.plot([x1, x2], [y1, y2], color=edge_color, lw=2, zorder=5)
            # Добавляем текстовую аннотацию с значением edges_label, если оно имеется
            edges_label = row.get('edges_label', None)
            if edges_label is not None:
                label = f"{edges_label:.2e}"
                ax.text(mid_x, mid_y, label, color="red", fontsize=9,
                        ha="center", va="center", zorder=6)

    # Добавляем подписи для узлов: номер шины и напряжение
    for bus in net.bus.index:
        if bus in net.bus_geodata.index and bus in net.res_bus.index:
            x, y = net.bus_geodata.loc[bus, ['x', 'y']]
            voltage = net.res_bus.loc[bus, 'vm_pu']
            label = f"{bus+1:2d}: {voltage:.2e}"
            ax.text(x, y, label, fontsize=9, color="orange", ha="center", va="center",
                    zorder=10)

    # Добавляем цветовую шкалу (colorbar) для отображения напряжения
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Voltage (p.u.)")

    plt.title(title)          # Устанавливаем заголовок графика
    plt.axis("off")           # Отключаем отображение осей
    fig.tight_layout()        # Автоматическая оптимизация расположения элементов графика
    plt.show()                # Отображаем финальный график
