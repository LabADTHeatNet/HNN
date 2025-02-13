from pathlib import Path
import random

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from tqdm.contrib.concurrent import process_map


# from PySide6.QtWidgets import QApplication
# app = QApplication([" "])


def generate_sample(case_creation_fn,
                    moded_node_num=[2, 4],
                    moded_node_val_range=[1.05, 1.2],
                    moded_edge_num=[2, 4],
                    moded_edge_val_range=[1.05, 1.1],
                    defect_edge_num=[1, 1],
                    defect_edge_val_range=[1.2, 1.4]):
    # Создание тестового примера электросети с модификацией параметров нагрузок и линий
    net = case_creation_fn()
    # Инициализация столбца 'moded' для линий: 0 означает отсутствие модификации
    net.line['moded'] = 0
    # Сохранение исходных значений сопротивления линии в столбец 'r_ohm_per_km_ideal'
    net.line['r_ohm_per_km_ideal'] = net.line['r_ohm_per_km']
    # Множество для хранения идентификаторов шин с изменёнными нагрузками
    modified_load_buses = set()
    # Если в сети присутствуют нагрузки, выполняем их модификацию
    if len(net.load) > 0:
        # Определение случайного количества нагрузок для изменения в пределах заданного диапазона
        num_load_mod = random.randint(moded_node_num[0], min(moded_node_num[1], len(net.load)))
        # Случайный выбор индексов нагрузок для модификации
        load_indices = random.sample(list(net.load.index), num_load_mod)
        for idx in load_indices:
            # Генерация случайного множителя в заданном диапазоне для изменения мощности нагрузки
            factor = random.uniform(*moded_node_val_range)
            # Модификация активной мощности нагрузки
            net.load.at[idx, 'p_mw'] *= factor
            # Получение идентификатора шины, к которой подключена нагрузка
            bus_id = net.load.at[idx, 'bus']
            # Добавление шины в множество модифицированных
            modified_load_buses.add(bus_id)
    # Если в сети имеются линии, выполняем их модификацию
    if len(net.line) > 0:
        # Получение списка всех индексов линий
        available_line_indices = list(net.line.index)
        # Определение количества линий для обычной модификации в заданном диапазоне
        num_line_mod = random.randint(moded_edge_num[0], min(moded_edge_num[1], len(net.line)))
        # Случайный выбор индексов линий для обычной модификации
        normal_line_mod_indices = random.sample(available_line_indices, num_line_mod)
        for idx in normal_line_mod_indices:
            # Генерация случайного множителя для изменения сопротивления линии
            factor = random.uniform(*moded_edge_val_range)
            # Модификация сопротивления линии
            net.line.at[idx, 'r_ohm_per_km'] *= factor
            # Установка флага модификации: 1 означает обычную модификацию
            net.line.loc[idx, 'moded'] = 1
        # Определение оставшихся линий, не затронутых обычной модификацией
        remaining_lines = list(set(net.line.index) - set(normal_line_mod_indices))
        # Определение количества линий для дефектной (более существенной) модификации
        num_line_def = random.randint(defect_edge_num[0], min(defect_edge_num[1], len(remaining_lines)))
        # Случайный выбор индексов линий для дефектной модификации
        normal_line_def_indices = random.sample(remaining_lines, num_line_def)
        for idx in normal_line_def_indices:
            # Генерация случайного множителя для дефектной модификации сопротивления линии
            factor = random.uniform(*defect_edge_val_range)
            # Применение модификации к сопротивлению линии
            net.line.at[idx, 'r_ohm_per_km'] *= factor
            # Установка флага модификации: 2 означает дефектную модификацию
            net.line.loc[idx, 'moded'] = 2
    # Возвращаем модифицированную сеть и множество шин с изменёнными нагрузками
    return net, modified_load_buses


def create_dfs(net, modified_load_buses):
    # Формирование таблицы с данными по узлам (шинам) электросети
    if hasattr(net, 'bus_geodata') and not net.bus_geodata.empty:
        # Если геоданные присутствуют, извлекаем координаты узлов
        pos_x = net.bus_geodata['x']
        pos_y = net.bus_geodata['y']
    else:
        # Если геоданные отсутствуют, задаём координаты по умолчанию (0)
        pos_x = pd.Series(0, index=net.bus.index)
        pos_y = pd.Series(0, index=net.bus.index)

    # Определение типа узлов: если столбец 'type' присутствует, используем его, иначе заполняем значениями 'bus'
    node_type = net.bus['type'] if 'type' in net.bus.columns else ['bus'] * len(net.bus)
    # Получение номинального напряжения узлов, если указано, иначе заполняем NaN
    vn_kv = net.bus['vn_kv'] if 'vn_kv' in net.bus.columns else pd.Series(np.nan, index=net.bus.index)
    # Определение статуса узлов: в работе или нет (по умолчанию True)
    in_service_bus = net.bus['in_service'] if 'in_service' in net.bus.columns else pd.Series(True, index=net.bus.index)
    # Извлечение рассчитанных значений напряжения (по модулю и углового значения) из результатов расчёта
    vm_pu = net.res_bus['vm_pu']
    va_degree = net.res_bus['va_degree']

    # Создание DataFrame с информацией по узлам, включающей идентификатор, координаты, тип, напряжение и статус работы
    nodes_df = pd.DataFrame({
        'id': net.bus.index,
        'pos_x': pos_x,
        'pos_y': pos_y,
        'node_type': node_type,
        'vn_kv': vn_kv,
        'in_service': in_service_bus,
        # Флаг модификации нагрузки: 1 если шина была изменена, иначе 0
        'moded': [1 if i in modified_load_buses else 0 for i in net.bus.index],
        'vm_pu': vm_pu,
        'va_degree': va_degree
    })

    # Добавление информации по нагрузкам, подключённым к узлам
    if not net.load.empty:
        # Группировка нагрузок по узлам с подсчётом суммарной активной и реактивной мощности
        load_sum = net.load.groupby('bus').agg({'p_mw': 'sum', 'q_mvar': 'sum'})
        # Подсчёт количества нагрузок на каждом узле
        load_count = net.load.groupby('bus').size()
        # Формирование строки с идентификаторами нагрузок для каждого узла
        load_ids = net.load.reset_index().groupby('bus').apply(lambda df: ','.join(df['index'].astype(str)), include_groups=False).rename('load_ids')
    else:
        # Если нагрузок нет, создаём заполненные нулями и пустыми строками структуры для последующего объединения
        load_sum = pd.DataFrame({'p_mw': pd.Series(0, index=net.bus.index),
                                 'q_mvar': pd.Series(0, index=net.bus.index)})
        load_count = pd.Series(0, index=net.bus.index)
        load_ids = pd.Series('', index=net.bus.index)

    # Добавление агрегированной информации по нагрузкам в DataFrame узлов
    nodes_df['p_load_mw'] = nodes_df['id'].map(load_sum['p_mw']).fillna(0)
    nodes_df['q_load_mvar'] = nodes_df['id'].map(load_sum['q_mvar']).fillna(0)
    nodes_df['load_count'] = nodes_df['id'].map(load_count).fillna(0).astype(int)
    nodes_df['load_ids'] = nodes_df['id'].map(load_ids).fillna('')

    # Добавление сведений по генераторам, подключённым к узлам
    if not net.res_gen.empty:
        # Обновление данных по реактивной мощности генераторов из результатов расчёта
        net.gen['q_mvar'] = net.res_gen['q_mvar']
        # Группировка генераторов по узлам с подсчётом суммарной мощности
        gen_sum = net.gen.groupby('bus').agg({'p_mw': 'sum', 'q_mvar': 'sum'}).rename(
            columns={'p_mw': 'p_gen_mw', 'q_mvar': 'q_gen_mvar'})
        # Подсчёт количества генераторов на каждом узле
        gen_count = net.gen.groupby('bus').size()
    else:
        # Если генераторов нет, создаём заполненные нулями структуры для объединения с данными узлов
        gen_sum = pd.DataFrame({'p_gen_mw': pd.Series(0, index=net.bus.index),
                                'q_gen_mvar': pd.Series(0, index=net.bus.index)})
        gen_count = pd.Series(0, index=net.bus.index)
    # Добавление агрегированной информации по генераторам в таблицу узлов
    nodes_df['p_gen_mw'] = nodes_df['id'].map(gen_sum['p_gen_mw']).fillna(0)
    nodes_df['q_gen_mvar'] = nodes_df['id'].map(gen_sum['q_gen_mvar']).fillna(0)
    nodes_df['gen_count'] = nodes_df['id'].map(gen_count).fillna(0).astype(int)

    # ---- Дополнительная информация: Shunt (шунты)
    if not net.shunt.empty:
        # Группировка шунтов по узлам с подсчётом суммарной реактивной мощности
        shunt_sum = net.shunt.groupby('bus').agg({'q_mvar': 'sum'})
        # Подсчёт количества шунтов на каждом узле
        shunt_count = net.shunt.groupby('bus').size()
        # Формирование строки с идентификаторами шунтов для каждого узла
        shunt_ids = net.shunt.reset_index().groupby('bus').apply(lambda df: ','.join(df['index'].astype(str)), include_groups=False).rename('shunt_ids')
    else:
        # Если шунтов нет, создаём структуры с нулевыми значениями
        shunt_sum = pd.DataFrame({'q_mvar': pd.Series(0, index=net.bus.index)})
        shunt_count = pd.Series(0, index=net.bus.index)
        shunt_ids = pd.Series('', index=net.bus.index)
    # Добавление информации по шунтам в DataFrame узлов
    nodes_df['shunt_q_mvar'] = nodes_df['id'].map(shunt_sum['q_mvar']).fillna(0)
    nodes_df['shunt_count'] = nodes_df['id'].map(shunt_count).fillna(0).astype(int)
    nodes_df['shunt_ids'] = nodes_df['id'].map(shunt_ids).fillna('')

    # ---- Дополнительная информация: External Grid (внешняя сеть)
    if not net.ext_grid.empty:
        # Группировка внешней сети по узлам с вычислением среднего напряжения
        ext_grid_avg = net.ext_grid.groupby('bus').agg({'vm_pu': 'mean'})
        # Подсчёт количества внешних сетей на каждом узле
        ext_grid_count = net.ext_grid.groupby('bus').size()
        # Формирование строки с идентификаторами внешних сетей для каждого узла
        ext_grid_ids = net.ext_grid.reset_index().groupby('bus').apply(lambda df: ','.join(df['index'].astype(str)), include_groups=False).rename('ext_grid_ids')
    else:
        # Если данных по внешней сети нет, создаём структуры с нулевыми значениями
        ext_grid_avg = pd.DataFrame({'vm_pu': pd.Series(0, index=net.bus.index)})
        ext_grid_count = pd.Series(0, index=net.bus.index)
        ext_grid_ids = pd.Series('', index=net.bus.index)
    # Добавление информации по внешней сети в таблицу узлов
    nodes_df['ext_grid_vm_pu'] = nodes_df['id'].map(ext_grid_avg['vm_pu']).fillna(0)
    nodes_df['ext_grid_count'] = nodes_df['id'].map(ext_grid_count).fillna(0).astype(int)
    nodes_df['ext_grid_ids'] = nodes_df['id'].map(ext_grid_ids).fillna('')

    # ---- Дополнительная информация: Poly Cost (полиномиальные затраты для генераторов и внешней сети)
    if not net.poly_cost.empty:
        # Инициализация DataFrame для подсчёта количества записей poly_cost для каждого узла
        poly_cost_df = pd.DataFrame({'poly_cost_count': pd.Series(0, index=net.bus.index)})
        if not net.gen.empty:
            # Отбор записей poly_cost, относящихся к генераторам
            gen_poly = net.poly_cost[net.poly_cost['et'] == 'gen']
            if not gen_poly.empty:
                # Отображение соответствия между элементами генераторов и их шинами
                gen_bus_map = net.gen['bus']
                gen_poly = gen_poly.copy()
                gen_poly['bus'] = gen_poly['element'].map(gen_bus_map)
                # Подсчёт количества записей poly_cost для генераторов по каждой шине
                gen_poly_count = gen_poly.groupby('bus').size()
                poly_cost_df['poly_cost_count'] = gen_poly_count
        # Заполнение пропусков нулевыми значениями
        poly_cost_df['poly_cost_count'] = poly_cost_df['poly_cost_count'].fillna(0)
        if not net.ext_grid.empty:
            # Отбор записей poly_cost, относящихся к внешней сети
            ext_poly = net.poly_cost[net.poly_cost['et'] == 'ext_grid']
            if not ext_poly.empty:
                # Отображение соответствия между элементами внешней сети и их шинами
                ext_bus_map = net.ext_grid['bus']
                ext_poly = ext_poly.copy()
                ext_poly['bus'] = ext_poly['element'].map(ext_bus_map)
                # Подсчёт количества записей poly_cost для внешней сети по каждой шине
                ext_poly_count = ext_poly.groupby('bus').size()
                poly_cost_df.loc[ext_poly_count.index, 'poly_cost_count'] += ext_poly_count
        # Добавление итогового количества записей poly_cost в DataFrame узлов
        nodes_df['poly_cost_count'] = nodes_df['id'].map(poly_cost_df['poly_cost_count']).fillna(0)
    else:
        # Если данных poly_cost нет, заполняем соответствующий столбец нулевыми значениями
        nodes_df['poly_cost_count'] = 0

    # Создание таблицы с данными по линиям (ребрам) электросети
    edges_line = net.line[['from_bus', 'to_bus', 'length_km']].copy()
    if 'r_ohm_per_km_ideal' in net.line.columns:
        # Добавление идеального значения сопротивления линии, если оно присутствует
        edges_line['r_ohm_per_km_ideal'] = net.line['r_ohm_per_km_ideal']
    else:
        # Если идеальное значение отсутствует, заполняем NaN
        edges_line['r_ohm_per_km_ideal'] = np.nan
    # Добавление текущего значения сопротивления линии и информации о модификации
    edges_line['r_ohm_per_km'] = net.line['r_ohm_per_km']

    edges_line['moded'] = net.line['moded']
    # Расчёт разницы между модифицированным и идеальным значением сопротивления
    edges_line['delta_r'] = edges_line['r_ohm_per_km'] - edges_line['r_ohm_per_km_ideal']
    # Установка типа ребра как 'line'
    edges_line['edge_type'] = 'line'

    # Если в сети присутствуют трансформаторы, формируем дополнительную таблицу для них
    if len(net.trafo) > 0:
        trafo_df = pd.DataFrame({
            'from_bus': net.trafo['hv_bus'],
            'to_bus': net.trafo['lv_bus'],
            'length_km': 0,
            'r_ohm_per_km': np.nan,
            'r_ohm_per_km_ideal': np.nan,
            'moded': 0,
            'delta_r': np.nan,
            'edge_type': 'trafo'
        })
        if hasattr(net, 'res_trafo') and not net.res_trafo.empty:
            # Добавление информации о загрузке трансформаторов, если доступна
            trafo_df['loading_percent'] = net.res_trafo['loading_percent'].values
        else:
            # Если данных о загрузке нет, заполняем NaN
            trafo_df['loading_percent'] = np.nan
    else:
        # Если трансформаторов нет, создаём пустой DataFrame
        trafo_df = pd.DataFrame()

    # Объединение таблиц линий и трансформаторов в единую таблицу ребер
    edges_df = pd.concat([edges_line, trafo_df], ignore_index=True)

    # Возврат сформированных таблиц узлов и ребер
    return nodes_df, edges_df


if __name__ == '__main__':
    # Основной блок выполнения: определение конфигураций наборов данных и генерация примеров
    dataset_cfg_list = [
        # Каждая конфигурация содержит функцию создания кейса и путь к директории для сохранения данных
        [pn.case14, Path('datasets/case14')],
        [pn.case118, Path('datasets/case118')]
    ]

    # Задание количества генерируемых примеров и максимального числа параллельных процессов
    num_samples = 1000
    max_workers = 16

    # Параметры для модификации узлов и линий: диапазоны количества и коэффициентов изменения
    moded_node_num = [2, 4]
    moded_node_val_range = [1.05, 1.2]
    moded_edge_num = [2, 4]
    moded_edge_val_range = [1.05, 1.2]
    defect_edge_num = [1, 1]
    defect_edge_val_range = [1.5, 2.0]

    def process(args):
        # Функция обработки одного примера: генерация, расчёт и сохранение данных
        fn, dataset_dir, idx = args
        # Генерация модифицированного примера электросети с заданными параметрами
        net, modified_load_buses = generate_sample(fn,
                                                   moded_node_num=moded_node_num,
                                                   moded_node_val_range=moded_node_val_range,
                                                   moded_edge_num=moded_edge_num,
                                                   moded_edge_val_range=moded_edge_val_range,
                                                   defect_edge_num=defect_edge_num,
                                                   defect_edge_val_range=defect_edge_val_range)
        # Выполнение расчёта (power flow) для модифицированной сети
        pp.runpp(net)
        # (Опционально) Вывод информации по полиномиальным затратам (закомментировано)
        # print(net.poly_cost)
        # Формирование таблиц с информацией по узлам и ребрам после проведения расчёта
        nodes_df, edges_df = create_dfs(net, modified_load_buses)

        # Сохранение данных по узлам и линиям в формате CSV с индексом в имени файла
        nodes_df.to_csv(dataset_dir / f'nodes_{idx:06d}.csv', index=False)
        edges_df.to_csv(dataset_dir / f'tubes_{idx:06d}.csv', index=False)
        # Возврат 0 для обозначения успешного завершения обработки
        return 0

    # Обработка каждой конфигурации набора данных
    for fn, dataset_dir in dataset_cfg_list:
        # Создание директории для сохранения данных, если она ещё не существует
        dataset_dir.mkdir(exist_ok=True)

        # Вывод сообщения о начале создания набора данных для текущей конфигурации
        print(f'Creation of: {dataset_dir}')
        # (Опционально) Генерация заданного количества примеров с использованием list comprehension (последовательная обработка)
        [process((fn, dataset_dir, idx)) for idx in range(0, num_samples)]
        # Параллельная генерация примеров с использованием process_map (закомментировано)
        # result = process_map(process, [(fn, dataset_dir, idx) for idx in range(0, num_samples)], max_workers=max_workers)

##########################
# Параметры в таблице nodes_df:
# id - идентификатор узла (шины) в электросети, соответствует индексу из таблицы net.bus.
# pos_x - координата узла по оси X, полученная из геоданных (bus_geodata) или равная 0, если геоданные отсутствуют.
# pos_y - координата узла по оси Y, полученная из геоданных (bus_geodata) или равная 0, если геоданные отсутствуют.
# node_type - тип узла; если задан, берётся из столбца 'type', иначе заполняется значением 'bus'.
# vn_kv - номинальное напряжение узла в кВ, берётся из столбца 'vn_kv' или устанавливается как NaN.
# in_service - флаг, указывающий, находится ли узел в эксплуатации (True) или нет (False).
# moded - флаг, показывающий, была ли изменена нагрузка на данном узле: 1, если шина модифицирована (нагрузка изменена), 0 – если нет.
# vm_pu - рассчитанное значение напряжения узла в pu (отношение напряжения к номинальному значению).
# va_degree - рассчитанный угол напряжения узла в градусах.
# p_load_mw - суммарная активная мощность (в МВт) нагрузок, подключённых к узлу.
# q_load_mvar - суммарная реактивная мощность (в Mvar) нагрузок, подключённых к узлу.
# load_count - количество нагрузок, подключённых к узлу.
# load_ids - строка, содержащая идентификаторы нагрузок, подключённых к узлу, разделённые запятыми.
# p_gen_mw - суммарная активная мощность генераторов (в МВт), подключённых к узлу.
# q_gen_mvar - суммарная реактивная мощность генераторов (в Mvar), подключённых к узлу.
# gen_count - количество генераторов, подключённых к узлу.
# shunt_q_mvar - суммарная реактивная мощность шунтов (в Mvar), подключённых к узлу.
# shunt_count - количество шунтов, подключённых к узлу.
# shunt_ids - строка, содержащая идентификаторы шунтов, подключённых к узлу, разделённые запятыми.
# ext_grid_vm_pu - среднее значение напряжения (в pu) для внешней сети, подключённой к узлу.
# ext_grid_count - количество подключённых к узлу внешних сетей.
# ext_grid_ids - строка, содержащая идентификаторы внешних сетей, подключённых к узлу, разделённые запятыми.
# poly_cost_count - количество записей полиномиальных затрат (poly_cost) для данного узла, включающее как генераторы, так и внешнюю сеть.
##########################

##########################
# Параметры в таблице edges_df:
# from_bus - идентификатор начальной шины (узла), откуда начинается ребро (линия или трансформатор).
# to_bus - идентификатор конечной шины (узла), где заканчивается ребро.
# length_km - длина ребра (линии) в километрах; для трансформаторов значение всегда 0.
# r_ohm_per_km_ideal - идеальное (исходное) сопротивление линии в омах на км до внесения модификаций.
# r_ohm_per_km - текущее значение сопротивления линии в омах на км после внесения модификаций.
# moded - флаг модификации ребра: 0 – без модификации, 1 – обычная модификация, 2 – дефектная модификация.
# delta_r - разница между текущим сопротивлением (r_ohm_per_km) и идеальным (r_ohm_per_km_ideal), показывающая изменение параметра.
# edge_type - тип ребра: 'line' для линий электропередачи и 'trafo' для трансформаторов.
# (Для трансформаторов могут дополнительно присутствовать другие параметры, например, loading_percent, показывающий процент загрузки трансформатора.)
##########################
