from pathlib import Path
import importlib
import copy
import json
import tqdm

import os
import numpy as np
import pandas as pd
import torch
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

from src.datasets import (
    prepare_data,
    data_to_tables,
    add_sections
)
from src.utils import (
    train,
    valid,
    weighted_mse_loss,
    FocalRegressionLoss,
    FocalLoss,
    MulticlassFocalLoss,
)
from src.plots import (
    draw_data
)


def _log_metrics(metrics, suffix, writer, epoch):
    """Логирует метрики в TensorBoard с указанным префиксом."""
    for key, value in metrics.items():
        writer.add_scalar(f"{key}/{suffix}", value, epoch)


def exp(cfg, project_name='HeatNet', run_clear_ml=False, log_dir=None):
    if run_clear_ml:
        from clearml import (
            Task,
            OutputModel
        )

    """Основная функция запуска эксперимента: обучение и валидация модели."""
    # Создание директории для логов
    if log_dir is None:
        log_dir = 'tmp'
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение конфигурации в файл
    cfg_dump = copy.copy(cfg)
    with open(log_dir / 'params.json', 'w') as f:
        json.dump(cfg_dump, f, indent=4)

    device = torch.device(cfg['utils']['device'])  # Устройство для вычислений (GPU/CPU)

    # Подготовка данных
    dataset, scalers, train_loader, val_loader, test_loader, ideal_dataset = prepare_data(cfg['dataset'], cfg['dataloader'], cfg['utils']['seed'])
    if cfg['dataset']['name'] == 'Termo_model_fwd_and_bwd':
        dataset= dataset[0]
    # Пример вывода информации о батче
    for batch in train_loader:
        print("Пример батча:")
        print(batch)
        break

    # Инициализация модели
    in_node_dim = dataset[0].x.shape[1]  # Размерность признаков узлов
    in_edge_dim = dataset[0].edge_attr.shape[1]  # Размерность признаков ребер
    # out_dim = dataset[0].edge_label.shape[-1]  # Размерность целевых меток

    # Динамический импорт класса модели
    model_fn = getattr(
        importlib.import_module(f"src.models.{cfg['model']['name']}"),
        cfg['model']['name'])

    def create_model():
        return model_fn(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            # out_dim=out_dim,
            **cfg['model']['kwargs']
        )
    model = create_model()
    model = model.to(device)

    # Инициализация оптимизатора и планировщика
    optimizer_fn = getattr(importlib.import_module('torch.optim'), cfg['optimizer']['name'])
    optimizer = optimizer_fn(model.parameters(), **cfg['optimizer']['kwargs'])

    if cfg['scheduler']['name'] is not None:
        scheduler_fn = getattr(importlib.import_module('torch.optim.lr_scheduler'), cfg['scheduler']['name'])
        scheduler = scheduler_fn(optimizer, **cfg['scheduler']['kwargs'])
    else:
        scheduler = None

    # Функция потерь
    if cfg['criterion']['name'] is not None:
        if cfg['criterion']['name'] == 'FocalRegressionLoss':
            criterion_fn = FocalRegressionLoss
        if cfg['criterion']['name'] == 'FocalLoss':
            criterion_fn = FocalLoss
        if cfg['criterion']['name'] == 'MulticlassFocalLoss':
            criterion_fn = MulticlassFocalLoss
        elif cfg['criterion']['name'] == 'weighted_mse_loss':
            criterion_fn = weighted_mse_loss
        else:
            criterion_fn = getattr(importlib.import_module('torch.nn'), cfg['criterion']['name'])
        if 'pos_weight' in cfg['criterion']['kwargs']:
            cfg['criterion']['kwargs']['pos_weight'] = torch.Tensor(
                cfg['criterion']['kwargs']['pos_weight']
            ).to(device)
        if 'weight' in cfg['criterion']['kwargs']:
            cfg['criterion']['kwargs']['weight'] = torch.Tensor(
                cfg['criterion']['kwargs']['weight']
            ).to(device)
        criterion = criterion_fn(**cfg['criterion']['kwargs'])
    else:
        criterion = None

    # Проверка формы вывода модели
    with torch.no_grad():
        if isinstance(batch, list):
            batch_fwd, batch_bwd = batch
            pred_fwd = model(batch_fwd.to(device))
            pred_bwd = model(batch_bwd.to(device))
            loss_fwd = criterion(pred_fwd, batch_fwd.edge_label)
            loss_bwd = criterion(pred_bwd, batch_bwd.edge_label)
            tmp_loss = (loss_fwd + loss_bwd) / 2
            pred_tmp = (pred_fwd + pred_bwd) / 2
        else:
            pred_tmp = model(batch.to(device))
            tmp_loss = criterion(pred_tmp, batch.edge_label)
    print("Размер вывода модели:", pred_tmp.shape)
    print("Тестовый лосс:", tmp_loss)

    # Интеграция с ClearML
    if run_clear_ml:
        task = Task.init(
            project_name=project_name,
            task_name=str(log_dir),
            output_uri=False
        )
        task.connect(cfg_dump)  # Логирование параметров
        output_model = OutputModel(task=task)
        output_model.update_design(config_dict=cfg_dump.get('model'))
    else:
        task = None

    # Логирование в TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    summary(model)
    print(model)

    edge_label_scaler = None # Скейлер для меток

    # Обучение модели
    best_score = torch.inf  # Лучшее значение метрики
    with tqdm.tqdm(total=cfg['train']['num_epochs'], desc="Epochs", unit="epoch") as pbar:
        for epoch in range(cfg['train']['num_epochs']):
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)  # Логирование lr

            # Обучение на тренировочных данных
            train_metrics = train(model, train_loader, optimizer, criterion, device, scaler=edge_label_scaler, max_norm=1e-2)
            # Валидация
            valid_metrics = valid(model, val_loader, criterion, device, scaler=edge_label_scaler)

            # Логирование метрик
            _log_metrics(train_metrics, "train", writer, epoch)
            _log_metrics(valid_metrics, "val", writer, epoch)

            # Сохранение лучшей модели
            if best_score > valid_metrics[cfg['train']['score_metric']]:
                best_epoch = epoch
                best_score = valid_metrics[cfg['train']['score_metric']]
                torch.save(model.state_dict(), log_dir / 'best_model.pth')

            # Обновление lr
            if scheduler is not None:
                scheduler.step()

            # Обновление прогресс-бара
            pbar.set_postfix({
                'best': f'{best_epoch+1:04d}',
                'LR': f'{optimizer.param_groups[0]["lr"]:7.1e}',
                'Train': '|'.join([f'{k} {v:7.1e}' for k, v in train_metrics.items()]),
                'Val': '|'.join([f'{k} {v:7.1e}' for k, v in valid_metrics.items()]),
            })
            pbar.update(1)

    # Загрузка лучшей модели для тестирования
    state_dict = torch.load(log_dir / 'best_model.pth', weights_only=True)
    model = create_model()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Оценка на тестовых данных
    test_metrics = valid(model, test_loader, criterion, device, scaler=edge_label_scaler)
    _log_metrics(test_metrics, "test", writer, 0)

    print(f"Тест: {'|'.join([f'{k} {v:7.1e}' for k, v in test_metrics.items()])}")

    writer.close()
    if run_clear_ml:
        task.close()  # Завершение задачи ClearML


def test_exp(exp_dir_path, results_dir_path, cfg, num_samples_to_draw=None):
    """Тестирование модели и сохранение результатов."""
    exp_dir_path = Path(exp_dir_path)
    results_dir_path = Path(results_dir_path)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg['utils']['device'])

    # 2) Подготовка данных
    cfg['dataset']['load'] = True
    dataset, scalers, _, _, test_loader, ideal_dataset = prepare_data(
        cfg['dataset'],
        cfg['dataloader'],
        cfg['utils']['seed']
    )
    names_dict= {0: 'fwd', 1: 'bwd'}
    def test(exp_dir_path, results_dir_path, cfg, dataset, scalers, test_loader, ideal_dataset, i = None):
        if i is not None:
            print(f"Тестирование {names_dict[i]} данных")
        # 3) Пример батча
        for batch in test_loader:
            print("Пример тестового батча:")
            if i is not None:
                print(batch[i])
            else:
                print(batch)
            break

        # 4) Инициализация модели
        in_node_dim = dataset[0].x.shape[1]
        in_edge_dim = dataset[0].edge_attr.shape[1]
        out_dim = dataset[0].edge_label.shape[-1]
        model_module = importlib.import_module(f"src.models.{cfg['model']['name']}")
        ModelClass = getattr(model_module, cfg['model']['name'])

        def create_model():
            return ModelClass(
                in_node_dim=in_node_dim,
                in_edge_dim=in_edge_dim,
                out_dim=out_dim,
                **cfg['model']['kwargs']
            )

        model = create_model().to(device)

        # 5) Функция потерь
        if cfg['criterion']['name'] is not None:
            if cfg['criterion']['name'] == 'FocalRegressionLoss':
                criterion_fn = FocalRegressionLoss
            if cfg['criterion']['name'] == 'FocalLoss':
                criterion_fn = FocalLoss
            if cfg['criterion']['name'] == 'MulticlassFocalLoss':
                criterion_fn = MulticlassFocalLoss
            elif cfg['criterion']['name'] == 'weighted_mse_loss':
                criterion_fn = weighted_mse_loss
            else:
                criterion_fn = getattr(importlib.import_module('torch.nn'), cfg['criterion']['name'])
            if 'pos_weight' in cfg['criterion']['kwargs']:
                cfg['criterion']['kwargs']['pos_weight'] = torch.Tensor(
                    cfg['criterion']['kwargs']['pos_weight']
                ).to(device)
            if 'weight' in cfg['criterion']['kwargs']:
                cfg['criterion']['kwargs']['weight'] = torch.Tensor(
                    cfg['criterion']['kwargs']['weight']
                ).to(device)
            criterion = criterion_fn(**cfg['criterion']['kwargs'])
        else:
            criterion = None

        # 6) Загрузка весов
        state = torch.load(exp_dir_path / 'best_model.pth', weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # 7) Регрессионная оценка
        edge_label_scaler = None
        test_metrics = valid(model, test_loader, criterion, device,
                            scaler=edge_label_scaler)
        print("Тест (классификация): " +
            "|".join(f"{k}={v:.3e}" for k, v in test_metrics.items()))

        # 8) Собираем предсказания по-ребру
        all_data = []
        all_predictions = []
        all_targets = []
        scalers['edge_label_scaler'] = None

        with torch.no_grad():
            for batch in test_loader:
                if i is None:
                    batch = batch.to(device)
                else:
                    batch = batch[i].to(device)
                    
                preds = model(batch)  # [batch_size, num_classes]
                pred_classes = preds.argmax(dim=1)  # [batch_size]
                
                all_predictions.extend(pred_classes.cpu().numpy())
                all_targets.extend(batch.edge_label.cpu().numpy())
                
                # Сохраняем предсказания для каждого графа
                offset = 0
                for d in batch.to_data_list():
                    d.edge_label_pred = pred_classes[offset:offset+1].cpu()  # [1] - класс для всего графа
                    offset += 1
                    all_data.append(d.cpu())

        def get_t_outside(sample):
            return int(sample.global_attrs[0][0].item())
        
        ideal_data_dict = dict()
        for sample in ideal_dataset:
            ideal_data_dict[get_t_outside(sample)] = sample

        def get_ideal_sample(sample):
            return ideal_data_dict.get(get_t_outside(sample), None)


        def get_tables(d, with_pred=True):
            """Получает таблицы узлов и ребер из графа."""
            nodes_df, edges_df = data_to_tables(
                d,
                node_attr=cfg['dataset']['node_attr'],
                edge_attr=cfg['dataset']['edge_attr'],
                edge_label=cfg['dataset']['edge_label'],
                scalers=scalers,
                edge_label_pred=[f"{v}_pred" for v in cfg['dataset']['edge_label']] if with_pred else None
            )
            return nodes_df, edges_df
        
        def get_denormed_data(d, nodes_df, edges_df, with_pred=True):
            """Получает денормализованные данные из графа."""
            denorm = d.clone().cpu()
            denorm.x = torch.tensor(
                nodes_df[cfg['dataset']['node_attr']].values, dtype=torch.float)
            denorm.edge_attr = torch.tensor(
                edges_df[cfg['dataset']['edge_attr']].values, dtype=torch.float)
            denorm.edge_label = torch.tensor(
                edges_df[cfg['dataset']['edge_label']].values, dtype=torch.float)
            if with_pred:
                denorm.edge_label_pred = torch.tensor(
                    edges_df[[f'{v}_pred' for v in cfg['dataset']['edge_label']]].values,
                    dtype=torch.float
                )
            return denorm

        # 11) Обработка каждого примера
        correct_predictions = []
        wrong_predictions = []
        for idx, d in enumerate(tqdm.tqdm(all_data, desc="Обработка примеров")):
            sample_name = Path(d.nodes_fp).stem
            
            # True и predicted классы
            true_class = d.edge_label.item()  # scalar
            pred_class = d.edge_label_pred.item()  # scalar
            nodes_df, edges_df = get_tables(d)
            denorm = get_denormed_data(d, nodes_df, edges_df)
            d = denorm
            # Сохраняем результаты
            result = {
                'sample_name': sample_name,
                'true_class': true_class,
                'pred_class': pred_class,
                'correct': true_class == pred_class
            }

            sample_dir = Path('.').joinpath(*(Path(d.nodes_fp).parts)[2:-1])
            sample_results_dir_path = results_dir_path / sample_dir
            sample_results_dir_path.mkdir(parents=True, exist_ok=True)

            out_nodes_path = sample_results_dir_path / Path(d.nodes_fp).with_suffix('.csv').name
            out_edges_path = sample_results_dir_path / Path(d.edges_fp).with_suffix('.csv').name
            nodes_df.to_csv(out_nodes_path, index=False)
            edges_df.to_csv(out_edges_path, index=False)
            
            if true_class == pred_class:
                correct_predictions.append(result)
            else:
                wrong_predictions.append(result)
                

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        print("Confusion Matrix:")
        print(cm)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions))

        # Accuracy
        accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
        print(f"Overall Accuracy: {accuracy:.4f}")
        unique_classes = np.unique(all_targets)
        last_class = max(unique_classes)  # Последний класс как negative
        
        # Преобразуем в бинарные метки
        binary_targets = np.array(all_targets) != last_class  # True для positive (все классы кроме последнего)
        binary_predictions = np.array(all_predictions) != last_class  # True для positive
        
        # Вычисляем метрики
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(binary_targets, binary_predictions, zero_division=0)
        recall = recall_score(binary_targets, binary_predictions, zero_division=0)
        f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
        
        print("\n" + "="*50)
        print("БИНАРНАЯ КЛАССИФИКАЦИЯ:")
        print(f"Positive классы: все кроме {last_class}")
        print(f"Negative класс: {last_class}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        # Дополнительная статистика для бинарной классификации
        tn = np.sum((~binary_targets) & (~binary_predictions))  # True Negative
        fp = np.sum((~binary_targets) & binary_predictions)     # False Positive
        fn = np.sum(binary_targets & (~binary_predictions))     # False Negative
        tp = np.sum(binary_targets & binary_predictions)        # True Positive
        
        print(f"\nМатрица ошибок (бинарная):")
        print(f"True Negative (TN): {tn}")
        print(f"False Positive (FP): {fp}")
        print(f"False Negative (FN): {fn}")
        print(f"True Positive (TP): {tp}")
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Specificity (TNR): {specificity:.4f}")
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"Negative Predictive Value (NPV): {npv:.4f}")
        # По классам
        unique_classes = np.unique(all_targets)
        for class_id in unique_classes:
            class_mask = np.array(all_targets) == class_id
            class_accuracy = (np.array(all_predictions)[class_mask] == class_id).mean()
            print(f"Class {class_id} Accuracy: {class_accuracy:.4f}")
            
        # Детальная статистика по ошибкам
        print("\nДетальная статистика по ошибкам:")
        print("=" * 50)

        # Получаем отсортированные уникальные классы
        unique_classes_sorted = sorted(unique_classes)
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes_sorted)}

        # Создаем красивую таблицу с статистикой ошибок
        error_stats = []
        for true_class in unique_classes_sorted:
            true_idx = class_to_index[true_class]
            for pred_class in unique_classes_sorted:
                pred_idx = class_to_index[pred_class]
                count = cm[true_idx, pred_idx]
                if true_class != pred_class:  # Только ошибочные предсказания
                    error_stats.append({
                        'Истинный класс': true_class,
                        'Предсказанный класс': pred_class,
                        'Количество ошибок': count,
                        'Доля от всех ошибок': f"{(count / cm.sum() * 100):.2f}%",
                        'Доля от класса': f"{(count / cm[true_idx].sum() * 100):.2f}%"
                    })

        # Сортируем по количеству ошибок (от больших к меньшим)
        error_stats.sort(key=lambda x: x['Количество ошибок'], reverse=True)

        # Выводим таблицу
        if error_stats:
            print("Топ ошибок (по количеству):")
            print("-" * 80)
            for i, stat in enumerate(error_stats[:10]):  # Показываем топ-10 ошибок
                print(f"{i+1:2d}. True:{stat['Истинный класс']} -> Pred:{stat['Предсказанный класс']}: "
                    f"{stat['Количество ошибок']:3d} ошибок "
                    f"({stat['Доля от всех ошибок']} от всех, "
                    f"{stat['Доля от класса']} от класса {stat['Истинный класс']})")
        else:
            print("Ошибок не обнаружено!")

        # Статистика по классам
        print("\nСтатистика по классам:")
        print("-" * 40)
        for class_id in unique_classes_sorted:
            class_idx = class_to_index[class_id]
            total_samples = cm[class_idx].sum()
            correct_predictions = cm[class_idx, class_idx]
            wrong_predictions = total_samples - correct_predictions
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            print(f"Класс {class_id}:")
            print(f"  Всего образцов: {total_samples}")
            print(f"  Правильно: {correct_predictions} ({accuracy:.2%})")
            print(f"  Ошибок: {wrong_predictions} ({(wrong_predictions/total_samples):.2%})")
            
            # Показываем, в какие классы ошибалась модель для этого класса
            wrong_distribution = []
            for pred_class in unique_classes_sorted:
                if pred_class != class_id:
                    pred_idx = class_to_index[pred_class]
                    wrong_count = cm[class_idx, pred_idx]
                    if wrong_count > 0:
                        wrong_distribution.append(f"{pred_class}({wrong_count})")
            
            if wrong_distribution:
                print(f"  Ошибочные предсказания: {', '.join(wrong_distribution)}")
            print()

        # Общая статистика
        total_samples = len(all_targets)
        total_errors = total_samples - np.trace(cm)
        overall_accuracy = np.trace(cm) / total_samples

        print("ОБЩАЯ СТАТИСТИКА:")
        print(f"Всего образцов: {total_samples}")
        print(f"Общая точность: {overall_accuracy:.2%}")
        print(f"Всего ошибок: {total_errors} ({(total_errors/total_samples):.2%})")

        # Самые частые типы ошибок
        if error_stats:
            most_common_error = error_stats[0]
            print(f"Самая частая ошибка: класс {most_common_error['Истинный класс']} -> "
                f"класс {most_common_error['Предсказанный класс']} "
                f"({most_common_error['Количество ошибок']} раз, "
                f"{most_common_error['Доля от всех ошибок']} от всех ошибок)")
        # Визуализация распределения классов (ДОБАВИТЬ)
        plt.figure(figsize=(10, 6))
        plt.hist(all_targets, bins=len(unique_classes), alpha=0.7, label='True')
        plt.hist(all_predictions, bins=len(unique_classes), alpha=0.7, label='Predicted')
        plt.xlabel('Class ID')
        plt.ylabel('Count')
        plt.title('Class Distribution - True vs Predicted')
        plt.legend()
        plt.savefig(results_dir_path / 'class_distribution.png')
        plt.close()
    
    if cfg['dataset']['name'] == 'Termo_model_fwd_and_bwd':
        test(exp_dir_path, Path(os.path.join(results_dir_path, "fwd")), cfg, dataset[0], scalers[0], test_loader, ideal_dataset[0], i=0)
        test(exp_dir_path, Path(os.path.join(results_dir_path, "bwd")), cfg, dataset[1], scalers[1], test_loader, ideal_dataset[1], i=1)
    else:
        test(exp_dir_path, results_dir_path, cfg, dataset, scalers, test_loader, ideal_dataset, i =None)