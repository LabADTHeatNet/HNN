import time
import datetime

import torch
import torch.nn.functional as F


def get_str_timestamp(timestamp=None):
    """Генерация строки с временной меткой в формате ГГГГММДД_ЧЧММСС."""
    if timestamp is None:
        timestamp = time.time()
    date_time = datetime.datetime.fromtimestamp(timestamp)
    return date_time.strftime("%Y%m%d_%H%M%S")


def compute_metrics(pred, target, scaler=None):
    """Вычисление метрик качества: MAE, MSE и RMSE (после денормализации)."""
    mae = F.l1_loss(pred, target).item()  # Средняя абсолютная ошибка
    mse = F.mse_loss(pred, target).item()  # Средняя квадратичная ошибка

    ret = {"MAE": mae, "MSE": mse}
    if scaler is not None:
        # Обратное преобразование предсказаний и целей
        real_pred = torch.Tensor(scaler.inverse_transform(pred.detach().numpy()))
        real_target = torch.Tensor(scaler.inverse_transform(target.detach().numpy()))
        real_mae = F.l1_loss(real_pred, real_target).item()
        real_mse = F.mse_loss(real_pred, real_target).item()
        ret["real_MAE"] = real_mae  # MAE в исходном масштабе
        ret["real_MSE"] = real_mse  # MSE в исходном масштабе
    return ret


def compute_metrics_cls(pred, target, scaler=None):
    """
    Для классификации по moded: считаем accuracy.
    pred: логиты [num_edges, 3]
    target: [num_edges] (значения 0,1,2)
    """
    with torch.no_grad():
        pred_labels = pred.argmax(dim=1)
        correct = (pred_labels == target).sum().item()
        total = target.size(0)
        acc_score = 1 - correct / total
    return {"Acc_score": acc_score}


def weighted_mse_loss(pred, target, weight):
    """Взвешенная MSE-функция потерь."""
    loss = F.mse_loss(pred, target, reduction='none')
    weighted_loss = loss * weight  # Применение весов к ошибкам
    return weighted_loss.mean()


def epoch(model, loader, optimizer, criterion, device, train=True, scaler=None):
    """Одна эпоха обучения или валидации."""
    total_loss = 0
    all_preds, all_targets = [], []
    for data in loader:
        data = data.to(device)
        if train:
            optimizer.zero_grad()

        # Прямой проход
        edge_pred = model(data)
        loss = criterion(edge_pred, data.edge_label)
        total_loss += loss.item()

        # Сохранение предсказаний и целей для метрик
        all_preds.append(edge_pred.cpu())
        all_targets.append(data.edge_label.cpu())

        if train:
            loss.backward()  # Обратное распространение
            optimizer.step()  # Обновление весов

    # Агрегация метрик
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = dict(Loss=total_loss / len(loader))
    metrics.update(compute_metrics(all_preds, all_targets, scaler=scaler))
    return metrics


def train(model, loader, optimizer, criterion, device, scaler=None):
    """Обучение модели на одном эпохе."""
    model.train()
    return epoch(model, loader, optimizer, criterion, device, train=True, scaler=scaler)


@torch.no_grad()
def valid(model, loader, criterion, device, scaler=None):
    """Валидация модели на одном эпохе."""
    model.eval()
    return epoch(model, loader, None, criterion, device, train=False, scaler=scaler)
