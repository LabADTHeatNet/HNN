import time
import datetime

import torch
import torch.nn.functional as F


def get_str_timestamp(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    date_time = datetime.datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
    return str_date_time


# Функции для метрик
def compute_metrics(pred, target):
    mae = F.l1_loss(pred.squeeze(), target).item()  # Mean Absolute Error
    mse = F.mse_loss(pred.squeeze(), target).item()  # Mean Squared Error
    # Root Mean Squared Error
    rmse = torch.sqrt(F.mse_loss(pred.squeeze(), target)).item()
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}

# Функция для взвешенной потери


def weighted_mse_loss(pred, target, weight):
    loss = F.mse_loss(pred, target, reduction='none')
    weighted_loss = loss * weight
    return weighted_loss.mean()

# Функция обучения


def epoch(model, loader, optimizer, criterion, device, train=True):
    total_loss = 0
    all_preds, all_targets = [], []
    for data in loader:
        data = data.to(device)
        if train:
            optimizer.zero_grad()
        edge_pred = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(edge_pred.squeeze(), data.edge_label)
        total_loss += loss.item()
        all_preds.append(edge_pred.cpu())
        all_targets.append(data.edge_label.cpu())

        if train:
            loss.backward()
            optimizer.step()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = dict(Loss=total_loss / len(loader))
    metrics.update(compute_metrics(all_preds, all_targets))
    return metrics

# Функция валидации и тестирования


def train(model, loader, optimizer, criterion, device):
    model.train()
    return epoch(model, loader, optimizer, criterion, device, train=True)


@torch.no_grad()
def valid(model, loader, criterion, device):
    model.eval()
    return epoch(model, loader, None, criterion, device, train=False)
