import numpy as np
import pandas as pd
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


class FocalRegressionLoss(torch.nn.Module):
    """
    Focal Loss для регрессии: добавляет к MSE модификатор, 
    усиливающий вклад больших ошибок.
    L = α * (1 - exp(-MSE))^γ * MSE
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Квадратичная ошибка
        se = (input - target) ** 2
        # Аналог “pt” из классического Focal Loss
        pt = torch.exp(-se)
        # Модифицированный весовой коэффициент
        modulator = (1 - pt) ** self.gamma
        loss = self.alpha * modulator * se

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(torch.nn.Module):
    """
    Binary Focal Loss:
      L = α * (1 - p_t)^γ * BCEWithLogitsLoss
    где p_t = sigmoid(logit) для положительного класса и 1 - sigmoid(logit) для отрицательного.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: float tensor 0.0 или 1.0, same shape as logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce_loss)  # это sigmoid(logits)*targets + (1-sigmoid(logits))*(1-targets)
        focal_term = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def epoch(model, loader, optimizer, criterion, device, train=True, scaler=None, max_norm=1.0):
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

            # ---- gradient clipping ----
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # ----------------------------

            optimizer.step()  # Обновление весов

    # Агрегация метрик
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = dict(Loss=total_loss / len(loader))
    metrics.update(compute_metrics(all_preds, all_targets, scaler=scaler))
    return metrics


def train(model, loader, optimizer, criterion, device, scaler=None, max_norm=1.0):
    """Обучение модели на одном эпохе."""
    model.train()
    return epoch(model, loader, optimizer, criterion, device, train=True, scaler=scaler, max_norm=max_norm)


@torch.no_grad()
def valid(model, loader, criterion, device, scaler=None):
    """Валидация модели на одном эпохе."""
    model.eval()
    return epoch(model, loader, None, criterion, device, train=False, scaler=scaler)


class IdealValueScaler:
    """
    Scaler that subtracts the per-index “ideal” value (mean over dataset)
    from each element of a DataFrame, and can reverse the operation.

    Usage:
      # 1) Передаём первый sample как DataFrame (N×M)
      scaler = IdealScaler(example_df)
      # 2) Собираем список всех sample DataFrame тех же N×M
      scaler.fit(list_of_dfs)
      # 3) Масштабируем любой DataFrame
      df_scaled = scaler.transform(df)
      # 4) Возвращаем оригинал
      df_orig   = scaler.inverse_transform(df_scaled)
    """

    def __init__(self, example_df: pd.DataFrame):
        # запоминаем структуру
        self.index = example_df.index
        self.columns = example_df.columns
        # инициализируем ideal_ нулями той же формы
        self.ideal_ = pd.DataFrame(
            data=np.zeros_like(example_df.values, dtype=float),
            index=self.index,
            columns=self.columns
        )
        self.fitted = False

    def fit(self, df: list[pd.DataFrame]):
        """
        Вычисляет ideal_ = mean(df.values) по списку DataFrame.
        Все df в списке должны иметь одинаковые index и columns.
        """

        # stack into array shape (S, N, M)
        arr = df.to_numpy().reshape(-1, len(self.index), len(self.columns))
        # compute mean over first axis → (N, M)
        mean_arr = arr.mean(axis=0)
        # store as DataFrame
        self.ideal_ = pd.DataFrame(mean_arr, index=self.index, columns=self.columns)
        self.fitted = True
        return self

    def transform(self, data) -> pd.DataFrame:
        """
        Вычитает ideal_ из входного DataFrame того же shape.
        """
        df_orig = None
        if type(data) is pd.DataFrame:
            df_orig, data = data.copy(), data.to_numpy()

        arr = data.reshape(-1, len(self.index), len(self.columns))
        for i, v in enumerate(arr):
            arr[i] = v - self.ideal_

        ret_data = arr.reshape(-1, len(self.columns))
        if df_orig is not None:
            ret_data = pd.DataFrame(
                data=ret_data,
                index=df_orig.index,
                columns=df_orig.columns
            )
        return ret_data

    def inverse_transform(self, data) -> pd.DataFrame:
        """
        Прибавляет ideal_ обратно.
        """
        df_orig = None
        if type(data) is pd.DataFrame:
            df_orig, data = data.copy(), data.to_numpy()

        arr = data.reshape(-1, len(self.index), len(self.columns))
        for i, v in enumerate(arr):
            arr[i] = v + self.ideal_

        ret_data = arr.reshape(-1, len(self.columns))
        if df_orig is not None:
            ret_data = pd.DataFrame(
                data=ret_data,
                index=df_orig.index,
                columns=df_orig.columns
            )
        return ret_data
