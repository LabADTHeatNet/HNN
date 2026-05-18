"""Evaluation metrics for baseline models, matching the GNN test_exp output.

Replicates the printed output format from exp.py:test() including:
- Full confusion matrix (44x44)
- Classification report
- Binary classification (class 43 = negative)
- Per-class accuracy, top error table
- Saves results to disk (metrics.json, confusion_matrix.npy, etc.)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def binary_metrics(y_true, y_pred, negative_class=43):
    """Compute binary classification metrics treating *negative_class* as negative.

    All other classes are collapsed into "positive".

    Returns:
        dict with keys: precision, recall, f1, tn, fp, fn, tp, specificity, npv.
    """
    binary_true = np.array(y_true) != negative_class
    binary_pred = np.array(y_pred) != negative_class

    tn = np.sum((~binary_true) & (~binary_pred))
    fp = np.sum((~binary_true) & binary_pred)
    fn = np.sum(binary_true & (~binary_pred))
    tp = np.sum(binary_true & binary_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv_ = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": specificity,
        "npv": npv_,
    }


def evaluate(y_true, y_pred, out_dir=None, num_classes=44):
    """Compute and print all metrics matching the GNN test_exp format.

    Args:
        y_true: list or 1-D array of true class labels (0 .. num_classes-1).
        y_pred: list or 1-D array of predicted class labels.
        y_pred_proba_full: optional [n, num_classes] probability array
            (from predict_proba_full). If provided, macro-avg metrics are
            also shown.
        out_dir: optional Path or str — if given, results are saved there.
        num_classes: total number of classes (default 44).

    Returns:
        metrics dict with scalar entries for logging.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out_dir = Path(out_dir) if out_dir else None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    unique_classes = sorted(set(y_true) | set(y_pred))

    print("Confusion Matrix:")
    print(cm)
    if out_dir:
        np.save(out_dir / "confusion_matrix.npy", cm)

    # --- Classification report ---
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, zero_division=0))
    if out_dir:
        report_str = classification_report(
            y_true, y_pred, labels=unique_classes, zero_division=0
        )
        (out_dir / "classification_report.txt").write_text(report_str)

    # --- Overall accuracy ---
    accuracy = (y_true == y_pred).mean()
    print(f"Overall Accuracy: {accuracy:.4f}")

    # --- Binary classification ---
    last_class = max(unique_classes) if unique_classes else num_classes - 1
    bin_metrics = binary_metrics(y_true, y_pred, negative_class=last_class)

    print("\n" + "=" * 50)
    print("БИНАРНАЯ КЛАССИФИКАЦИЯ:")
    print(f"Positive классы: все кроме {last_class}")
    print(f"Negative класс: {last_class}")
    print(f"Precision: {bin_metrics['precision']:.4f}")
    print(f"Recall: {bin_metrics['recall']:.4f}")
    print(f"F1-score: {bin_metrics['f1']:.4f}")
    print(f"\nМатрица ошибок (бинарная):")
    print(f"True Negative (TN): {bin_metrics['tn']}")
    print(f"False Positive (FP): {bin_metrics['fp']}")
    print(f"False Negative (FN): {bin_metrics['fn']}")
    print(f"True Positive (TP): {bin_metrics['tp']}")
    print(f"Specificity (TNR): {bin_metrics['specificity']:.4f}")
    print(f"Negative Predictive Value (NPV): {bin_metrics['npv']:.4f}")

    # --- Per-class accuracy ---
    print("\nPer-class Accuracy:")
    for cls in unique_classes:
        mask = y_true == cls
        cls_acc = (y_pred[mask] == cls).mean()
        print(f"  Class {cls} Accuracy: {cls_acc:.4f}")

    # --- Top error table ---
    print("\nДетальная статистика по ошибкам:")
    print("=" * 50)
    error_stats = []
    for true_cls in unique_classes:
        for pred_cls in unique_classes:
            if true_cls != pred_cls:
                idx_t = unique_classes.index(true_cls)
                idx_p = unique_classes.index(pred_cls)
                count = cm[idx_t, idx_p]
                if count > 0:
                    error_stats.append({
                        "true": true_cls,
                        "pred": pred_cls,
                        "count": count,
                        "pct_all": count / cm.sum() * 100,
                        "pct_class": count / cm[idx_t].sum() * 100,
                    })
    error_stats.sort(key=lambda x: x["count"], reverse=True)

    if error_stats:
        print("Топ ошибок (по количеству):")
        print("-" * 80)
        for i, s in enumerate(error_stats[:10]):
            print(
                f"{i+1:2d}. True:{s['true']:4d} -> Pred:{s['pred']:4d}: "
                f"{s['count']:3d} ошибок "
                f"({s['pct_all']:.2f}% от всех, "
                f"{s['pct_class']:.2f}% от класса {s['true']})"
            )
    else:
        print("Ошибок не обнаружено!")

    # # --- Per-class statistics ---
    print("\nСтатистика по классам:")
    print("-" * 40)
    for cls in unique_classes:
        idx = unique_classes.index(cls)
        total = cm[idx].sum()
        correct = cm[idx, idx]
        wrong = total - correct
        cls_acc = correct / total if total > 0 else 0.0
        pct_str = f"{wrong/total:.2%}" if total > 0 else "N/A"
        print(f"  Класс {cls}:")
        print(f"    Всего образцов: {total}")
        print(f"    Правильно: {correct} ({cls_acc:.2%})")
        print(f"    Ошибок: {wrong} ({pct_str})")
        # wrong distribution
        dist = []
        for pred_cls in unique_classes:
            if pred_cls != cls:
                p_idx = unique_classes.index(pred_cls)
                c = cm[idx, p_idx]
                if c > 0:
                    dist.append(f"{pred_cls}({c})")
        if dist:
            print(f"    Ошибочные предсказания: {', '.join(dist)}")
        print()

    # --- Overall statistics ---
    total_samples = len(y_true)
    total_errors = total_samples - np.trace(cm)
    overall_acc = np.trace(cm) / total_samples
    print("ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего образцов: {total_samples}")
    print(f"Общая точность: {overall_acc:.2%}")
    print(f"Всего ошибок: {total_errors} ({(total_errors/total_samples):.2%})")

    if error_stats:
        s = error_stats[0]
        print(f"Самая частая ошибка: класс {s['true']} -> "
              f"класс {s['pred']} ({s['count']} раз, "
              f"{s['pct_all']:.2f}% от всех ошибок)")

    # --- Class distribution plot ---
    if out_dir:
        plt.figure(figsize=(10, 6))
        bins = np.arange(num_classes + 1) - 0.5
        plt.hist(y_true, bins=bins, alpha=0.7, label="True")
        plt.hist(y_pred, bins=bins, alpha=0.7, label="Predicted")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("Class Distribution - True vs Predicted")
        plt.xticks(range(0, num_classes, max(1, num_classes // 10)))
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "class_distribution.png", dpi=150)
        plt.close()

    # --- Save summary metrics ---
    metrics = {
        "accuracy": float(accuracy),
        "binary_precision": float(bin_metrics["precision"]),
        "binary_recall": float(bin_metrics["recall"]),
        "binary_f1": float(bin_metrics["f1"]),
        "binary_specificity": float(bin_metrics["specificity"]),
        "binary_npv": float(bin_metrics["npv"]),
        "total_samples": int(total_samples),
        "total_errors": int(total_errors),
        "num_classes": int(len(unique_classes)),
    }

    if out_dir:
        (out_dir / "predictions.csv").write_text(
            "true_class,pred_class\n"
            + "\n".join(f"{t},{p}" for t, p in zip(y_true, y_pred))
        )
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics