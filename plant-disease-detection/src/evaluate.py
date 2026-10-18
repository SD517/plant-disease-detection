"""Test-set evaluation: accuracy, precision/recall/F1, confusion matrix, speed, size."""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def predict_dataset(model, ds):
    """Return (y_true, probabilities, ms_per_image). Only model time is measured."""
    y_true, probs, elapsed, n = [], [], 0.0, 0
    for x, y in ds:
        t = time.perf_counter()
        p = model.predict_on_batch(x)
        elapsed += time.perf_counter() - t
        probs.append(np.asarray(p))
        y_true.append(y.numpy())
        n += len(y)
    return np.concatenate(y_true), np.concatenate(probs), 1000.0 * elapsed / max(n, 1)


def compute_metrics(y_true, probs, class_names) -> dict:
    y_pred = probs.argmax(axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = float(np.mean([t in row for t, row in zip(y_true, top3)]))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "top3_accuracy": top3_acc,
        "precision_macro": float(p_m), "recall_macro": float(r_m),
        "f1_macro": float(f_m), "f1_weighted": float(f_w),
        "report": report, "confusion_matrix": cm,
    }


def top_confusions(cm, class_names, k: int = 10):
    """The k most frequent wrong (true -> predicted) pairs."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:k]
    return [(class_names[i], class_names[j], int(off[i, j])) for i, j in idx if off[i, j] > 0]


def model_size_mb(path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)
