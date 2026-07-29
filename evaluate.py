# evaluate.py – zunifikowane metryki dla zbiorow MedMNIST

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

SCALAR_METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "auc_macro_ovr",
    "ap_macro",
    "mcc",
    "cohen_kappa",
    "ece",
    "brier",
    "min_per_class_recall",
    "min_per_class_auc",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float64)[labels.astype(int)]


def macro_ovr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if probs.shape[1] == 2:
            return _safe_float(roc_auc_score(labels, probs[:, 1]))
        return _safe_float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def macro_average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return _safe_float(
            average_precision_score(
                one_hot(labels, probs.shape[1]), probs, average="macro"
            )
        )
    except ValueError:
        return float("nan")


def per_class_recall(
    labels: np.ndarray, preds: np.ndarray, num_classes: int
) -> np.ndarray:
    return np.asarray(
        recall_score(
            labels,
            preds,
            labels=list(range(num_classes)),
            average=cast(Any, None),
            zero_division=cast(Any, 0),
        ),
        dtype=np.float64,
    )


def per_class_f1(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    return np.asarray(
        f1_score(
            labels,
            preds,
            labels=list(range(num_classes)),
            average=cast(Any, None),
            zero_division=cast(Any, 0),
        ),
        dtype=np.float64,
    )


def per_class_ovr_auc(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    aucs = []
    for class_idx in range(probs.shape[1]):
        try:
            aucs.append(
                _safe_float(
                    roc_auc_score(
                        (labels == class_idx).astype(int), probs[:, class_idx]
                    )
                )
            )
        except ValueError:
            aucs.append(float("nan"))
    return np.asarray(aucs, dtype=np.float64)


def calibration_bins(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dzieli predykcje na kubelki wg pewnosci i zwraca (accuracy, pewnosc, licznosc)
    dla kazdego kubelka. Ostatni kubelek jest domkniety z prawej strony, wiec
    pewnosc rowna 1.0 tez jest liczona. Puste kubelki maja NaN w dwoch pierwszych.
    """
    confidences = probs.max(axis=1)
    correctness = (probs.argmax(axis=1) == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for bin_idx, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (confidences >= left) & (confidences < right)
        if bin_idx == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        counts[bin_idx] = int(in_bin.sum())
        if counts[bin_idx]:
            accs[bin_idx] = correctness[in_bin].mean()
            confs[bin_idx] = confidences[in_bin].mean()

    return accs, confs, counts


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    accs, confs, counts = calibration_bins(labels, probs, n_bins=n_bins)
    filled = counts > 0
    weights = counts[filled] / len(labels)
    return _safe_float(np.sum(weights * np.abs(accs[filled] - confs[filled])))


def brier_score_multiclass(labels: np.ndarray, probs: np.ndarray) -> float:
    labels_oh = one_hot(labels, probs.shape[1])
    return _safe_float(np.mean(np.mean((probs - labels_oh) ** 2, axis=1)))


def scalar_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    ece_bins: int = 15,
) -> dict[str, float]:
    recalls = per_class_recall(labels, preds, probs.shape[1])
    aucs = per_class_ovr_auc(labels, probs)
    return {
        "accuracy": _safe_float(accuracy_score(labels, preds)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(labels, preds)),
        "f1_macro": _safe_float(
            f1_score(labels, preds, average="macro", zero_division=cast(Any, 0))
        ),
        "auc_macro_ovr": macro_ovr_auc(labels, probs),
        "ap_macro": macro_average_precision(labels, probs),
        "mcc": _safe_float(matthews_corrcoef(labels, preds)),
        "cohen_kappa": _safe_float(cohen_kappa_score(labels, preds)),
        "ece": expected_calibration_error(labels, probs, n_bins=ece_bins),
        "brier": brier_score_multiclass(labels, probs),
        "min_per_class_recall": _safe_float(np.nanmin(recalls)),
        "min_per_class_auc": _safe_float(np.nanmin(aucs)),
    }


def bootstrap_scalar_cis(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    ece_bins: int = 15,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in SCALAR_METRIC_NAMES}

    for _ in range(n_resamples):
        idx = rng.integers(0, len(labels), size=len(labels))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_metrics = scalar_metrics(
                labels[idx], preds[idx], probs[idx], ece_bins=ece_bins
            )
        for name, value in sample_metrics.items():
            if not np.isnan(value):
                values[name].append(value)

    cis: dict[str, tuple[float, float]] = {}
    for name in SCALAR_METRIC_NAMES:
        if values[name]:
            lo, hi = np.percentile(np.asarray(values[name]), [2.5, 97.5])
            cis[name] = (_safe_float(lo), _safe_float(hi))
        else:
            cis[name] = (float("nan"), float("nan"))
    return cis


def collect_predictions(
    model, test_loader, device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.reshape(-1).long()

            outputs = model(imgs)
            probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_preds, dtype=np.int64),
        np.asarray(all_probs, dtype=np.float64),
    )


def evaluate_model(
    model,
    test_loader,
    device,
    class_names: list[str],
    ece_bins: int = 15,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    labels, preds, probs = collect_predictions(model, test_loader, device)
    num_classes = len(class_names)
    metrics = scalar_metrics(labels, preds, probs, ece_bins=ece_bins)
    cis = bootstrap_scalar_cis(
        labels,
        preds,
        probs,
        ece_bins=ece_bins,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    per_class = {
        "recall": per_class_recall(labels, preds, num_classes),
        "f1": per_class_f1(labels, preds, num_classes),
        "auc_ovr": per_class_ovr_auc(labels, probs),
    }

    print("\nUnified test metrics:")
    for name in SCALAR_METRIC_NAMES:
        lo, hi = cis[name]
        print(f"  {name:<22}: {metrics[name]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print("\nPer-class metrics:")
    for idx, class_name in enumerate(class_names):
        print(
            f"  {idx:02d} {class_name:<42} "
            f"recall={per_class['recall'][idx]:.4f} "
            f"f1={per_class['f1'][idx]:.4f} "
            f"auc_ovr={per_class['auc_ovr'][idx]:.4f}"
        )

    return {
        "metrics": metrics,
        "cis": cis,
        "per_class": per_class,
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }
