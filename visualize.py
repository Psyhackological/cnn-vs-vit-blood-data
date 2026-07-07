# visualize.py – wykresy treningu i diagnostyki

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from evaluate import brier_score_multiclass, expected_calibration_error, one_hot


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _save(fig: plt.Figure, path_no_ext: str, dpi: int = 150) -> None:
    fig.savefig(f"{path_no_ext}.png", dpi=dpi)
    fig.savefig(f"{path_no_ext}.pdf")
    print(f"Zapisano: {path_no_ext}.png oraz {path_no_ext}.pdf")


def plot_history(
    history: dict[str, Any],
    model_name: str,
    dataset_name: str,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f"Training history – {dataset_name} – {model_name}", fontsize=14)

    for ax, key_tr, key_val, title in (
        (axes[0], "train_loss", "val_loss", "Loss (Cross-Entropy)"),
        (axes[1], "train_acc", "val_acc", "Accuracy"),
    ):
        ax.plot(epochs, history[key_tr], label="Train")
        ax.plot(epochs, history[key_val], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    axes[2].plot(epochs, history["val_auc"], label="Val macro OvR AUC")
    axes[2].set_title("Validation AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    path = os.path.join(
        save_dir, f"{_safe_name(dataset_name)}_{_safe_name(model_name)}_history"
    )
    _save(fig, path)
    plt.close(fig)


def _plot_confusion(
    ax: plt.Axes,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    order: list[int] | None,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    if order is not None:
        cm = cm[np.ix_(order, order)]
        class_names = [class_names[i] for i in order]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Row-normalized confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j] * 100:.0f}%",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=6,
            )


def _curve_classes(num_classes: int) -> list[int]:
    return [1] if num_classes == 2 else list(range(num_classes))


def _macro_curve(
    curves: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    grid = np.linspace(0.0, 1.0, 101)
    if not curves:
        return grid, np.full_like(grid, np.nan), float("nan")
    interp = [np.interp(grid, x, y) for x, y, _ in curves]
    return (
        grid,
        np.mean(interp, axis=0),
        float(np.mean([score for _, _, score in curves])),
    )


def _plot_roc(
    ax: plt.Axes,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> None:
    labels_oh = one_hot(labels, probs.shape[1])
    curves = []
    for idx in _curve_classes(probs.shape[1]):
        if len(np.unique(labels_oh[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels_oh[:, idx], probs[:, idx])
        score = float(auc(fpr, tpr))
        curves.append((fpr, tpr, score))
        ax.plot(
            fpr, tpr, linewidth=1, alpha=0.55, label=f"{class_names[idx]} ({score:.3f})"
        )

    x, y, score = _macro_curve(curves)
    ax.plot(x, y, color="black", linewidth=2.5, label=f"macro AUC={score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC OvR + macro")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")


def _plot_pr(
    ax: plt.Axes,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> None:
    labels_oh = one_hot(labels, probs.shape[1])
    curves = []
    for idx in _curve_classes(probs.shape[1]):
        if len(np.unique(labels_oh[:, idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels_oh[:, idx], probs[:, idx])
        score = float(average_precision_score(labels_oh[:, idx], probs[:, idx]))
        order = np.argsort(recall)
        curves.append((recall[order], precision[order], score))
        ax.plot(
            recall,
            precision,
            linewidth=1,
            alpha=0.55,
            label=f"{class_names[idx]} ({score:.3f})",
        )

    x, y, score = _macro_curve(curves)
    ax.plot(x, y, color="black", linewidth=2.5, label=f"macro AP={score:.3f}")
    ax.axhline(
        labels_oh[:, _curve_classes(probs.shape[1])].mean(),
        linestyle="--",
        color="gray",
        linewidth=1,
    )
    ax.set_title("Precision–Recall OvR + macro")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7, loc="lower left")


def _plot_reliability(
    ax: plt.Axes,
    labels: np.ndarray,
    probs: np.ndarray,
    ece_bins: int,
) -> None:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    accs = np.full(ece_bins, np.nan)
    confs = np.full(ece_bins, np.nan)
    counts = np.zeros(ece_bins, dtype=int)

    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= left) & (conf < right)
        if i == ece_bins - 1:
            mask = (conf >= left) & (conf <= right)
        counts[i] = int(mask.sum())
        if np.any(mask):
            accs[i] = correct[mask].mean()
            confs[i] = conf[mask].mean()

    valid = ~np.isnan(accs)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(confs[valid], accs[valid], marker="o", linewidth=2)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Empirical accuracy",
        title="Reliability diagram",
    )
    ax.twinx().bar(centers, counts, width=1.0 / ece_bins, alpha=0.18, color="tab:gray")
    ax.text(
        0.04,
        0.96,
        f"ECE={expected_calibration_error(labels, probs, ece_bins):.3f}\n"
        f"Brier={brier_score_multiclass(labels, probs):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )


def plot_diagnostic_panel(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    model_name: str,
    dataset_name: str,
    save_dir: str,
    ece_bins: int = 15,
    class_order: list[int] | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(f"Diagnostics – {dataset_name} – {model_name}", fontsize=14)
    _plot_confusion(axes[0, 0], labels, preds, class_names, class_order)
    _plot_roc(axes[0, 1], labels, probs, class_names)
    _plot_pr(axes[1, 0], labels, probs, class_names)
    _plot_reliability(axes[1, 1], labels, probs, ece_bins)
    path = os.path.join(
        save_dir,
        f"{_safe_name(dataset_name)}_{_safe_name(model_name)}_diagnostic_panel",
    )
    _save(fig, path)
    plt.close(fig)
