"""Generate one 2x2 diagnostic panel per saved (dataset, model) result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from evaluate import (
    brier_score_multiclass,
    calibration_bins,
    expected_calibration_error,
)
from utils import safe_name


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    metric_files = sorted(results_dir.glob("*_metrics.json"))
    if metric_files:
        rows = []
        for path in metric_files:
            try:
                with path.open() as f:
                    row = json.load(f)
                if isinstance(row, dict) and "dataset" in row and "model" in row:
                    rows.append(row)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Skipping unreadable metrics file {path}: {exc}")
        return rows

    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No *_metrics.json or summary.json found in {results_dir}."
        )
    with summary_path.open() as f:
        summary = json.load(f)
    if isinstance(summary, dict):
        summary = summary.get("results", [summary])
    if not isinstance(summary, list):
        raise TypeError(f"Unexpected summary format in {summary_path}")
    return [row for row in summary if isinstance(row, dict)]


def _prediction_path(row: dict[str, Any], results_dir: Path) -> Path | None:
    artifact = row.get("test_predictions_file")
    if not artifact:
        return None
    path = Path(artifact)
    candidates = (
        [path]
        if path.is_absolute()
        else [
            Path.cwd() / path,
            results_dir / path.name,
            results_dir.parent / path,
        ]
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _load_predictions(
    row: dict[str, Any], results_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    path = _prediction_path(row, results_dir)
    if path is None:
        print(f"Skipping {row['dataset']}/{row['model']}: prediction NPZ is missing")
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["labels"], dtype=int).reshape(-1)
            preds = np.asarray(data["preds"], dtype=int).reshape(-1)
            probs = np.asarray(data["probs"], dtype=float)
            names = (
                data["class_names"].astype(str).tolist()
                if "class_names" in data.files
                else row.get("per_class", {}).get("class_names", [])
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Skipping unreadable prediction file {path}: {exc}")
        return None

    if probs.ndim != 2 or len(labels) != len(preds) or len(labels) != len(probs):
        print(f"Skipping malformed prediction file {path}")
        return None
    if len(names) != probs.shape[1]:
        names = [str(i) for i in range(probs.shape[1])]
    return labels, preds, probs, names


def _save(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.svg")


def _plot_confusion(
    ax: Any,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    sort_by_frequency: bool,
) -> None:
    n_classes = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    if sort_by_frequency:
        order = np.argsort(-cm.sum(axis=1))
        cm = cm[np.ix_(order, order)]
        class_names = [class_names[i] for i in order]

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )
    annotations = np.asarray(
        [
            [f"{cm[i, j]}\n{normalized[i, j] * 100:.0f}%" for j in range(n_classes)]
            for i in range(n_classes)
        ]
    )
    sns.heatmap(
        normalized,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={"size": 7},
    )
    ax.set_title("Row-normalized confusion matrix", loc="left", weight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, rotation=0, fontsize=7)


def _curve_classes(n_classes: int) -> list[int]:
    return [1] if n_classes == 2 else list(range(n_classes))


def _plot_roc(
    ax: Any,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> None:
    classes = _curve_classes(probs.shape[1])
    curves: list[tuple[np.ndarray, np.ndarray, float]] = []
    for class_idx in classes:
        binary_labels = labels == class_idx
        if np.unique(binary_labels).size < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, class_idx])
        class_auc = float(auc(fpr, tpr))
        curves.append((fpr, tpr, class_auc))
        ax.plot(fpr, tpr, linewidth=1, alpha=0.65, label=f"{class_names[class_idx]} ({class_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Chance")
    if curves:
        grid = np.linspace(0, 1, 101)
        if probs.shape[1] == 2:
            macro_auc = curves[0][2]
            macro_tpr = np.interp(grid, curves[0][0], curves[0][1])
        else:
            macro_tpr = np.mean(
                [np.interp(grid, fpr, tpr) for fpr, tpr, _ in curves], axis=0
            )
            macro_tpr[0], macro_tpr[-1] = 0.0, 1.0
            macro_auc = float(np.mean([class_auc for _, _, class_auc in curves]))
        ax.plot(grid, macro_tpr, color="black", linewidth=2.5, label=f"Macro AUC ({macro_auc:.3f})")

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title("One-vs-rest ROC", loc="left", weight="bold")
    ax.legend(fontsize=6, loc="lower right", frameon=True)


def _plot_precision_recall(
    ax: Any,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> None:
    classes = _curve_classes(probs.shape[1])
    curves: list[tuple[np.ndarray, np.ndarray, float]] = []
    for class_idx in classes:
        binary_labels = labels == class_idx
        if np.unique(binary_labels).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(binary_labels, probs[:, class_idx])
        ap = float(average_precision_score(binary_labels, probs[:, class_idx]))
        order = np.argsort(recall)
        recall, precision = recall[order], precision[order]
        curves.append((recall, precision, ap))
        ax.plot(
            recall,
            precision,
            linewidth=1,
            alpha=0.65,
            label=f"{class_names[class_idx]} (AP={ap:.3f})",
        )

    if curves:
        grid = np.linspace(0, 1, 101)
        if probs.shape[1] == 2:
            macro_precision = np.interp(grid, curves[0][0], curves[0][1])
            macro_ap = curves[0][2]
        else:
            macro_precision = np.mean(
                [np.interp(grid, recall, precision) for recall, precision, _ in curves],
                axis=0,
            )
            macro_ap = float(np.mean([ap for _, _, ap in curves]))
        ax.plot(grid, macro_precision, color="black", linewidth=2.5, label=f"Macro AP ({macro_ap:.3f})")
        prevalence = float(np.mean([np.mean(labels == idx) for idx in classes]))
        ax.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label="Prevalence")

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.set_title("One-vs-rest precision–recall", loc="left", weight="bold")
    ax.legend(fontsize=6, loc="lower left", frameon=True)


def _plot_reliability(
    ax: Any,
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> None:
    accuracies, confidences, counts = calibration_bins(labels, probs, n_bins=n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = counts > 0

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Perfect calibration")
    ax.plot(confidences[valid], accuracies[valid], marker="o", linewidth=2, color="tab:blue", label="Model")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence", ylabel="Empirical accuracy")
    ax.set_title("Calibration (15 bins)", loc="left", weight="bold")

    count_ax = ax.twinx()
    count_ax.bar(centers, counts, width=1 / n_bins, color="0.5", alpha=0.16, zorder=0)
    count_ax.set_ylabel("Predictions per bin", color="0.5", fontsize=8)
    count_ax.tick_params(axis="y", labelsize=7, colors="0.5")
    count_ax.grid(False)

    ece = expected_calibration_error(labels, probs, n_bins=n_bins)
    brier = brier_score_multiclass(labels, probs)
    ax.text(
        0.04,
        0.96,
        f"ECE {ece:.3f}\nBrier {brier:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    ax.legend(fontsize=7, loc="lower right")


def _plot_run(
    row: dict[str, Any],
    result_dir: Path,
    output_dir: Path,
    n_bins: int,
) -> bool:
    loaded = _load_predictions(row, result_dir)
    if loaded is None:
        return False
    labels, preds, probs, class_names = loaded

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.12, top=0.92, wspace=0.45, hspace=0.4)
    model_name = str(row["model"])
    dataset = str(row["dataset"])
    fig.suptitle(f"{dataset}  |  {model_name}", fontsize=14, weight="bold")
    _plot_confusion(
        axes[0, 0],
        labels,
        preds,
        class_names,
        sort_by_frequency=dataset.lower() == "dermamnist",
    )
    _plot_roc(axes[0, 1], labels, probs, class_names)
    _plot_precision_recall(axes[1, 0], labels, probs, class_names)
    _plot_reliability(axes[1, 1], labels, probs, n_bins=n_bins)

    stem = f"{safe_name(dataset)}_{safe_name(model_name)}_diagnostics"
    _save(fig, output_dir, stem)
    return True


def create_plots(results_dir: Path, output_dir: Path, n_bins: int = 15) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    rows = _load_rows(results_dir)
    if not rows:
        raise ValueError(f"No valid results found in {results_dir}")

    generated = sum(_plot_run(row, results_dir, output_dir, n_bins) for row in rows)
    if generated == 0:
        raise FileNotFoundError(
            f"No prediction NPZ files found for results in {results_dir}."
        )
    print(f"Created {generated} per-run diagnostic panels in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "plots")
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()
    create_plots(args.results_dir, args.output_dir, n_bins=args.bins)


if __name__ == "__main__":
    main()
