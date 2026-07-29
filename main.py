# main.py – punkt wejscia, trenuje i porownuje modele na zbiorach MedMNIST

from __future__ import annotations

import gc
import json
import os
import time
from typing import Any

import numpy as np
import torch

from config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    ECE_BINS,
    LR,
    MODELS,
    NUM_EPOCHS,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from dataset import get_dataset_metadata, get_loaders
from evaluate import SCALAR_METRIC_NAMES, evaluate_model
from models import get_model
from train import run_training
from utils import safe_name, set_seed
from visualize import plot_diagnostics, plot_history


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def _metric_row(
    dataset_key: str,
    model_name: str,
    model_type: str,
    metadata,
    history: dict[str, Any],
    checkpoint_path: str,
    train_time: float,
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset_key,
        "model": model_name,
        "model_type": model_type,
        "num_classes": metadata.num_classes,
        "best_epoch": history["best_epoch"],
        "best_val_auc_macro_ovr": _round(history["best_val_auc"]),
        "checkpoint": checkpoint_path,
        "train_time_s": round(train_time, 1),
        "per_class": {
            "class_names": metadata.class_names,
            **{k: eval_result["per_class"][k] for k in ("recall", "f1", "auc_ovr")},
        },
    }
    for name in SCALAR_METRIC_NAMES:
        row[name] = _round(eval_result["metrics"][name])
        row[f"{name}_ci"] = [_round(v) for v in eval_result["cis"][name]]
    return row


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 94}\n  PODSUMOWANIE POROWNANIA\n{'=' * 94}")
    print(
        f"{'Dataset':<15} {'Model':<38} {'BalAcc':>7} {'F1':>7} "
        f"{'AUC':>7} {'AP':>7} {'MCC':>7} {'ECE':>7}"
    )
    print("-" * 94)
    for row in rows:
        vals = [
            row[m]
            for m in (
                "balanced_accuracy",
                "f1_macro",
                "auc_macro_ovr",
                "ap_macro",
                "mcc",
                "ece",
            )
        ]
        print(
            f"{row['dataset']:<15} {row['model']:<38} "
            + " ".join("    nan" if v is None else f"{v:7.4f}" for v in vals)
        )


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Urzadzenie: {device}\n")
    summary_rows: list[dict[str, Any]] = []

    for dataset_key, dataset_cfg in DATASETS.items():
        metadata_hint = get_dataset_metadata(dataset_key)
        print(
            f"\n{'#' * 70}\n  Dataset: {metadata_hint.display_name} ({dataset_key})\n{'#' * 70}"
        )

        for model_name, model_cfg in MODELS.items():
            print(
                f"\n{'=' * 70}\n  Dataset: {dataset_key} | Model: {model_name}\n{'=' * 70}"
            )

            # Kazdy przebieg startuje z tego samego ziarna, wiec wynik modelu
            # nie zalezy od jego pozycji w petli.
            set_seed(SEED)
            img_size = int(dataset_cfg.get("img_size", model_cfg["img_size"]))
            model = get_model(model_name, metadata_hint.num_classes, img_size).to(
                device
            )
            train_loader, val_loader, test_loader, metadata = get_loaders(
                dataset_key, model, img_size, BATCH_SIZE, seed=SEED
            )

            prefix = f"{safe_name(dataset_key)}_{safe_name(model_name)}"
            checkpoint_path = os.path.join(RESULTS_DIR, f"{prefix}_best.pth")
            t0 = time.time()
            history = run_training(
                model,
                train_loader,
                val_loader,
                NUM_EPOCHS,
                LR,
                WEIGHT_DECAY,
                device,
                checkpoint_path=checkpoint_path,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                accumulation_steps=ACCUMULATION_STEPS,
            )
            train_time = time.time() - t0

            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            eval_result = evaluate_model(
                model,
                test_loader,
                device,
                metadata.class_names,
                ece_bins=ECE_BINS,
                bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
            class_order = (
                metadata.class_frequency_order
                if dataset_cfg.get("sort_classes_by_frequency")
                else None
            )
            plot_history(history, model_name, dataset_key, RESULTS_DIR)
            plot_diagnostics(
                eval_result["labels"],
                eval_result["preds"],
                eval_result["probs"],
                metadata.class_names,
                model_name,
                dataset_key,
                RESULTS_DIR,
                ece_bins=ECE_BINS,
                class_order=class_order,
            )

            row = _metric_row(
                dataset_key,
                model_name,
                str(model_cfg["type"]),
                metadata,
                history,
                checkpoint_path,
                train_time,
                eval_result,
            )
            summary_rows.append(row)
            with open(os.path.join(RESULTS_DIR, f"{prefix}_metrics.json"), "w") as f:
                json.dump(_jsonable(row), f, indent=2)

            del model, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()

    _print_summary(summary_rows)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(_jsonable(summary_rows), f, indent=2)
    print(f"\nWyniki zapisane w: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
