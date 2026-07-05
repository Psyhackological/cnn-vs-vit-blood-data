# main.py – punkt wejscia, trenuje i porownuje wszystkie modele

import gc
import json
import os
import time

import torch

from config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LR,
    MODELS,
    NUM_CLASSES,
    NUM_EPOCHS,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from dataset import get_loaders
from evaluate import evaluate_model
from models import get_model
from train import run_training
from visualize import plot_confusion_matrix, plot_history


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Urzadzenie: {device}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}

    for model_name, cfg in MODELS.items():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n{'=' * 55}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 55}")

        img_size = int(cfg["img_size"])
        model = get_model(model_name, NUM_CLASSES, img_size).to(device)
        train_loader, val_loader, test_loader = get_loaders(model, img_size, BATCH_SIZE)
        checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")

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
            best_val_auc = history["best_val_auc"]
            best_val_auc_text = "n/a" if best_val_auc is None else f"{best_val_auc:.4f}"
            print(
                f"Zaladowano najlepszy checkpoint: {checkpoint_path} "
                f"(epoch={history['best_epoch']}, val_auc={best_val_auc_text})"
            )

        metrics = evaluate_model(model, test_loader, device)

        plot_history(history, model_name, RESULTS_DIR)
        plot_confusion_matrix(metrics["confusion_matrix"], model_name, RESULTS_DIR)

        summary[model_name] = {
            "accuracy": round(metrics["accuracy"], 4),
            "f1_macro": round(metrics["f1"], 4),
            "auc_roc_macro_ovr": round(metrics["auc"], 4),
            "best_val_auc_macro_ovr": (
                None
                if history["best_val_auc"] is None
                else round(history["best_val_auc"], 4)
            ),
            "best_epoch": history["best_epoch"],
            "checkpoint": checkpoint_path,
            "train_time_s": round(train_time, 1),
        }

    print(f"\n{'=' * 55}")
    print("  PODSUMOWANIE POROWNANIA")
    print(f"{'=' * 55}")
    print(f"{'Model':<30} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Czas(s)':>9}")
    print("-" * 55)
    for name, m in summary.items():
        print(
            f"{name:<30} {m['accuracy']:>6.4f} {m['f1_macro']:>6.4f} "
            f"{m['auc_roc_macro_ovr']:>6.4f} {m['train_time_s']:>9.1f}"
        )

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWyniki zapisane w: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
