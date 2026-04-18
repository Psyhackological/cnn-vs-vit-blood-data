# main.py – punkt wejscia, trenuje i porownuje wszystkie modele

import os
import time
import torch
import json

from config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY, NUM_CLASSES, MODELS, RESULTS_DIR
from dataset import get_loaders
from models import get_model
from train import run_training
from evaluate import evaluate_model
from visualize import plot_history, plot_confusion_matrix


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Urzadzenie: {device}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  Model: {model_name}")
        print(f"{'='*55}")

        img_size = cfg["img_size"]
        train_loader, val_loader, test_loader = get_loaders(img_size, BATCH_SIZE)

        model = get_model(model_name, NUM_CLASSES, img_size).to(device)

        t0 = time.time()
        history = run_training(
            model, train_loader, val_loader,
            NUM_EPOCHS, LR, WEIGHT_DECAY, device
        )
        train_time = time.time() - t0

        metrics = evaluate_model(model, test_loader, device)

        plot_history(history, model_name, RESULTS_DIR)
        plot_confusion_matrix(metrics["confusion_matrix"], model_name, RESULTS_DIR)

        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR, f"{model_name}.pth"))

        summary[model_name] = {
            "accuracy": round(metrics["accuracy"], 4),
            "f1_macro": round(metrics["f1"], 4),
            "auc_roc":  round(metrics["auc"], 4),
            "train_time_s": round(train_time, 1),
        }

    print(f"\n{'='*55}")
    print("  PODSUMOWANIE POROWNANIA")
    print(f"{'='*55}")
    print(f"{'Model':<30} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Czas(s)':>9}")
    print("-" * 55)
    for name, m in summary.items():
        print(f"{name:<30} {m['accuracy']:>6.4f} {m['f1_macro']:>6.4f} "
              f"{m['auc_roc']:>6.4f} {m['train_time_s']:>9.1f}")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWyniki zapisane w: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

