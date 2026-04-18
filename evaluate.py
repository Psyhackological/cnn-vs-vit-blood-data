# evaluate.py – metryki na zbiorze testowym

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix,
)


CLASS_NAMES = [
    "Basophil", "Eosinophil", "Erythroblast",
    "Immature Gran.", "Lymphocyte",
    "Monocyte", "Neutrophil", "Platelet",
]


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average="macro")
    auc    = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES
    )

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print(f"AUC-ROC (OvR)  : {auc:.4f}")
    print("\nClassification Report:")
    print(report)

    return {"accuracy": acc, "f1": f1, "auc": auc,
            "confusion_matrix": cm, "report": report}

