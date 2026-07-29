# train.py – petla treningowa z zapisem historii

from __future__ import annotations

import math
from typing import Any

import torch
from tqdm import tqdm

from evaluate import macro_ovr_auc


def _cuda_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _macro_ovr_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    auc = macro_ovr_auc(labels.numpy(), probs.numpy())
    if math.isnan(auc):
        class_counts = torch.bincount(labels, minlength=probs.shape[1]).tolist()
        row_sums = probs.sum(dim=1)
        print(
            "Val AUC unavailable: "
            f"class_counts={class_counts} | "
            f"prob_row_sum_range=({row_sums.min().item():.6f}, {row_sums.max().item():.6f})"
        )
    return auc


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    amp_dtype,
    accumulation_steps: int = 1,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    accumulation_steps = max(1, accumulation_steps)

    optimizer.zero_grad(set_to_none=True)
    for step, (imgs, labels) in enumerate(
        tqdm(loader, leave=False, desc="  train"), start=1
    ):
        imgs = imgs.to(device)
        labels = labels.reshape(-1).long().to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.reshape(-1).long().to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            all_labels.append(labels.cpu())
            all_probs.append(torch.softmax(outputs.float(), dim=1).cpu())

    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)
    auc = _macro_ovr_auc(labels, probs)

    return total_loss / total, correct / total, auc


def run_training(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    weight_decay,
    device,
    checkpoint_path: str | None = None,
    early_stopping_patience: int | None = None,
    accumulation_steps: int = 1,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    amp_dtype = _cuda_amp_dtype()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )

    if device.type == "cuda":
        print(
            f"AMP dtype: {amp_dtype} | "
            f"GradScaler: {'on' if scaler.is_enabled() else 'off'} | "
            f"accumulation_steps={max(1, accumulation_steps)}"
        )

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }
    best_val_auc = -math.inf
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_dtype,
            accumulation_steps,
        )
        va_loss, va_acc, va_auc = validate(
            model, val_loader, criterion, device, amp_dtype
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_auc"].append(va_auc)

        if math.isnan(va_auc):
            improved = best_epoch == 0 or va_loss < best_val_loss
        else:
            improved = va_auc > best_val_auc

        if improved:
            if not math.isnan(va_auc):
                best_val_auc = va_auc
            best_val_loss = va_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        best_text = " *best*" if improved else ""
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} "
            f"AUC(macro OvR): {va_auc:.4f}{best_text}"
        )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping: "
                f"brak poprawy Val AUC przez {early_stopping_patience} epok."
            )
            break

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = None if math.isinf(best_val_auc) else best_val_auc
    return history
