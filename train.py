# train.py – petla treningowa z zapisem historii

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from evaluate import macro_ovr_auc


def _cuda_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_torch_save(value: Any, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}.")
    os.close(fd)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_json_save(value: Any, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}.")
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(_jsonable(value), output, indent=2, allow_nan=False)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _rng_state(train_loader) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": (
            train_loader.generator.get_state()
            if getattr(train_loader, "generator", None) is not None
            else None
        ),
    }
    return state


def _restore_rng_state(state: dict[str, Any], train_loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator_state = state["loader_generator"]
    if generator_state is not None and getattr(train_loader, "generator", None) is not None:
        train_loader.generator.set_state(generator_state)


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
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    accumulation_steps = max(1, accumulation_steps)

    optimizer.zero_grad(set_to_none=True)
    for step, (imgs, labels) in enumerate(
        tqdm(loader, leave=False, desc="  train"), start=1
    ):
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        labels = labels.reshape(-1).long().to(
            device, non_blocking=device.type == "cuda"
        )

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

        total_loss += loss.detach() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum()
        total += imgs.size(0)

    loss_sum, correct_sum = torch.stack(
        (total_loss, correct.to(total_loss.dtype))
    ).cpu().tolist()
    return loss_sum / total, correct_sum / total


def validate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            labels = labels.reshape(-1).long().to(
                device, non_blocking=device.type == "cuda"
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.detach() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum()
            total += imgs.size(0)

            all_labels.append(labels)
            all_probs.append(torch.softmax(outputs.float(), dim=1))

    loss_sum, correct_sum = torch.stack(
        (total_loss, correct.to(total_loss.dtype))
    ).cpu().tolist()
    labels = torch.cat(all_labels).cpu()
    probs = torch.cat(all_probs).cpu()
    auc = _macro_ovr_auc(labels, probs)

    return loss_sum / total, correct_sum / total, auc


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
    resume_path: str | None = None,
    history_path: str | None = None,
    resume_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
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
    completed_epoch = 0
    elapsed_before_resume = 0.0

    if resume_path is not None and os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("resume_config") != resume_config:
            raise ValueError(
                f"Resume checkpoint settings do not match current run: {resume_path}. "
                "Keep the original settings or move the resume file aside to restart."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best_val_auc = state["best_val_auc"]
        best_val_loss = state["best_val_loss"]
        best_epoch = state["best_epoch"]
        epochs_without_improvement = state["epochs_without_improvement"]
        completed_epoch = state["completed_epoch"]
        elapsed_before_resume = state["elapsed_train_seconds"]
        _restore_rng_state(state["rng"], train_loader)
        print(f"Resuming from completed epoch {completed_epoch}/{num_epochs}")

    already_stopped = (
        early_stopping_patience is not None
        and early_stopping_patience > 0
        and epochs_without_improvement >= early_stopping_patience
    )
    first_epoch = num_epochs + 1 if already_stopped else completed_epoch + 1
    for epoch in range(first_epoch, num_epochs + 1):
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
                _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        stop_early = (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        )
        best_text = " *best*" if improved else ""
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} "
            f"AUC(macro OvR): {va_auc:.4f}{best_text}"
        )

        history["best_epoch"] = best_epoch
        history["best_val_auc"] = None if math.isinf(best_val_auc) else best_val_auc
        elapsed_train_seconds = elapsed_before_resume + time.monotonic() - started_at
        history["train_time_s"] = elapsed_train_seconds
        if resume_path is not None:
            _atomic_torch_save(
                {
                    "resume_config": resume_config,
                    "completed_epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best_val_auc": best_val_auc,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "elapsed_train_seconds": elapsed_train_seconds,
                    "rng": _rng_state(train_loader),
                },
                resume_path,
            )
        if history_path is not None:
            _atomic_json_save(history, history_path)

        if stop_early:
            print(
                "Early stopping: "
                f"brak poprawy Val AUC przez {early_stopping_patience} epok."
            )
            break

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = None if math.isinf(best_val_auc) else best_val_auc
    history["train_time_s"] = elapsed_before_resume + time.monotonic() - started_at
    if history_path is not None:
        _atomic_json_save(history, history_path)
    return history
