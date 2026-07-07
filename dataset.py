# dataset.py – ladowanie zbiorow MedMNIST przez medmnist

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import medmnist
import numpy as np
import torch
from medmnist import INFO
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torch.utils.data import DataLoader

SUPPORTED_MEDMNIST_SIZES = {28, 64, 128, 224}


@dataclass(frozen=True)
class DatasetMetadata:
    key: str
    display_name: str
    task: str
    num_classes: int
    class_names: list[str]
    n_channels: int
    class_frequency_order: list[int]


def _resolve_spatial_size(input_size: int | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) == 2:
        return input_size
    if len(input_size) == 3:
        return input_size[-2], input_size[-1]
    raise ValueError(f"Nieprawidlowy input_size z konfiguracji timm: {input_size}")


def _dataset_class(dataset_key: str):
    if dataset_key not in INFO:
        raise KeyError(f"Nieznany zbior MedMNIST: {dataset_key}")
    info = cast(dict[str, Any], INFO[dataset_key])
    return getattr(medmnist, str(info["python_class"]))


def _class_names(dataset_key: str) -> list[str]:
    info = cast(dict[str, Any], INFO[dataset_key])
    labels = cast(dict[str, str], info["label"])
    return [labels[str(i)] for i in range(len(labels))]


def _frequency_order(labels: np.ndarray, num_classes: int) -> list[int]:
    counts = np.bincount(labels.reshape(-1).astype(int), minlength=num_classes)
    return np.argsort(-counts).astype(int).tolist()


def get_transforms(model: nn.Module, img_size: int):
    data_cfg = resolve_data_config(
        {"input_size": (3, img_size, img_size)},
        model=model,
    )

    train_tf = create_transform(
        **data_cfg,
        is_training=True,
        scale=(0.8, 1.0),
        hflip=0.5,
        color_jitter=0.0,
        auto_augment=None,
        re_prob=0.0,
    )
    val_tf = create_transform(**data_cfg, is_training=False)

    print(
        "Data config: "
        f"input_size={data_cfg['input_size']} | "
        f"mean={data_cfg['mean']} | std={data_cfg['std']} | "
        f"interpolation={data_cfg['interpolation']} | crop_pct={data_cfg['crop_pct']}"
    )
    return train_tf, val_tf, data_cfg


def get_dataset_metadata(
    dataset_key: str,
    train_labels: np.ndarray | None = None,
) -> DatasetMetadata:
    info = cast(dict[str, Any], INFO[dataset_key])
    labels = cast(dict[str, str], info["label"])
    num_classes = len(labels)
    if train_labels is None:
        class_frequency_order = list(range(num_classes))
    else:
        class_frequency_order = _frequency_order(train_labels, num_classes)

    return DatasetMetadata(
        key=dataset_key,
        display_name=str(info["python_class"]),
        task=str(info["task"]),
        num_classes=num_classes,
        class_names=_class_names(dataset_key),
        n_channels=int(info["n_channels"]),
        class_frequency_order=class_frequency_order,
    )


def get_loaders(
    dataset_key: str,
    model: nn.Module,
    img_size: int,
    batch_size: int,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], DatasetMetadata]:
    train_tf, val_tf, data_cfg = get_transforms(model, img_size)
    input_h, input_w = _resolve_spatial_size(data_cfg["input_size"])

    if input_h != input_w:
        raise ValueError(
            f"MedMNIST wymaga kwadratowego rozmiaru, otrzymano {input_h}x{input_w}"
        )
    if input_h not in SUPPORTED_MEDMNIST_SIZES:
        raise ValueError(
            "MedMNIST udostepnia natywne rozmiary tylko dla "
            f"{sorted(SUPPORTED_MEDMNIST_SIZES)}, otrzymano {input_h}."
        )

    ds_cls = _dataset_class(dataset_key)
    metadata = get_dataset_metadata(dataset_key)
    as_rgb = metadata.n_channels == 1

    train_ds = ds_cls(
        split="train",
        size=input_h,
        download=True,
        transform=train_tf,
        as_rgb=as_rgb,
    )
    val_ds = ds_cls(
        split="val",
        size=input_h,
        download=True,
        transform=val_tf,
        as_rgb=as_rgb,
    )
    test_ds = ds_cls(
        split="test",
        size=input_h,
        download=True,
        transform=val_tf,
        as_rgb=as_rgb,
    )
    metadata = get_dataset_metadata(dataset_key, np.asarray(train_ds.labels))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    print(
        f"Dataset: {metadata.display_name} | classes={metadata.num_classes} | "
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, metadata
