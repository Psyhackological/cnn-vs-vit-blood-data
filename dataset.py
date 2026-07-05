# dataset.py – ladowanie BloodMNIST przez medmnist

from __future__ import annotations

import torch
from medmnist import BloodMNIST
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader

SUPPORTED_BLOODMNIST_SIZES = {28, 64, 128, 224}


def _resolve_spatial_size(input_size: int | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) == 2:
        return input_size
    if len(input_size) == 3:
        return input_size[-2], input_size[-1]
    raise ValueError(f"Nieprawidlowy input_size z konfiguracji timm: {input_size}")


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


def get_loaders(model: nn.Module, img_size: int, batch_size: int):
    train_tf, val_tf, data_cfg = get_transforms(model, img_size)
    input_h, input_w = _resolve_spatial_size(data_cfg["input_size"])

    if input_h != input_w:
        raise ValueError(
            f"BloodMNIST wymaga kwadratowego rozmiaru, otrzymano {input_h}x{input_w}"
        )
    if input_h not in SUPPORTED_BLOODMNIST_SIZES:
        raise ValueError(
            "MedMNIST udostepnia natywne rozmiary BloodMNIST tylko dla "
            f"{sorted(SUPPORTED_BLOODMNIST_SIZES)}, otrzymano {input_h}."
        )

    train_ds = BloodMNIST(
        split="train", size=input_h, download=True, transform=train_tf
    )
    val_ds = BloodMNIST(split="val", size=input_h, download=True, transform=val_tf)
    test_ds = BloodMNIST(split="test", size=input_h, download=True, transform=val_tf)

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

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
