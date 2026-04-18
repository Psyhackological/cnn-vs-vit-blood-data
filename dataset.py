# dataset.py – ladowanie BloodMNIST przez medmnist

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from medmnist import BloodMNIST


def get_transforms(img_size: int):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=4),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])
    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])
    return train_tf, val_tf


def get_loaders(img_size: int, batch_size: int):
    train_tf, val_tf = get_transforms(img_size)

    train_ds = BloodMNIST(split="train", download=True, transform=train_tf)
    val_ds   = BloodMNIST(split="val",   download=True, transform=val_tf)
    test_ds  = BloodMNIST(split="test",  download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader

