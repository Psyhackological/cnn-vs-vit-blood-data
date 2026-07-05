# models.py – tworzenie modeli przez timm

import timm
import torch.nn as nn


def _uses_absolute_pos_embed(model_name: str) -> bool:
    transformer_prefixes = (
        "beit",
        "cait",
        "deit",
        "eva",
        "flexivit",
        "pit",
        "swin",
        "vit",
    )
    return model_name.startswith(transformer_prefixes)


def get_model(model_name: str, num_classes: int, img_size: int) -> nn.Module:
    """
    Tworzy pre-trenowany model z timm i dostosowuje glowice klasyfikacyjna.
    Dla modeli transformerowych przekazuje img_size, zeby timm mogl dopasowac
    osadzenia pozycyjne do rozdzielczosci eksperymentu.
    """
    if _uses_absolute_pos_embed(model_name):
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            img_size=img_size,
        )
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name} | Parametry: {total_params:.1f}M")
    return model
