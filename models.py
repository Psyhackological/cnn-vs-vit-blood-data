# models.py – tworzenie modeli przez timm

import timm
import torch.nn as nn


def get_model(model_name: str, num_classes: int, img_size: int) -> nn.Module:
    """
    Tworzy pre-trenowany model z timm i dostosowuje glowice klasyfikacyjna.
    Obsluguje CNN (ResNet, EfficientNet) i ViT.
    """
    is_vit = model_name.startswith("vit") or model_name.startswith("deit")

    if is_vit:
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

