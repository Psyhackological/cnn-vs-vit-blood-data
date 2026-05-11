# config.py – wszystkie hiperparametry w jednym miejscu

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda"
BATCH_SIZE = 16  # 16 bezpieczne dla 52.9M modelu na 8GB VRAM
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 8
IMG_SIZE_CNN = 224
IMG_SIZE_VIT = 224
IMG_SIZE = 224

MODELS = {
    "tf_efficientnetv2_m.in21k": {
        "type": "cnn",
        "img_size": IMG_SIZE,
        "source": "timm",
    },
    "vit_base_patch16_224.augreg_in21k": {
        "type": "vit",
        "img_size": IMG_SIZE_VIT,
        "source": "timm",
    },
}

RESULTS_DIR = "results"
