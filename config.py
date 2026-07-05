# config.py – wszystkie hiperparametry w jednym miejscu

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda"
BATCH_SIZE = 16  # 16 bezpieczne dla 52.9M modelu na 8GB VRAM
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-2
ACCUMULATION_STEPS = 1
EARLY_STOPPING_PATIENCE = 7
NUM_CLASSES = 8
IMG_SIZE_CNN = 224
IMG_SIZE_VIT = 224
IMG_SIZE = 224

MODELS = {
    "tf_efficientnetv2_s.in21k": {
        "type": "cnn",
        "img_size": IMG_SIZE,
        "source": "timm",
    },
    "deit3_small_patch16_224.fb_in22k_ft_in1k": {
        "type": "vit",
        "img_size": IMG_SIZE_VIT,
        "source": "timm",
    },
    "convnextv2_base.fcmae": {
        "type": "cnn",
        "img_size": IMG_SIZE,
        "source": "timm",
    },
    "eva02_base_patch14_224.mim_in22k": {
        "type": "vit",
        "img_size": IMG_SIZE_VIT,
        "source": "timm",
    },
}

RESULTS_DIR = "results"
