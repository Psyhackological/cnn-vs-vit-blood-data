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
IMG_SIZE = 224  # MedMNIST udostepnia natywnie tylko 28 / 64 / 128 / 224
IMG_SIZE_CNN = IMG_SIZE  # CNN przyjmuja dowolny rozmiar wejscia
IMG_SIZE_VIT = IMG_SIZE  # ViT: timm interpoluje osadzenia pozycyjne pod ten rozmiar
ECE_BINS = 15
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
SEED = 42  # ziarno treningu: init glowicy, kolejnosc batchy, augmentacja

# "img_size" jest opcjonalne – ustaw je tylko, gdy dany zbior ma byc trenowany
# w innej rozdzielczosci niz deklaruje model w MODELS.
DATASETS = {
    "bloodmnist": {
        "sort_classes_by_frequency": False,
    },
    "dermamnist": {
        "sort_classes_by_frequency": True,
    },
    "pneumoniamnist": {
        "sort_classes_by_frequency": False,
    },
    "pathmnist": {
        "sort_classes_by_frequency": False,
    },
}

MODELS = {
    "tf_efficientnetv2_s.in21k": {
        "type": "cnn",
        "img_size": IMG_SIZE_CNN,
        "source": "timm",
    },
    "deit3_small_patch16_224.fb_in22k_ft_in1k": {
        "type": "vit",
        "img_size": IMG_SIZE_VIT,
        "source": "timm",
    },
    "convnextv2_base.fcmae": {
        "type": "cnn",
        "img_size": IMG_SIZE_CNN,
        "source": "timm",
    },
    "eva02_base_patch14_224.mim_in22k": {
        "type": "vit",
        "img_size": IMG_SIZE_VIT,
        "source": "timm",
    },
}

RESULTS_DIR = "results"
