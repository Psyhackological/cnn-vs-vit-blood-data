# config.py – wszystkie hiperparametry w jednym miejscu

DEVICE = "cuda"          # "cpu" jesli brak GPU
BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 8
IMG_SIZE = 28            # BloodMNIST: 28x28
IMG_SIZE_VIT = 224       # ViT wymaga wiekszych obrazow

MODELS = {
    "resnet50":               {"img_size": IMG_SIZE},
    "efficientnet_b0":        {"img_size": IMG_SIZE},
    "vit_small_patch16_224":  {"img_size": IMG_SIZE_VIT},
    "vit_base_patch16_224":   {"img_size": IMG_SIZE_VIT},
}

RESULTS_DIR = "results"

