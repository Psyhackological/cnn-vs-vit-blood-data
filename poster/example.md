# Porównanie CNN vs ViT – BloodMNIST *(z odniesieniami do źródeł)*

---

## 1. Architektura CNN vs ViT

### CNN (Sieć Konwolucyjna)
- Przetwarza obraz **lokalnie** przez filtry konwolucyjne przesuwane po obrazie
- Kluczowe bloki: `Conv2d` → `BatchNorm` → `ReLU` → `MaxPool2d` → `Linear`
- Cechy: lokalność, translacyjna niezmienniczość, mała liczba parametrów, szybki trening

### ViT (Vision Transformer)
- Dzieli obraz na **patche** (np. 16×16 px), traktuje je jak tokeny NLP — zgodnie z oryginalną pracą Dosovitskiy et al. *"An Image is Worth 16×16 Words"* [^4]
- Kluczowe bloki: Patch Embedding → Multi-Head Self-Attention → FFN → LayerNorm → CLS token
- Mechanizm self-attention pochodzi z pracy Vaswani et al. *"Attention Is All You Need"* [^4]

| Cecha | CNN | ViT |
|---|---|---|
| Receptive field | Lokalny | Globalny |
| Indukcja lokalności | Tak | Nie |
| Parametry | ~25 M (ResNet-50) | ~86 M (ViT-B/16) |
| Trening od zera | Dobry | Słabszy bez pre-treningu |
| Z pre-treningiem ImageNet | Dobry | Porównywalny lub lepszy |

> Matsoukas et al. wykazali, że **CNN trenowane od zera osiągają lepsze wyniki**, ale pre-trenowane ViT (ImageNet, supervised i self-supervised) **dorównują CNN** i stanowią realną alternatywę w medycynie. [^3]

---

## 2. Metryki porównawcze

| Metryka | Opis | Zastosowanie |
|---|---|---|
| Accuracy | Odsetek poprawnych predykcji | Główna miara jakości |
| F1-score (macro) | Średnia harmoniczna Precision i Recall per klasa | Niezbalansowane klasy |
| AUC-ROC | Pole pod krzywą ROC (One-vs-Rest) | Pewność predykcji |
| Loss (Cross-Entropy) | Wartość funkcji straty na val/test | Monitorowanie treningu |
| Czas treningu | Sekundy/epoka na GPU | Efektywność obliczeniowa |
| Liczba parametrów | Miliony parametrów modelu | Złożoność modelu |

> ⚠️ Reinke et al. (*Nature Methods*, 2023) zidentyfikowali **37 źródeł pułapek** w walidacji modeli obrazowych — m.in. niezbalansowanie klas, słaba jakość obrazów, nieodpowiedni dobór metryk. Badacze często sięgają po popularne metryki, które mogą być **całkowicie nieodpowiednie** dla danego problemu. [^1]

> Większość prac stosuje accuracy i F1-score jako standardowe metryki porównawcze CNN vs ViT. [^6]

**Protokół eksperymentu:**
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-2
- Scheduler: CosineAnnealingLR
- Batch size: 64, Epoki: 30
- Augmentacja: RandomHorizontalFlip, RandomCrop, Normalize

---

## 3. Próbka danych BloodMNIST

**Charakterystyka zbioru** (Yang et al., *Scientific Data*, 2023): [^2]

- Źródło: MedMNIST v2 (Zenodo) — kolekcja 12 zbiorów 2D i 6 zbiorów 3D, łącznie ~708K obrazów 2D
- Obrazy: **17 092 RGB**, oryginalnie 360×363 px → center-crop 200×200 → resize **28×28 px**
- Podział: Train 70% / Val 10% / Test 20%
- Zadanie: Klasyfikacja 8-klasowa
- Komórki pochodzą od zdrowych dawców (bez infekcji, chorób hematologicznych/onkologicznych)
- Licencja: CC BY 4.0

**8 klas komórek krwi:**
1. Basophil (BAS)
2. Eosinophil (EOS)
3. Erythroblast (ERY)
4. Immature Granulocytes (IMM)
5. Lymphocyte (LYM)
6. Monocyte (MON)
7. Neutrophil (NEU)
8. Platelet (PLT)

> Na MedMNIST2D ResNet-18 i ResNet-50 osiągają wyniki porównywalne z Google AutoML Vision, przy czym **ResNet-18 często przewyższa ResNet-50** przy rozdzielczości 28×28. Wyższa rozdzielczość (224×224) konsekwentnie daje lepsze AUC i ACC. [^2]

```python
from medmnist import BloodMNIST
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

train_ds = BloodMNIST(split="train", download=True, transform=transform)
val_ds   = BloodMNIST(split="val",   download=True, transform=transform)
test_ds  = BloodMNIST(split="test",  download=True, transform=transform)
```

---

## 4. Przykładowe modele PyTorch (timm)

Biblioteka **timm** (Wightman, 2019) dostarcza setki pre-trenowanych modeli gotowych do fine-tuningu. [^4]

```python
import timm, torch

# CNN: ResNet-50
cnn_model = timm.create_model("resnet50", pretrained=True, num_classes=8)

# ViT: ViT-Base/16
vit_model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=8,
    img_size=28
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = cnn_model.to(device)
vit_model = vit_model.to(device)

optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(30):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        loss = criterion(cnn_model(imgs), labels)
        loss.backward()
        optimizer.step()
```

| Model | Parametry | timm ID |
|---|---|---|
| ResNet-50 | ~25 M | `resnet50` |
| EfficientNet-B0 | ~5.3 M | `efficientnet_b0` |
| ViT-Base/16 | ~86 M | `vit_base_patch16_224` |
| ViT-Small/16 | ~22 M | `vit_small_patch16_224` |
| DeiT-Small | ~22 M | `deit_small_patch16_224` |

---

## 5. Produkcyjne narzędzia

| Biblioteka | Rola |
|---|---|
| `torch` | Framework Deep Learning |
| `torchvision` | Transformacje obrazów |
| `timm` | Pre-trenowane modele (Wightman) [^4] |
| `medmnist` | Pobieranie BloodMNIST (Yang et al.) [^2] |
| `numpy` | Operacje numeryczne |
| `matplotlib` | Wizualizacja wyników |
| `scikit-learn` | Metryki: F1, AUC, Confusion Matrix |
| `tqdm` | Pasek postępu treningu |

```bash
pip install torch torchvision timm medmnist scikit-learn matplotlib tqdm
```

**Śledzenie eksperymentów:** TensorBoard, Weights & Biases (`wandb`), MLflow

---

## 6. Podsumowanie i konkluzja

| Kryterium | CNN (ResNet-50) | ViT (ViT-B/16) |
|---|---|---|
| Accuracy (oczekiwana) | ~90–93% | ~91–95% |
| Czas treningu/epoka | Szybszy | Wolniejszy |
| Liczba parametrów | ~25 M | ~86 M |
| Małe obrazy (28×28) | Dobry natywnie | Wymaga adaptacji |
| Fine-tuning | Łatwy | Wymaga uwagi |

**Wnioski poparte literaturą:**

- **CNN trenowane od zera** osiągają lepsze wyniki niż ViT bez pre-treningu; jednak **pre-trenowane ViT dorównują CNN** w zadaniach medycznych (Matsoukas et al., 2023). [^3]

- Na zbiorach komórek krwi **Google ViT osiągnął 100% accuracy** na wysokiej jakości zbiorach PBC i 88.36% na trudnym, zaszumionym zbiorze BCCD — przewyższając wszystkie testowane modele CNN (ImageNet ILSVRC). [^5]

- CNN (modele ImageNet) wykazywały **overfitting na małych zbiorach**, podczas gdy ViT pozostawał stabilny niezależnie od rozmiaru danych. [^5]

- Przegląd literatury (Maurício et al., 2023) potwierdza, że **ViT przewyższa CNN** w wielu zadaniach medycznych (RTG klatki piersiowej, USG piersi, rak skóry), choć przy małej liczbie obrazów i niskiej jakości może zawodzić. [^6]

- Na BloodMNIST przy rozdzielczości 28×28 **ResNet-18 często bije ResNet-50** — mniejszy model lepiej generalizuje na małych obrazach. [^2]

- Dobór metryk jest krytyczny — samo accuracy może być mylące przy niezbalansowanych klasach; zalecane jest stosowanie **F1-score (macro) i AUC-ROC** (Reinke et al., *Nature Methods*, 2023). [^1]

[^1]: [Understanding metric-related pitfalls in image analysis validation | Nature Methods](https://doi.org/10.1038/s41592-023-02150-0) (21%)
[^2]: [MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification | Scientific Data](https://doi.org/10.1038/s41597-022-01721-8) (20%)
[^3]: [[2303.07034] Pretrained ViTs Yield Versatile Representations For Medical Images](https://doi.org/10.48550/arXiv.2303.07034) (16%)
[^4]: [message-7652a2fa-2f95-4701-9b67-dfb7be242966](message-7652a2fa-2f95-4701-9b67-dfb7be242966.txt) (16%)
[^5]: [White Blood Cell Classification: Convolutional Neural Network (CNN) and Vision Transformer (ViT) under Medical Microscope](https://doi.org/10.3390/a16110525) (14%)
[^6]: [Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review](https://doi.org/10.3390/app13095521) (13%)
