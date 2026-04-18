// ============================================================
//  Plakat A0 – CNN vs ViT na BloodMNIST
//  Kompilacja: typst compile poster.typ
// ============================================================

#set page(
  width: 841mm,
  height: 1189mm,
  margin: (top: 20mm, bottom: 20mm, left: 20mm, right: 20mm),
  fill: rgb(&quot;#f8f9fa&quot;),
)

#set text(font: &quot;New Computer Modern&quot;, size: 22pt, lang: &quot;pl&quot;)
#set par(justify: true, leading: 0.65em)

// ── Kolory ──────────────────────────────────────────────────
#let accent   = rgb(&quot;#1a73e8&quot;)
#let accent2  = rgb(&quot;#e84c1a&quot;)
#let darkbg   = rgb(&quot;#1e2a3a&quot;)
#let lightbg  = rgb(&quot;#eef3fb&quot;)
#let cardbg   = white
#let border   = rgb(&quot;#c8d8f0&quot;)

// ── Pomocnicze funkcje ───────────────────────────────────────
#let section(title, color: accent, body) = block(
  width: 100%,
  fill: cardbg,
  stroke: (left: 6pt + color, rest: 1pt + border),
  radius: 6pt,
  inset: 18pt,
  below: 16pt,
)[
  #text(weight: &quot;bold&quot;, size: 26pt, fill: color)[#title]
  #v(8pt)
  #body
]

#let badge(txt, color: accent) = box(
  fill: color.lighten(80%),
  stroke: 1pt + color,
  radius: 4pt,
  inset: (x: 8pt, y: 4pt),
)[#text(size: 18pt, fill: color, weight: &quot;bold&quot;)[#txt]]

#let kv(k, v) = grid(
  columns: (auto, 1fr),
  gutter: 6pt,
  text(weight: &quot;bold&quot;)[#k:],
  text[#v],
)

#let codebox(code) = block(
  width: 100%,
  fill: rgb(&quot;#1e2a3a&quot;),
  radius: 6pt,
  inset: 14pt,
  below: 10pt,
  text(font: &quot;Courier New&quot;, size: 17pt, fill: rgb(&quot;#a8d8a8&quot;),
    raw(code, lang: &quot;python&quot;)
  )
)

// ── NAGŁÓWEK ────────────────────────────────────────────────
#block(
  width: 100%,
  fill: darkbg,
  radius: 10pt,
  inset: 28pt,
  below: 22pt,
)[
  #align(center)[
    #text(size: 48pt, weight: &quot;bold&quot;, fill: white)[
      Porównanie skuteczności sieci konwolucyjnych (CNN)\
      oraz Vision Transformer (ViT)\
      w klasyfikacji obrazów medycznych
    ]
    #v(10pt)
    #text(size: 30pt, fill: rgb(&quot;#90caf9&quot;))[
      Na przykładzie zbioru danych #badge(&quot;BloodMNIST&quot;, color: rgb(&quot;#e84c1a&quot;))
    ]
    #v(14pt)
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 10pt,
      badge(&quot;Deep Learning&quot;, color: accent),
      badge(&quot;Klasyfikacja medyczna&quot;, color: accent2),
      badge(&quot;PyTorch · timm · MedMNIST&quot;, color: rgb(&quot;#2e7d32&quot;)),
    )
  ]
]

// ── DWIE KOLUMNY ────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr),
  gutter: 22pt,

  // ╔══════════════════════════════╗
  // ║       LEWA KOLUMNA           ║
  // ╚══════════════════════════════╝
  [
    // ── 1. Architektura CNN vs ViT ──────────────────────────
    #section(&quot;1 · Architektura CNN vs ViT&quot;, color: accent)[

      #grid(
        columns: (1fr, 1fr),
        gutter: 14pt,

        block(fill: lightbg, radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 24pt, fill: accent)[Sieć konwolucyjna (CNN)]
          #v(6pt)
          Przetwarza obraz lokalnie przez *filtry konwolucyjne* przesuwane po całym obrazie.

          #v(8pt)
          *Kluczowe bloki:*

- Warstwa konwolucyjna (`Conv2d`)
- Normalizacja wsadowa (`BatchNorm`)
- Aktywacja (`ReLU`)
- Pooling (`MaxPool2d`)
- Klasyfikator (`Linear`)

          #v(8pt)
          *Cechy:*

- Indukcja lokalności i translacyjnej niezmienniczości
- Mała liczba parametrów przy małych obrazach
- Szybki trening na CPU/GPU
- Sprawdzona w medycynie (ResNet, EfficientNet)
        ],

        block(fill: rgb(&quot;#fff3e0&quot;), radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 24pt, fill: accent2)[Vision Transformer (ViT)]
          #v(6pt)
          Dzieli obraz na *patche* (np. 16×16 px), traktuje je jak tokeny i przetwarza mechanizmem *self-attention*.

          #v(8pt)
          *Kluczowe bloki:*

- Patch Embedding
- Multi-Head Self-Attention (MHSA)
- Feed-Forward Network (FFN)
- Layer Normalization
- CLS token → klasyfikator

          #v(8pt)
          *Cechy:*

- Globalne zależności od pierwszej warstwy
- Wymaga dużych zbiorów lub pre-treningu
- Większa liczba parametrów
- Doskonałe wyniki przy fine-tuningu
        ],
      )

      #v(10pt)
      #align(center)[
        #block(fill: lightbg, radius: 6pt, inset: 12pt, width: 90%)[
          #text(size: 19pt)[
            *CNN* → lokalne wzorce → hierarchia cech → klasyfikacja\
            *ViT* → patche → sekwencja tokenów → attention → klasyfikacja
          ]
        ]
      ]
    ]

    // ── 2. Metryki porównawcze ──────────────────────────────
    #section(&quot;2 · Metryki porównawcze&quot;, color: rgb(&quot;#6a1b9a&quot;))[

      Oba modele oceniane są w *identycznych warunkach* (ten sam zbiór, te same hiperparametry bazowe).

      #v(10pt)
      #table(
        columns: (auto, 1fr, 1fr),
        fill: (col, row) =&gt; if row == 0 { rgb(&quot;#6a1b9a&quot;) } else if calc.odd(row) { rgb(&quot;#f3e5f5&quot;) } else { white },
        stroke: 0.5pt + rgb(&quot;#ce93d8&quot;),
        inset: 10pt,
        align: (left, center, center),
        table.header(
          text(fill: white, weight: &quot;bold&quot;)[Metryka],
          text(fill: white, weight: &quot;bold&quot;)[Opis],
          text(fill: white, weight: &quot;bold&quot;)[Zastosowanie],
        ),

        [*Accuracy*],        [Odsetek poprawnych predykcji],          [Główna miara jakości],
        [*Top-1 Accuracy*],  [Najwyższy wynik klasy = etykieta],      [Standard benchmarku],
        [*Loss (CE)*],       [Cross-Entropy na zbiorze val/test],      [Monitorowanie treningu],
        [*F1-score (macro)*],[Średnia harmoniczna P i R per klasa],    [Niezbalansowane klasy],
        [*AUC-ROC*],         [Pole pod krzywą ROC (OvR)],             [Pewność predykcji],
        [*Czas treningu*],   [Sekundy/epoka na GPU],                  [Efektywność obliczeniowa],
        [*Liczba param.*],   [Miliony parametrów modelu],             [Złożoność modelu],
      )

      #v(10pt)
      #block(fill: rgb(&quot;#f3e5f5&quot;), radius: 6pt, inset: 12pt)[
        *Protokół eksperymentu:*

- Optimizer: AdamW, lr = 1e-4, weight decay = 1e-2
- Scheduler: CosineAnnealingLR
- Batch size: 64 · Epoki: 30
- Augmentacja: RandomHorizontalFlip, RandomCrop, Normalize
- Środowisko: PyTorch 2.x, CUDA 12
      ]
    ]

    // ── 3. Próbka danych BloodMNIST ────────────────────────
    #section(&quot;3 · Próbka danych BloodMNIST&quot;, color: rgb(&quot;#00695c&quot;))[

      #grid(
        columns: (1fr, 1fr),
        gutter: 14pt,

        block(fill: rgb(&quot;#e0f2f1&quot;), radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 22pt, fill: rgb(&quot;#00695c&quot;))[Charakterystyka zbioru]
          #v(8pt)
          #kv(&quot;Źródło&quot;, &quot;MedMNIST v2 (Zenodo)&quot;)
          #kv(&quot;Obrazy&quot;, &quot;17 092 RGB, 28×28 px&quot;)
          #kv(&quot;Podział&quot;, &quot;Train 70% · Val 10% · Test 20%&quot;)
          #kv(&quot;Zadanie&quot;, &quot;Klasyfikacja 8-klasowa&quot;)
          #kv(&quot;Licencja&quot;, &quot;CC BY 4.0&quot;)
          #kv(&quot;Dostęp&quot;, &quot;pip install medmnist&quot;)
        ],

        block(fill: rgb(&quot;#e0f2f1&quot;), radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 22pt, fill: rgb(&quot;#00695c&quot;))[8 klas komórek krwi]
          #v(8pt)
          #for (i, cls) in (
            &quot;Basophil (BAS)&quot;,
            &quot;Eosinophil (EOS)&quot;,
            &quot;Erythroblast (ERY)&quot;,
            &quot;Immature Granulocytes (IMM)&quot;,
            &quot;Lymphocyte (LYM)&quot;,
            &quot;Monocyte (MON)&quot;,
            &quot;Neutrophil (NEU)&quot;,
            &quot;Platelet (PLT)&quot;,
          ).enumerate() [
            #box(fill: accent.lighten(80%), radius: 3pt, inset: (x:5pt,y:2pt))[
              #text(size: 16pt, fill: accent, weight: &quot;bold&quot;)[#str(i+1)]
            ]
            #h(4pt)#text(size: 19pt)[#cls] \
          ]
        ],
      )

      #v(10pt)
      #codebox(&quot;from medmnist import BloodMNIST
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
])

train_ds = BloodMNIST(split=\&quot;train\&quot;,
                      download=True,
                      transform=transform)
val_ds   = BloodMNIST(split=\&quot;val\&quot;,   download=True,
                      transform=transform)
test_ds  = BloodMNIST(split=\&quot;test\&quot;,  download=True,
                      transform=transform)

print(f\&quot;Train: {len(train_ds)} | Val: {len(val_ds)}\&quot;
      f\&quot; | Test: {len(test_ds)}\&quot;)&quot;)
    ]
  ],

  // ╔══════════════════════════════╗
  // ║       PRAWA KOLUMNA          ║
  // ╚══════════════════════════════╝
  [
    // ── 4. Przykładowe modele PyTorch ──────────────────────
    #section(&quot;4 · Przykładowe modele PyTorch (timm)&quot;, color: rgb(&quot;#e65100&quot;))[

      Biblioteka *timm* (PyTorch Image Models) dostarcza setki pre-trenowanych modeli gotowych do fine-tuningu.

      #v(8pt)
      #codebox(&quot;import timm, torch

# CNN: ResNet-50
cnn_model = timm.create_model(
    \&quot;resnet50\&quot;,
    pretrained=True,
    num_classes=8,   # BloodMNIST: 8 klas
)

# ViT: ViT-Base/16
vit_model = timm.create_model(
    \&quot;vit_base_patch16_224\&quot;,
    pretrained=True,
    num_classes=8,
    img_size=28,     # dopasowanie do 28x28
)

# Przeniesienie na GPU
device = torch.device(\&quot;cuda\&quot; if torch.cuda.is_available()
                       else \&quot;cpu\&quot;)
cnn_model = cnn_model.to(device)
vit_model = vit_model.to(device)

# Petla treningowa (szkielet)
optimizer = torch.optim.AdamW(
    cnn_model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(30):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.squeeze().long()
        optimizer.zero_grad()
        loss = criterion(cnn_model(imgs), labels)
        loss.backward()
        optimizer.step()&quot;)

      #v(8pt)
      #table(
        columns: (auto, 1fr, 1fr),
        fill: (col, row) =&gt; if row == 0 { rgb(&quot;#e65100&quot;) } else if calc.odd(row) { rgb(&quot;#fff3e0&quot;) } else { white },
        stroke: 0.5pt + rgb(&quot;#ffb74d&quot;),
        inset: 10pt,
        align: (left, center, center),
        table.header(
          text(fill: white, weight: &quot;bold&quot;)[Model],
          text(fill: white, weight: &quot;bold&quot;)[Parametry],
          text(fill: white, weight: &quot;bold&quot;)[timm ID],
        ),

        [ResNet-50],          [~25 M],   [`resnet50`],
        [EfficientNet-B0],    [~5.3 M],  [`efficientnet_b0`],
        [ViT-Base/16],        [~86 M],   [`vit_base_patch16_224`],
        [ViT-Small/16],       [~22 M],   [`vit_small_patch16_224`],
        [DeiT-Small],         [~22 M],   [`deit_small_patch16_224`],
      )
    ]

    // ── 5. Produkcyjne narzędzia ────────────────────────────
    #section(&quot;5 · Produkcyjne narzędzia&quot;, color: rgb(&quot;#1565c0&quot;))[

      #grid(
        columns: (1fr, 1fr),
        gutter: 14pt,

        block(fill: lightbg, radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 22pt, fill: accent)[Biblioteki Python]
          #v(8pt)
          #table(
            columns: (auto, 1fr),
            stroke: none,
            inset: (x: 6pt, y: 5pt),
            fill: (col, row) =&gt; if calc.odd(row) { lightbg } else { white },
            [`torch`],       [Framework DL],
            [`torchvision`], [Transformacje obrazów],
            [`timm`],        [Pre-trenowane modele],
            [`medmnist`],    [Pobieranie BloodMNIST],
            [`numpy`],       [Operacje numeryczne],
            [`matplotlib`],  [Wizualizacja wyników],
            [`scikit-learn`],[Metryki: F1, AUC, CM],
            [`tqdm`],        [Pasek postępu treningu],
          )
        ],

        block(fill: lightbg, radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 22pt, fill: accent)[Środowisko i instalacja]
          #v(8pt)
          #codebox(&quot;pip install torch torchvision
pip install timm medmnist
pip install scikit-learn
pip install matplotlib tqdm

# Wersje testowane:
# Python  3.10+
# PyTorch 2.2+
# CUDA    12.1
# timm    1.0+&quot;)
          #v(8pt)
          *Śledzenie eksperymentów:*

- TensorBoard (`tensorboard`)
- Weights &amp; Biases (`wandb`)
- MLflow (`mlflow`)
        ],
      )

      #v(8pt)
      #block(fill: rgb(&quot;#e3f2fd&quot;), radius: 6pt, inset: 12pt)[
        *Pipeline eksperymentu:*
        #v(4pt)
        #grid(
          columns: (1fr,) * 5,
          gutter: 6pt,
          align: center,
          block(fill: accent, radius: 4pt, inset: 8pt)[#text(fill:white, size:17pt)[*Dane*\nBloodMNIST]],
          block(fill: accent, radius: 4pt, inset: 8pt)[#text(fill:white, size:17pt)[*Prepro-\ncessing*]],
          block(fill: accent, radius: 4pt, inset: 8pt)[#text(fill:white, size:17pt)[*Model*\nCNN/ViT]],
          block(fill: accent, radius: 4pt, inset: 8pt)[#text(fill:white, size:17pt)[*Trening*\n30 epok]],
          block(fill: accent2, radius: 4pt, inset: 8pt)[#text(fill:white, size:17pt)[*Ewaluacja*\nMetryki]],
        )
      ]
    ]

    // ── 6. Podsumowanie i konkluzja ─────────────────────────
    #section(&quot;6 · Podsumowanie i konkluzja&quot;, color: rgb(&quot;#2e7d32&quot;))[

      #grid(
        columns: (1fr, 1fr),
        gutter: 14pt,

        block(fill: rgb(&quot;#e8f5e9&quot;), radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 22pt, fill: rgb(&quot;#2e7d32&quot;))[Oczekiwane wyniki]
          #v(8pt)
          #table(
            columns: (auto, 1fr, 1fr),
            fill: (col, row) =&gt; if row == 0 { rgb(&quot;#2e7d32&quot;) } else if calc.odd(row) { rgb(&quot;#e8f5e9&quot;) } else { white },
            stroke: 0.5pt + rgb(&quot;#81c784&quot;),
            inset: 9pt,
            align: (left, center, center),
            table.header(
              text(fill:white,weight:&quot;bold&quot;)[Kryterium],
              text(fill:white,weight:&quot;bold&quot;)[CNN],
              text(fill:white,weight:&quot;bold&quot;)[ViT],
            ),

            [Accuracy],       [~90–93%], [~91–95%],
            [Czas/epoka],     [szybszy], [wolniejszy],
            [Param. (M)],     [~25],     [~86],
            [Fine-tuning],    [łatwy],   [wymaga uwagi],
            [Małe obrazy],    [dobry],   [wymaga adapt.],
          )
        ],

        block(fill: rgb(&quot;#e8f5e9&quot;), radius: 6pt, inset: 14pt)[
          #text(weight: &quot;bold&quot;, size: 22pt, fill: rgb(&quot;#2e7d32&quot;))[Wnioski]
          #v(8pt)

- *CNN* (ResNet) sprawdza się doskonale na małych obrazach 28×28 – mniej parametrów, szybszy trening, dobra dokładność.

- *ViT* osiąga wyższe accuracy przy fine-tuningu z ImageNet, lecz wymaga większych obrazów (224×224) i dłuższego treningu.

- Na BloodMNIST różnica accuracy między CNN a ViT jest *niewielka* (~1–3 pp), ale ViT lepiej modeluje globalne zależności morfologiczne komórek.

- W zastosowaniach produkcyjnych CNN pozostaje *bardziej praktyczny* ze względu na niższy koszt obliczeniowy.
        ],
      )

      #v(10pt)
      #block(
        fill: darkbg,
        radius: 8pt,
        inset: 16pt,
        width: 100%,
      )[
        #align(center)[
          #text(fill: white, size: 21pt)[
            *Oba podejścia osiągają wysoką skuteczność na BloodMNIST.*\
            Wybór modelu zależy od dostępnych zasobów obliczeniowych\
            i wymagań dotyczących interpretowalności.
          ]
          #v(10pt)
          #grid(
            columns: (1fr, 1fr),
            gutter: 10pt,
            align: center,
            block(fill: accent, radius: 6pt, inset: 10pt)[
              #text(fill: white, weight: &quot;bold&quot;)[Źródło danych]\
              #text(fill: rgb(&quot;#90caf9&quot;), size: 18pt)[zenodo.org/records/10519652]
            ],
            block(fill: accent2, radius: 6pt, inset: 10pt)[
              #text(fill: white, weight: &quot;bold&quot;)[Modele]\
              #text(fill: rgb(&quot;#ffccbc&quot;), size: 18pt)[github.com/huggingface/pytorch-image-models]
            ],
          )
        ]
      ]
    ]
  ],
)

// ── STOPKA ──────────────────────────────────────────────────
#v(1fr)
#block(
  width: 100%,
  fill: darkbg,
  radius: 8pt,
  inset: 14pt,
)[
  #align(center)[
    #text(fill: rgb(&quot;#90caf9&quot;), size: 18pt)[
      Yang et al., *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023 ·
      Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021 ·
      Wightman R., *PyTorch Image Models (timm)*, 2019
    ]
  ]
]
