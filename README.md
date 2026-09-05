# Handwritten Azerbaijani Letters Classification

A convolutional neural network that classifies handwritten Azerbaijani letters from 32×32 binary bitmaps, built with TensorFlow/Keras. Two architectures are trained and compared, and the test set is analysed to explain a persistent gap between validation and test accuracy.

Azerbaijani uses a Latin alphabet with characters not found in English — **ə, ğ, ı, ö, ş, ü, ç** — which puts it outside the reach of any pretrained handwriting model. This project trains from scratch on a purpose-built dataset.

---

## Dataset

Two files in a custom `#`-delimited format:

| File | Rows | Per class |
|---|---|---|
| `HandChars32_150_Train.txt` | 4,200 | 150 |
| `HandChars32_50_Test.txt` | 1,400 | 50 |

Both are perfectly balanced across **28 letter classes**. The `32` in the filename refers to the image dimensions, not the number of letters.

### Format

```
PatN#PatType#PatProb#SizeH#SizeW#Data
00#00#10#32#32#0000000000000001100000...
```

| Field | Meaning |
|---|---|
| `PatN` | **Class label** — which letter (0–27) |
| `PatType` | Pattern variant — see *Distribution shift* below |
| `PatProb` | Sample index within the pattern group |
| `SizeH`, `SizeW` | Image dimensions, always 32×32 |
| `Data` | 1,024 characters of `0`/`1` — the flattened bitmap |

Images are already binary, so no rescaling or normalization is needed. `Data` is parsed character by character, cast to `int`, and reshaped to 32×32 using the `SizeH`/`SizeW` fields rather than hard-coded values.

### Splits

The provided training file is split 76/24 into train and validation (`random_state=42`), with the provided test file held out entirely:

| Split | Shape |
|---|---|
| Train | `(3192, 32, 32, 1)` |
| Validation | `(1008, 32, 32, 1)` |
| Test | `(1400, 32, 32, 1)` |

Labels are one-hot encoded with `LabelBinarizer`.

---

## Architectures

Both designs share the same convolutional trunk and differ only in the classifier head.

### Design 1 — wide head

```
Conv2D(32, 5×5, same, relu)  →  MaxPool(2×2)  →  Dropout(0.5)
Conv2D(64, 5×5, same, relu)  →  MaxPool(2×2)  →  Dropout(0.5)
Flatten  →  Dense(256, relu)  →  Dropout(0.5)  →  Dense(28, softmax)
```

Optimizer: **RMSprop** · 30 epochs · **1,108,124 parameters**

### Design 2 — deep narrow head

```
[ same conv trunk ]
Flatten  →  Dense(100)  →  Drop(0.5)  →  Dense(200)  →  Drop(0.5)
         →  Dense(100)  →  Drop(0.5)  →  Dense(28, softmax)
```

Optimizer: **Adam** (`lr=0.001`) · 30 epochs · **504,924 parameters**

Three dropout layers stacked between narrow dense layers hold Design 2's *training* accuracy down to ~0.82 while validation sits at ~0.95 — dropout is active during training and disabled at evaluation, so the usual train/val gap runs backwards here.

`ModelCheckpoint` monitors `val_loss` and saves the best weights (`FirstDesignBestModel.h5`, `SecondDesignBestModel.h5`), so evaluation uses the best epoch rather than the last one.

---

## Results

### On the full test set

| Model | Best val loss | Val accuracy | **Test accuracy** | Test loss |
|---|---|---|---|---|
| Design 1 | 0.129 | ~0.963 | **0.8500** | 0.761 |
| Design 2 | 0.167 | ~0.956 | **0.8471** | 0.630 |

The two designs land within 0.3 percentage points of each other despite Design 2 having less than half the parameters — the convolutional trunk is doing most of the work, and the shape of the classifier head barely matters.

Design 2 scores marginally *lower* accuracy but noticeably *lower* loss (0.630 vs 0.761), meaning its wrong answers are less confidently wrong. If calibrated probabilities matter downstream, Design 2 is the better model despite the accuracy column.

### Distribution shift

Validation accuracy sits around 96% while test accuracy is 85% — an 11-point drop that ordinary overfitting doesn't explain, since both models were checkpointed on validation loss.

Grouping by `PatType` finds the cause:

| PatType | Train | Test |
|---|---|---|
| 0 | 4,200 | 1,207 |
| 1 | — | 182 |
| 2 | — | 9 |
| 3 | — | 2 |

**Every training sample is `PatType` 0, but 193 test samples (13.8%) are types 1–3.** The models were never shown these pattern variants and are being asked to generalize to a category of input that does not appear in training at all.

Re-evaluating on the `PatType == 0` subset only:

| Model | Full test set | PatType 0 only | Gain |
|---|---|---|---|
| Design 1 | 0.8500 | **0.9064** | +5.6 pts |
| Design 2 | 0.8471 | **0.9105** | +6.3 pts |

So roughly half the validation-to-test gap is attributable to unseen pattern types rather than to model quality. On the distribution it was actually trained on, the CNN reaches about 91%.

This is the most useful result in the project: it separates "the model is weak" from "the model is being tested on something it never saw," and those call for completely different fixes.

---

## Running it

```bash
git clone https://github.com/RovshanBayramRB/Handwritten-Azerbaijani-Letters-Classification.git
cd Handwritten-Azerbaijani-Letters-Classification
pip install tensorflow pandas numpy scikit-learn matplotlib
jupyter notebook "Handwritten Azerbaijani Letters Classification.ipynb"
```

Run cells top to bottom — the *Extra* section loads the `.h5` checkpoints written during training, so both models must be trained first. The checkpoints are not committed.

Each model takes roughly 6 minutes for 30 epochs (~12 s/epoch on Colab CPU).

**Compatibility note:** `load_model` on `.h5` files works on Keras 2. On Keras 3 (TensorFlow 2.16+), switch `ModelCheckpoint` to the `.keras` extension.

---

## Repository structure

```
.
├── Handwritten Azerbaijani Letters Classification.ipynb
├── HandChars32_150_Train.txt   # 4,200 samples, 150 per class
├── HandChars32_50_Test.txt     # 1,400 samples, 50 per class
└── README.md
```

---

## Known issues and next steps

- **`LabelBinarizer` is re-fit on each split.** Cells use `fit_transform` on validation and test rather than `transform`. It works here because all 28 classes appear in every split, but it will silently produce misaligned columns on any subset where a class is missing. `transform` is the correct call.
- **`validation_steps=10` has no effect** when `validation_data` is passed as in-memory arrays; it applies to generators. Harmless, but misleading to read.
- **No data augmentation.** Small rotations, shifts, and elastic distortions are the standard remedy for handwriting datasets this size, and would likely help most against the unseen `PatType` variants.
- **Dropout of 0.5 after convolutional layers is aggressive.** Spatial dropout, or a lower rate on the conv blocks with 0.5 reserved for the dense head, is the more usual configuration.
- **No confusion matrix.** With 28 classes including visually similar pairs, knowing *which* letters get confused would be more actionable than the aggregate accuracy.
- **Batch normalization and a learning rate schedule** are the obvious cheap wins that neither design uses.
