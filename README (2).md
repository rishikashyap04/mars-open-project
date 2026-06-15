# &#x20;Deepfake Audio Detection

A deep learning system that classifies speech recordings as either **Genuine (Human)** or **Deepfake (AI-Generated)** using a lightweight CNN with channel attention, log-mel spectrograms, probability calibration, and fine-grained threshold search.

\---

## &#x20;Project Description

With advances in generative AI enabling highly realistic synthetic speech, detecting deepfake audio has become a critical security challenge. This project develops a robust binary classifier capable of distinguishing real human speech from AI-generated audio, trained on the **Fake-or-Real Dataset** (Kaggle) and optimized to meet strict accuracy and EER thresholds.

\---

## &#x20;Performance Results

|Metric|Result|Threshold|
|-|-|-|
|Overall Accuracy|**82.78%**|≥ 80% |
|Equal Error Rate (EER)|**10.19%**|≤ 12% |
|F1 Score|**85.94%**|≥ 80% |
|Genuine (Per-Class Accuracy)|**85.95%**|≥ 75% |
|Deepfake (Per-Class Accuracy)|**79.75%**|≥ 75% |

### Confusion Matrix

||Predicted Genuine|Predicted Deepfake|
|-|-|-|
|**Actual Genuine**|1946|318|
|**Actual Deepfake**|480|1890|

### Training Summary

* **Epochs trained:** 18
* **Best validation accuracy:** 99.92%
* **Best validation loss:** 0.1236

\---

## &#x20;Repository Structure

```
deepfake-audio-detection/
│
├── notebooks/
│   └── deepfake\_audio\_detection\_fixed.ipynb  # Full training \& evaluation notebook
│
├── model/
│   └── best\_model.pth                        # Trained model weights
│
├── app/
│   └── streamlit\_app.py                      # Streamlit web application
│
├── scripts/
│   └── predict.py                            # CLI script to test new audio samples
│
├── reports/
│   ├── performance\_report.json               # Evaluation metrics
│   ├── confusion\_matrix.png                  # Confusion matrix visualization
│   └── training\_curves.png                   # Loss / accuracy / balanced accuracy plots
│
├── requirements.txt
└── README.md
```

\---

## &#x20;Methodology

### 1\. Dataset \& Splits

* **Dataset:** [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) — `for-norm` variant
* **Splits used:** `training/`, `validation/`, `testing/` sub-directories
* **Label mapping:** `real/` → Genuine (0), `fake/` → Deepfake (1)

\---

### 2\. Class Balancing (Capped Oversampling + Data-Driven Weights)

A two-stage approach is used to handle class imbalance without introducing bias:

**Capped oversampling:** The minority class is oversampled with replacement, but capped at a maximum ratio of **3×** to prevent the model from memorizing duplicated clips.

**Data-driven class weights:** After oversampling, residual imbalance is corrected using class weights computed from the actual resulting counts:

```
weight\[class] = (n0 + n1) / (2 × n\_class)
```

These weights are fed into the loss function, replacing the previous hard-coded `alpha=0.75` which incorrectly assumed Deepfake was always the minority class.

\---

### 3\. Preprocessing \& Feature Extraction

Each audio file is processed as follows:

* **Resampling:** All audio loaded at **16,000 Hz**, mono
* **Fixed length:** Padded with zeros or truncated to exactly **4 seconds** (64,000 samples)
* **Log-Mel Spectrogram:**

  * `n\_mels = 128` mel filter banks
  * `n\_fft = 1024`, `hop\_length = 256`
  * Converted to dB scale with `librosa.power\_to\_db`
  * Min-max normalized to `\[0, 1]`
* **Caching:** All mel spectrograms are pre-computed and stored in RAM before training begins — every epoch reuses cached data with zero disk I/O

\---

### 4\. Data Augmentation (SpecAugment)

Applied on-the-fly during training only:

* **Frequency masking:** 2 masks of up to 20 mel bins each, zeroed out
* **Time masking:** 2 masks of up to 30 time steps each, zeroed out
* **Gaussian noise:** With 40% probability, small Gaussian noise (`σ = 0.02`) is added to the spectrogram and clipped to `\[0, 1]`

\---

### 5\. Model Architecture — `AudioCNN`

A lightweight CNN with residual blocks and channel attention:

```
Input: Log-Mel Spectrogram \[1 × 128 × T]
    │
    ├── Stem: Conv2d(1→32, 3×3) + BN + ReLU + MaxPool(2) + Dropout2d(0.1)
    │
    ├── Block 1: Conv2d(32→64, 3×3) + BN + ReLU
    │            + ResBlock(64) + MaxPool(2) + Dropout2d(0.1)
    │
    ├── Block 2: Conv2d(64→128, 3×3) + BN + ReLU
    │            + ResBlock(128) + MaxPool(2) + Dropout2d(0.2)
    │
    ├── Block 3: Conv2d(128→128, 3×3) + BN + ReLU
    │            + AdaptiveAvgPool2d(1×1)
    │
    └── Classifier: Linear(128→64) + ReLU + Dropout(0.5) → Linear(64→2)
```

**ResBlock** (used inside Blocks 1 \& 2):

```
x → Conv2d(3×3) + BN + ReLU → Conv2d(3×3) + BN
                                     ↓
                          ChannelAttention (squeeze-and-excite)
                                     ↓
                             x + attended output → ReLU
```

**ChannelAttention:** Squeeze via `AdaptiveAvgPool2d(1)` → two FC layers with ReLU and Sigmoid → channel-wise scale.

Total parameters: \~lightweight (suitable for fast training on a single GPU).

\---

### 6\. Loss Function — Focal Loss

```
FL(p\_t) = α\_t · (1 − p\_t)^γ · CE(p\_t)
```

* **γ = 2.0** (focusing parameter — down-weights easy examples)
* **α\_t** = data-driven class weights from §2 (adapts automatically to the actual class distribution, not hard-coded)

\---

### 7\. Training Configuration

|Hyperparameter|Value|
|-|-|
|Optimizer|AdamW|
|Learning rate|1e-3|
|Weight decay|1e-4|
|Batch size|64|
|Max epochs|30|
|Early stopping patience|7 epochs|
|LR scheduler|ReduceLROnPlateau (factor=0.5, patience=2)|
|Mixed precision|AMP (torch.cuda.amp) — \~2× faster on GPU|
|Gradient clipping|max norm = 1.0|

**Checkpoint selection:** Best model saved by **validation balanced accuracy** (mean per-class accuracy), not raw validation loss. This directly targets the per-class accuracy requirements and prevents saving a checkpoint that has collapsed onto the majority class.

\---

### 8\. Probability Calibration — Temperature Scaling

After training, the raw model logits are calibrated using **temperature scaling**:

```
calibrated\_probs = softmax(logits / T)
```

Temperature `T` is found by minimizing validation NLL over a grid search from 0.25 to 5.0 (step 0.05). This corrects systematic probability compression (where `P(deepfake)` was clustered in a narrow range for all samples), restoring the output probabilities to a meaningful `\[0, 1]` spread without changing the ranking of predictions (so EER is preserved).

\---

### 9\. Threshold Search (on Calibrated Probabilities)

A fine-grained grid search over thresholds from 0.01 to 0.99 in steps of **0.001** is run on calibrated validation probabilities. The selected threshold is the one that satisfies **all** of the following simultaneously:

* Accuracy ≥ 80%
* F1 Score ≥ 80%
* Genuine per-class accuracy ≥ 75%
* Deepfake per-class accuracy ≥ 75%

If multiple thresholds qualify, the one with the highest balanced accuracy is chosen. If none qualify, the balanced-accuracy maximizer is used as a fallback.

\---

### 10\. Inference

At inference time, the same calibration temperature and threshold are applied:

```python
probs = softmax(model(mel) / T)
pred  = "Deepfake" if probs\[1] >= threshold else "Genuine"
confidence = probs\[predicted\_class]
```

Both `calibration\_temperature` and `threshold` are saved in `performance\_report.json` alongside all metrics.

\---

## &#x20;Pipeline

```
Raw Audio (.wav)
        ↓
  Load \& Resample (16kHz, mono, 4 seconds)
        ↓
  Log-Mel Spectrogram (128 mels, n\_fft=1024, hop=256) + Min-Max Normalize
        ↓
  \[Training only] SpecAugment (freq mask + time mask + Gaussian noise)
        ↓
  AudioCNN (Stem → ResBlock × 2 + Channel Attention → Classifier)
        ↓
  Temperature Scaling (calibrated probabilities)
        ↓
  Threshold Classification
        ↓
  "Genuine (Human)" or "Deepfake (AI-Generated)" + Confidence Score
```

\---

## &#x20;Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `librosa`, `soundfile`, `scikit-learn`, `tqdm`, `seaborn`, `streamlit`, `matplotlib`

### Training

Open and run the notebook on Kaggle (GPU T4 recommended):

```bash
jupyter notebook notebooks/deepfake\_audio\_detection\_fixed.ipynb
```

> The notebook is self-contained and requires no internet access after the dataset is downloaded.

### Testing a New Audio Sample (CLI)

```bash
python scripts/predict.py --audio path/to/audio.wav
```

**Example output:**

```
File       : sample.wav
Prediction : Deepfake (AI-Generated)
Confidence : 91.4%
```

\---

## &#x20;Streamlit Web App

```bash
streamlit run app/streamlit\_app.py
```

**Features:**

* Upload audio in `.wav`, `.mp3`, or `.flac` format
* Returns: **Genuine (Human)** or **Deepfake (AI-Generated)**
* Displays confidence score as a percentage

\---

## &#x20;Dataset

**Primary Dataset:** [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

* Variant used: `for-norm` (normalized audio)
* Splits: `training/`, `validation/`, `testing/`

**Reference Benchmark:** [ASVspoof 2019](https://www.asvspoof.org/index2019.html)

* Available for cross-dataset generalization evaluation

\---

## 

