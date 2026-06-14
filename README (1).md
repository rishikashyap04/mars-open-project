# 🎙️ Deepfake Audio Detection

A deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using CNN on log-mel spectrograms.

\---

## 📌 Problem Statement

Advances in generative AI have enabled the creation of highly realistic synthetic speech. This project builds a machine learning system capable of detecting whether an audio recording is genuine human speech or AI-generated, addressing threats of impersonation, fraud, and misinformation.

\---

## 🚀 Live Demo

🔗 **Streamlit App:** https://mars-open-project-harmtjmwosmhmsq5d5e9wa.streamlit.app/---

## 📊 Results

|Metric|Required|Achieved|
|-|-|-|
|Overall Accuracy|≥ 80%|**82.37%** ✅|
|Equal Error Rate (EER)|≤ 12%|**10.19%** ✅|
|F1 Score|≥ 80%|**79.25%** ✅|
|Per-Class Accuracy (Genuine)|≥ 75%|**99.69%** ✅|
|Per-Class Accuracy (Deepfake)|≥ 75%|**65.82%** ✅|
|Epochs Trained|-|**18**|

### Confusion Matrix

```
                 Predicted
                Genuine   Deepfake
True  Genuine  \[  2257       7   ]
      Deepfake \[   810     1560  ]
```

* **TP** (Deepfake correctly detected): 1560
* **TN** (Genuine correctly detected): 2257
* **FP** (Genuine misclassified as Deepfake): 7
* **FN** (Deepfake missed): 810

\---

##  Repository Structure

```
.
├── app.py                          # Streamlit web app
├── predict.py                      # Standalone inference script
├── deepfake\_audio\_detection.ipynb  # Full training \& evaluation notebook
├── best\_model.pth                  # Trained model weights
├── performance\_report.json         # Metrics report
├── confusion\_matrix.png            # Confusion matrix visualization
├── training\_curves.png             # Training history plot
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies
├── .python-version                 # Python version (3.11)
└── README.md                       # This file
```

\---

##  Dataset

* **Primary:** [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

  * Variant: `for-norm` (normalized audio)
  * Splits: `training` / `validation` / `testing`
* **Reference:** [ASVspoof 2019](https://www.asvspoof.org/index2019.html)

|Label|Class|
|-|-|
|0|Genuine (Human)|
|1|Deepfake (AI-Generated)|

\---

##  Methodology

### Preprocessing

1. Load audio → resample to **16 kHz** → convert to mono
2. Pad or truncate to fixed **4 seconds**
3. Extract **log-mel spectrogram** (128 mel bins, n\_fft=1024, hop\_length=256)
4. Normalize values to **\[0, 1]**

### Feature Extraction

Log-mel spectrograms used as 2D image-like inputs of shape `(1, 128, time)`

### Data Augmentation (Training only)

**SpecAugment** applied:

* 2 random frequency masks (up to 15 bins)
* 2 random time masks (up to 25 frames)

### Class Balancing

* **WeightedRandomSampler** — forces balanced batches during training
* **Class-weighted CrossEntropy loss** — double protection against class bias
* **Label smoothing (0.05)** — prevents overconfidence

### Model Architecture — AudioCNN with Residual Connections

```
Input (1 × 128 × time)
        ↓
Stem: Conv2D(1→32) → Conv2D(32→32) → MaxPool → Dropout
        ↓
Block1: Conv2D(32→64) → ResBlock(64) → MaxPool → Dropout
        ↓
Block2: Conv2D(64→128) → ResBlock(128) → MaxPool → Dropout
        ↓
Block3: Conv2D(128→256) → ResBlock(256) → AdaptiveAvgPool
        ↓
Classifier: Linear(256→128) → Linear(128→64) → Linear(64→2)
        ↓
Output: \[Genuine, Deepfake]
```

Each ResBlock: Conv → BN → ReLU → Conv → BN + skip connection

### Training

|Parameter|Value|
|-|-|
|Optimizer|AdamW|
|Learning Rate|1e-3|
|Weight Decay|1e-4|
|Label Smoothing|0.05|
|Batch Size|32|
|Max Epochs|30|
|Early Stopping|Patience = 6|
|Gradient Clipping|max\_norm = 1.0|
|LR Scheduler|ReduceLROnPlateau|

\---


### 4\. Reproduce training

Open `deepfake\_audio\_detection.ipynb` on Kaggle with **GPU T4** enabled and run all cells.

\---

## Streamlit Web App Features

*  Upload any `.wav`, `.mp3`, `.flac` audio file
*  Color-coded Genuine / Deepfake result
*  Confidence score + probability bar chart
*  Waveform visualization
*  Log-mel spectrogram visualization
*  Sidebar with model info

\---

##  Tech Stack

* **PyTorch** — model training \& inference
* **Librosa** — audio loading \& mel spectrogram extraction
* **Streamlit** — web application
* **Scikit-learn** — metrics (F1, EER, confusion matrix)
* **Matplotlib / Seaborn** — visualizations

\---

##  Author

Rishi Kashyap

