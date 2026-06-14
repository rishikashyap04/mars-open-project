import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os

# ---------------------------
# Config
# ---------------------------
SR = 16000
DURATION = 4.0
N_SAMPLES = int(SR * DURATION)
N_MELS = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Model
# ---------------------------
class AudioCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ---------------------------
# Audio utils
# ---------------------------
def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return y


def to_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=256)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-6)
    return logmel.astype(np.float32)


@st.cache_resource
def load_model(model_path="best_model.pth"):
    model = AudioCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, audio_path):
    y = load_audio(audio_path)
    mel = to_logmel(y)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
    pred = int(probs.argmax())
    label = "Deepfake (AI-Generated)" if pred == 1 else "Genuine (Human)"
    confidence = float(probs[pred])
    return label, confidence, probs.cpu().numpy(), y, mel


def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(8, 2))
    times = np.linspace(0, DURATION, len(y))
    ax.plot(times, y, color='#1f77b4', linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim([0, DURATION])
    fig.tight_layout()
    return fig


def plot_melspectrogram(mel):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bins")
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    return fig


def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    classes = ['Genuine\n(Human)', 'Deepfake\n(AI-Generated)']
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.barh(classes, probs * 100, color=colors, edgecolor='white', height=0.4)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{p*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Class Probabilities")
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🎙️ Deepfake Audio Detection</h1>
    <p style='text-align: center; color: grey;'>Upload an audio file to detect whether it is Genuine (Human) or Deepfake (AI-Generated)</p>
    <hr>
""", unsafe_allow_html=True)

model_path = "best_model.pth"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Place it in the same directory as this app.")
    st.stop()

model = load_model(model_path)

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a CNN trained on log-mel spectrograms to classify audio as Genuine or Deepfake.")
    st.markdown("**Model:** AudioCNN")
    st.markdown("**Input:** Any audio file (.wav, .mp3, .flac)")
    st.markdown("**Output:** Label + Confidence Score")
    st.markdown("---")
    st.markdown("**Dataset:** Fake-or-Real Dataset (for-norm)")
    st.markdown("**Features:** 64-bin Log-Mel Spectrogram")

uploaded_file = st.file_uploader("📂 Upload an audio file", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)
    st.markdown("---")

    with st.spinner("🔍 Analyzing audio..."):
        try:
            label, confidence, probs, y, mel = predict(model, tmp_path)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            os.remove(tmp_path)
            st.stop()

    os.remove(tmp_path)

    st.subheader("🔎 Detection Result")
    col1, col2 = st.columns(2)

    with col1:
        if "Deepfake" in label:
            st.error(f"### 🔴 {label}")
        else:
            st.success(f"### 🟢 {label}")
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")

    with col2:
        prob_fig = plot_probabilities(probs)
        st.pyplot(prob_fig)
        plt.close()

    st.markdown("---")

    st.subheader("📊 Audio Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Waveform**")
        wave_fig = plot_waveform(y)
        st.pyplot(wave_fig)
        plt.close()

    with col4:
        st.markdown("**Log-Mel Spectrogram**")
        mel_fig = plot_melspectrogram(mel)
        st.pyplot(mel_fig)
        plt.close()

    st.markdown("---")
    st.caption("Deepfake Audio Detection | CNN on Log-Mel Spectrograms | Problem Statement 2")
