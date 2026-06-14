#!/usr/bin/env python3
"""
Deepfake Audio Detection — Inference Script
MARS Open Projects 2026

Usage
-----
# Single file
python predict.py --audio path/to/audio.wav --model models/best_model.pt

# Batch (CSV with a column 'filepath')
python predict.py --csv files.csv --model models/best_model.pt --output predictions.csv

# CSV with arbitrary column name
python predict.py --csv files.csv --path_col audio_path --model models/best_model.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import timm
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Re-use the model class from train_pipeline (or define inline if run standalone)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from train_pipeline import DeepfakeAudioDetector, Config
except ImportError:
    # Inline definitions so predict.py works standalone
    from dataclasses import dataclass, field

    @dataclass
    class Config:
        sample_rate: int  = 16_000
        duration:    float= 4.0
        n_mels:      int  = 128
        n_fft:       int  = 1024
        hop_length:  int  = 256
        backbone:    str  = "efficientnet_b0"
        pretrained:  bool = False
        device:      str  = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    class DeepfakeAudioDetector(nn.Module):
        def __init__(self, backbone="efficientnet_b0", pretrained=False, num_classes=2, dropout=0.3):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                               in_chans=1, num_classes=0, global_pool="avg")
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 256), nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(256, num_classes),
            )
        def forward(self, x):
            return self.head(self.backbone(x))


LABEL_NAMES = {0: "Real (Genuine)", 1: "Fake (AI-Generated)"}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (mirrors AudioDataset._load_and_preprocess)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_audio(path: str, cfg: Config) -> torch.Tensor:
    """
    Load an audio file and return a normalised mel-spectrogram tensor of
    shape (1, 1, n_mels, T) ready for model inference.
    """
    waveform, sr = torchaudio.load(path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # Resample
    if sr != cfg.sample_rate:
        waveform = T.Resample(sr, cfg.sample_rate)(waveform)

    # Pad / centre-crop
    target = int(cfg.sample_rate * cfg.duration)
    n = waveform.shape[1]
    if n < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - n))
    else:
        start = (n - target) // 2
        waveform = waveform[:, start : start + target]

    # Mel spectrogram
    mel = T.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )(waveform)
    mel = T.AmplitudeToDB(top_db=80)(mel)

    # Normalise to [-1, 1]
    mn, mx = mel.min(), mel.max()
    if mx > mn:
        mel = 2.0 * (mel - mn) / (mx - mn) - 1.0

    return mel.unsqueeze(0)   # (1, 1, n_mels, T)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> Tuple[DeepfakeAudioDetector, Config]:
    """Load a checkpoint saved by train_pipeline.py."""
    ckpt = torch.load(model_path, map_location="cpu")
    
    saved_cfg = ckpt.get("config", {})
    cfg = Config()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = DeepfakeAudioDetector(
        backbone=cfg.backbone, pretrained=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    model: DeepfakeAudioDetector,
    cfg: Config,
    audio_path: str,
    device: str = "cpu",
) -> dict:
    """
    Predict a single audio file.

    Returns
    -------
    dict with keys: filepath, label, label_name, confidence, fake_score, real_score
    """
    tensor = preprocess_audio(audio_path, cfg).to(device)
    model = model.to(device)

    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx   = int(probs.argmax().item())
    fake_score = float(probs[1].item())
    real_score = float(probs[0].item())

    return {
        "filepath":   audio_path,
        "label":      pred_idx,
        "label_name": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "fake_score": fake_score,
        "real_score": real_score,
    }


@torch.no_grad()
def predict_batch(
    model: DeepfakeAudioDetector,
    cfg: Config,
    paths: List[str],
    device: str = "cpu",
    batch_size: int = 32,
) -> List[dict]:
    """Predict a list of audio file paths in batches."""
    model = model.to(device)
    results = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Predicting"):
        batch_paths = paths[i : i + batch_size]
        tensors = []
        failed  = []

        for p in batch_paths:
            try:
                tensors.append(preprocess_audio(p, cfg))
            except Exception as e:
                print(f"  [WARN] Could not load {p}: {e}")
                failed.append(p)

        if not tensors:
            for p in failed:
                results.append({
                    "filepath": p, "label": -1,
                    "label_name": "ERROR", "confidence": 0.0,
                    "fake_score": 0.0, "real_score": 0.0,
                })
            continue

        batch = torch.cat(tensors, dim=0).to(device)    # (B, 1, n_mels, T)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)

        for j, path in enumerate(batch_paths):
            if path in failed:
                results.append({
                    "filepath": path, "label": -1,
                    "label_name": "ERROR", "confidence": 0.0,
                    "fake_score": 0.0, "real_score": 0.0,
                })
            else:
                p_vec    = probs[j - len(failed)]
                pred_idx = int(p_vec.argmax().item())
                results.append({
                    "filepath":   path,
                    "label":      pred_idx,
                    "label_name": LABEL_NAMES[pred_idx],
                    "confidence": float(p_vec[pred_idx].item()),
                    "fake_score": float(p_vec[1].item()),
                    "real_score": float(p_vec[0].item()),
                })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        sys.exit(f"[ERROR] Model file not found: {args.model}")

    print(f"Loading model from {args.model} …")
    model, cfg = load_model(args.model)
    device = cfg.device
    print(f"  Backbone: {cfg.backbone}  |  Device: {device}\n")

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.audio:
        result = predict_single(model, cfg, args.audio, device)
        print("─" * 50)
        print(f"  File       : {result['filepath']}")
        print(f"  Prediction : {result['label_name']}")
        print(f"  Confidence : {result['confidence']*100:.2f}%")
        print(f"  Real score : {result['real_score']:.4f}")
        print(f"  Fake score : {result['fake_score']:.4f}")
        print("─" * 50)
        return

    # ── CSV batch mode ────────────────────────────────────────────────────────
    if args.csv:
        df = pd.read_csv(args.csv)
        col = args.path_col

        if col not in df.columns:
            sys.exit(f"[ERROR] Column '{col}' not found in CSV. "
                     f"Available: {list(df.columns)}")

        paths   = df[col].tolist()
        results = predict_batch(model, cfg, paths, device, batch_size=args.batch_size)
        out_df  = pd.DataFrame(results)

        # Merge with original dataframe (preserves any ground-truth columns)
        out_df = df.merge(
            out_df[["filepath", "label", "label_name", "confidence",
                    "fake_score", "real_score"]],
            left_on=col, right_on="filepath", how="left",
        )

        output_path = args.output or "predictions.csv"
        out_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")

        # Summary
        valid = out_df[out_df["label"] >= 0]
        n_real = (valid["label"] == 0).sum()
        n_fake = (valid["label"] == 1).sum()
        n_err  = (out_df["label"] == -1).sum()
        print(f"  Total   : {len(out_df)}")
        print(f"  Real    : {n_real}")
        print(f"  Fake    : {n_fake}")
        if n_err:
            print(f"  Errors  : {n_err}")

        # Optional accuracy if ground-truth column provided
        if args.gt_col and args.gt_col in out_df.columns:
            from sklearn.metrics import accuracy_score, classification_report
            gt_map = {"real": 0, "genuine": 0, "fake": 1, "spoof": 1}
            y_true = valid[args.gt_col].astype(str).str.lower().map(gt_map)
            y_pred = valid["label"]
            mask   = y_true.notna()
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                print(f"\n  Accuracy vs '{args.gt_col}': {acc*100:.2f}%")
                print(classification_report(
                    y_true[mask], y_pred[mask],
                    target_names=["Real", "Fake"], zero_division=0,
                ))
        return

    print("[ERROR] Provide --audio <file> or --csv <file>")
    print("Run python predict.py --help for usage.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deepfake Audio Detector — Inference")
    ap.add_argument("--model",      type=str, required=True,
                    help="Path to best_model.pt checkpoint")
    ap.add_argument("--audio",      type=str, default=None,
                    help="Single audio file for inference")
    ap.add_argument("--csv",        type=str, default=None,
                    help="CSV file containing audio file paths")
    ap.add_argument("--path_col",   type=str, default="filepath",
                    help="Column name in CSV that holds file paths (default: filepath)")
    ap.add_argument("--gt_col",     type=str, default=None,
                    help="Optional column in CSV with ground-truth labels for accuracy reporting")
    ap.add_argument("--output",     type=str, default="predictions.csv",
                    help="Output CSV path (batch mode)")
    ap.add_argument("--batch_size", type=int, default=32)
    main(ap.parse_args())
