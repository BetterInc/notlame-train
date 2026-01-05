# notlame-train

Training pipeline for **notlame** - a neural MP3 encoder that replaces LAME's psychoacoustic model with a learned neural network.

## What This Does

Traditional MP3 encoders like LAME use hand-crafted psychoacoustic models to decide how many bits to allocate to each frequency band. This project trains a neural network (**PsychoNet**) to make these decisions instead.

**Input:** 576 MDCT coefficients (one MP3 granule)
**Output:** 21 scalefactor values (one per frequency band, range 0-15)

### MP3 Scalefactor Semantics
- **LOW scalefactor (0-5):** Fine quantization, more bits, higher quality
- **HIGH scalefactor (10-15):** Coarse quantization, fewer bits, lower quality

## Quick Start

```bash
# 1. Setup environment
make setup

# 2. Download audio data
make download-gtzan

# 3. Validate and prepare data
make validate
make prepare

# 4. Train model
make train

# 5. Evaluate against LAME
make evaluate

# 6. Export to ONNX
make export
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)
- ~10GB disk space for datasets
- LAME encoder (`apt install lame`) for evaluation

## Detailed Pipeline

### Step 1: Setup Environment

```bash
make setup
```

Creates a Python virtual environment and installs dependencies:
- PyTorch, torchaudio
- NumPy, SciPy, librosa
- TensorBoard
- ONNX, ONNX Runtime

### Step 2: Download Audio Data

| Command | Dataset | Size | Hours |
|---------|---------|------|-------|
| `make download` | Test samples | ~50MB | <1 min |
| `make download-gtzan` | GTZAN music | 1.2GB | 8 hrs |
| `make download-fma` | FMA-small | 7.2GB | 66 hrs |
| `make download-librispeech` | LibriSpeech | 6.3GB | 100 hrs |

```bash
# Recommended for first training
make download-gtzan

# List all available datasets
make download-list
```

### Step 3: Validate Audio Files

```bash
make validate
```

Checks audio files for:
- Valid format (WAV, FLAC, MP3, OGG)
- Sample rate >= 22050 Hz
- Duration >= 5 seconds
- No clipping or excessive silence

Output: `validation_report.json`

### Step 4: Prepare Dataset

```bash
make prepare WORKERS=8
```

Converts audio to MDCT frames:
- Frame size: 1152 samples (MP3 standard)
- Hop size: 576 samples (50% overlap)
- Output: 576 MDCT coefficients per frame
- Files saved to `data/processed/*.npy`

### Step 5: Train Model

```bash
# Basic training
make train

# With custom parameters
make train DEVICE=cuda BATCH_SIZE=64 TRAIN_STEPS=100000

# Resume from checkpoint
make train-resume
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | cuda | cuda, cuda:0, cpu |
| `BATCH_SIZE` | 32 | Batch size |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `TRAIN_STEPS` | 100000 | Total steps |
| `MODEL_VARIANT` | default | lite, default, large |
| `CACHE` | 1 | Cache data in GPU memory |

**Monitor with TensorBoard:**

```bash
make tensorboard
# Open http://localhost:6006
```

**Checkpoints saved:**
- `checkpoints/latest.pt` - Most recent
- `checkpoints/best.pt` - Best validation loss
- `checkpoints/step_*.pt` - Periodic saves

### Step 6: Evaluate Model

```bash
make evaluate
```

Compares trained model against LAME at 192 kbps:

| Metric | Description | Target |
|--------|-------------|--------|
| SNR | Signal-to-Noise Ratio (dB) | > 20 |
| MR-STFT | Multi-resolution STFT distance | < 0.5 |
| Mel | Mel spectrogram distance | < 1.0 |
| ViSQOL | Perceptual MOS (1-5) | > 4.0 |

Output: `evaluation_report.json`

### Step 7: Export to ONNX

```bash
make export
```

Exports model to `models/psycho_v1.onnx` for use in notlame-lib (Rust).

## Architecture

### PsychoNet Model

```
Input: (batch, 576) MDCT coefficients
    │
    ├── Per-band feature extraction (21 bands)
    │
    ├── Global context encoder
    │
    ├── Cross-band attention (3 transformer layers)
    │
    └── Scalefactor head
            │
Output: (batch, 21) scalefactors [0-15]
```

**Model Variants:**

| Variant | Parameters | Use Case |
|---------|------------|----------|
| `lite` | 165k | Mobile, fast inference |
| `default` | 265k | Balanced |
| `large` | 1M | Maximum quality |

### Training Pipeline

1. **MDCT Transform:** Convert audio frames to frequency domain
2. **Neural Prediction:** PsychoNet predicts scalefactors from MDCT coefficients
3. **Differentiable Quantization:** Simulate MP3 quantization with straight-through estimator
4. **Rate-Distortion Loss:** Balance quality (distortion) vs compression (bitrate)

### Loss Function

```python
loss = distortion + rate_weight * rate

# distortion: Perceptually-weighted MDCT reconstruction error
# rate: Penalty for using too many bits (low scalefactors)
```

## Project Structure

```
notlame-train/
├── Makefile                  # Build automation
├── notlame_train/
│   ├── model.py              # PsychoNet architecture
│   ├── differentiable_mp3.py # MDCT + quantization
│   ├── losses.py             # Loss functions
│   ├── dataset.py            # Data loading
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation metrics
│   └── export_onnx.py        # ONNX export
├── scripts/
│   ├── download_data.py      # Dataset download
│   └── prepare_dataset.py    # Audio → MDCT
├── data/
│   ├── raw/                  # Audio files
│   └── processed/            # MDCT .npy files
├── checkpoints/              # Model checkpoints
├── models/                   # Exported ONNX
└── runs/                     # TensorBoard logs
```

## Testing

```bash
# Run all module tests
make test

# Quick training test (50 steps, CPU)
make test-quick
```

## Troubleshooting

### "No .npy files found"
Run `make prepare` to convert audio to MDCT format.

### "CUDA out of memory"
Reduce batch size: `make train BATCH_SIZE=16`

### "LAME not found"
Install LAME: `apt install lame` (Linux) or `brew install lame` (macOS)

### Poor evaluation results
- Ensure you trained for enough steps (100k minimum)
- Check that `best.pt` checkpoint exists
- Try training with more data (`make download-fma`)

### Slow training
- Enable GPU: `make train DEVICE=cuda`
- Enable caching: `make train CACHE=1`
- Use more workers: `make train WORKERS=8`

## After Training

Copy the exported model to notlame-lib:

```bash
cp models/psycho_v1.onnx ../notlame-lib/models/
```

## Technical Details

### MDCT (Modified Discrete Cosine Transform)
- Frame size: 1152 samples
- Output: 576 coefficients
- Window: Sine window (Princen-Bradley)
- Perfect reconstruction via 50% overlap-add

### Scalefactor Bands (44.1 kHz)
21 bands covering 576 MDCT coefficients:
```
[0-4, 4-8, 8-12, 12-16, 16-20, 20-24, 24-30, 30-36, 36-44, 44-52,
 52-62, 62-74, 74-90, 90-110, 110-134, 134-162, 162-196, 196-238,
 238-288, 288-342, 342-576]
```

### MP3 Quantization Formula
```
quantized = sign(x) * round(|x|^0.75 / step)
step = 2^(scalefactor / 4)
```

## License

MIT
