# How To Train notlame

Simple guide to train the neural MP3 encoder.

## Quick Start

```bash
# 1. Setup environment
make setup

# 2. Download music
make download-gtzan

# 3. Prepare data
make prepare

# 4. Run tests (verify code works)
make test

# 5. Train
make train

# 6. Check results
make evaluate
```

That's it!

---

## What Happens During Training

### Step 1: Data Preparation

```
Audio files (.wav, .mp3, .flac)
         ↓
    [MDCT Transform]
         ↓
MDCT coefficients (.npy files)
```

We pre-compute MDCT coefficients to speed up training.

### Step 2: Training Loop

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│   Load 4 MDCT frames                                   │
│           ↓                                            │
│   Model predicts scalefactors (22 values per frame)   │
│           ↓                                            │
│   Quantize using scalefactors                          │
│           ↓                                            │
│   Reconstruct audio (overlap-add)                      │
│           ↓                                            │
│   Compare: reconstructed vs original                   │
│           ↓                                            │
│   Calculate loss (how bad does it sound?)              │
│           ↓                                            │
│   Update model to reduce loss                          │
│           ↓                                            │
│   Repeat 100,000 times                                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Step 3: What the Model Learns

The model learns **where to spend bits**:

```
Scalefactor = 0  → Best quality (more bits)
Scalefactor = 15 → Worst quality (fewer bits)

The model learns:
- Use low scalefactors for important sounds
- Use high scalefactors for masked/quiet sounds
- Balance quality vs file size
```

---

## Testing

Run the test suite before training to verify everything works:

```bash
# Run all 141 tests
make test

# Verify we beat LAME at all 9 standard bitrates
make test-beat-lame
```

Current test results:
- **STFT Loss**: Beat LAME on 8/9 bitrates
- **Mel Loss**: Beat LAME on 9/9 bitrates
- **Both metrics**: Beat LAME on 8/9 bitrates

---

## Training Commands

### Basic Training
```bash
make train
```

### Background Training (recommended)
```bash
nohup make train > nohup.out 2>&1 &

# Watch progress
tail -f nohup.out
```

### Resume After Interruption
```bash
make train-resume
```

### Monitor with TensorBoard
```bash
make tensorboard
# Open http://localhost:6006
```

### Custom Settings
```bash
make train BATCH_SIZE=64 TRAIN_STEPS=200000
```

---

## Datasets

| Dataset | Size | Time | Quality | Command |
|---------|------|------|---------|---------|
| GTZAN | 1.2GB | ~1 hour train | Testing | `make download-gtzan` |
| FMA-small | 7.2GB | ~3 hours train | Good | `make download-fma` |
| FMA-large | 93GB | ~24 hours train | Best | `make download-fma-large` |

**Recommendation:** Start with GTZAN to test, use FMA-large for production.

---

## Evaluation

After training, compare against LAME:

```bash
make evaluate
```

Output:
```
Metric          notlame    LAME      Winner
──────────────────────────────────────────
SNR (dB)        35.0       29.5      notlame  ✓
MR-STFT         0.45       0.55      notlame  ✓
Mel Distance    0.10       0.12      notlame  ✓
```

**Goal:** Beat LAME on all metrics!

---

## Export for Production

```bash
make export
```

Creates `models/psycho_v1.onnx` for use in the notlame encoder.

---

## Troubleshooting

### Out of GPU memory
```bash
make train BATCH_SIZE=16
```

### Training too slow
```bash
make train CACHE=1  # Load data to GPU (default)
```

### Want to see what's happening
```bash
make tensorboard
```

### Model not improving
- Train longer: `TRAIN_STEPS=200000`
- Use more data: `make download-fma-large`
- Check learning rate: `LEARNING_RATE=3e-4`

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4GB VRAM | 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Disk | 10GB | 100GB (for FMA-large) |
| Time | 1 hour (GTZAN) | 24 hours (FMA-large) |
