# notlame-train

Train a neural network to replace LAME's psychoacoustic model for MP3 encoding.

## Quick Start

```bash
make setup              # Install dependencies
make download-gtzan     # Download music (1.2GB)
make prepare            # Convert to MDCT
make train              # Train model
make test               # Run 141 tests (verify code)
make evaluate           # Compare vs LAME
make export             # Export to ONNX
```

## What It Does

```
Traditional MP3:  Audio → LAME psychoacoustic model → scalefactors → quantize
notlame:          Audio → Neural network (PsychoNet) → scalefactors → quantize
```

The neural network learns WHERE to spend bits for best perceptual quality.

## Documentation

- **[How To Train](docs/HOW_TO_TRAIN.md)** - Simple training guide
- **[Technical Details](docs/TRAINING.md)** - Loss functions, architecture, research
- **[Improvements](docs/IMPROVEMENTS.md)** - Future enhancements planned

## Test Suite

141 tests verify the training pipeline before you train:

```bash
make test               # Run all tests
make test-beat-lame     # Verify we beat LAME at all 9 bitrates
```

Current status: **Beats LAME on 8/9 bitrates (STFT), 9/9 bitrates (Mel)**

## Requirements

- Python 3.8+
- CUDA GPU (recommended)
- LAME encoder (`apt install lame`)

## After Training

Copy the model to notlame encoder:

```bash
cp models/psycho_v1.onnx ../notlame-lib/models/
```

## License

MIT
