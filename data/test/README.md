# Test Audio Samples

This directory contains 20 audio samples for evaluating the notlame neural MP3 encoder.

## Source: GTZAN Genre Collection

These samples are from the **GTZAN Genre Collection**, a widely-used dataset for music genre classification research.

### Citation

```
Tzanetakis, G., & Cook, P. (2002).
Musical genre classification of audio signals.
IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.
```

### Dataset Information

- **Original source**: http://marsyas.info/downloads/datasets.html
- **License**: For research purposes
- **Format**: 22050 Hz, mono, WAV
- **Duration**: 30 seconds per track

### Samples Included

2 tracks from each of the 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

### Usage

These samples are used to evaluate the notlame encoder against LAME at various metrics:
- SNR (Signal-to-Noise Ratio)
- MR-STFT (Multi-Resolution STFT Distance)
- Mel Spectrogram Distance

Run evaluation with:
```bash
make evaluate
```

### Note on Sample Rate

The GTZAN samples are 22050 Hz. The notlame model (trained at 44100 Hz) will
automatically resample them for processing and resample back for comparison.
