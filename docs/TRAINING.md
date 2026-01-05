# Neural Scalefactor Prediction for MP3 Encoding
## Training Methodology and Perceptual Loss Design

**notlame-train: A Differentiable MP3 Psychoacoustic Model**

---

## Abstract

We present a training methodology for neural prediction of MP3 scalefactors that optimizes for perceptual audio quality. Traditional MP3 encoders like LAME use hand-crafted psychoacoustic models to determine quantization parameters. We replace this with a learned model that directly optimizes perceptual metrics. Our approach combines insights from recent neural audio codec research (DAC, EnCodec, SoundStream) with the specific constraints of MP3 encoding. We use multi-scale mel spectrogram losses with L1+L2 combination, multi-resolution STFT losses with log and linear magnitude terms, and a rate-distortion penalty to balance quality against compression. Our training pipeline operates entirely in the MDCT domain, matching MP3's native representation, with differentiable quantization enabling end-to-end gradient flow. We provide thorough documentation of loss function design choices based on extensive literature review.

## 1. Introduction

MP3 encoding requires determining 21 scalefactors per frame, each controlling the quantization step size for a frequency band. The relationship is:

```
step_size = 2^(scalefactor / 4)
```

- **Low scalefactor** = fine quantization = more bits = higher quality
- **High scalefactor** = coarse quantization = fewer bits = more compression

Traditional encoders (LAME, libmp3lame) use psychoacoustic models based on masking thresholds and equal-loudness contours. These models are carefully tuned but fundamentally hand-crafted. Our goal is to learn this mapping directly from data, optimizing for perceptual quality metrics.

### 1.1 Challenges

Unlike neural audio codecs (DAC, EnCodec) which learn the entire encoder-decoder pipeline, we operate within MP3's fixed structure:

1. **Fixed transform**: MDCT with 576 coefficients per frame (1152 samples at 44.1kHz)
2. **Fixed quantization**: Power-law quantization with scalefactor-controlled step sizes
3. **Short frames**: Only 1152 samples (~26ms) limits the time-frequency resolution of perceptual losses
4. **No learned codebook**: We predict continuous scalefactors, not discrete codes

### 1.2 Contributions

- A multi-scale perceptual loss formulation adapted for short MP3 frames
- Combination of MDCT-domain and audio-domain losses for robust training
- Rate-distortion penalty that encourages compression without explicit bitrate modeling
- Thorough documentation of design choices based on neural audio codec literature

## 2. Related Work

### 2.1 Neural Audio Codecs

**SoundStream** [Zeghidour et al., 2021] introduced end-to-end neural audio compression with residual vector quantization. Key insights:
- Multi-scale waveform discriminators improve perceptual quality
- Adversarial losses are critical for phase reconstruction
- Feature matching loss stabilizes GAN training

**EnCodec** [Défossez et al., 2022] extended SoundStream with:
- Multi-scale STFT discriminator
- Loss balancer for gradient-scale decoupling
- Support for variable bitrates via quantizer dropout

**DAC (Improved RVQGAN)** [Kumar et al., 2023] achieved state-of-the-art quality with:
- Multi-scale mel reconstruction loss with window lengths [32, 64, 128, 256, 512, 1024, 2048]
- Loss weights: λ_mel = 15.0, λ_feat = 2.0, λ_adv = 1.0
- Snake activation for periodic inductive bias
- L2-normalized codebook lookup

### 2.2 Key Insight: Discriminators Are Optional

While adversarial training significantly improves neural codec quality, recent work shows that carefully designed reconstruction losses can achieve competitive results:

**MelCap** [2025] demonstrates that:
- Multi-scale prevents over-smoothing
- Two-stage training (perceptual loss → Gram-matrix loss) stabilizes learning
- Perceptual loss can substitute for feature matching loss

This is particularly relevant for our use case, where we predict quantization parameters rather than generate waveforms directly.

## 3. Method

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MDCT Coefficients ──► PsychoNet ──► Scalefactors (21)      │
│       (576)              │              │                    │
│         │                │              │                    │
│         ▼                │              ▼                    │
│   ┌─────────────┐        │      ┌──────────────┐            │
│   │   Inverse   │        │      │ Differentiable│            │
│   │    MDCT     │        │      │  Quantization │            │
│   └──────┬──────┘        │      └───────┬──────┘            │
│          │               │              │                    │
│          ▼               │              ▼                    │
│    Original Audio        │      Quantized MDCT              │
│       (1152)             │          (576)                   │
│          │               │              │                    │
│          │               │              ▼                    │
│          │               │      ┌──────────────┐            │
│          │               │      │   Inverse    │            │
│          │               │      │    MDCT      │            │
│          │               │      └───────┬──────┘            │
│          │               │              │                    │
│          ▼               │              ▼                    │
│   ┌──────────────────────┴──────────────────────┐           │
│   │            Perceptual Losses                 │           │
│   │  • Multi-Resolution STFT (log + linear mag)  │           │
│   │  • Multi-Scale Mel (L1 + L2)                 │           │
│   │  • MDCT Reconstruction                       │           │
│   └──────────────────────────────────────────────┘           │
│                          │                                   │
│                          ▼                                   │
│                   Rate Penalty ◄── Target SF                 │
│                          │                                   │
│                          ▼                                   │
│                    Total Loss                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Differentiable Quantization

MP3 quantization is inherently non-differentiable (floor operation). We use the straight-through estimator:

```python
# Forward: actual quantization
quantized = sign(x) * floor(abs(x) / step + 0.5) * step

# Backward: pass gradients through as if identity
grad_input = grad_output  # STE
```

The audio scale factor converts MDCT coefficients to appropriate magnitude:

```python
audio_scale = 100.0  # Empirically tuned
# SF=0: ~47 dB SNR, SF=15: ~25 dB SNR
```

This ensures scalefactors have meaningful impact on quality. Too large a scale (e.g., 4096) makes quantization too fine, eliminating the quality-compression tradeoff.

## 4. Loss Function Design

### 4.1 Multi-Scale Mel Spectrogram Loss

Based on DAC's formulation, we compute mel spectrograms at multiple window sizes to capture both fine temporal detail and broad spectral structure.

**Window lengths**: [32, 64, 128, 256, 512]

Adapted from DAC's [32, 64, 128, 256, 512, 1024, 2048] for our 1152-sample frames. Larger windows would produce only 1-2 STFT frames, providing insufficient gradient signal.

**Hop length**: window_length / 4 (75% overlap)

Following DAC and HiFi-GAN conventions for smooth spectral transitions.

**Loss formulation** (L1 + L2):

```python
def forward(self, x: Tensor, y: Tensor) -> Tensor:
    l1_total = 0.0
    l2_total = 0.0

    for loss_fn in self.losses:  # One per window size
        x_mel = loss_fn.mel_spectrogram(x)
        y_mel = loss_fn.mel_spectrogram(y)
        l1_total += F.l1_loss(x_mel, y_mel)
        l2_total += F.mse_loss(x_mel, y_mel)

    # L1 preserves detail, L2 provides stability
    return (l1_total / n) + 0.5 * (l2_total / n)
```

**Rationale for L1 + L2**:
- L1 loss preserves fine spectral detail (sharp edges)
- L2 loss provides training stability (smooth gradients)
- MelCap [2025] shows this combination prevents over-smoothing

### 4.2 Multi-Resolution STFT Loss

We compute STFT losses at multiple resolutions, combining spectral convergence and magnitude losses.

**FFT sizes**: [64, 128, 256, 512]

Matched to our short frames. Each resolution captures different time-frequency tradeoffs.

**Loss components**:

```python
def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    x_mag = self.stft(x)
    y_mag = self.stft(y)

    # Spectral convergence: normalized Frobenius norm
    sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    # Log magnitude: captures dynamics across frequency range
    log_mag_loss = F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))

    # Linear magnitude: captures spectral shape
    lin_mag_loss = F.l1_loss(x_mag, y_mag)

    # Combined (auraloss recommendation)
    mag_loss = log_mag_loss + 0.5 * lin_mag_loss

    return sc_loss, mag_loss
```

**Rationale for log + linear**:
- Log magnitude balances low and high energy components (dB-like)
- Linear magnitude directly penalizes spectral shape errors
- auraloss library documents this combination as best practice

### 4.3 MDCT Domain Loss

Direct loss on MDCT coefficients with perceptual weighting by band:

| Bands | Frequency Range | Weight | Rationale |
|-------|-----------------|--------|-----------|
| 0-6   | Bass/Low-mids   | 2.0-3.0 | Highest perceptual importance |
| 7-14  | Mids            | 1.1-1.5 | Moderate importance |
| 15-20 | Highs           | 0.5-1.0 | Lower sensitivity |

Weights are based on equal-loudness contours and critical band importance.

### 4.4 Rate-Distortion Penalty

Without explicit rate control, the model learns to use SF=0 everywhere (maximum quality, no compression). We add a penalty encouraging compression:

```python
sf_mean = scalefactors.mean()
rate_penalty = torch.relu(target_sf - sf_mean)
```

**Target SF**: 7.5 (midpoint of [0, 15] range)

This encourages the model to learn: use low SF only where it matters for quality, use high SF elsewhere to save bits.

### 4.5 Loss Weights

Based on DAC/LRAC research, adapted for our non-adversarial setting:

| Loss Component | Weight | Rationale |
|----------------|--------|-----------|
| MDCT reconstruction | 0.1 | Anchor loss, prevents drift |
| MR-STFT | 1.0 | Spectral fidelity |
| Multi-scale Mel | 15.0 | Primary perceptual loss (DAC uses 15-45) |
| Rate penalty | 0.1 | Encourages compression |

**Key insight**: Without discriminators, mel loss weight should be higher to compensate for missing adversarial signal. DAC uses λ_mel = 15.0 with discriminators; we use the same without.

## 5. Training Configuration

### 5.1 Optimizer

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)
```

### 5.2 Learning Rate Schedule

Cosine annealing with warm restarts:

```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10000,   # Restart every 10k steps
    T_mult=2,    # Double period after each restart
)
```

### 5.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Essential for stability given the multi-loss formulation.

### 5.4 Recommended Settings

```bash
make train DEVICE=cuda BATCH_SIZE=64 CACHE=1 TRAIN_STEPS=100000
```

- **Batch size 64**: Good gradient estimation, fits in ~8GB VRAM
- **CACHE=1**: Cache dataset in memory for faster training
- **100k steps**: Sufficient for convergence on moderate datasets

## 6. Verification

Our `make test-losses` command verifies the loss formulation before training:

### 6.1 Quality Sensitivity Test

| Scalefactor | SNR (dB) | MR-STFT | Mel Loss |
|-------------|----------|---------|----------|
| 0 (best quality) | 50.7 | 0.27 | 0.14 |
| 7.5 (target) | 39.5 | 0.43 | 0.54 |
| 15 (most compressed) | 28.3 | 0.82 | 2.41 |

**Expected behavior**: All metrics should degrade monotonically with increasing SF.

### 6.2 Gradient Flow Test

```
✓ Gradients flow to scalefactors (grad norm: 0.0097)
✓ Mel loss dominates (weighted: 6.44 vs STFT: 0.42)
✓ Gradients are substantial (norm: 2.86)
```

### 6.3 Training Step Simulation

Verifies complete forward-backward pass with all loss components.

## 7. Comparison with Neural Audio Codecs

| Aspect | DAC/EnCodec | Our Approach |
|--------|-------------|--------------|
| **Task** | Full encoder-decoder | Scalefactor prediction |
| **Transform** | Learned | Fixed MDCT |
| **Quantization** | VQ codebook | Power-law with SF |
| **Discriminators** | Yes (critical) | No |
| **Frame length** | Variable | Fixed 1152 samples |
| **Mel windows** | [32...2048] | [32...512] |
| **Mel weight** | 15.0 | 15.0 |
| **Target bitrate** | 8-24 kbps | ~192 kbps (MP3) |

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **No discriminator**: May result in some over-smoothing artifacts
2. **Single frame**: No temporal context across frames
3. **Fixed target SF**: Could learn adaptive rate control
4. **No stereo modeling**: Mono processing only

### 8.2 Future Directions

1. **Add lightweight discriminator**: Multi-scale STFT discriminator could improve high-frequency detail
2. **Temporal modeling**: RNN/Transformer for cross-frame dependencies
3. **Adaptive rate**: Learn bitrate allocation per-frame
4. **Joint stereo**: Model mid-side stereo for better compression

## 9. Conclusion

We present a training methodology for neural MP3 scalefactor prediction that combines insights from state-of-the-art neural audio codecs with the constraints of MP3's fixed structure. Our multi-scale perceptual loss formulation, adapted for short frames and operating without discriminators, enables end-to-end training of a learned psychoacoustic model. The careful documentation of design choices, grounded in extensive literature review, provides a foundation for further improvements.

## References

1. Kumar, R., et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS 2023.
2. Défossez, A., et al. "High Fidelity Neural Audio Compression." arXiv:2210.13438, 2022.
3. Zeghidour, N., et al. "SoundStream: An End-to-End Neural Audio Codec." IEEE TASLP, 2021.
4. Lee, S., et al. "BigVGAN: A Universal Neural Vocoder with Large-Scale Training." arXiv:2206.04658, 2022.
5. Kong, J., et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." NeurIPS 2020.
6. Steinmetz, C. "auraloss: Audio-focused loss functions in PyTorch." GitHub, 2020.
7. "Baseline Systems for the 2025 Low-Resource Audio Codec Challenge." arXiv:2510.00264, 2025.
8. "MelCap: A Unified Single-Codebook Neural Codec." arXiv:2510.01903, 2025.

---

*Document generated from notlame-train research, January 2026*
