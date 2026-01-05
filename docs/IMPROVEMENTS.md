# Planned Improvements

Implementation plan for notlame-train enhancements.

---

## Current Status

| Feature | Status |
|---------|--------|
| Beat LAME (Mel loss) | 9/9 bitrates |
| Beat LAME (STFT loss) | 8/9 bitrates |
| Perceptual weights | Tuned [0.65, 1.35] |
| Stereo M/S | Implemented |
| Test suite | 141 tests passing |

---

## High Priority

### 1. Temporal Modeling

**Problem**: Model sees one frame at a time. No context from previous frames. Audio has temporal dependencies (transients, sustains, phrases).

**Solution**: Add LSTM or Transformer layers for cross-frame dependencies.

```
Current:  frame₁ → SF₁,  frame₂ → SF₂  (independent)
Improved: [frame₁, frame₂, ...] → LSTM → [SF₁, SF₂, ...]  (context-aware)
```

**Implementation:**

```python
class PsychoNetTemporal(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, temporal_dim=128):
        super().__init__()

        # Existing band processing
        self.band_encoder = PsychoNet(hidden_dim, num_layers)

        # NEW: Temporal context
        self.temporal = nn.LSTM(
            input_size=22,           # scalefactors per frame
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,     # Causal for streaming
        )

        # Output projection
        self.output = nn.Linear(temporal_dim, 22)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, 576) MDCT coefficients
            hidden: Optional LSTM hidden state for streaming

        Returns:
            scalefactors: (batch, seq_len, 22)
            hidden: Updated hidden state
        """
        batch, seq_len, _ = x.shape

        # Process each frame through band encoder
        frame_features = []
        for t in range(seq_len):
            out = self.band_encoder(x[:, t, :])
            frame_features.append(out['scalefactors'])

        # Stack: (batch, seq_len, 22)
        frame_features = torch.stack(frame_features, dim=1)

        # Temporal processing
        temporal_out, hidden = self.temporal(frame_features, hidden)

        # Output scalefactors
        scalefactors = torch.sigmoid(self.output(temporal_out)) * 15

        return {'scalefactors': scalefactors, 'hidden': hidden}
```

**Training changes:**
- Already loading 4 consecutive frames (good!)
- Increase to 8-16 frames for better temporal context
- Process all frames through temporal model
- Loss computed on full sequence

**Benefits:**
- Smoother scalefactor transitions
- Better transient handling
- Context-aware bit allocation

---

### 2. Train on Larger Dataset

**Problem**: Currently using GTZAN (1005 files, ~8 hours). Limited genre diversity.

**Solution**: Train on FMA-large (93GB, ~900 hours of music).

```bash
make download-fma-large
make prepare
make train TRAIN_STEPS=500000
```

| Dataset | Size | Hours | Genres | Quality |
|---------|------|-------|--------|---------|
| GTZAN | 1.2GB | 8 | 10 | Testing |
| FMA-small | 7.2GB | 66 | 8 | Good |
| FMA-large | 93GB | 900 | 161 | Best |

**Benefits:**
- Better generalization across genres
- More robust to different audio characteristics
- Production-quality model

---

## Medium Priority

### 3. Adaptive Rate Control

**Problem**: Fixed target scalefactor (7.5) for all content. Complex passages need more bits, simple passages need fewer.

**Solution**: Model predicts per-frame rate target based on content complexity.

```python
class PsychoNetAdaptive(nn.Module):
    def __init__(self, ...):
        # ... existing layers ...

        # NEW: Rate predictor
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output 0-1
        )

    def forward(self, x):
        # ... existing processing ...

        # Predict target rate (0-1 maps to SF range 3-12)
        rate_target = self.rate_head(global_features)
        target_sf = 3 + rate_target * 9  # Range [3, 12]

        return {
            'scalefactors': scalefactors,
            'target_sf': target_sf,
        }
```

**Benefits:**
- More bits for complex audio (orchestral, transients)
- Fewer bits for simple audio (speech, silence)
- Better overall quality/compression tradeoff

---

### 4. Real Audio Evaluation

**Problem**: Current tests use synthetic signals. Need listening tests on real music.

**Solution**: Encode real audio files and compare vs LAME.

```bash
# After training
make evaluate

# Manual listening test
make encode-samples
# Listen to outputs in samples/encoded/
```

**Metrics to track:**
- PESQ (perceptual quality)
- POLQA (newer perceptual metric)
- ABX listening tests (blind comparison)
- Spectrograms (visual inspection)

---

### 5. Short Block Support

**Problem**: Currently long blocks only (1152 samples, ~26ms). Transients (drums, attacks) benefit from short blocks (384 samples, ~9ms).

**Solution**: Add block switching logic.

```python
def detect_transient(frame, prev_frame):
    """Detect if frame contains transient."""
    energy_ratio = frame.abs().max() / (prev_frame.abs().max() + 1e-10)
    return energy_ratio > 3.0  # Threshold

def process_frame(frame, prev_frame, model):
    if detect_transient(frame, prev_frame):
        # Use 3 short blocks instead of 1 long block
        return process_short_blocks(frame, model)
    else:
        return process_long_block(frame, model)
```

**Benefits:**
- Better transient preservation (drums, percussion)
- Reduced pre-echo artifacts
- Matches LAME's block switching behavior

---

## Low Priority

### 6. Discriminator (GAN Training)

**Problem**: Reconstruction losses can cause over-smoothing. Missing fine high-frequency detail.

**Solution**: Add multi-scale STFT discriminator.

```python
class MultiScaleSTFTDiscriminator(nn.Module):
    """Discriminator operating on STFT magnitudes at multiple scales."""

    def __init__(self, fft_sizes=[256, 512, 1024, 2048]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(fft_size) for fft_size in fft_sizes
        ])

    def forward(self, x):
        outputs = []
        features = []
        for disc in self.discriminators:
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features
```

**Training:**
```python
# Adversarial loss
adv_loss = sum(F.relu(1 - out).mean() for out in disc_fake_outputs)

# Feature matching loss
feat_loss = sum(F.l1_loss(fake, real.detach())
               for fake, real in zip(fake_features, real_features))

# Combined
g_loss = reconstruction_loss + 0.1 * adv_loss + 2.0 * feat_loss
```

**Benefits:**
- Sharper high frequencies
- More realistic audio texture
- Reduced over-smoothing

**Risks:**
- Training instability
- Mode collapse
- Requires careful hyperparameter tuning

---

### 7. Stereo Correlation Optimization

**Status**: Stereo M/S already implemented in `model.py` (PsychoNetStereo).

**Potential improvement**: Add correlation-aware bit allocation.

```python
# Current: shared weights for M and S
# Improved: reduce side channel bits when highly correlated

correlation = compute_correlation(mid, side)
side_sf_boost = correlation * 5  # More compression when correlated
side_sf = base_side_sf + side_sf_boost
```

**Benefits:**
- Better stereo compression
- More bits for uncorrelated content (wide stereo)
- Fewer bits for correlated content (centered vocals)

---

## Implementation Order

1. **Larger dataset** - Easy, just download and train
2. **Real audio evaluation** - Validate current model quality
3. **Temporal modeling** - Biggest potential quality improvement
4. **Adaptive rate** - Easy add-on after temporal
5. **Short blocks** - For transient-heavy music
6. **Discriminator** - Last, most complex

---

## Quick Wins

These can be done immediately:

```bash
# 1. Train on more data
make download-fma-large && make prepare && make train TRAIN_STEPS=500000

# 2. Evaluate current model
make evaluate

# 3. Run full test suite
make test
```

---

## Research References

- **Temporal**: SoundStream, EnCodec use temporal modeling
- **Adaptive rate**: Variable bitrate (VBR) in modern codecs
- **Discriminator**: DAC, HiFi-GAN, BigVGAN
- **Short blocks**: LAME source code, ISO MP3 spec
