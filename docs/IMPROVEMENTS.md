# Planned Improvements

Implementation plan for notlame-train enhancements.

---

## 1. Temporal Modeling (Priority: HIGH)

### Problem
Model sees one frame at a time. No context from previous frames. Audio has temporal dependencies (transients, sustains, phrases).

### Solution
Add LSTM or Transformer layers for cross-frame dependencies.

### Implementation

```python
# Current model (per-frame):
Input: (batch, 576)  →  PsychoNet  →  (batch, 21) scalefactors

# New model (temporal):
Input: (batch, seq_len, 576)  →  PsychoNet + LSTM  →  (batch, seq_len, 21)
```

**Changes to model.py:**

```python
class PsychoNetTemporal(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, temporal_dim=128):
        super().__init__()

        # Existing band processing
        self.band_encoder = PsychoNet(hidden_dim, num_layers)

        # NEW: Temporal context
        self.temporal = nn.LSTM(
            input_size=21,           # scalefactors per frame
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,     # Causal for streaming
        )

        # Output projection
        self.output = nn.Linear(temporal_dim, 21)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, 576) MDCT coefficients
            hidden: Optional LSTM hidden state for streaming

        Returns:
            scalefactors: (batch, seq_len, 21)
            hidden: Updated hidden state
        """
        batch, seq_len, _ = x.shape

        # Process each frame through band encoder
        frame_features = []
        for t in range(seq_len):
            out = self.band_encoder(x[:, t, :])
            frame_features.append(out['scalefactors'])

        # Stack: (batch, seq_len, 21)
        frame_features = torch.stack(frame_features, dim=1)

        # Temporal processing
        temporal_out, hidden = self.temporal(frame_features, hidden)

        # Output scalefactors
        scalefactors = torch.sigmoid(self.output(temporal_out)) * 15

        return {'scalefactors': scalefactors, 'hidden': hidden}
```

**Changes to training:**
- Already loading 4 consecutive frames (good!)
- Increase to 8-16 frames for better temporal context
- Process all frames through temporal model
- Loss computed on full sequence

**Benefits:**
- Smoother scalefactor transitions
- Better transient handling
- Context-aware bit allocation

---

## 2. Adaptive Rate Control (Priority: MEDIUM)

### Problem
Fixed target scalefactor (7.5) for all content. Complex passages need more bits, simple passages need fewer.

### Solution
Model predicts per-frame rate target based on content complexity.

### Implementation

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

**Training changes:**

```python
# Adaptive rate penalty
target_sf = output['target_sf']  # Model's prediction
actual_sf = scalefactors.mean(dim=-1)  # Actual mean SF

# Penalize deviation from predicted target
rate_loss = F.mse_loss(actual_sf, target_sf.detach())

# Also penalize extreme targets (regularization)
target_reg = torch.relu(3 - target_sf) + torch.relu(target_sf - 12)
```

**Benefits:**
- More bits for complex audio
- Fewer bits for simple audio
- Better overall quality/compression tradeoff

---

## 3. Joint Stereo Support (Priority: MEDIUM)

### Problem
Mono processing only. Stereo audio is common.

### Solution
Support mid-side stereo encoding.

### Implementation

**Mid-Side Transform:**
```python
def to_mid_side(left, right):
    mid = (left + right) / 2
    side = (left - right) / 2
    return mid, side

def from_mid_side(mid, side):
    left = mid + side
    right = mid - side
    return left, right
```

**Stereo Model:**
```python
class PsychoNetStereo(nn.Module):
    def __init__(self, ...):
        # Shared encoder for mid and side
        self.encoder = PsychoNet(...)

        # Separate heads for mid/side
        self.mid_head = nn.Linear(hidden_dim, 21)
        self.side_head = nn.Linear(hidden_dim, 21)

        # Stereo correlation predictor
        self.correlation = nn.Linear(hidden_dim * 2, 1)

    def forward(self, mid_coeffs, side_coeffs):
        # Encode both channels
        mid_features = self.encoder.extract_features(mid_coeffs)
        side_features = self.encoder.extract_features(side_coeffs)

        # Predict scalefactors
        mid_sf = torch.sigmoid(self.mid_head(mid_features)) * 15
        side_sf = torch.sigmoid(self.side_head(side_features)) * 15

        # Side channel often needs fewer bits when correlated
        correlation = torch.sigmoid(self.correlation(
            torch.cat([mid_features, side_features], dim=-1)
        ))

        # Reduce side channel bits when highly correlated
        side_sf = side_sf + correlation * 5  # Increase SF = fewer bits

        return {
            'mid_sf': mid_sf,
            'side_sf': side_sf,
            'correlation': correlation,
        }
```

**Data pipeline changes:**
- Load stereo audio
- Convert to mid-side
- Compute MDCT for both channels
- Train on paired mid/side

**Benefits:**
- Stereo support
- Better compression (correlated channels)
- Compatible with MP3 joint stereo mode

---

## 4. Discriminator (Priority: LOW)

### Problem
Reconstruction losses can cause over-smoothing. Missing fine high-frequency detail.

### Solution
Add multi-scale STFT discriminator (GAN training).

### Implementation

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


class STFTDiscriminator(nn.Module):
    """Single-scale STFT discriminator."""

    def __init__(self, fft_size, hop_size=None, channels=32):
        super().__init__()
        hop_size = hop_size or fft_size // 4

        self.stft = lambda x: torch.stft(
            x, fft_size, hop_size, return_complex=True
        )

        # Conv layers on STFT magnitude
        n_bins = fft_size // 2 + 1
        self.convs = nn.Sequential(
            nn.Conv2d(1, channels, (3, 9), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels * 2, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels * 4, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 4, 1, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        # Compute STFT magnitude
        stft = self.stft(x)
        mag = torch.abs(stft).unsqueeze(1)  # (batch, 1, freq, time)

        # Get intermediate features for feature matching loss
        features = []
        h = mag
        for layer in self.convs:
            h = layer(h)
            features.append(h)

        return h, features
```

**Training changes:**

```python
# Generator (our model) update
fake_audio = reconstruct(model_output)
real_audio = original_audio

disc_fake_outputs, disc_fake_features = discriminator(fake_audio)
disc_real_outputs, disc_real_features = discriminator(real_audio)

# Adversarial loss (generator wants discriminator to think fake is real)
adv_loss = sum(F.relu(1 - out).mean() for out in disc_fake_outputs)

# Feature matching loss (match intermediate features)
feat_loss = 0
for fake_feat, real_feat in zip(disc_fake_features, disc_real_features):
    feat_loss += F.l1_loss(fake_feat, real_feat.detach())

# Total generator loss
g_loss = reconstruction_loss + 0.1 * adv_loss + 2.0 * feat_loss

# Discriminator update (separate optimizer)
d_real = sum(F.relu(1 - out).mean() for out in disc_real_outputs)
d_fake = sum(F.relu(1 + out).mean() for out in disc_fake_outputs)
d_loss = d_real + d_fake
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

## Implementation Order

1. **Temporal modeling** - Do first, biggest quality improvement
2. **Adaptive rate** - Easy add-on, can do with temporal
3. **Stereo** - After mono model is solid
4. **Discriminator** - Last, most complex, needs stable base

---

## Timeline

| Phase | Feature | Est. Effort |
|-------|---------|-------------|
| 1 | Temporal modeling | 2-3 days |
| 2 | Adaptive rate | 1 day |
| 3 | Stereo support | 2-3 days |
| 4 | Discriminator | 3-5 days |

Start with Phase 1 after current training completes and evaluation shows improvement.
