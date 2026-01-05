"""PsychoNet: Neural psychoacoustic model for MP3 encoding.

Replaces LAME's psychoacoustic model with a learned approach.
Input: 576 MDCT coefficients (one granule)
Output: 22 scalefactor band allocations + masking thresholds
"""

import torch
import torch.nn as nn


# =============================================================================
# MP3 Constants (single source of truth)
# =============================================================================
FRAME_SIZE = 1152         # Samples per MP3 frame (2 granules)
MDCT_SIZE = 576           # MDCT coefficients per frame (FRAME_SIZE // 2)
HOP_SIZE = 576            # 50% overlap

# Scalefactor band boundaries (long blocks, 44.1kHz)
# 22 bands covering all 576 MDCT coefficients (indices 0-575)
SCALEFACTOR_BANDS_LONG = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44,
    52, 62, 74, 90, 110, 134, 162, 196, 238, 288,
    342, 418, 576
]
NUM_BANDS = len(SCALEFACTOR_BANDS_LONG) - 1  # 22 bands


class FrequencyAttention(nn.Module):
    """Attention mechanism across frequency bands."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class BandProcessor(nn.Module):
    """Process each scalefactor band."""

    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PsychoNet(nn.Module):
    """Neural psychoacoustic model.

    Architecture:
    1. Band-wise feature extraction from MDCT coefficients
    2. Cross-band attention for masking effects
    3. Output: scalefactor allocations + masking thresholds

    Input: (batch, 576) - MDCT coefficients
    Output: (batch, 44) - 22 scalefactors + 22 thresholds
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_bands = NUM_BANDS
        self.hidden_dim = hidden_dim

        # Band boundaries as buffer (not trainable)
        bands = torch.tensor(SCALEFACTOR_BANDS_LONG, dtype=torch.long)
        self.register_buffer("band_boundaries", bands)

        # Per-band feature extractors
        # Each band has different size, so we use separate projections
        self.band_projections = nn.ModuleList()
        for i in range(NUM_BANDS):
            band_size = SCALEFACTOR_BANDS_LONG[i + 1] - SCALEFACTOR_BANDS_LONG[i]
            self.band_projections.append(
                nn.Sequential(
                    nn.Linear(band_size, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            )

        # Global context (energy, spectral features)
        self.global_encoder = nn.Sequential(
            nn.Linear(576, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Cross-band transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attn": FrequencyAttention(hidden_dim, num_heads),
                    "norm1": nn.LayerNorm(hidden_dim),
                    "ffn": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 4, hidden_dim),
                        nn.Dropout(dropout),
                    ),
                    "norm2": nn.LayerNorm(hidden_dim),
                })
            )

        # Output head - scalefactors only (thresholds removed - didn't work)
        self.scalefactor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def extract_band_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from each scalefactor band.

        Args:
            x: (batch, 576) MDCT coefficients

        Returns:
            (batch, NUM_BANDS, hidden_dim) band features
        """
        band_features = []

        for i in range(self.num_bands):
            start = SCALEFACTOR_BANDS_LONG[i]
            end = SCALEFACTOR_BANDS_LONG[i + 1]
            band = x[:, start:end]

            # Project to hidden dim
            features = self.band_projections[i](band)
            band_features.append(features)

        return torch.stack(band_features, dim=1)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: (batch, 576) MDCT coefficients

        Returns:
            dict with:
                - scalefactors: (batch, 21) values in [0, 15]
        """
        # Extract band-wise features
        band_features = self.extract_band_features(x)  # (B, 21, H)

        # Add global context
        global_ctx = self.global_encoder(x)  # (B, H)
        band_features = band_features + global_ctx.unsqueeze(1)

        # Cross-band attention layers
        for layer in self.layers:
            # Self-attention
            attn_out = layer["attn"](layer["norm1"](band_features))
            band_features = band_features + attn_out

            # FFN
            ffn_out = layer["ffn"](layer["norm2"](band_features))
            band_features = band_features + ffn_out

        # Output scalefactors
        scalefactors = self.scalefactor_head(band_features).squeeze(-1)  # (B, 21)

        # Constrain scalefactors to valid MP3 range [0, 15]
        # MP3 quantization: step = 2^(sf/4)
        # LOW scalefactor = small step = fine quantization = better quality (more bits)
        # HIGH scalefactor = large step = coarse quantization = lower quality (fewer bits)
        scalefactors = torch.sigmoid(scalefactors) * 15

        return {
            "scalefactors": scalefactors,
        }

    def get_quantization_params(self, x: torch.Tensor) -> torch.Tensor:
        """Get scalefactors for quantization.

        Args:
            x: (batch, 576) MDCT coefficients

        Returns:
            (batch, 21) scalefactors
        """
        out = self.forward(x)
        return out["scalefactors"]


class PsychoNetLite(nn.Module):
    """Lightweight version for faster inference.

    ~50k parameters vs ~200k for full PsychoNet.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.num_bands = NUM_BANDS

        # Simple band aggregation
        self.band_agg = nn.Sequential(
            nn.Linear(576, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * NUM_BANDS),
        )

        # Per-band processing
        self.band_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Output - scalefactors only
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]

        # Aggregate to band features
        features = self.band_agg(x)  # (B, H * 21)
        features = features.view(batch_size, self.num_bands, -1)  # (B, 21, H)

        # Process each band
        features = self.band_net(features)

        # Output scalefactors
        scalefactors = self.output(features).squeeze(-1)  # (B, 21)
        scalefactors = torch.sigmoid(scalefactors) * 15

        return {
            "scalefactors": scalefactors,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(variant: str = "default", **kwargs) -> nn.Module:
    """Create a PsychoNet model.

    Args:
        variant: "default", "lite", or "large"
        **kwargs: Additional arguments for the model

    Returns:
        PsychoNet model
    """
    if variant == "lite":
        return PsychoNetLite(**kwargs)
    elif variant == "large":
        return PsychoNet(hidden_dim=128, num_layers=4, **kwargs)
    else:
        return PsychoNet(**kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing PsychoNet models...")

    batch_size = 4
    x = torch.randn(batch_size, 576)

    for variant in ["default", "lite", "large"]:
        model = create_model(variant)
        out = model(x)

        print(f"\n{variant}:")
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  Scalefactors shape: {out['scalefactors'].shape}")
        print(f"  Scalefactors range: [{out['scalefactors'].min():.2f}, {out['scalefactors'].max():.2f}]")
