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

# Scalefactor band boundaries (long blocks) for different sample rates
# From ISO/IEC 11172-3 (MPEG-1 Layer III)
# 22 bands covering all 576 MDCT coefficients (indices 0-575)
SCALEFACTOR_BANDS_BY_SR = {
    # MPEG-1 sample rates
    44100: [0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576],
    48000: [0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576],
    32000: [0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576],
}
SUPPORTED_SAMPLE_RATES = list(SCALEFACTOR_BANDS_BY_SR.keys())
DEFAULT_SAMPLE_RATE = 44100

# Default bands (44.1kHz for backwards compatibility)
SCALEFACTOR_BANDS_LONG = SCALEFACTOR_BANDS_BY_SR[DEFAULT_SAMPLE_RATE]
NUM_BANDS = len(SCALEFACTOR_BANDS_LONG) - 1  # 22 bands


def get_scalefactor_bands(sample_rate: int) -> list:
    """Get scalefactor band boundaries for a given sample rate."""
    if sample_rate in SCALEFACTOR_BANDS_BY_SR:
        return SCALEFACTOR_BANDS_BY_SR[sample_rate]
    # Fall back to nearest supported sample rate
    nearest = min(SUPPORTED_SAMPLE_RATES, key=lambda sr: abs(sr - sample_rate))
    return SCALEFACTOR_BANDS_BY_SR[nearest]


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
                - scalefactors: (batch, NUM_BANDS) values in [0, 15]
        """
        # Extract band-wise features
        band_features = self.extract_band_features(x)  # (B, NUM_BANDS, H)

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
        scalefactors = self.scalefactor_head(band_features).squeeze(-1)  # (B, NUM_BANDS)

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
            (batch, NUM_BANDS) scalefactors
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
        features = self.band_agg(x)  # (B, H * NUM_BANDS)
        features = features.view(batch_size, self.num_bands, -1)  # (B, NUM_BANDS, H)

        # Process each band
        features = self.band_net(features)

        # Output scalefactors
        scalefactors = self.output(features).squeeze(-1)  # (B, NUM_BANDS)
        scalefactors = torch.sigmoid(scalefactors) * 15

        return {
            "scalefactors": scalefactors,
        }


class PsychoNetStereo(nn.Module):
    """Stereo wrapper for PsychoNet.

    Processes Mid and Side channels through shared weights.
    Optionally uses separate side channel processing for better compression
    (side channel often needs fewer bits when channels are correlated).

    Input: (batch, 2, 576) - Mid and Side MDCT coefficients
    Output: (batch, 2, 22) scalefactors for M and S channels
    """

    def __init__(self, base_model: nn.Module, shared_weights: bool = True):
        super().__init__()

        self.shared_weights = shared_weights
        self.mid_model = base_model

        if not shared_weights:
            # Separate model for side channel (can learn different allocation)
            # Side channel often has less energy and can use coarser quantization
            import copy
            self.side_model = copy.deepcopy(base_model)
        else:
            self.side_model = base_model

    def forward(self, x: torch.Tensor) -> dict:
        """Process stereo MDCT coefficients.

        Args:
            x: (batch, 2, 576) Mid and Side MDCT coefficients
               or (batch, 576) mono coefficients

        Returns:
            dict with 'scalefactors': (batch, 2, 22) or (batch, 22) for mono
        """
        # Handle mono input
        if x.dim() == 2:
            return self.mid_model(x)

        # Split Mid and Side
        mid = x[:, 0, :]  # (batch, 576)
        side = x[:, 1, :]  # (batch, 576)

        # Process each channel
        mid_out = self.mid_model(mid)
        side_out = self.side_model(side)

        # Stack results
        mid_sf = mid_out["scalefactors"]  # (batch, 22)
        side_sf = side_out["scalefactors"]  # (batch, 22)

        scalefactors = torch.stack([mid_sf, side_sf], dim=1)  # (batch, 2, 22)

        return {
            "scalefactors": scalefactors,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(variant: str = "default", stereo: bool = False, **kwargs) -> nn.Module:
    """Create a PsychoNet model.

    Args:
        variant: "default", "lite", "large", or "stereo"
        stereo: Wrap model for stereo processing
        **kwargs: Additional arguments for the model

    Returns:
        PsychoNet model (or PsychoNetStereo if stereo=True)
    """
    if variant == "lite":
        base = PsychoNetLite(**kwargs)
    elif variant == "large":
        base = PsychoNet(hidden_dim=128, num_layers=4, **kwargs)
    else:
        base = PsychoNet(**kwargs)

    if stereo:
        return PsychoNetStereo(base, shared_weights=True)
    return base
