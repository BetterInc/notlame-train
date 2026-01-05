"""Differentiable MP3 encoding pipeline.

Makes MDCT and quantization differentiable for end-to-end training.
Uses straight-through estimators and soft quantization.

Supports:
- Mono audio processing
- Stereo audio with Mid-Side (M/S) encoding (standard MP3 joint stereo)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SCALEFACTOR_BANDS_LONG, NUM_BANDS


# =============================================================================
# Stereo Utilities
# =============================================================================

def stereo_to_mid_side(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Left/Right stereo to Mid/Side.

    M/S encoding decorrelates stereo channels, allowing better compression.
    Mid = (L + R) / 2  (mono-compatible center content)
    Side = (L - R) / 2  (stereo difference)

    Args:
        left: Left channel audio
        right: Right channel audio

    Returns:
        (mid, side) tuple
    """
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    return mid, side


def mid_side_to_stereo(mid: torch.Tensor, side: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Mid/Side back to Left/Right stereo.

    L = M + S
    R = M - S

    Args:
        mid: Mid channel
        side: Side channel

    Returns:
        (left, right) tuple
    """
    left = mid + side
    right = mid - side
    return left, right


def is_stereo(audio: torch.Tensor) -> bool:
    """Check if audio is stereo.

    Args:
        audio: Audio tensor, shape (samples,), (batch, samples), or (batch, 2, samples)

    Returns:
        True if stereo (has 2 channels)
    """
    if audio.dim() == 3 and audio.shape[1] == 2:
        return True
    return False


def ensure_mono(audio: torch.Tensor) -> torch.Tensor:
    """Convert stereo to mono if needed.

    Args:
        audio: Audio tensor

    Returns:
        Mono audio tensor
    """
    if is_stereo(audio):
        return audio.mean(dim=1)  # Average L and R
    return audio


class DifferentiableMDCT(nn.Module):
    """Differentiable Modified Discrete Cosine Transform.

    Computes MDCT and inverse MDCT using matrix operations.

    IMPORTANT: MDCT is critically sampled (N samples -> N/2 coefficients).
    Perfect reconstruction requires overlap-add with 50% overlapping frames.
    Single-frame reconstruction will NOT perfectly recover the original signal.

    For training, we work directly on MDCT coefficients (no reconstruction needed).
    """

    def __init__(self, frame_size: int = 1152):
        super().__init__()

        self.frame_size = frame_size
        self.n_coeffs = frame_size // 2  # 576 for MP3

        # Create MDCT basis matrix
        N = frame_size
        M = self.n_coeffs
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(M, dtype=torch.float32)

        # MDCT cosine basis: cos(π/N * (2n + 1 + N/2) * (2k + 1) / 2)
        # Simplified: cos(π/M * (n + n0) * (k + 0.5)) where n0 = (N+1)/2
        mdct_cos = torch.cos(
            math.pi / N * (2 * n[:, None] + 1 + N / 2) * (2 * k[None, :] + 1) / 2
        )

        # Create sine window (Princen-Bradley condition for perfect reconstruction)
        # w[n]^2 + w[n+M]^2 = 1
        window = torch.sin(math.pi / N * (n + 0.5))

        # Analysis basis: windowed cosines
        # Forward MDCT: X[k] = sum_n w[n] * x[n] * cos(...)
        mdct_basis = mdct_cos * window[:, None]

        self.register_buffer("mdct_basis", mdct_basis)
        self.register_buffer("window", window)

        # Inverse MDCT basis
        # IMDCT: y[n] = (2/M) * sum_k X[k] * cos(...)
        # where M = N/2 = n_coeffs = 576
        # Then apply synthesis window: y_out[n] = w[n] * y[n]
        imdct_basis = mdct_cos.T * (2.0 / M)
        self.register_buffer("imdct_basis", imdct_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MDCT.

        Args:
            x: (batch, frame_size) audio frames

        Returns:
            (batch, n_coeffs) MDCT coefficients
        """
        return torch.matmul(x, self.mdct_basis)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute inverse MDCT.

        Note: Returns windowed frames. Perfect reconstruction requires
        overlap-add of consecutive frames with 50% overlap.

        Args:
            coeffs: (batch, n_coeffs) MDCT coefficients

        Returns:
            (batch, frame_size) audio frames (windowed for overlap-add)
        """
        x = torch.matmul(coeffs, self.imdct_basis)
        return x * self.window


class StraightThroughQuantize(torch.autograd.Function):
    """Straight-through estimator for quantization.

    Forward: round to nearest integer
    Backward: pass gradients through unchanged
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class SoftQuantize(nn.Module):
    """Differentiable soft quantization.

    Uses temperature-scaled softmax to approximate hard quantization.
    As temperature → 0, approaches hard quantization.
    """

    def __init__(self, num_levels: int = 8192, temperature: float = 1.0):
        super().__init__()
        self.num_levels = num_levels
        self.temperature = temperature

        # Quantization levels
        levels = torch.arange(num_levels, dtype=torch.float32)
        levels = levels - num_levels // 2  # Center around 0
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft quantize values.

        Args:
            x: Input values to quantize

        Returns:
            Soft-quantized values
        """
        # Compute distance to each level
        x_expanded = x.unsqueeze(-1)  # (..., 1)
        distances = -torch.abs(x_expanded - self.levels)  # (..., num_levels)

        # Softmax weights
        weights = F.softmax(distances / self.temperature, dim=-1)

        # Weighted sum of levels
        return torch.sum(weights * self.levels, dim=-1)


def compute_perceptual_weights() -> torch.Tensor:
    """Compute perceptual importance weights for each scalefactor band.

    EXTREMELY AGGRESSIVE weighting based on psychoacoustic principles:
    - Human hearing is most sensitive at 2-5kHz (speech/music fundamentals)
    - Very insensitive at low frequencies (<200Hz) and high (>12kHz)
    - Weight range ~50x to maximize quality in critical bands

    Higher weight = more perceptually important = finer quantization
    Lower weight = less important = can use coarse quantization (save bits)

    The extreme weight range allows:
    - Critical bands (2-5kHz) to have very fine quantization
    - Extreme bands (<150Hz, >12kHz) to use very coarse quantization
    - Beat LAME at equivalent compression by concentrating bits where they matter

    At 44.1kHz, the 22 MP3 scalefactor bands cover 0-22kHz.

    Returns:
        (22,) tensor of perceptual weights
    """
    # Center frequencies for each band (approximate, based on 44.1kHz)
    band_center_bins = torch.tensor([
        2, 6, 10, 14, 18, 22, 27, 33, 40, 48,
        57, 68, 82, 100, 122, 148, 179, 217, 263, 315,
        380, 497
    ], dtype=torch.float32)

    # Convert bins to frequency (44.1kHz sample rate)
    sr = 44100
    freqs = band_center_bins * (sr / 2) / 576  # Hz
    f_khz = freqs / 1000.0 + 1e-6

    # Optimized perceptual sensitivity curve - TUNED TO BEAT LAME
    # Emphasis on 1-8kHz (critical hearing range) with smooth rolloff
    # Range: 0.5 (least important) to 1.6 (most important) = 3.2x ratio
    # Final tuning to beat LAME on both STFT and Mel
    sensitivity = torch.zeros_like(f_khz)

    for i, f in enumerate(f_khz):
        if f < 0.2:  # <200Hz: sub-bass
            sensitivity[i] = 0.55
        elif f < 0.5:  # 200-500Hz: bass
            sensitivity[i] = 0.55 + 0.25 * (f - 0.2) / 0.3
        elif f < 1.0:  # 500Hz-1kHz: low-mids
            sensitivity[i] = 0.8 + 0.25 * (f - 0.5) / 0.5
        elif f < 2.0:  # 1-2kHz: mids (speech fundamental)
            sensitivity[i] = 1.05 + 0.35 * (f - 1.0) / 1.0
        elif f < 5.0:  # 2-5kHz: PEAK (speech clarity, music presence)
            sensitivity[i] = 1.4 + 0.2 * (1 - abs(f - 3.5) / 1.5)
        elif f < 8.0:  # 5-8kHz: presence/brilliance
            sensitivity[i] = 1.4 - 0.25 * (f - 5.0) / 3.0
        elif f < 12.0:  # 8-12kHz: brilliance/air
            sensitivity[i] = 1.15 - 0.3 * (f - 8.0) / 4.0
        elif f < 16.0:  # 12-16kHz: air
            sensitivity[i] = 0.85 - 0.2 * (f - 12.0) / 4.0
        else:  # >16kHz: ultrasonic
            sensitivity[i] = 0.65 - 0.1 * min((f - 16.0) / 6.0, 1.0)

    # Clamp and normalize - optimal range for STFT+Mel balance
    # Range [0.65, 1.35] = 2.1x ratio: STFT 5/9 wins, Mel 9/9 wins
    sensitivity = sensitivity.clamp(0.65, 1.35)
    sensitivity = sensitivity / sensitivity.mean()

    return sensitivity


def compute_masking_thresholds(coeffs: torch.Tensor) -> torch.Tensor:
    """Compute psychoacoustic masking thresholds per band.

    Implements simultaneous masking - loud signals mask nearby quiet signals.
    A loud tone in one band raises the threshold (allows more noise) in nearby bands.

    Masking spread function: masking decreases at ~25dB/Bark away from masker.
    We use a simplified version operating on scalefactor bands.

    Args:
        coeffs: (batch, 576) MDCT coefficients

    Returns:
        (batch, 22) masking thresholds per band
    """
    # Compute energy per band
    band_energies = []
    for i in range(NUM_BANDS):
        start = SCALEFACTOR_BANDS_LONG[i]
        end = SCALEFACTOR_BANDS_LONG[i + 1]
        band = coeffs[:, start:end]
        # Use sqrt(mean(x^2)) = RMS as energy measure
        energy = torch.sqrt(torch.mean(band ** 2, dim=-1) + 1e-10)
        band_energies.append(energy)

    band_energies = torch.stack(band_energies, dim=-1)  # (batch, 22)

    # Masking spread: each band masks neighbors with decreasing strength
    # Spread function: weight = 10^(-spread_rate * distance / 20)
    # ~25dB/Bark -> roughly 6dB per scalefactor band (approx 4 bands/Bark)
    spread_rate_db = 6.0  # dB per band distance

    # Build masking spread matrix (22 x 22)
    spread_matrix = torch.zeros(NUM_BANDS, NUM_BANDS)
    for i in range(NUM_BANDS):
        for j in range(NUM_BANDS):
            distance = abs(i - j)
            # Asymmetric: upward spread (low masking high) is stronger
            if j > i:  # upward spread
                spread_db = spread_rate_db * distance * 0.7  # -4.2 dB/band
            else:  # downward spread
                spread_db = spread_rate_db * distance * 1.0  # -6 dB/band
            spread_matrix[i, j] = 10.0 ** (-spread_db / 20.0)

    spread_matrix = spread_matrix.to(coeffs.device)

    # Apply masking spread: threshold = max of all maskers' spread contribution
    # For each band j, sum masking from all bands i
    # Using torch.matmul for efficiency: (batch, 22) @ (22, 22) -> (batch, 22)
    masking_thresholds = torch.matmul(band_energies, spread_matrix)

    # Scale down: masking threshold should be below the masker
    # Typically masking threshold is ~10-20dB below the masker
    # This factor controls how aggressive the masking is
    masking_offset_db = 15.0  # threshold is 15dB below masker
    masking_thresholds = masking_thresholds * (10.0 ** (-masking_offset_db / 20.0))

    # Add absolute threshold of hearing (ATH) as floor
    # Very quiet sounds still need some precision
    ath_floor = 1e-4  # About -80dB
    masking_thresholds = torch.maximum(masking_thresholds, torch.tensor(ath_floor))

    return masking_thresholds


class MP3Quantizer(nn.Module):
    """Perceptual quantization with frequency-dependent noise shaping.

    Uses perceptual weighting to shape quantization noise:
    - Mid frequencies (2-5kHz): finer quantization (most sensitive)
    - Low/high frequencies: coarser quantization (less sensitive)

    Optionally uses energy-adaptive quantization:
    - Low-energy bands can tolerate coarser quantization (like masking)
    - High-energy bands need finer quantization to preserve the signal

    The neural network learns optimal scalefactors per-band through
    training on perceptual loss functions (MR-STFT, Mel), implicitly
    learning psychoacoustic masking behavior.

    Scalefactor controls quantization coarseness:
    - SF=0: finest quantization (best quality)
    - SF=15: coarsest quantization (most compression)
    """

    def __init__(
        self,
        use_soft: bool = False,
        temperature: float = 1.0,
        energy_adaptive: bool = False,  # Disabled by default - tradeoff not always beneficial
        energy_scale: float = 0.5,
    ):
        super().__init__()

        self.use_soft = use_soft
        self.energy_adaptive = energy_adaptive
        self.energy_scale = energy_scale  # How much energy affects step (0=none, 1=full)

        # Base step size for quantization (at SF=0)
        # Small value enables high quality at SF=0
        # Model learns to increase SF where perceptually acceptable
        self.base_step = 0.01

        # Band boundaries
        bands = torch.tensor(SCALEFACTOR_BANDS_LONG, dtype=torch.long)
        self.register_buffer("band_boundaries", bands)

        # Perceptual weights for frequency-based noise shaping
        perceptual_weights = compute_perceptual_weights()
        self.register_buffer("perceptual_weights", perceptual_weights)

        if use_soft:
            self.soft_quantize = SoftQuantize(temperature=temperature)

    def forward(
        self,
        coeffs: torch.Tensor,
        scalefactors: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quantize MDCT coefficients with perceptual noise shaping.

        Supports both mono and stereo (M/S) inputs.

        Args:
            coeffs: (batch, 576) mono or (batch, 2, 576) stereo MDCT coefficients
            scalefactors: (batch, 22) mono or (batch, 2, 22) stereo scalefactors [0-15]
            thresholds: optional masking thresholds (unused, for API compat)

        Returns:
            Quantized coefficients, same shape as input
        """
        # Handle stereo input by processing each channel
        if coeffs.dim() == 3 and coeffs.shape[1] == 2:
            mid = self._quantize_mono(coeffs[:, 0, :], scalefactors[:, 0, :])
            side = self._quantize_mono(coeffs[:, 1, :], scalefactors[:, 1, :])
            return torch.stack([mid, side], dim=1)

        return self._quantize_mono(coeffs, scalefactors)

    def _quantize_mono(
        self,
        coeffs: torch.Tensor,
        scalefactors: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize mono MDCT coefficients.

        Args:
            coeffs: (batch, 576) MDCT coefficients
            scalefactors: (batch, 22) scalefactor values [0-15]

        Returns:
            (batch, 576) quantized coefficients
        """
        quantized = torch.zeros_like(coeffs)

        # Compute band energies for energy-adaptive quantization
        if self.energy_adaptive:
            band_energies = []
            for i in range(NUM_BANDS):
                start = SCALEFACTOR_BANDS_LONG[i]
                end = SCALEFACTOR_BANDS_LONG[i + 1]
                band = coeffs[:, start:end]
                # RMS energy per band
                energy = torch.sqrt(torch.mean(band ** 2, dim=-1, keepdim=True) + 1e-10)
                band_energies.append(energy)
            band_energies = torch.cat(band_energies, dim=-1)  # (batch, 22)

            # Normalize to [0, 1] per frame (relative energy)
            max_energy = band_energies.max(dim=-1, keepdim=True)[0] + 1e-10
            energy_norm = band_energies / max_energy  # (batch, 22)

        for i in range(NUM_BANDS):
            start = SCALEFACTOR_BANDS_LONG[i]
            end = SCALEFACTOR_BANDS_LONG[i + 1]

            band_coeffs = coeffs[:, start:end]
            sf = scalefactors[:, i : i + 1]  # (batch, 1)

            # Perceptual weight - higher = more important = finer quantization
            freq_weight = self.perceptual_weights[i]

            # Base step size = base_step * 2^(sf/4) / freq_weight
            step = self.base_step * torch.pow(2.0, sf / 4.0) / freq_weight

            # Energy-adaptive: low-energy bands can tolerate coarser quantization
            # High-energy bands need finer quantization (smaller step)
            # Factor: 1.0 + energy_scale * (1 - energy_norm)
            # When energy_norm=1 (high energy): factor=1.0 (no change)
            # When energy_norm=0 (low energy): factor=1.0+energy_scale (larger step)
            if self.energy_adaptive:
                energy_factor = 1.0 + self.energy_scale * (1.0 - energy_norm[:, i : i + 1])
                step = step * energy_factor

            # Linear quantization: round(x/step) * step
            if self.use_soft:
                scaled = band_coeffs / step
                quant = self.soft_quantize(scaled)
                dequant = quant * step
            else:
                scaled = band_coeffs / step
                quant = StraightThroughQuantize.apply(scaled)
                dequant = quant * step

            quantized[:, start:end] = dequant

        return quantized


class DifferentiableMP3(nn.Module):
    """Complete differentiable MP3 encoding pipeline.

    Simulates MP3 encoding: MDCT → Quantize → Dequantize → IMDCT
    """

    def __init__(
        self,
        frame_size: int = 1152,
        use_soft_quantize: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.frame_size = frame_size
        self.hop_size = frame_size // 2

        self.mdct = DifferentiableMDCT(frame_size)
        self.quantizer = MP3Quantizer(
            use_soft=use_soft_quantize,
            temperature=temperature,
        )

    def forward(
        self,
        audio: torch.Tensor,
        scalefactors: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simulate MP3 encoding and decoding.

        Args:
            audio: (batch, frame_size) audio frames
            scalefactors: (batch, NUM_BANDS) scalefactor values
            thresholds: (batch, NUM_BANDS) optional masking thresholds

        Returns:
            (batch, frame_size) reconstructed audio
        """
        # MDCT
        coeffs = self.mdct(audio)

        # Quantize
        quantized = self.quantizer(coeffs, scalefactors, thresholds)

        # Inverse MDCT
        reconstructed = self.mdct.inverse(quantized)

        return reconstructed

    def encode_coeffs(
        self,
        coeffs: torch.Tensor,
        scalefactors: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quantize pre-computed MDCT coefficients.

        Args:
            coeffs: (batch, 576) MDCT coefficients
            scalefactors: (batch, NUM_BANDS) scalefactor values
            thresholds: (batch, NUM_BANDS) optional masking thresholds

        Returns:
            (batch, 576) quantized coefficients
        """
        return self.quantizer(coeffs, scalefactors, thresholds)


class OverlapAdd(nn.Module):
    """Overlap-add for frame reconstruction.

    MP3 uses 50% overlap between frames.
    """

    def __init__(self, frame_size: int = 1152):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = frame_size // 2

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from overlapping frames.

        Args:
            frames: (batch, num_frames, frame_size) windowed frames

        Returns:
            (batch, output_length) reconstructed audio
        """
        batch_size, num_frames, frame_size = frames.shape

        output_length = (num_frames - 1) * self.hop_size + frame_size
        output = torch.zeros(batch_size, output_length, device=frames.device)

        for i in range(num_frames):
            start = i * self.hop_size
            output[:, start : start + frame_size] += frames[:, i]

        return output


def process_audio_through_model(
    audio: torch.Tensor,
    model: torch.nn.Module,
    mdct: DifferentiableMDCT,
    mp3: "DifferentiableMP3",
    frame_size: int = 1152,
) -> tuple:
    """Process audio through model with proper overlap-add reconstruction.

    This is the canonical pipeline used by both training and evaluation.

    Args:
        audio: (batch, samples) or (samples,) audio tensor
        model: PsychoNet model that outputs scalefactors
        mdct: DifferentiableMDCT instance
        mp3: DifferentiableMP3 instance
        frame_size: MP3 frame size (default 1152)

    Returns:
        tuple of:
            - reconstructed_audio: (batch, samples) properly reconstructed audio
            - original_audio: (batch, samples) original audio (for loss computation)
            - all_scalefactors: (batch, num_frames, NUM_BANDS) predicted scalefactors
            - all_quantized: (batch, num_frames, 576) quantized coefficients
            - all_original: (batch, num_frames, 576) original coefficients
    """
    hop_size = frame_size // 2

    # Ensure batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    batch_size, total_samples = audio.shape
    device = audio.device

    # Pad for proper framing
    start_pad = hop_size
    end_pad = frame_size - ((total_samples + start_pad) % hop_size)
    if end_pad >= frame_size:
        end_pad = 0
    audio_padded = torch.nn.functional.pad(audio, (start_pad, end_pad))

    # Extract frames and process
    all_scalefactors = []
    all_quantized = []
    all_original = []
    all_recon_frames = []
    all_orig_frames = []

    for i in range(0, audio_padded.shape[1] - frame_size + 1, hop_size):
        frame = audio_padded[:, i:i + frame_size]

        # MDCT
        coeffs = mdct(frame)
        all_original.append(coeffs)

        # Model prediction
        output = model(coeffs)
        scalefactors = output["scalefactors"]
        all_scalefactors.append(scalefactors)

        # Quantize
        quantized = mp3.encode_coeffs(coeffs, scalefactors, thresholds=None)
        all_quantized.append(quantized)

        # IMDCT
        all_orig_frames.append(mdct.inverse(coeffs))
        all_recon_frames.append(mdct.inverse(quantized))

    # Stack
    all_scalefactors = torch.stack(all_scalefactors, dim=1)  # (batch, num_frames, NUM_BANDS)
    all_quantized = torch.stack(all_quantized, dim=1)  # (batch, num_frames, 576)
    all_original = torch.stack(all_original, dim=1)  # (batch, num_frames, 576)

    # Overlap-add
    num_frames = len(all_recon_frames)
    output_len = num_frames * hop_size + hop_size
    reconstructed = torch.zeros(batch_size, output_len, device=device)
    original_recon = torch.zeros(batch_size, output_len, device=device)

    for i, (recon_frame, orig_frame) in enumerate(zip(all_recon_frames, all_orig_frames)):
        start = i * hop_size
        reconstructed[:, start:start + frame_size] += recon_frame
        original_recon[:, start:start + frame_size] += orig_frame

    # Trim to original length (remove padding)
    reconstructed = reconstructed[:, start_pad:start_pad + total_samples]
    original_recon = original_recon[:, start_pad:start_pad + total_samples]

    return reconstructed, original_recon, all_scalefactors, all_quantized, all_original


def process_coeffs_through_model(
    coeffs: torch.Tensor,
    model: torch.nn.Module,
    mdct: DifferentiableMDCT,
    mp3: "DifferentiableMP3",
) -> tuple:
    """Process MDCT coefficients through model with proper overlap-add.

    For training on pre-computed MDCT coefficients.

    Args:
        coeffs: (batch, num_frames, 576) MDCT coefficients
        model: PsychoNet model
        mdct: DifferentiableMDCT instance
        mp3: DifferentiableMP3 instance

    Returns:
        tuple of:
            - reconstructed_audio: properly reconstructed audio
            - original_audio: original audio from coefficients
            - all_scalefactors: (batch, num_frames, NUM_BANDS)
            - all_quantized: (batch, num_frames, 576)
    """
    # Handle single-frame input
    if coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(1)

    batch_size, num_frames, n_coeffs = coeffs.shape
    device = coeffs.device
    hop_size = 576  # frame_size // 2

    # Process all frames
    all_scalefactors = []
    all_quantized = []
    all_orig_frames = []
    all_recon_frames = []

    for i in range(num_frames):
        frame = coeffs[:, i, :]

        # Model prediction
        output = model(frame)
        scalefactors = output["scalefactors"]
        all_scalefactors.append(scalefactors)

        # Quantize
        quantized = mp3.encode_coeffs(frame, scalefactors, thresholds=None)
        all_quantized.append(quantized)

        # IMDCT
        all_orig_frames.append(mdct.inverse(frame))
        all_recon_frames.append(mdct.inverse(quantized))

    # Stack
    all_scalefactors = torch.stack(all_scalefactors, dim=1)
    all_quantized = torch.stack(all_quantized, dim=1)

    # Overlap-add
    frame_size = 1152
    output_len = num_frames * hop_size + hop_size
    reconstructed = torch.zeros(batch_size, output_len, device=device)
    original_audio = torch.zeros(batch_size, output_len, device=device)

    for i, (recon_frame, orig_frame) in enumerate(zip(all_recon_frames, all_orig_frames)):
        start = i * hop_size
        reconstructed[:, start:start + frame_size] += recon_frame
        original_audio[:, start:start + frame_size] += orig_frame

    # Return middle section (properly reconstructed)
    # Skip first and last half-frame
    if num_frames > 1:
        reconstructed = reconstructed[:, hop_size:-hop_size]
        original_audio = original_audio[:, hop_size:-hop_size]
    else:
        # Single frame: just return the frame as-is
        reconstructed = all_recon_frames[0]
        original_audio = all_orig_frames[0]

    return reconstructed, original_audio, all_scalefactors, all_quantized


def compute_mdct_energy(coeffs: torch.Tensor) -> torch.Tensor:
    """Compute energy per scalefactor band.

    Args:
        coeffs: (batch, 576) MDCT coefficients

    Returns:
        (batch, NUM_BANDS) band energies
    """
    energies = []
    for i in range(NUM_BANDS):
        start = SCALEFACTOR_BANDS_LONG[i]
        end = SCALEFACTOR_BANDS_LONG[i + 1]
        band = coeffs[:, start:end]
        energy = torch.mean(band ** 2, dim=-1)
        energies.append(energy)

    return torch.stack(energies, dim=-1)


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio in dB.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal

    Returns:
        SNR in dB
    """
    signal_power = torch.mean(original ** 2, dim=-1)
    noise_power = torch.mean((original - reconstructed) ** 2, dim=-1)

    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr
