"""Perceptual loss functions for audio quality.

Multi-resolution STFT and mel spectrogram losses for training.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NUM_BANDS, SCALEFACTOR_BANDS_LONG


class STFTLoss(nn.Module):
    """STFT-based loss function.

    Computes spectral convergence and log magnitude loss.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
    ):
        super().__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size

        # Hann window
        window = torch.hann_window(win_size)
        self.register_buffer("window", window)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude.

        Args:
            x: (batch, time) audio signal

        Returns:
            (batch, freq, frames) magnitude spectrogram
        """
        # Ensure 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Compute STFT
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

        # Magnitude
        return torch.abs(spec)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT losses.

        Args:
            x: (batch, time) predicted signal
            y: (batch, time) target signal

        Returns:
            (spectral_convergence, magnitude) losses

        Research notes (auraloss, DAC):
        - Spectral convergence: normalized Frobenius norm
        - Log magnitude: L1 on log spectrogram (captures dynamics)
        - Linear magnitude: L1 on linear spectrogram (captures shape)
        - Combined log+linear works best per auraloss docs
        """
        x_mag = self.stft(x)
        y_mag = self.stft(y)

        # Spectral convergence: ||y_mag - x_mag||_F / ||y_mag||_F
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)

        # Log magnitude loss (captures dynamics across frequency range)
        log_x = torch.log(x_mag + 1e-8)
        log_y = torch.log(y_mag + 1e-8)
        log_mag_loss = F.l1_loss(log_x, log_y)

        # Linear magnitude loss (captures spectral shape)
        lin_mag_loss = F.l1_loss(x_mag, y_mag)

        # Combine log and linear (auraloss recommendation)
        mag_loss = log_mag_loss + 0.5 * lin_mag_loss

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss.

    Combines STFT losses at multiple time-frequency resolutions.
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_sizes: List[int] = [512, 1024, 2048],
    ):
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)

        self.losses = nn.ModuleList()
        for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
            self.losses.append(STFTLoss(fft, hop, win))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-resolution STFT losses.

        Args:
            x: (batch, time) predicted signal
            y: (batch, time) target signal

        Returns:
            (spectral_convergence, log_magnitude) total losses
        """
        sc_total = 0.0
        mag_total = 0.0

        for loss_fn in self.losses:
            sc, mag = loss_fn(x, y)
            sc_total += sc
            mag_total += mag

        # Average over resolutions
        n = len(self.losses)
        return sc_total / n, mag_total / n


def compute_mel_perceptual_weights(n_mels: int, f_min: float, f_max: float) -> torch.Tensor:
    """Compute perceptual importance weights for mel bands.

    Based on equal-loudness contours and speech/music importance:
    - Peak sensitivity at 2-5kHz (speech fundamentals, music presence)
    - Lower sensitivity at extremes (<200Hz, >12kHz)

    Args:
        n_mels: Number of mel bands
        f_min: Minimum frequency (Hz)
        f_max: Maximum frequency (Hz)

    Returns:
        (n_mels,) tensor of perceptual weights, normalized to mean=1
    """
    # Compute center frequencies for each mel band
    def hz_to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_centers = torch.linspace(mel_min, mel_max, n_mels)
    hz_centers = torch.tensor([mel_to_hz(m.item()) for m in mel_centers])

    # Compute perceptual weights based on frequency
    weights = torch.zeros(n_mels)
    for i, f in enumerate(hz_centers):
        f_khz = f.item() / 1000.0

        if f_khz < 0.15:  # <150Hz: very low sensitivity
            weights[i] = 0.3
        elif f_khz < 0.3:  # 150-300Hz: low
            weights[i] = 0.3 + 0.4 * (f_khz - 0.15) / 0.15
        elif f_khz < 0.5:  # 300-500Hz: moderate-low
            weights[i] = 0.7 + 0.3 * (f_khz - 0.3) / 0.2
        elif f_khz < 1.0:  # 500Hz-1kHz: moderate
            weights[i] = 1.0 + 0.4 * (f_khz - 0.5) / 0.5
        elif f_khz < 2.0:  # 1-2kHz: high
            weights[i] = 1.4 + 0.4 * (f_khz - 1.0) / 1.0
        elif f_khz < 5.0:  # 2-5kHz: PEAK sensitivity
            weights[i] = 1.8 + 0.2 * (1 - abs(f_khz - 3.5) / 1.5)
        elif f_khz < 8.0:  # 5-8kHz: high
            weights[i] = 1.8 - 0.4 * (f_khz - 5.0) / 3.0
        elif f_khz < 12.0:  # 8-12kHz: moderate
            weights[i] = 1.4 - 0.5 * (f_khz - 8.0) / 4.0
        else:  # >12kHz: low
            weights[i] = 0.9 - 0.4 * min((f_khz - 12.0) / 8.0, 1.0)

    # Normalize to mean=1
    weights = weights.clamp(0.3, 2.0)
    weights = weights / weights.mean()

    return weights


class MelSpectrogramLoss(nn.Module):
    """Mel spectrogram loss using librosa-compatible computation.

    Matches the evaluation metrics exactly for consistent training/eval.
    Optionally applies perceptual weighting to emphasize critical frequency bands.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        use_perceptual_weight: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.use_perceptual_weight = use_perceptual_weight

        # Use librosa's mel filterbank for exact match with evaluation
        # Ensure n_mels is valid for the FFT size to avoid empty filters
        n_freqs = n_fft // 2 + 1
        if n_mels > n_freqs - 2:
            # Too many mel bands for this FFT size - reduce
            self.n_mels = max(8, n_freqs - 2)

        try:
            import librosa
            mel_basis = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=self.n_mels,
                fmin=f_min,
                fmax=self.f_max,
            )
            # Check for empty filters and warn if found
            empty_count = (mel_basis.sum(axis=1) == 0).sum()
            if empty_count > 0:
                # This shouldn't happen with proper n_mels, but handle gracefully
                mel_basis = mel_basis[mel_basis.sum(axis=1) > 0]
                self.n_mels = mel_basis.shape[0]
            self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
            self._use_librosa = True
        except ImportError:
            # Fallback to custom filterbank
            mel_basis = self._create_mel_filterbank()
            self.register_buffer("mel_basis", mel_basis)
            self._use_librosa = False

        # STFT window
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        # Perceptual weights for mel bands
        if use_perceptual_weight:
            perceptual_weights = compute_mel_perceptual_weights(
                self.n_mels, f_min, self.f_max
            )
            self.register_buffer("perceptual_weights", perceptual_weights)
        else:
            self.register_buffer("perceptual_weights", torch.ones(self.n_mels))

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix (fallback if librosa unavailable)."""
        n_freqs = self.n_fft // 2 + 1

        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])

        bin_points = torch.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).long()

        filterbank = torch.zeros(self.n_mels, n_freqs)

        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram (librosa-compatible).

        Args:
            x: (batch, time) audio signal

        Returns:
            (batch, n_mels, frames) log mel spectrogram
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # STFT
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

        # Power spectrogram (matches librosa)
        power = torch.abs(spec) ** 2

        # Apply mel filterbank: (n_mels, n_freqs) @ (batch, n_freqs, frames)
        # Need to transpose for matmul
        power_t = power.transpose(-2, -1)  # (batch, frames, n_freqs)
        mel = torch.matmul(power_t, self.mel_basis.T)  # (batch, frames, n_mels)
        mel = mel.transpose(-2, -1)  # (batch, n_mels, frames)

        # Log scale (same as evaluation)
        log_mel = torch.log(mel + 1e-8)

        return log_mel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram L1 loss with perceptual weighting.

        Args:
            x: (batch, time) predicted signal
            y: (batch, time) target signal

        Returns:
            Perceptually-weighted L1 loss between log mel spectrograms
        """
        x_mel = self.mel_spectrogram(x)
        y_mel = self.mel_spectrogram(y)

        # Compute weighted L1 loss
        # weights: (n_mels,) -> (1, n_mels, 1) for broadcasting
        weights = self.perceptual_weights.view(1, -1, 1)
        weighted_diff = weights * torch.abs(x_mel - y_mel)

        return weighted_diff.mean()


class MultiScaleMelLoss(nn.Module):
    """Multi-scale mel spectrogram loss with perceptual weighting.

    Uses multiple window sizes for multi-resolution coverage.
    Applies perceptual weighting to emphasize critical frequency bands.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        window_lengths: List[int] = [32, 64, 128, 256, 512],
        n_mels: int = 64,
        use_l2: bool = False,  # Default to False to match evaluation
        use_perceptual_weight: bool = True,
    ):
        super().__init__()
        self.use_l2 = use_l2
        self.use_perceptual_weight = use_perceptual_weight

        self.losses = nn.ModuleList()
        for win_len in window_lengths:
            hop = win_len // 4
            # Librosa mel filterbank needs smaller n_mels than (n_fft/2+1)
            # Empirically: n_mels ~= n_fft/8 works without empty filters at 44.1kHz
            max_safe_mels = max(8, win_len // 8)
            actual_mels = min(n_mels, max_safe_mels)
            self.losses.append(
                MelSpectrogramLoss(
                    sample_rate=sample_rate,
                    n_fft=win_len,
                    hop_length=hop,
                    n_mels=actual_mels,
                    use_perceptual_weight=use_perceptual_weight,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale mel loss with perceptual weighting.

        Args:
            x: (batch, time) predicted signal
            y: (batch, time) target signal

        Returns:
            Average perceptually-weighted loss across all scales
        """
        total_loss = 0.0

        for loss_fn in self.losses:
            # Use the forward method which applies perceptual weighting
            total_loss += loss_fn(x, y)
            if self.use_l2:
                x_mel = loss_fn.mel_spectrogram(x)
                y_mel = loss_fn.mel_spectrogram(y)
                weights = loss_fn.perceptual_weights.view(1, -1, 1)
                weighted_mse = weights * (x_mel - y_mel) ** 2
                total_loss += 0.5 * weighted_mse.mean()

        return total_loss / len(self.losses)


class PerceptualLoss(nn.Module):
    """Combined perceptual loss for audio.

    Combines multi-resolution STFT and mel spectrogram losses.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        stft_weight: float = 1.0,
        mel_weight: float = 1.0,
        time_weight: float = 0.1,
    ):
        super().__init__()

        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.time_weight = time_weight

        # Multi-resolution STFT
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_sizes=[512, 1024, 2048],
        )

        # Mel spectrogram
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        """Compute combined perceptual loss.

        Args:
            x: (batch, time) predicted signal
            y: (batch, time) target signal

        Returns:
            dict with individual and total losses
        """
        # Multi-resolution STFT
        sc_loss, mag_loss = self.stft_loss(x, y)
        stft_total = sc_loss + mag_loss

        # Mel spectrogram
        mel_loss = self.mel_loss(x, y)

        # Time domain
        time_loss = F.l1_loss(x, y)

        # Total
        total = (
            self.stft_weight * stft_total
            + self.mel_weight * mel_loss
            + self.time_weight * time_loss
        )

        return {
            "total": total,
            "stft_sc": sc_loss,
            "stft_mag": mag_loss,
            "mel": mel_loss,
            "time": time_loss,
        }


class MDCTLoss(nn.Module):
    """Loss directly on MDCT coefficients with perceptual weighting.

    Uses psychoacoustic-inspired weighting:
    - Low frequencies (bands 0-6) are most important perceptually
    - Mid frequencies (bands 7-14) are moderately important
    - High frequencies (bands 15-20) are least important

    Based on equal-loudness contours and critical band importance.
    """

    def __init__(self, use_perceptual_weights: bool = True):
        super().__init__()

        if use_perceptual_weights:
            # Create perceptual weights for each of 576 coefficients
            # Lower bands = higher weight (more perceptually important)
            weights = torch.zeros(576)

            # Perceptual importance by band (empirically derived)
            # Bands 0-6: highest importance (bass, fundamentals)
            # Bands 7-14: medium importance (mids, harmonics)
            # Bands 15-21: lower importance (highs, less sensitive)
            band_importance = [
                3.0, 3.0, 2.5, 2.5, 2.0, 2.0, 1.8,  # Bands 0-6: bass/low-mids
                1.5, 1.5, 1.3, 1.3, 1.2, 1.2, 1.1, 1.1,  # Bands 7-14: mids
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,  # Bands 15-21: highs (added band 21)
            ]

            for i in range(NUM_BANDS):
                start = SCALEFACTOR_BANDS_LONG[i]
                end = SCALEFACTOR_BANDS_LONG[i + 1]
                weights[start:end] = band_importance[i]

            # Normalize so mean weight = 1.0
            weights = weights / weights.mean()
            self.register_buffer("weights", weights)
        else:
            self.register_buffer("weights", torch.ones(576))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptually-weighted MDCT coefficient loss.

        Args:
            x: (batch, 576) predicted MDCT coefficients
            y: (batch, 576) target MDCT coefficients

        Returns:
            Weighted L1 loss
        """
        # Expand weights to batch size
        weights = self.weights.unsqueeze(0).expand_as(x)
        return torch.mean(weights * torch.abs(x - y))


class BitrateLoss(nn.Module):
    """Loss to encourage efficient bit allocation (compression).

    MP3 scalefactor semantics:
    - LOW scalefactor = fine quantization = MORE bits = higher quality
    - HIGH scalefactor = coarse quantization = FEWER bits = lower quality

    This loss encourages compression by:
    1. Penalizing LOW scalefactors (which waste bits)
    2. Allowing low SF only on high-energy bands (where bits matter)
    3. Encouraging high SF on low-energy bands (save bits on unimportant content)
    """

    def __init__(self, target_avg_sf: float = 8.0):
        super().__init__()
        self.target_avg_sf = target_avg_sf

    def forward(
        self,
        scalefactors: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bitrate penalty.

        Args:
            scalefactors: (batch, NUM_BANDS) predicted scalefactors [0-15]
            energy: (batch, NUM_BANDS) band energies

        Returns:
            Penalty for inefficient bit allocation (using too many bits)
        """
        # Average scalefactor
        avg_sf = torch.mean(scalefactors)

        # Penalize LOW scalefactors (below target = using too many bits)
        # Higher SF = fewer bits = more efficient compression
        rate_penalty = F.relu(self.target_avg_sf - avg_sf)

        # Penalize LOW SF on low-energy bands (wasting bits on unimportant content)
        # High-energy bands (energy_norm ~1) can have low SF (spend bits where needed)
        # Low-energy bands (energy_norm ~0) should have high SF (save bits)
        energy_norm = energy / (energy.max(dim=-1, keepdim=True)[0] + 1e-8)
        # (15 - sf) is high when sf is low; (1 - energy_norm) is high when energy is low
        waste_penalty = torch.mean((15.0 - scalefactors) * (1 - energy_norm))

        return rate_penalty + 0.1 * waste_penalty


class CombinedLoss(nn.Module):
    """Complete training loss.

    Combines perceptual quality and bitrate efficiency.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        perceptual_weight: float = 1.0,
        bitrate_weight: float = 0.01,
    ):
        super().__init__()

        self.perceptual_weight = perceptual_weight
        self.bitrate_weight = bitrate_weight

        self.perceptual_loss = PerceptualLoss(sample_rate=sample_rate)
        self.bitrate_loss = BitrateLoss()

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        scalefactors: torch.Tensor,
        band_energy: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            pred_audio: Predicted audio
            target_audio: Target audio
            scalefactors: Predicted scalefactors
            band_energy: Band energies

        Returns:
            dict with all loss components
        """
        # Perceptual loss
        perceptual = self.perceptual_loss(pred_audio, target_audio)

        # Bitrate loss
        bitrate = self.bitrate_loss(scalefactors, band_energy)

        # Total
        total = (
            self.perceptual_weight * perceptual["total"]
            + self.bitrate_weight * bitrate
        )

        return {
            "total": total,
            "perceptual": perceptual["total"],
            "bitrate": bitrate,
            **{f"perceptual_{k}": v for k, v in perceptual.items() if k != "total"},
        }


class RateDistortionLoss(nn.Module):
    """Rate-distortion tradeoff loss for MDCT-domain training.

    Balances reconstruction quality against bitrate efficiency:
    - Distortion: Perceptually-weighted MDCT reconstruction loss
    - Rate: Penalizes low scalefactors (which use more bits)

    The model learns to find the optimal tradeoff:
    - Low SF = better quality but more bits
    - High SF = worse quality but fewer bits
    """

    def __init__(
        self,
        rate_weight: float = 0.01,
        use_perceptual_weights: bool = True,
    ):
        """Initialize rate-distortion loss.

        Args:
            rate_weight: Weight for rate (bitrate) penalty. Higher = more compression.
                         0.001 = high quality, 0.1 = aggressive compression
            use_perceptual_weights: Use perceptual band weighting for distortion
        """
        super().__init__()
        self.rate_weight = rate_weight
        self.mdct_loss = MDCTLoss(use_perceptual_weights=use_perceptual_weights)

    def forward(
        self,
        quantized: torch.Tensor,
        original: torch.Tensor,
        scalefactors: torch.Tensor,
    ) -> dict:
        """Compute rate-distortion loss.

        Args:
            quantized: (batch, 576) quantized/reconstructed MDCT coefficients
            original: (batch, 576) original MDCT coefficients
            scalefactors: (batch, NUM_BANDS) predicted scalefactors [0-15]

        Returns:
            dict with loss components:
                - total: combined loss
                - distortion: reconstruction quality loss
                - rate: bitrate penalty
                - sf_mean: mean scalefactor (for monitoring)
        """
        # Distortion loss (perceptually weighted)
        distortion = self.mdct_loss(quantized, original)

        # Rate loss: penalize LOW scalefactors (which use more bits)
        # In MP3: step = 2^(sf/4), so low sf = small step = fine quantization = more bits
        # We want to encourage HIGHER scalefactors for efficiency
        # Rate proxy: average (15 - sf) - this increases when sf is low
        sf_mean = torch.mean(scalefactors)
        rate = torch.mean(15.0 - scalefactors)  # High when sf is low

        # Energy-adaptive rate: penalize more for high-energy bands using low SF
        # (high-energy bands with low SF = lots of bits spent)
        band_energies = []
        for i in range(NUM_BANDS):
            start = SCALEFACTOR_BANDS_LONG[i]
            end = SCALEFACTOR_BANDS_LONG[i + 1]
            band_energy = torch.mean(original[:, start:end] ** 2, dim=1)
            band_energies.append(band_energy)
        band_energies = torch.stack(band_energies, dim=1)  # (batch, NUM_BANDS)

        # Normalize energies
        energy_norm = band_energies / (band_energies.max(dim=1, keepdim=True)[0] + 1e-8)

        # Adaptive rate: high-energy bands with low SF get extra penalty
        adaptive_rate = torch.mean(energy_norm * (15.0 - scalefactors))

        # Total loss
        total = distortion + self.rate_weight * (rate + adaptive_rate)

        return {
            "total": total,
            "distortion": distortion,
            "rate": rate,
            "adaptive_rate": adaptive_rate,
            "sf_mean": sf_mean,
        }
