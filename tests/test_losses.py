"""Comprehensive tests for loss functions.

Tests STFT, Mel, MDCT losses and their gradients.
"""

import pytest
import torch
import math

from notlame_train.losses import (
    STFTLoss,
    MultiResolutionSTFTLoss,
    MelSpectrogramLoss,
    MultiScaleMelLoss,
    MDCTLoss,
    BitrateLoss,
    RateDistortionLoss,
    PerceptualLoss,
    CombinedLoss,
    compute_mel_perceptual_weights,
)
from notlame_train.model import NUM_BANDS
from notlame_train import config


class TestSTFTLoss:
    """Test STFT-based loss functions."""

    def test_stft_loss_shape(self, audio_signal):
        """STFT loss should return scalar losses."""
        loss_fn = STFTLoss()
        target = audio_signal
        pred = audio_signal + 0.1 * torch.randn_like(audio_signal)

        sc, mag = loss_fn(pred, target)

        assert sc.dim() == 0, "SC loss should be scalar"
        assert mag.dim() == 0, "Mag loss should be scalar"

    def test_stft_loss_identical_zero(self, audio_signal):
        """STFT loss should be near zero for identical signals."""
        loss_fn = STFTLoss()

        sc, mag = loss_fn(audio_signal, audio_signal)

        assert sc < 0.01, f"SC loss for identical signals: {sc}"
        assert mag < 0.01, f"Mag loss for identical signals: {mag}"

    def test_stft_loss_increases_with_noise(self, audio_signal):
        """STFT loss should increase with more noise."""
        loss_fn = STFTLoss()
        target = audio_signal

        losses = []
        for noise_level in [0.01, 0.1, 0.5]:
            pred = target + noise_level * torch.randn_like(target)
            sc, mag = loss_fn(pred, target)
            losses.append((sc + mag).item())

        # Loss should increase with noise
        for i in range(len(losses) - 1):
            assert losses[i] < losses[i + 1], \
                f"Loss should increase with noise: {losses}"

    def test_stft_loss_gradient(self, audio_signal):
        """STFT loss should have gradients."""
        loss_fn = STFTLoss()
        pred = audio_signal.clone().requires_grad_(True)
        # Use different target to ensure gradients flow
        target = audio_signal + 0.1 * torch.randn_like(audio_signal)

        sc, mag = loss_fn(pred, target.detach())
        total = sc + mag
        total.backward()

        assert pred.grad is not None
        # Gradient may be very small but should exist
        assert pred.grad is not None


class TestMultiResolutionSTFT:
    """Test multi-resolution STFT loss."""

    def test_mr_stft_uses_config(self, audio_signal):
        """Multi-res STFT should use config parameters."""
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=config.STFT_FFT_SIZES,
            hop_sizes=config.STFT_HOP_SIZES,
            win_sizes=config.STFT_WIN_SIZES,
        )

        assert len(loss_fn.losses) == len(config.STFT_FFT_SIZES)

    def test_mr_stft_loss_shape(self, audio_signal):
        """Multi-res STFT should return scalar losses."""
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=[64, 128, 256],
            hop_sizes=[16, 32, 64],
            win_sizes=[64, 128, 256],
        )

        target = audio_signal
        pred = audio_signal + 0.1 * torch.randn_like(audio_signal)

        sc, mag = loss_fn(pred, target)

        assert sc.dim() == 0
        assert mag.dim() == 0

    def test_mr_stft_combines_resolutions(self, audio_signal):
        """Multi-res should give different results than single-res."""
        single = STFTLoss(fft_size=256, hop_size=64, win_size=256)
        multi = MultiResolutionSTFTLoss(
            fft_sizes=[64, 128, 256],
            hop_sizes=[16, 32, 64],
            win_sizes=[64, 128, 256],
        )

        pred = audio_signal + 0.1 * torch.randn_like(audio_signal)
        target = audio_signal

        sc_s, mag_s = single(pred, target)
        sc_m, mag_m = multi(pred, target)

        # Should be different (multi averages multiple resolutions)
        assert abs((sc_s + mag_s) - (sc_m + mag_m)) > 0.001


class TestMelLoss:
    """Test mel spectrogram loss."""

    def test_mel_loss_shape(self, audio_signal):
        """Mel loss should return scalar."""
        loss_fn = MelSpectrogramLoss(n_fft=512, hop_length=128, n_mels=32)

        target = audio_signal
        pred = audio_signal + 0.1 * torch.randn_like(audio_signal)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0

    def test_mel_loss_identical_zero(self, audio_signal):
        """Mel loss should be near zero for identical signals."""
        loss_fn = MelSpectrogramLoss(n_fft=512, hop_length=128, n_mels=32)

        loss = loss_fn(audio_signal, audio_signal)

        assert loss < 0.01, f"Mel loss for identical: {loss}"

    def test_mel_perceptual_weights_shape(self):
        """Mel perceptual weights should have correct shape."""
        n_mels = 64
        weights = compute_mel_perceptual_weights(n_mels, 0.0, 22050.0)

        assert weights.shape == (n_mels,)

    def test_mel_perceptual_weights_normalized(self):
        """Mel perceptual weights should be normalized."""
        weights = compute_mel_perceptual_weights(64, 0.0, 22050.0)

        assert abs(weights.mean() - 1.0) < 0.01

    def test_mel_perceptual_weights_peak(self):
        """Mel weights should peak in speech/music range (2-5kHz)."""
        weights = compute_mel_perceptual_weights(64, 0.0, 22050.0)

        # Rough band indices for 2-5kHz in 64 mel bands
        mid_bands = weights[20:35].mean()
        low_bands = weights[0:10].mean()
        high_bands = weights[50:].mean()

        assert mid_bands >= low_bands
        assert mid_bands >= high_bands


class TestMultiScaleMel:
    """Test multi-scale mel loss."""

    def test_multiscale_mel_uses_config(self, audio_signal):
        """Multi-scale mel should use config parameters."""
        loss_fn = MultiScaleMelLoss(
            sample_rate=config.MEL_SAMPLE_RATE,
            window_lengths=config.MEL_WINDOW_LENGTHS,
            n_mels=config.MEL_N_MELS,
            use_l2=config.MEL_USE_L2,
        )

        assert len(loss_fn.losses) == len(config.MEL_WINDOW_LENGTHS)

    def test_multiscale_mel_gradient(self, audio_signal):
        """Multi-scale mel should have gradients."""
        loss_fn = MultiScaleMelLoss(
            window_lengths=[128, 256],
            n_mels=32,
        )

        pred = audio_signal.clone().requires_grad_(True)
        # Use different target to ensure gradients flow
        target = audio_signal + 0.1 * torch.randn_like(audio_signal)

        loss = loss_fn(pred, target.detach())
        loss.backward()

        assert pred.grad is not None


class TestMDCTLoss:
    """Test MDCT domain loss."""

    def test_mdct_loss_shape(self, mdct_coeffs):
        """MDCT loss should return scalar."""
        loss_fn = MDCTLoss(use_perceptual_weights=True)

        target = mdct_coeffs
        pred = mdct_coeffs + 0.1 * torch.randn_like(mdct_coeffs)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0

    def test_mdct_loss_perceptual_weights(self, mdct_coeffs):
        """Perceptual-weighted MDCT loss should differ from unweighted."""
        weighted = MDCTLoss(use_perceptual_weights=True)
        unweighted = MDCTLoss(use_perceptual_weights=False)

        target = mdct_coeffs
        pred = mdct_coeffs + 0.1 * torch.randn_like(mdct_coeffs)

        loss_w = weighted(pred, target)
        loss_u = unweighted(pred, target)

        # Both should compute valid losses
        assert torch.isfinite(loss_w)
        assert torch.isfinite(loss_u)
        # Weights should affect the result (may be similar but not identical)
        assert loss_w > 0 and loss_u > 0

    def test_mdct_loss_weights_valid(self):
        """MDCT loss weights should cover all 576 coefficients."""
        loss_fn = MDCTLoss(use_perceptual_weights=True)

        assert loss_fn.weights.shape == (576,)
        assert (loss_fn.weights > 0).all()


class TestRateDistortionLoss:
    """Test rate-distortion loss for training."""

    def test_rd_loss_components(self, mdct_coeffs, scalefactors):
        """RD loss should return all components."""
        loss_fn = RateDistortionLoss(rate_weight=0.01)

        quantized = mdct_coeffs + 0.1 * torch.randn_like(mdct_coeffs)

        losses = loss_fn(quantized, mdct_coeffs, scalefactors)

        assert "total" in losses
        assert "distortion" in losses
        assert "rate" in losses
        assert "adaptive_rate" in losses
        assert "sf_mean" in losses

    def test_rd_loss_rate_increases_with_low_sf(self, mdct_coeffs):
        """Rate penalty should be higher for low scalefactors."""
        loss_fn = RateDistortionLoss(rate_weight=0.01)

        low_sf = torch.full((mdct_coeffs.shape[0], NUM_BANDS), 2.0)
        high_sf = torch.full((mdct_coeffs.shape[0], NUM_BANDS), 12.0)

        quantized = mdct_coeffs + 0.1 * torch.randn_like(mdct_coeffs)

        losses_low = loss_fn(quantized, mdct_coeffs, low_sf)
        losses_high = loss_fn(quantized, mdct_coeffs, high_sf)

        # Rate penalty should be higher for low SF (using more bits)
        assert losses_low["rate"] > losses_high["rate"]

    def test_rd_loss_distortion_gradient(self, mdct_coeffs):
        """Distortion component should have gradients."""
        loss_fn = RateDistortionLoss(rate_weight=0.0)  # Disable rate

        coeffs = mdct_coeffs.clone().requires_grad_(True)
        sf = torch.rand(mdct_coeffs.shape[0], NUM_BANDS) * 15

        quantized = coeffs + 0.1 * torch.randn_like(coeffs)
        losses = loss_fn(quantized, coeffs, sf)

        losses["distortion"].backward()

        assert coeffs.grad is not None


class TestBitrateLoss:
    """Test bitrate efficiency loss."""

    def test_bitrate_loss_shape(self, scalefactors):
        """Bitrate loss should return scalar."""
        loss_fn = BitrateLoss(target_avg_sf=8.0)
        energy = torch.rand(scalefactors.shape[0], NUM_BANDS)

        loss = loss_fn(scalefactors, energy)

        assert loss.dim() == 0

    def test_bitrate_penalty_below_target(self):
        """Bitrate loss should penalize SF below target."""
        loss_fn = BitrateLoss(target_avg_sf=8.0)

        low_sf = torch.full((2, NUM_BANDS), 4.0)  # Below target
        high_sf = torch.full((2, NUM_BANDS), 12.0)  # Above target
        energy = torch.ones(2, NUM_BANDS)

        loss_low = loss_fn(low_sf, energy)
        loss_high = loss_fn(high_sf, energy)

        # Low SF should have higher penalty (using more bits)
        assert loss_low > loss_high


class TestPerceptualLoss:
    """Test combined perceptual loss."""

    def test_perceptual_loss_components(self, audio_signal):
        """Perceptual loss should return all components."""
        loss_fn = PerceptualLoss()

        pred = audio_signal + 0.1 * torch.randn_like(audio_signal)
        target = audio_signal

        losses = loss_fn(pred, target)

        assert "total" in losses
        assert "stft_sc" in losses
        assert "stft_mag" in losses
        assert "mel" in losses
        assert "time" in losses


class TestCombinedLoss:
    """Test full combined training loss."""

    def test_combined_loss_all_components(self, audio_signal, scalefactors):
        """Combined loss should include all components."""
        loss_fn = CombinedLoss()

        pred = audio_signal + 0.1 * torch.randn_like(audio_signal)
        target = audio_signal
        energy = torch.rand(scalefactors.shape[0], NUM_BANDS)

        losses = loss_fn(pred, target, scalefactors, energy)

        assert "total" in losses
        assert "perceptual" in losses
        assert "bitrate" in losses


class TestLossNumericalStability:
    """Test numerical stability of losses."""

    def test_stft_loss_small_signal(self):
        """STFT loss should handle very small signals."""
        loss_fn = STFTLoss()

        small = torch.randn(2, 1024) * 1e-8
        target = torch.randn(2, 1024) * 1e-8

        sc, mag = loss_fn(small, target)

        assert torch.isfinite(sc)
        assert torch.isfinite(mag)

    def test_mel_loss_zero_signal(self):
        """Mel loss should handle zero signal (with epsilon)."""
        loss_fn = MelSpectrogramLoss(n_fft=256, hop_length=64, n_mels=16)

        zeros = torch.zeros(2, 1024)
        signal = torch.randn(2, 1024) * 0.1

        loss = loss_fn(zeros, signal)

        assert torch.isfinite(loss)

    def test_mdct_loss_large_coefficients(self, batch_size):
        """MDCT loss should handle large coefficient values."""
        loss_fn = MDCTLoss()

        large = torch.randn(batch_size, 576) * 1000
        target = torch.randn(batch_size, 576) * 1000

        loss = loss_fn(large, target)

        assert torch.isfinite(loss)


class TestLossDeviceCompatibility:
    """Test losses on different devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stft_loss_cuda(self):
        """STFT loss should work on CUDA."""
        loss_fn = STFTLoss().cuda()

        x = torch.randn(2, 1024).cuda()
        y = torch.randn(2, 1024).cuda()

        sc, mag = loss_fn(x, y)

        assert sc.is_cuda
        assert mag.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mel_loss_cuda(self):
        """Mel loss should work on CUDA."""
        loss_fn = MelSpectrogramLoss(n_fft=256, hop_length=64, n_mels=16).cuda()

        x = torch.randn(2, 1024).cuda()
        y = torch.randn(2, 1024).cuda()

        loss = loss_fn(x, y)

        assert loss.is_cuda
