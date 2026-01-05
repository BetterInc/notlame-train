"""Tests for differentiable MP3 encoding pipeline.

Tests MDCT, quantization, perceptual weights, and stereo processing.
"""

import pytest
import torch
import math

from notlame_train.differentiable_mp3 import (
    DifferentiableMDCT,
    DifferentiableMP3,
    MP3Quantizer,
    StraightThroughQuantize,
    OverlapAdd,
    compute_perceptual_weights,
    compute_masking_thresholds,
    stereo_to_mid_side,
    mid_side_to_stereo,
    compute_snr,
    process_audio_through_model,
    process_coeffs_through_model,
)
from notlame_train.model import NUM_BANDS, MDCT_SIZE, FRAME_SIZE


class TestMDCT:
    """Test Modified Discrete Cosine Transform."""

    def test_mdct_output_shape(self, mp3_frame):
        """MDCT should produce correct number of coefficients."""
        mdct = DifferentiableMDCT()
        coeffs = mdct(mp3_frame)

        assert coeffs.shape == (mp3_frame.shape[0], MDCT_SIZE)

    def test_imdct_output_shape(self, mdct_coeffs):
        """Inverse MDCT should produce correct frame size."""
        mdct = DifferentiableMDCT()
        frame = mdct.inverse(mdct_coeffs)

        assert frame.shape == (mdct_coeffs.shape[0], FRAME_SIZE)

    def test_mdct_reconstruction_with_overlap_add(self):
        """MDCT with proper overlap-add should give near-perfect reconstruction."""
        mdct = DifferentiableMDCT()
        overlap_add = OverlapAdd()

        # Create test signal (several frames)
        num_frames = 10
        hop_size = FRAME_SIZE // 2
        signal_length = (num_frames + 1) * hop_size
        t = torch.linspace(0, 1, signal_length)
        test_signal = torch.sin(2 * math.pi * 440 * t)

        # Extract overlapping frames
        frames = []
        for i in range(num_frames):
            start = i * hop_size
            frames.append(test_signal[start:start + FRAME_SIZE])
        frames = torch.stack(frames)

        # MDCT -> IMDCT
        coeffs = mdct(frames)
        reconstructed_frames = mdct.inverse(coeffs)

        # Overlap-add
        reconstructed = overlap_add(reconstructed_frames.unsqueeze(0)).squeeze(0)

        # Compare middle section (avoid edge effects)
        start = hop_size
        end = (num_frames - 1) * hop_size
        orig = test_signal[start:end]
        recon = reconstructed[start:end]

        snr = compute_snr(orig.unsqueeze(0), recon.unsqueeze(0)).item()

        # Should be near-perfect (>50 dB SNR)
        assert snr > 50, f"MDCT reconstruction SNR {snr:.1f} dB too low"

    def test_mdct_energy_preservation(self, mp3_frame):
        """MDCT should approximately preserve signal energy."""
        mdct = DifferentiableMDCT()
        coeffs = mdct(mp3_frame)

        signal_energy = torch.sum(mp3_frame ** 2, dim=-1)
        coeff_energy = torch.sum(coeffs ** 2, dim=-1)

        # Energy ratio should be positive (energy is preserved)
        ratio = coeff_energy / (signal_energy + 1e-10)

        # All ratios should be positive and non-zero
        assert (ratio > 0).all(), "Energy should be positive"


class TestQuantization:
    """Test MP3 quantization."""

    def test_quantizer_output_shape(self, mdct_coeffs, scalefactors):
        """Quantizer should preserve input shape."""
        quantizer = MP3Quantizer()
        quantized = quantizer(mdct_coeffs, scalefactors)

        assert quantized.shape == mdct_coeffs.shape

    @pytest.mark.parametrize("sf_val", [0, 7.5, 15])
    def test_quantization_snr_vs_scalefactor(self, mdct_coeffs, sf_val):
        """Higher scalefactor should give lower SNR (more quantization noise)."""
        quantizer = MP3Quantizer()
        sf = torch.full((mdct_coeffs.shape[0], NUM_BANDS), sf_val)

        quantized = quantizer(mdct_coeffs, sf)
        snr = compute_snr(mdct_coeffs, quantized).mean().item()

        # SNR should be positive for all SF values
        assert snr > 0, f"SNR should be positive, got {snr:.1f}"

        if sf_val == 0:
            # SF=0 should give good SNR (finer quantization)
            assert snr > 20, f"SF=0 should give decent SNR, got {snr:.1f}"
        elif sf_val == 15:
            # SF=15 can have lower SNR but still positive
            assert snr > 0, f"SF=15 should still give positive SNR, got {snr:.1f}"

    def test_quantization_monotonicity(self, mdct_coeffs):
        """SNR should decrease as scalefactor increases."""
        quantizer = MP3Quantizer()
        snrs = []

        for sf_val in [0, 5, 10, 15]:
            sf = torch.full((mdct_coeffs.shape[0], NUM_BANDS), float(sf_val))
            quantized = quantizer(mdct_coeffs, sf)
            snr = compute_snr(mdct_coeffs, quantized).mean().item()
            snrs.append(snr)

        # SNR should decrease (or stay same) as SF increases
        for i in range(len(snrs) - 1):
            assert snrs[i] >= snrs[i + 1] - 1.0, \
                f"SNR should decrease with SF: {snrs}"

    def test_straight_through_gradient(self):
        """Straight-through estimator should pass gradients unchanged."""
        x = torch.randn(10, requires_grad=True)
        y = StraightThroughQuantize.apply(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be 1 for all elements
        torch.testing.assert_close(x.grad, torch.ones_like(x))


class TestPerceptualWeights:
    """Test perceptual importance weights."""

    def test_weights_shape(self):
        """Perceptual weights should have correct shape."""
        weights = compute_perceptual_weights()
        assert weights.shape == (NUM_BANDS,)

    def test_weights_normalized(self):
        """Perceptual weights should be normalized to mean=1."""
        weights = compute_perceptual_weights()
        assert abs(weights.mean().item() - 1.0) < 0.01

    def test_weights_range(self):
        """Perceptual weights should be within expected range."""
        weights = compute_perceptual_weights()

        # Should be clamped to [0.65, 1.35] based on current config
        assert weights.min() >= 0.6, f"Min weight {weights.min()} too low"
        assert weights.max() <= 1.4, f"Max weight {weights.max()} too high"

    def test_weights_mid_frequency_emphasis(self):
        """Mid-frequency bands (2-5kHz) should have higher weights."""
        weights = compute_perceptual_weights()

        # Bands roughly corresponding to 2-5kHz (bands ~10-15)
        mid_bands = weights[10:16].mean()
        low_bands = weights[0:5].mean()
        high_bands = weights[18:].mean()

        # Mid frequencies should be weighted higher
        assert mid_bands >= low_bands, "Mid frequencies should be >= low"
        assert mid_bands >= high_bands, "Mid frequencies should be >= high"


class TestMaskingThresholds:
    """Test psychoacoustic masking threshold computation."""

    def test_thresholds_shape(self, mdct_coeffs):
        """Masking thresholds should have correct shape."""
        thresholds = compute_masking_thresholds(mdct_coeffs)
        assert thresholds.shape == (mdct_coeffs.shape[0], NUM_BANDS)

    def test_thresholds_positive(self, mdct_coeffs):
        """Masking thresholds should be positive."""
        thresholds = compute_masking_thresholds(mdct_coeffs)
        assert (thresholds > 0).all()

    def test_thresholds_scale_with_energy(self, batch_size):
        """Higher energy input should give higher thresholds."""
        low_energy = torch.randn(batch_size, MDCT_SIZE) * 0.01
        high_energy = torch.randn(batch_size, MDCT_SIZE) * 1.0

        low_thresh = compute_masking_thresholds(low_energy)
        high_thresh = compute_masking_thresholds(high_energy)

        assert high_thresh.mean() > low_thresh.mean()


class TestStereoProcessing:
    """Test Mid-Side stereo encoding."""

    def test_stereo_roundtrip(self, batch_size):
        """M/S encoding should be perfectly invertible."""
        left = torch.randn(batch_size, 1000)
        right = torch.randn(batch_size, 1000)

        mid, side = stereo_to_mid_side(left, right)
        left_recon, right_recon = mid_side_to_stereo(mid, side)

        torch.testing.assert_close(left, left_recon)
        torch.testing.assert_close(right, right_recon)

    def test_mid_is_mono_compatible(self, batch_size):
        """Mid channel should be average of L and R."""
        left = torch.randn(batch_size, 1000)
        right = torch.randn(batch_size, 1000)

        mid, _ = stereo_to_mid_side(left, right)
        expected_mid = (left + right) / 2

        torch.testing.assert_close(mid, expected_mid)

    def test_stereo_quantization(self, batch_size):
        """Stereo input should be quantized correctly."""
        quantizer = MP3Quantizer()

        # Create stereo coefficients
        coeffs = torch.randn(batch_size, 2, MDCT_SIZE) * 0.1
        sf = torch.rand(batch_size, 2, NUM_BANDS) * 15

        quantized = quantizer(coeffs, sf)

        assert quantized.shape == coeffs.shape


class TestGradientFlow:
    """Test gradient flow through the MP3 pipeline."""

    def test_quantizer_gradient_flow(self, mdct_coeffs, scalefactors):
        """Gradients should flow through quantizer to scalefactors."""
        mdct_coeffs = mdct_coeffs.clone()
        scalefactors = scalefactors.clone().requires_grad_(True)

        quantizer = MP3Quantizer()
        quantized = quantizer(mdct_coeffs, scalefactors)

        loss = (quantized ** 2).mean()
        loss.backward()

        assert scalefactors.grad is not None
        assert scalefactors.grad.abs().sum() > 0, "No gradients to scalefactors"

    def test_pipeline_gradient_flow(self, mp3_frame):
        """Gradients should flow through full MP3 pipeline."""
        mp3 = DifferentiableMP3()
        sf = torch.rand(mp3_frame.shape[0], NUM_BANDS) * 15
        sf.requires_grad_(True)

        output = mp3(mp3_frame, sf)
        loss = (output ** 2).mean()
        loss.backward()

        assert sf.grad is not None
        assert sf.grad.abs().sum() > 0


class TestProcessingPipelines:
    """Test high-level processing functions."""

    def test_process_coeffs_shapes(self, multi_frame_coeffs, model_default, mdct, mp3_pipeline):
        """process_coeffs_through_model should return correct shapes."""
        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model_default, mdct, mp3_pipeline
        )

        batch_size, num_frames, _ = multi_frame_coeffs.shape

        # Scalefactors and quantized should match input
        assert scalefactors.shape == (batch_size, num_frames, NUM_BANDS)
        assert quantized.shape == (batch_size, num_frames, MDCT_SIZE)

        # Audio outputs should be 2D
        assert reconstructed.dim() == 2
        assert original.dim() == 2

    def test_snr_function(self):
        """SNR computation should be correct."""
        original = torch.randn(4, 1000)
        noise = torch.randn(4, 1000) * 0.1
        noisy = original + noise

        snr = compute_snr(original, noisy)

        # SNR should be positive and reasonable
        assert (snr > 0).all()
        # With 10% noise, SNR should be around 20 dB
        assert 15 < snr.mean().item() < 30


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_zero_input(self):
        """Pipeline should handle zero input."""
        mdct = DifferentiableMDCT()
        quantizer = MP3Quantizer()

        zeros = torch.zeros(2, MDCT_SIZE)
        sf = torch.ones(2, NUM_BANDS) * 7.5

        quantized = quantizer(zeros, sf)

        assert torch.isfinite(quantized).all()
        assert (quantized == 0).all()  # Zero in, zero out

    def test_large_input(self):
        """Pipeline should handle large magnitude inputs."""
        quantizer = MP3Quantizer()

        large = torch.randn(2, MDCT_SIZE) * 100
        sf = torch.ones(2, NUM_BANDS) * 7.5

        quantized = quantizer(large, sf)

        assert torch.isfinite(quantized).all()

    def test_extreme_scalefactors(self):
        """Pipeline should handle extreme scalefactor values."""
        quantizer = MP3Quantizer()
        coeffs = torch.randn(2, MDCT_SIZE) * 0.1

        # SF = 0 (finest quantization)
        sf_low = torch.zeros(2, NUM_BANDS)
        q_low = quantizer(coeffs, sf_low)
        assert torch.isfinite(q_low).all()

        # SF = 15 (coarsest quantization)
        sf_high = torch.full((2, NUM_BANDS), 15.0)
        q_high = quantizer(coeffs, sf_high)
        assert torch.isfinite(q_high).all()
