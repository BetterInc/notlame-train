"""Tests for PsychoNet model architecture.

Tests model creation, forward pass, output shapes, and gradient flow.
"""

import pytest
import torch

from notlame_train.model import (
    PsychoNet,
    PsychoNetLite,
    PsychoNetStereo,
    create_model,
    count_parameters,
    NUM_BANDS,
    MDCT_SIZE,
    SCALEFACTOR_BANDS_BY_SR,
    SUPPORTED_SAMPLE_RATES,
    get_scalefactor_bands,
)


class TestModelCreation:
    """Test model instantiation and variants."""

    def test_create_default_model(self):
        """Default model should be PsychoNet."""
        model = create_model("default")
        assert isinstance(model, PsychoNet)

    def test_create_lite_model(self):
        """Lite variant should be PsychoNetLite."""
        model = create_model("lite")
        assert isinstance(model, PsychoNetLite)

    def test_create_large_model(self):
        """Large variant should be PsychoNet with more capacity."""
        model = create_model("large")
        assert isinstance(model, PsychoNet)
        # Large should have more parameters than default
        default = create_model("default")
        assert count_parameters(model) > count_parameters(default)

    def test_create_stereo_model(self):
        """Stereo flag should wrap in PsychoNetStereo."""
        model = create_model("default", stereo=True)
        assert isinstance(model, PsychoNetStereo)

    def test_model_parameter_counts(self):
        """Model variants should have reasonable parameter counts."""
        lite = create_model("lite")
        default = create_model("default")
        large = create_model("large")

        lite_params = count_parameters(lite)
        default_params = count_parameters(default)
        large_params = count_parameters(large)

        # Sanity checks - all models should have reasonable size
        assert 10_000 < lite_params < 500_000, f"Lite has {lite_params} params"
        assert 100_000 < default_params < 1_000_000, f"Default has {default_params} params"
        assert default_params < large_params, "Large should be bigger than default"


class TestModelForward:
    """Test model forward pass."""

    @pytest.mark.parametrize("variant", ["default", "lite", "large"])
    def test_forward_output_shape(self, mdct_coeffs, variant):
        """Forward pass should return correct output shape."""
        model = create_model(variant)
        batch_size = mdct_coeffs.shape[0]

        output = model(mdct_coeffs)

        assert "scalefactors" in output
        assert output["scalefactors"].shape == (batch_size, NUM_BANDS)

    @pytest.mark.parametrize("variant", ["default", "lite"])
    def test_scalefactors_range(self, mdct_coeffs, variant):
        """Scalefactors should be in valid MP3 range [0, 15]."""
        model = create_model(variant)

        output = model(mdct_coeffs)
        sf = output["scalefactors"]

        assert sf.min() >= 0.0, f"Min SF {sf.min()} < 0"
        assert sf.max() <= 15.0, f"Max SF {sf.max()} > 15"

    def test_forward_deterministic(self, mdct_coeffs, model_default):
        """Model should be deterministic in eval mode."""
        model_default.eval()

        with torch.no_grad():
            out1 = model_default(mdct_coeffs)
            out2 = model_default(mdct_coeffs)

        torch.testing.assert_close(out1["scalefactors"], out2["scalefactors"])

    def test_forward_batch_independence(self, model_default):
        """Each sample in batch should be processed independently."""
        torch.manual_seed(42)
        x1 = torch.randn(1, MDCT_SIZE) * 0.1
        x2 = torch.randn(1, MDCT_SIZE) * 0.1

        model_default.eval()
        with torch.no_grad():
            # Process individually
            out1 = model_default(x1)["scalefactors"]
            out2 = model_default(x2)["scalefactors"]

            # Process as batch
            batch = torch.cat([x1, x2], dim=0)
            out_batch = model_default(batch)["scalefactors"]

        torch.testing.assert_close(out1, out_batch[0:1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out2, out_batch[1:2], atol=1e-5, rtol=1e-5)


class TestModelGradients:
    """Test gradient flow through model."""

    @pytest.mark.parametrize("variant", ["default", "lite"])
    def test_gradients_flow(self, mdct_coeffs, variant):
        """Gradients should flow to all model parameters."""
        model = create_model(variant)
        model.train()

        output = model(mdct_coeffs)
        loss = output["scalefactors"].mean()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # At least some gradients should be non-zero
                # (not all will be for every input)

    def test_gradient_magnitude(self, mdct_coeffs, model_default):
        """Gradients should have reasonable magnitude (not exploding/vanishing)."""
        model_default.train()

        output = model_default(mdct_coeffs)
        loss = output["scalefactors"].mean()
        loss.backward()

        grad_norms = []
        for name, param in model_default.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        max_grad = max(grad_norms)
        min_grad = min(grad_norms)

        # Gradients shouldn't explode or completely vanish
        assert max_grad < 100.0, f"Gradient explosion: max={max_grad}"
        # At least some gradients should be meaningful
        assert max_grad > 1e-10, f"Gradients too small: max={max_grad}"


class TestStereoModel:
    """Test stereo (M/S) model wrapper."""

    def test_stereo_forward_shape(self, batch_size):
        """Stereo model should handle (batch, 2, 576) input."""
        model = create_model("default", stereo=True)
        x = torch.randn(batch_size, 2, MDCT_SIZE) * 0.1

        output = model(x)

        assert output["scalefactors"].shape == (batch_size, 2, NUM_BANDS)

    def test_stereo_mono_fallback(self, mdct_coeffs):
        """Stereo model should handle mono (batch, 576) input."""
        model = create_model("default", stereo=True)

        output = model(mdct_coeffs)

        # Mono input should give mono output
        assert output["scalefactors"].shape == (mdct_coeffs.shape[0], NUM_BANDS)

    def test_stereo_channel_independence(self, batch_size):
        """Mid and Side channels can have different scalefactors."""
        model = create_model("default", stereo=True)

        # Create input with very different M and S channels
        mid = torch.randn(batch_size, MDCT_SIZE) * 0.1
        side = torch.randn(batch_size, MDCT_SIZE) * 0.5  # Different magnitude
        x = torch.stack([mid, side], dim=1)

        output = model(x)
        sf = output["scalefactors"]

        mid_sf = sf[:, 0, :]
        side_sf = sf[:, 1, :]

        # Channels should potentially have different scalefactors
        # (exact same would be suspicious)
        diff = (mid_sf - side_sf).abs().mean()
        assert diff > 0.01, "M/S channels have identical scalefactors"


class TestModelDevice:
    """Test model on different devices."""

    def test_model_to_device(self, device, mdct_coeffs):
        """Model should work on specified device."""
        model = create_model("default").to(device)
        x = mdct_coeffs.to(device)

        output = model(x)

        # Check device type matches (cuda:0 should match cuda)
        assert output["scalefactors"].device.type == device.type

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda(self):
        """Model should work on CUDA."""
        model = create_model("default").cuda()
        x = torch.randn(2, MDCT_SIZE).cuda() * 0.1

        output = model(x)

        assert output["scalefactors"].is_cuda


class TestScalefactorBands:
    """Test scalefactor band definitions per sample rate."""

    def test_all_sample_rates_defined(self):
        """All MPEG-1 sample rates should have band definitions."""
        assert 44100 in SCALEFACTOR_BANDS_BY_SR
        assert 48000 in SCALEFACTOR_BANDS_BY_SR
        assert 32000 in SCALEFACTOR_BANDS_BY_SR

    @pytest.mark.parametrize("sr", SUPPORTED_SAMPLE_RATES)
    def test_bands_have_22_bands(self, sr):
        """Each sample rate should have 22 scalefactor bands."""
        bands = SCALEFACTOR_BANDS_BY_SR[sr]
        assert len(bands) == 23, f"SR {sr}: expected 23 boundaries, got {len(bands)}"
        assert len(bands) - 1 == NUM_BANDS

    @pytest.mark.parametrize("sr", SUPPORTED_SAMPLE_RATES)
    def test_bands_start_at_zero(self, sr):
        """Bands should start at coefficient 0."""
        bands = SCALEFACTOR_BANDS_BY_SR[sr]
        assert bands[0] == 0

    @pytest.mark.parametrize("sr", SUPPORTED_SAMPLE_RATES)
    def test_bands_end_at_576(self, sr):
        """Bands should end at coefficient 576 (MDCT_SIZE)."""
        bands = SCALEFACTOR_BANDS_BY_SR[sr]
        assert bands[-1] == MDCT_SIZE

    @pytest.mark.parametrize("sr", SUPPORTED_SAMPLE_RATES)
    def test_bands_monotonically_increasing(self, sr):
        """Band boundaries should be monotonically increasing."""
        bands = SCALEFACTOR_BANDS_BY_SR[sr]
        for i in range(len(bands) - 1):
            assert bands[i] < bands[i + 1], f"SR {sr}: band {i} not increasing"

    def test_get_scalefactor_bands_known(self):
        """get_scalefactor_bands should return correct bands for known rates."""
        assert get_scalefactor_bands(44100) == SCALEFACTOR_BANDS_BY_SR[44100]
        assert get_scalefactor_bands(48000) == SCALEFACTOR_BANDS_BY_SR[48000]
        assert get_scalefactor_bands(32000) == SCALEFACTOR_BANDS_BY_SR[32000]

    def test_get_scalefactor_bands_fallback(self):
        """get_scalefactor_bands should fall back to nearest for unknown rates."""
        # 96000 should fall back to 48000 (nearest)
        bands_96k = get_scalefactor_bands(96000)
        assert bands_96k == SCALEFACTOR_BANDS_BY_SR[48000]

        # 22050 should fall back to 32000 (nearest)
        bands_22k = get_scalefactor_bands(22050)
        assert bands_22k == SCALEFACTOR_BANDS_BY_SR[32000]

    def test_band_widths_differ_by_sr(self):
        """Different sample rates should have different band widths."""
        bands_44 = SCALEFACTOR_BANDS_BY_SR[44100]
        bands_48 = SCALEFACTOR_BANDS_BY_SR[48000]

        # They should be different (not identical)
        assert bands_44 != bands_48, "44.1kHz and 48kHz bands should differ"

        # But same number of bands
        assert len(bands_44) == len(bands_48)
