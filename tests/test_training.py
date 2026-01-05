"""Integration tests for training pipeline.

Tests that verify the full training loop works correctly.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from notlame_train.model import create_model, NUM_BANDS
from notlame_train.differentiable_mp3 import (
    DifferentiableMP3,
    DifferentiableMDCT,
    process_coeffs_through_model,
)
from notlame_train.losses import (
    MultiResolutionSTFTLoss,
    MultiScaleMelLoss,
    RateDistortionLoss,
    MDCTLoss,
)
from notlame_train import config


class TestTrainingStep:
    """Test individual training step components."""

    @pytest.fixture
    def training_setup(self):
        """Set up model and loss functions for training."""
        model = create_model("default")
        mp3 = DifferentiableMP3()
        mdct = DifferentiableMDCT()

        stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=config.STFT_FFT_SIZES,
            hop_sizes=config.STFT_HOP_SIZES,
            win_sizes=config.STFT_WIN_SIZES,
        )
        mel_loss = MultiScaleMelLoss(
            sample_rate=config.MEL_SAMPLE_RATE,
            window_lengths=config.MEL_WINDOW_LENGTHS,
            n_mels=config.MEL_N_MELS,
            use_l2=config.MEL_USE_L2,
        )
        rd_loss = RateDistortionLoss(rate_weight=0.0, use_perceptual_weights=True)

        return {
            "model": model,
            "mp3": mp3,
            "mdct": mdct,
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "rd_loss": rd_loss,
        }

    def test_forward_pass(self, training_setup, multi_frame_coeffs):
        """Model forward pass should work with training data."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        batch_size, num_frames, _ = multi_frame_coeffs.shape

        assert scalefactors.shape == (batch_size, num_frames, NUM_BANDS)
        assert quantized.shape == multi_frame_coeffs.shape

    def test_loss_computation(self, training_setup, multi_frame_coeffs):
        """All losses should compute without error."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        stft_loss = training_setup["stft_loss"]
        mel_loss = training_setup["mel_loss"]
        rd_loss = training_setup["rd_loss"]

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        # MDCT-domain loss
        mdct_l = rd_loss(quantized[:, 0, :], multi_frame_coeffs[:, 0, :], scalefactors[:, 0, :])

        # Perceptual losses
        sc, mag = stft_loss(reconstructed, original)
        mel = mel_loss(reconstructed, original)

        # All should be finite
        assert torch.isfinite(mdct_l["total"])
        assert torch.isfinite(sc)
        assert torch.isfinite(mag)
        assert torch.isfinite(mel)

    def test_gradient_flow_end_to_end(self, training_setup, multi_frame_coeffs):
        """Gradients should flow from losses to model parameters."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        stft_loss = training_setup["stft_loss"]
        mel_loss = training_setup["mel_loss"]

        model.train()

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        # Compute losses
        sc, mag = stft_loss(reconstructed, original)
        mel = mel_loss(reconstructed, original)

        total_loss = (sc + mag) + 15.0 * mel
        total_loss.backward()

        # Check gradients exist
        grad_count = 0
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                grad_norm += param.grad.norm().item()

        assert grad_count > 0, "No gradients computed"
        assert grad_norm > 0.001, f"Gradients too small: {grad_norm}"

    def test_optimizer_step(self, training_setup, multi_frame_coeffs):
        """Optimizer should update model parameters."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        mel_loss = training_setup["mel_loss"]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Get initial parameters
        initial_params = {name: p.clone() for name, p in model.named_parameters()}

        model.train()

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        loss = mel_loss(reconstructed, original)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        params_changed = 0
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-8):
                params_changed += 1

        assert params_changed > 0, "No parameters were updated"

    def test_loss_decreases_over_steps(self, training_setup, multi_frame_coeffs):
        """Loss should generally decrease over training steps."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        mel_loss = training_setup["mel_loss"]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        losses = []

        for step in range(20):
            reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
                multi_frame_coeffs, model, mdct, mp3
            )

            loss = mel_loss(reconstructed, original)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease overall (first vs last)
        # Allow some tolerance for noise
        assert losses[-1] < losses[0] * 1.1, \
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestLossWeights:
    """Test that loss weights from config are applied correctly."""

    @pytest.fixture
    def training_setup(self):
        """Set up model and loss functions for training."""
        model = create_model("default")
        mp3 = DifferentiableMP3()
        mdct = DifferentiableMDCT()

        stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=config.STFT_FFT_SIZES,
            hop_sizes=config.STFT_HOP_SIZES,
            win_sizes=config.STFT_WIN_SIZES,
        )
        mel_loss = MultiScaleMelLoss(
            sample_rate=config.MEL_SAMPLE_RATE,
            window_lengths=config.MEL_WINDOW_LENGTHS,
            n_mels=config.MEL_N_MELS,
            use_l2=config.MEL_USE_L2,
        )
        rd_loss = RateDistortionLoss(rate_weight=0.0, use_perceptual_weights=True)

        return {
            "model": model,
            "mp3": mp3,
            "mdct": mdct,
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "rd_loss": rd_loss,
        }

    @pytest.fixture
    def multi_frame_coeffs(self):
        """Generate multi-frame MDCT coefficients."""
        torch.manual_seed(42)
        return torch.randn(4, 4, 576) * 0.1

    def test_config_weights_exist(self):
        """Config should have all required loss weights."""
        assert "mdct" in config.LOSS_WEIGHTS
        assert "stft" in config.LOSS_WEIGHTS
        assert "mel" in config.LOSS_WEIGHTS
        assert "rate" in config.LOSS_WEIGHTS

    def test_mel_dominates(self, training_setup, multi_frame_coeffs):
        """Mel loss should dominate the total loss (per config)."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        stft_loss = training_setup["stft_loss"]
        mel_loss = training_setup["mel_loss"]
        rd_loss = training_setup["rd_loss"]

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        # Compute individual losses
        mdct_l = rd_loss(quantized[:, 0, :], multi_frame_coeffs[:, 0, :], scalefactors[:, 0, :])
        sc, mag = stft_loss(reconstructed, original)
        mel = mel_loss(reconstructed, original)

        # Apply weights
        weighted_mdct = config.LOSS_WEIGHTS["mdct"] * mdct_l["distortion"].item()
        weighted_stft = config.LOSS_WEIGHTS["stft"] * (sc + mag).item()
        weighted_mel = config.LOSS_WEIGHTS["mel"] * mel.item()

        # Mel should be the dominant term
        assert weighted_mel >= weighted_stft, \
            f"Mel ({weighted_mel:.4f}) should dominate STFT ({weighted_stft:.4f})"


class TestCheckpointing:
    """Test model checkpointing."""

    def test_save_load_checkpoint(self, model_default):
        """Model should be saveable and loadable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.pt"

            # Save
            torch.save(model_default.state_dict(), path)

            # Load into new model
            loaded_model = create_model("default")
            loaded_model.load_state_dict(torch.load(path))

            # Parameters should match
            for (n1, p1), (n2, p2) in zip(
                model_default.named_parameters(),
                loaded_model.named_parameters()
            ):
                assert n1 == n2
                torch.testing.assert_close(p1, p2)

    def test_checkpoint_contains_all_components(self, model_default):
        """Full checkpoint should include optimizer and scheduler state."""
        optimizer = torch.optim.AdamW(model_default.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000
        )

        checkpoint = {
            "model_state_dict": model_default.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": 100,
            "epoch": 5,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pt"
            torch.save(checkpoint, path)

            loaded = torch.load(path)

            assert "model_state_dict" in loaded
            assert "optimizer_state_dict" in loaded
            assert "scheduler_state_dict" in loaded
            assert loaded["global_step"] == 100
            assert loaded["epoch"] == 5


class TestNumericalStability:
    """Test training stability."""

    @pytest.fixture
    def training_setup(self):
        """Set up model and loss functions for training."""
        model = create_model("default")
        mp3 = DifferentiableMP3()
        mdct = DifferentiableMDCT()
        mel_loss = MultiScaleMelLoss(
            sample_rate=config.MEL_SAMPLE_RATE,
            window_lengths=config.MEL_WINDOW_LENGTHS,
            n_mels=config.MEL_N_MELS,
        )
        return {
            "model": model,
            "mp3": mp3,
            "mdct": mdct,
            "mel_loss": mel_loss,
        }

    @pytest.fixture
    def multi_frame_coeffs(self):
        """Generate multi-frame MDCT coefficients."""
        torch.manual_seed(42)
        return torch.randn(4, 4, 576) * 0.1

    def test_no_nan_in_forward(self, training_setup, multi_frame_coeffs):
        """Forward pass should not produce NaN."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        assert torch.isfinite(reconstructed).all()
        assert torch.isfinite(scalefactors).all()
        assert torch.isfinite(quantized).all()

    def test_no_nan_in_gradients(self, training_setup, multi_frame_coeffs):
        """Gradients should not contain NaN."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        mel_loss = training_setup["mel_loss"]

        model.train()

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        loss = mel_loss(reconstructed, original)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"NaN gradient in {name}"

    def test_gradient_clipping(self, training_setup, multi_frame_coeffs):
        """Gradient clipping should limit gradient magnitude."""
        model = training_setup["model"]
        mp3 = training_setup["mp3"]
        mdct = training_setup["mdct"]
        mel_loss = training_setup["mel_loss"]

        model.train()

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            multi_frame_coeffs, model, mdct, mp3
        )

        loss = mel_loss(reconstructed, original)
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Check all gradients are bounded
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm * 1.1, \
            f"Gradient norm {total_norm} exceeds max {max_norm}"


class TestDeviceHandling:
    """Test training on different devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_on_cuda(self, multi_frame_coeffs):
        """Training should work on CUDA."""
        device = torch.device("cuda")

        model = create_model("default").to(device)
        mp3 = DifferentiableMP3().to(device)
        mdct = DifferentiableMDCT().to(device)
        mel_loss = MultiScaleMelLoss(
            sample_rate=config.MEL_SAMPLE_RATE,
            window_lengths=config.MEL_WINDOW_LENGTHS,
            n_mels=config.MEL_N_MELS,
        ).to(device)

        coeffs = multi_frame_coeffs.to(device)

        model.train()

        reconstructed, original, scalefactors, quantized = process_coeffs_through_model(
            coeffs, model, mdct, mp3
        )

        loss = mel_loss(reconstructed, original)
        loss.backward()

        # Check everything is on CUDA
        assert reconstructed.is_cuda
        assert loss.is_cuda
