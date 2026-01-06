"""Tests for configuration consistency.

Verifies that config values are valid and consistent.
"""

import pytest

from notlame_train import config
from notlame_train.model import NUM_BANDS, MDCT_SIZE, FRAME_SIZE


class TestSTFTConfig:
    """Test STFT configuration validity."""

    def test_stft_sizes_match(self):
        """FFT, hop, and window sizes should have same length."""
        assert len(config.STFT_FFT_SIZES) == len(config.STFT_HOP_SIZES)
        assert len(config.STFT_FFT_SIZES) == len(config.STFT_WIN_SIZES)

    def test_stft_sizes_valid(self):
        """FFT sizes should be powers of 2."""
        for fft_size in config.STFT_FFT_SIZES:
            assert fft_size > 0
            assert (fft_size & (fft_size - 1)) == 0, f"{fft_size} not power of 2"

    def test_stft_hop_less_than_win(self):
        """Hop size should be <= window size for overlap."""
        for hop, win in zip(config.STFT_HOP_SIZES, config.STFT_WIN_SIZES):
            assert hop <= win, f"hop {hop} > win {win}"

    def test_stft_fits_in_mp3_frame(self):
        """STFT window should fit within MP3 frame."""
        for win_size in config.STFT_WIN_SIZES:
            assert win_size <= FRAME_SIZE, \
                f"Window {win_size} > frame {FRAME_SIZE}"


class TestMelConfig:
    """Test Mel spectrogram configuration."""

    def test_mel_sample_rate(self):
        """Sample rate should be valid."""
        assert config.MEL_SAMPLE_RATE > 0
        assert config.MEL_SAMPLE_RATE == 44100  # Standard CD quality

    def test_mel_window_lengths(self):
        """Window lengths should be positive."""
        for win_len in config.MEL_WINDOW_LENGTHS:
            assert win_len > 0

    def test_mel_n_mels(self):
        """Number of mel bands should be reasonable."""
        assert config.MEL_N_MELS > 0
        assert config.MEL_N_MELS <= 128  # Typical range


class TestLossWeights:
    """Test loss weight configuration."""

    def test_all_weights_present(self):
        """All required loss weights should exist."""
        required = ["mdct", "stft", "mel", "rate"]
        for name in required:
            assert name in config.LOSS_WEIGHTS, f"Missing weight: {name}"

    def test_weights_positive(self):
        """All weights should be non-negative."""
        for name, weight in config.LOSS_WEIGHTS.items():
            assert weight >= 0, f"Negative weight for {name}: {weight}"

    def test_mel_weight_dominant(self):
        """Mel weight should be highest (perceptual focus)."""
        mel = config.LOSS_WEIGHTS["mel"]
        stft = config.LOSS_WEIGHTS["stft"]
        mdct = config.LOSS_WEIGHTS["mdct"]

        assert mel >= stft, "Mel should be >= STFT weight"
        assert mel >= mdct, "Mel should be >= MDCT weight"


class TestTargets:
    """Test evaluation target configuration."""

    def test_target_scalefactor(self):
        """Target SF should be in valid range."""
        assert 0 <= config.TARGET_SCALEFACTOR <= 15

    def test_sf_range(self):
        """SF range should be valid."""
        min_sf, max_sf = config.SF_RANGE
        assert min_sf == 0
        assert max_sf == 15

    def test_eval_targets_present(self):
        """Evaluation targets should be defined."""
        assert "visqol" in config.EVAL_TARGETS
        assert "snr" in config.EVAL_TARGETS
        assert "mr_stft" in config.EVAL_TARGETS
        assert "mel" in config.EVAL_TARGETS


class TestTrainingConfig:
    """Test training hyperparameter configuration."""

    def test_learning_rate(self):
        """Learning rate should be reasonable."""
        lr = config.TRAINING["lr"]
        assert 1e-6 <= lr <= 1e-2, f"LR {lr} out of typical range"

    def test_weight_decay(self):
        """Weight decay should be small."""
        wd = config.TRAINING["weight_decay"]
        assert 0 <= wd <= 0.1, f"Weight decay {wd} too large"

    def test_grad_clip(self):
        """Gradient clipping should be positive."""
        clip = config.TRAINING["grad_clip"]
        assert clip > 0


class TestModelConstants:
    """Test model constants are consistent."""

    def test_frame_size(self):
        """Frame size should be 1152 (MP3 standard)."""
        assert FRAME_SIZE == 1152

    def test_mdct_size(self):
        """MDCT size should be half of frame size."""
        assert MDCT_SIZE == FRAME_SIZE // 2
        assert MDCT_SIZE == 576

    def test_num_bands(self):
        """Should have 22 scalefactor bands."""
        assert NUM_BANDS == 22


class TestSampleRateConfig:
    """Test sample rate configuration."""

    def test_supported_sample_rates(self):
        """Supported sample rates should include common formats."""
        assert 44100 in config.SUPPORTED_SAMPLE_RATES
        assert 48000 in config.SUPPORTED_SAMPLE_RATES
        assert 32000 in config.SUPPORTED_SAMPLE_RATES

    def test_default_sample_rate(self):
        """Default sample rate should be supported."""
        assert config.DEFAULT_SAMPLE_RATE in config.SUPPORTED_SAMPLE_RATES

    def test_model_sample_rate(self):
        """Model sample rate should match default."""
        assert config.MODEL_SAMPLE_RATE == config.DEFAULT_SAMPLE_RATE
        assert config.MODEL_SAMPLE_RATE == 44100

    def test_resample_same_rate(self):
        """Resampling same rate should return input."""
        import numpy as np
        audio = np.random.randn(1000).astype(np.float32)
        result = config.resample_audio(audio, 44100, 44100)
        assert result is audio  # Should be same object

    def test_resample_44100_to_48000(self):
        """Resampling 44.1kHz to 48kHz should increase length."""
        import numpy as np
        audio = np.random.randn(44100).astype(np.float32)
        result = config.resample_audio(audio, 44100, 48000)
        expected_len = int(44100 * 48000 / 44100)
        assert abs(len(result) - expected_len) <= 1

    def test_resample_48000_to_44100(self):
        """Resampling 48kHz to 44.1kHz should decrease length."""
        import numpy as np
        audio = np.random.randn(48000).astype(np.float32)
        result = config.resample_audio(audio, 48000, 44100)
        expected_len = int(48000 * 44100 / 48000)
        assert abs(len(result) - expected_len) <= 1

    def test_resample_roundtrip(self):
        """Resampling roundtrip should roughly preserve signal."""
        import numpy as np
        # Create a simple sine wave
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine

        # Roundtrip: 44100 -> 48000 -> 44100
        up = config.resample_audio(audio, 44100, 48000)
        down = config.resample_audio(up, 48000, 44100)

        # Should be similar length
        min_len = min(len(audio), len(down))
        # Allow some tolerance due to resampling artifacts
        correlation = np.corrcoef(audio[:min_len], down[:min_len])[0, 1]
        assert correlation > 0.99, f"Correlation {correlation} too low"


class TestConfigConsistency:
    """Test configuration is internally consistent."""

    def test_target_sf_in_range(self):
        """Target SF should be within SF range."""
        min_sf, max_sf = config.SF_RANGE
        assert min_sf <= config.TARGET_SCALEFACTOR <= max_sf

    def test_stft_config_for_audio(self):
        """STFT config should work with sample rate."""
        for win_len in config.MEL_WINDOW_LENGTHS:
            # Hop length is typically win_len // 4
            hop_len = win_len // 4
            # Should be able to process at least a few frames
            min_samples = win_len + hop_len
            assert min_samples < config.MEL_SAMPLE_RATE, \
                f"Window {win_len} too large for sample rate"
