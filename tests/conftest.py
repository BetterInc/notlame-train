"""Pytest configuration and shared fixtures for notlame-train tests."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device (prefer CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device("cpu")


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def mdct_coeffs(batch_size):
    """Generate random MDCT coefficients (batch, 576)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 576) * 0.1


@pytest.fixture
def multi_frame_coeffs(batch_size):
    """Generate multi-frame MDCT coefficients (batch, num_frames, 576)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 4, 576) * 0.1


@pytest.fixture
def audio_signal(batch_size):
    """Generate test audio signal (batch, samples)."""
    torch.manual_seed(42)
    length = 4096  # Short for fast tests
    t = torch.linspace(0, 1, length)
    # Mix of frequencies for realistic test
    signal = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t)
    signal = signal.unsqueeze(0).expand(batch_size, -1)
    return signal + 0.01 * torch.randn_like(signal)


@pytest.fixture
def mp3_frame(batch_size):
    """Generate MP3-sized frame (batch, 1152)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 1152) * 0.1


@pytest.fixture
def scalefactors(batch_size):
    """Generate random scalefactors (batch, 22) in valid range [0, 15]."""
    torch.manual_seed(42)
    from notlame_train.model import NUM_BANDS
    return torch.rand(batch_size, NUM_BANDS) * 15


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def model_default():
    """Create default PsychoNet model."""
    from notlame_train.model import create_model
    return create_model("default")


@pytest.fixture
def model_lite():
    """Create lite PsychoNet model."""
    from notlame_train.model import create_model
    return create_model("lite")


# =============================================================================
# Pipeline Fixtures
# =============================================================================

@pytest.fixture
def mdct():
    """Create DifferentiableMDCT instance."""
    from notlame_train.differentiable_mp3 import DifferentiableMDCT
    return DifferentiableMDCT()


@pytest.fixture
def mp3_pipeline():
    """Create DifferentiableMP3 pipeline."""
    from notlame_train.differentiable_mp3 import DifferentiableMP3
    return DifferentiableMP3()


# =============================================================================
# Loss Fixtures
# =============================================================================

@pytest.fixture
def stft_loss():
    """Create multi-resolution STFT loss."""
    from notlame_train.losses import MultiResolutionSTFTLoss
    from notlame_train import config
    return MultiResolutionSTFTLoss(
        fft_sizes=config.STFT_FFT_SIZES,
        hop_sizes=config.STFT_HOP_SIZES,
        win_sizes=config.STFT_WIN_SIZES,
    )


@pytest.fixture
def mel_loss():
    """Create multi-scale mel loss."""
    from notlame_train.losses import MultiScaleMelLoss
    from notlame_train import config
    return MultiScaleMelLoss(
        sample_rate=config.MEL_SAMPLE_RATE,
        window_lengths=config.MEL_WINDOW_LENGTHS,
        n_mels=config.MEL_N_MELS,
        use_l2=config.MEL_USE_L2,
    )


# =============================================================================
# Test Data Directory
# =============================================================================

@pytest.fixture
def data_dir():
    """Get path to processed data directory."""
    return Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture
def has_data(data_dir):
    """Check if training data is available."""
    return data_dir.exists() and any(data_dir.glob("*.npy"))
