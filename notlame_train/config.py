"""Centralized configuration for training and evaluation.

All loss parameters, FFT sizes, and hyperparameters in one place
to ensure consistency between training and evaluation.
"""

# =============================================================================
# Multi-Resolution STFT Loss Configuration
# =============================================================================
# Must fit within 1152-sample MP3 frames
STFT_FFT_SIZES = [64, 128, 256, 512]
STFT_HOP_SIZES = [16, 32, 64, 128]
STFT_WIN_SIZES = [64, 128, 256, 512]

# =============================================================================
# Sample Rate Configuration
# =============================================================================
# Supported sample rates (MPEG-1 Layer III)
SUPPORTED_SAMPLE_RATES = [44100, 48000, 32000]
DEFAULT_SAMPLE_RATE = 44100
MODEL_SAMPLE_RATE = 44100  # Model is trained on this sample rate


def resample_audio(audio, sr_in: int, sr_out: int):
    """Resample audio to target sample rate.

    Args:
        audio: numpy array of audio samples
        sr_in: Input sample rate
        sr_out: Output sample rate

    Returns:
        Resampled audio array
    """
    if sr_in == sr_out:
        return audio

    import numpy as np
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)
    except ImportError:
        from scipy import signal
        num_samples = int(len(audio) * sr_out / sr_in)
        return signal.resample(audio, num_samples).astype(np.float32)

# =============================================================================
# Multi-Scale Mel Loss Configuration (DAC-style)
# =============================================================================
MEL_SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # Use default for training
# Minimum 128 to avoid empty mel filters in librosa
MEL_WINDOW_LENGTHS = [128, 256, 512]
MEL_N_MELS = 64
MEL_USE_L2 = False  # Pure L1 to match evaluation metrics

# =============================================================================
# Loss Weights (based on DAC/LRAC research)
# =============================================================================
LOSS_WEIGHTS = {
    "mdct": 0.1,      # MDCT reconstruction (anchor)
    "stft": 1.0,      # MR-STFT (log + linear magnitude)
    "mel": 15.0,      # Multi-scale Mel (primary perceptual loss)
    "rate": 1.5,      # Rate penalty - push for more compression learning
}

# =============================================================================
# Rate-Distortion Configuration
# =============================================================================
# Target SF for ~192kbps quality (LAME uses ~4-6 for good bands)
# Lower = better quality, higher = more compression
TARGET_SCALEFACTOR = 5.5  # Was 7.5, reduced for better quality target

# =============================================================================
# MP3 Configuration
# =============================================================================
# All MP3 structure constants (FRAME_SIZE, MDCT_SIZE, NUM_BANDS, etc.)
# are in model.py - import from there to avoid duplication
SF_RANGE = (0, 15)        # Scalefactor range

# =============================================================================
# Training Configuration
# =============================================================================
TRAINING = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "scheduler_T0": 10000,
    "scheduler_Tmult": 2,
}

# =============================================================================
# Evaluation Targets
# =============================================================================
EVAL_TARGETS = {
    "visqol": 4.0,      # MOS 1-5, >4.0 is good
    "snr": 20.0,        # dB, >20 is good
    "mr_stft": 0.5,     # Lower is better, <0.5 is good
    "mel": 1.0,         # Lower is better, <1.0 is good
    "win_rate": 0.50,   # >50% means better than LAME
}
