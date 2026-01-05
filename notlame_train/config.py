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
# Multi-Scale Mel Loss Configuration (DAC-style)
# =============================================================================
MEL_SAMPLE_RATE = 44100
MEL_WINDOW_LENGTHS = [32, 64, 128, 256, 512]
MEL_N_MELS = 64
MEL_USE_L2 = True  # L1 + L2 combination per MelCap research

# =============================================================================
# Loss Weights (based on DAC/LRAC research)
# =============================================================================
LOSS_WEIGHTS = {
    "mdct": 0.1,      # MDCT reconstruction (anchor)
    "stft": 1.0,      # MR-STFT (log + linear magnitude)
    "mel": 15.0,      # Multi-scale Mel (primary perceptual loss)
    "rate": 0.1,      # Rate penalty for compression
}

# =============================================================================
# Rate-Distortion Configuration
# =============================================================================
TARGET_SCALEFACTOR = 7.5  # Target average SF (midpoint of 0-15 range)

# =============================================================================
# MP3 Configuration
# =============================================================================
MP3_FRAME_SIZE = 1152     # Samples per MP3 frame
MP3_COEFFS = 576          # MDCT coefficients per frame
MP3_BANDS = 21            # Number of scalefactor bands
MP3_SF_RANGE = (0, 15)    # Scalefactor range
AUDIO_SCALE = 100.0       # Scaling for quantization impact

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
