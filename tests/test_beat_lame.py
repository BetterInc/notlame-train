"""Tests that verify our perceptual quantizer beats LAME.

These tests compare our perceptual weighting approach against
LAME-style uniform quantization across multiple bitrates.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from notlame_train.differentiable_mp3 import (
    DifferentiableMP3,
    DifferentiableMDCT,
    compute_perceptual_weights,
)
from notlame_train.losses import MultiResolutionSTFTLoss, MultiScaleMelLoss
from notlame_train.model import NUM_BANDS, SCALEFACTOR_BANDS_LONG
from notlame_train import config


# Bitrate to scalefactor mapping (empirically determined)
# Higher SF = more compression = lower bitrate
BITRATE_TO_SF = {
    320: 3.0,   # Highest quality
    256: 4.5,
    224: 5.0,
    192: 5.5,   # Common "high quality"
    160: 6.5,
    128: 8.0,   # Common default
    112: 9.0,
    96: 10.0,
    64: 12.5,   # Low bitrate
}


class LAMEQuantizer:
    """Simulates LAME-style uniform quantization (no perceptual weighting)."""

    def __init__(self):
        self.base_step = 0.01

    def quantize(self, coeffs: torch.Tensor, scalefactors: torch.Tensor) -> torch.Tensor:
        """Quantize with uniform step size across all bands."""
        quantized = torch.zeros_like(coeffs)

        for i in range(NUM_BANDS):
            start = SCALEFACTOR_BANDS_LONG[i]
            end = SCALEFACTOR_BANDS_LONG[i + 1]
            band = coeffs[:, start:end]
            sf = scalefactors[:, i:i + 1]

            # Uniform step - no perceptual weighting
            step = self.base_step * torch.pow(2.0, sf / 4.0)
            quantized[:, start:end] = torch.round(band / step) * step

        return quantized


class OursQuantizer:
    """Our perceptual quantizer with frequency-dependent weighting."""

    def __init__(self):
        self.base_step = 0.01
        self.perceptual_weights = compute_perceptual_weights()

    def quantize(self, coeffs: torch.Tensor, scalefactors: torch.Tensor) -> torch.Tensor:
        """Quantize with perceptual weighting."""
        quantized = torch.zeros_like(coeffs)

        for i in range(NUM_BANDS):
            start = SCALEFACTOR_BANDS_LONG[i]
            end = SCALEFACTOR_BANDS_LONG[i + 1]
            band = coeffs[:, start:end]
            sf = scalefactors[:, i:i + 1]

            # Perceptual weighting: higher weight = finer quantization
            weight = self.perceptual_weights[i]
            step = self.base_step * torch.pow(2.0, sf / 4.0) / weight
            quantized[:, start:end] = torch.round(band / step) * step

        return quantized


@pytest.fixture(scope="module")
def test_coefficients():
    """Load test MDCT coefficients from data directory."""
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    if not data_dir.exists():
        pytest.skip("No training data available - run 'make prepare' first")

    files = sorted(data_dir.glob("*.npy"))[:20]
    if not files:
        pytest.skip("No .npy files found in data/processed/")

    all_coeffs = []
    for f in files:
        data = np.load(f)
        all_coeffs.append(torch.from_numpy(data[:4]).float())

    return torch.cat(all_coeffs, dim=0)


@pytest.fixture(scope="module")
def loss_functions():
    """Create loss functions for comparison."""
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
    return stft_loss, mel_loss


@pytest.fixture(scope="module")
def mdct():
    """Create MDCT instance for audio conversion."""
    return DifferentiableMDCT()


class TestBeatLAME:
    """Tests that verify we beat LAME at various bitrates."""

    def test_perceptual_weights_configured(self):
        """Verify perceptual weights are properly configured."""
        weights = compute_perceptual_weights()

        assert weights.shape == (NUM_BANDS,)

        # Check range is as expected [0.65, 1.35] for 2.1x ratio
        assert weights.min() >= 0.6, f"Min weight {weights.min()} below expected"
        assert weights.max() <= 1.4, f"Max weight {weights.max()} above expected"

        # Check normalized
        assert abs(weights.mean() - 1.0) < 0.05

    @pytest.mark.parametrize("kbps,sf_val", list(BITRATE_TO_SF.items()))
    def test_beat_lame_mel(self, kbps, sf_val, test_coefficients, loss_functions, mdct):
        """Our quantizer should beat LAME on Mel loss at each bitrate."""
        stft_loss, mel_loss = loss_functions
        coeffs = test_coefficients

        lame = LAMEQuantizer()
        ours = OursQuantizer()

        sf = torch.full((coeffs.shape[0], NUM_BANDS), sf_val)

        # Quantize with both approaches
        q_lame = lame.quantize(coeffs, sf)
        q_ours = ours.quantize(coeffs, sf)

        # Convert to audio
        audio_orig = mdct.inverse(coeffs)
        audio_lame = mdct.inverse(q_lame)
        audio_ours = mdct.inverse(q_ours)

        # Compute mel loss
        mel_lame = mel_loss(audio_lame, audio_orig).item()
        mel_ours = mel_loss(audio_ours, audio_orig).item()

        assert mel_ours <= mel_lame, \
            f"Failed to beat LAME on Mel at {kbps}kbps: ours={mel_ours:.4f} vs LAME={mel_lame:.4f}"

    def test_beat_lame_majority_stft(self, test_coefficients, loss_functions, mdct):
        """Our quantizer should beat LAME on STFT at majority of bitrates."""
        stft_loss, mel_loss = loss_functions
        coeffs = test_coefficients

        lame = LAMEQuantizer()
        ours = OursQuantizer()

        stft_wins = 0

        for kbps, sf_val in BITRATE_TO_SF.items():
            sf = torch.full((coeffs.shape[0], NUM_BANDS), sf_val)

            q_lame = lame.quantize(coeffs, sf)
            q_ours = ours.quantize(coeffs, sf)

            audio_orig = mdct.inverse(coeffs)
            audio_lame = mdct.inverse(q_lame)
            audio_ours = mdct.inverse(q_ours)

            sc_l, mag_l = stft_loss(audio_lame, audio_orig)
            sc_o, mag_o = stft_loss(audio_ours, audio_orig)

            stft_lame = (sc_l + mag_l).item()
            stft_ours = (sc_o + mag_o).item()

            if stft_ours < stft_lame:
                stft_wins += 1

        # Should win at least 5/9 bitrates (majority)
        assert stft_wins >= 5, \
            f"Failed to beat LAME on STFT majority: only {stft_wins}/9 wins"

    def test_beat_lame_both_majority(self, test_coefficients, loss_functions, mdct):
        """Our quantizer should beat LAME on BOTH metrics at majority of bitrates."""
        stft_loss, mel_loss = loss_functions
        coeffs = test_coefficients

        lame = LAMEQuantizer()
        ours = OursQuantizer()

        both_wins = 0

        for kbps, sf_val in BITRATE_TO_SF.items():
            sf = torch.full((coeffs.shape[0], NUM_BANDS), sf_val)

            q_lame = lame.quantize(coeffs, sf)
            q_ours = ours.quantize(coeffs, sf)

            audio_orig = mdct.inverse(coeffs)
            audio_lame = mdct.inverse(q_lame)
            audio_ours = mdct.inverse(q_ours)

            # STFT
            sc_l, mag_l = stft_loss(audio_lame, audio_orig)
            sc_o, mag_o = stft_loss(audio_ours, audio_orig)
            stft_lame = (sc_l + mag_l).item()
            stft_ours = (sc_o + mag_o).item()

            # Mel
            mel_lame = mel_loss(audio_lame, audio_orig).item()
            mel_ours = mel_loss(audio_ours, audio_orig).item()

            if stft_ours < stft_lame and mel_ours < mel_lame:
                both_wins += 1

        # Should win on both at least 5/9 bitrates
        assert both_wins >= 5, \
            f"Failed to beat LAME on BOTH metrics: only {both_wins}/9 wins"

    def test_beat_lame_192kbps(self, test_coefficients, loss_functions, mdct):
        """Must beat LAME at the common 192kbps setting."""
        stft_loss, mel_loss = loss_functions
        coeffs = test_coefficients
        sf_val = BITRATE_TO_SF[192]

        lame = LAMEQuantizer()
        ours = OursQuantizer()

        sf = torch.full((coeffs.shape[0], NUM_BANDS), sf_val)

        q_lame = lame.quantize(coeffs, sf)
        q_ours = ours.quantize(coeffs, sf)

        audio_orig = mdct.inverse(coeffs)
        audio_lame = mdct.inverse(q_lame)
        audio_ours = mdct.inverse(q_ours)

        # Both metrics
        sc_l, mag_l = stft_loss(audio_lame, audio_orig)
        sc_o, mag_o = stft_loss(audio_ours, audio_orig)
        mel_lame = mel_loss(audio_lame, audio_orig).item()
        mel_ours = mel_loss(audio_ours, audio_orig).item()

        assert mel_ours < mel_lame, \
            f"Failed to beat LAME on Mel at 192kbps: {mel_ours:.4f} vs {mel_lame:.4f}"

        # STFT should also be close or better
        stft_lame = (sc_l + mag_l).item()
        stft_ours = (sc_o + mag_o).item()
        assert stft_ours < stft_lame * 1.05, \
            f"STFT significantly worse at 192kbps: {stft_ours:.4f} vs {stft_lame:.4f}"


class TestBeatLAMESummary:
    """Summary test with detailed output."""

    def test_full_comparison_report(self, test_coefficients, loss_functions, mdct, capsys):
        """Generate full comparison report against LAME."""
        stft_loss, mel_loss = loss_functions
        coeffs = test_coefficients

        lame = LAMEQuantizer()
        ours = OursQuantizer()

        weights = compute_perceptual_weights()
        print(f"\nPerceptual weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"Weight ratio: {weights.max()/weights.min():.1f}x\n")

        print("=" * 70)
        print(f"{'kbps':>6} | {'SF':>4} | {'STFT Ours':>10} {'STFT LAME':>10} | "
              f"{'Mel Ours':>9} {'Mel LAME':>9} | Result")
        print("=" * 70)

        stft_wins = 0
        mel_wins = 0
        both_wins = 0

        for kbps, sf_val in sorted(BITRATE_TO_SF.items(), reverse=True):
            sf = torch.full((coeffs.shape[0], NUM_BANDS), sf_val)

            q_lame = lame.quantize(coeffs, sf)
            q_ours = ours.quantize(coeffs, sf)

            audio_orig = mdct.inverse(coeffs)
            audio_lame = mdct.inverse(q_lame)
            audio_ours = mdct.inverse(q_ours)

            sc_l, mag_l = stft_loss(audio_lame, audio_orig)
            sc_o, mag_o = stft_loss(audio_ours, audio_orig)
            stft_lame = (sc_l + mag_l).item()
            stft_ours = (sc_o + mag_o).item()

            mel_lame = mel_loss(audio_lame, audio_orig).item()
            mel_ours = mel_loss(audio_ours, audio_orig).item()

            stft_better = stft_ours < stft_lame
            mel_better = mel_ours < mel_lame

            if stft_better:
                stft_wins += 1
            if mel_better:
                mel_wins += 1
            if stft_better and mel_better:
                both_wins += 1

            if stft_better and mel_better:
                result = "BOTH WIN"
            elif stft_better:
                result = "STFT win"
            elif mel_better:
                result = "Mel win"
            else:
                result = "LAME"

            print(f"{kbps:>6} | {sf_val:>4.1f} | {stft_ours:>10.4f} {stft_lame:>10.4f} | "
                  f"{mel_ours:>9.4f} {mel_lame:>9.4f} | {result}")

        print("=" * 70)
        print(f"\nSummary: STFT {stft_wins}/9, Mel {mel_wins}/9, BOTH {both_wins}/9")

        # Assert we meet minimum requirements
        assert mel_wins >= 8, f"Must win Mel at 8+ bitrates, got {mel_wins}"
        assert stft_wins >= 5, f"Must win STFT at 5+ bitrates, got {stft_wins}"
        assert both_wins >= 5, f"Must win BOTH at 5+ bitrates, got {both_wins}"

        print("\nAll assertions passed!")
