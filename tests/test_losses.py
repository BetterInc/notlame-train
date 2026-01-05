#!/usr/bin/env python3
"""Test SNR, MR-STFT, and Mel losses on multiple files."""

import sys
import glob
import numpy as np
import torch

from notlame_train.differentiable_mp3 import DifferentiableMP3, DifferentiableMDCT
from notlame_train.losses import MultiResolutionSTFTLoss, MultiScaleMelLoss
from notlame_train.model import NUM_BANDS
from notlame_train import config


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute SNR in dB."""
    signal_power = (original ** 2).mean()
    noise_power = ((original - reconstructed) ** 2).mean()
    return 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()


def test_losses():
    """Test all losses on multiple files."""
    print("=" * 60)
    print("Testing SNR, MR-STFT, and Mel losses")
    print("=" * 60)

    # Load test files
    files = sorted(glob.glob('data/processed/*.npy'))[:10]
    if not files:
        print("ERROR: No .npy files found in data/processed/")
        print("Run 'make prepare' first")
        return False

    print(f"\nTesting on {len(files)} files...")

    # Setup
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

    for sf_val in [0, 7.5, 15]:
        print(f"\n--- Scalefactor = {sf_val} ---")

        snr_list = []
        stft_list = []
        mel_list = []

        for f in files:
            # Load MDCT coefficients
            data = np.load(f)
            coeffs = torch.from_numpy(data[:4]).float()  # 4 frames

            # Quantize with fixed scalefactor
            sf = torch.full((4, NUM_BANDS), sf_val)
            quantized = mp3.encode_coeffs(coeffs, sf, thresholds=None)

            # Convert to audio
            orig_audio = mdct.inverse(coeffs)
            recon_audio = mdct.inverse(quantized)

            # SNR
            snr = compute_snr(coeffs, quantized)
            snr_list.append(snr)

            # MR-STFT
            sc, mag = stft_loss(recon_audio, orig_audio)
            stft_list.append((sc + mag).item())

            # Mel
            mel = mel_loss(recon_audio, orig_audio)
            mel_list.append(mel.item())

        # Aggregate
        snr_mean = np.mean(snr_list)
        stft_mean = np.mean(stft_list)
        mel_mean = np.mean(mel_list)

        print(f"  SNR:     {snr_mean:.1f} dB (std={np.std(snr_list):.1f})")
        print(f"  MR-STFT: {stft_mean:.4f} (std={np.std(stft_list):.4f})")
        print(f"  Mel:     {mel_mean:.4f} (std={np.std(mel_list):.4f})")

    # Verify losses change with scalefactor
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)

    # Test SF=0 vs SF=15
    coeffs = torch.from_numpy(np.load(files[0])[:4]).float()

    sf_low = torch.full((4, NUM_BANDS), 0.0)
    sf_high = torch.full((4, NUM_BANDS), 15.0)

    q_low = mp3.encode_coeffs(coeffs, sf_low, thresholds=None)
    q_high = mp3.encode_coeffs(coeffs, sf_high, thresholds=None)

    audio_orig = mdct.inverse(coeffs)
    audio_low = mdct.inverse(q_low)
    audio_high = mdct.inverse(q_high)

    snr_low = compute_snr(coeffs, q_low)
    snr_high = compute_snr(coeffs, q_high)

    sc_low, mag_low = stft_loss(audio_low, audio_orig)
    sc_high, mag_high = stft_loss(audio_high, audio_orig)
    stft_low = (sc_low + mag_low).item()
    stft_high = (sc_high + mag_high).item()

    mel_low = mel_loss(audio_low, audio_orig).item()
    mel_high = mel_loss(audio_high, audio_orig).item()

    print("\nSF=0 vs SF=15 comparison:")
    print(f"  SNR:     {snr_low:.1f} vs {snr_high:.1f} dB  (diff: {snr_low - snr_high:.1f})")
    print(f"  MR-STFT: {stft_low:.4f} vs {stft_high:.4f}  (diff: {stft_high - stft_low:.4f})")
    print(f"  Mel:     {mel_low:.4f} vs {mel_high:.4f}  (diff: {mel_high - mel_low:.4f})")

    # Check expected behavior
    checks = []

    # SNR should be higher for SF=0
    if snr_low > snr_high + 10:
        print("\n✓ SNR: SF=0 gives higher SNR (correct)")
        checks.append(True)
    else:
        print(f"\n✗ SNR: Expected SF=0 > SF=15 by >10 dB, got {snr_low - snr_high:.1f}")
        checks.append(False)

    # MR-STFT should be lower for SF=0
    if stft_low < stft_high:
        print("✓ MR-STFT: SF=0 gives lower loss (correct)")
        checks.append(True)
    else:
        print(f"✗ MR-STFT: Expected SF=0 < SF=15, got {stft_low:.4f} vs {stft_high:.4f}")
        checks.append(False)

    # Mel should be lower for SF=0
    if mel_low < mel_high:
        print("✓ Mel: SF=0 gives lower loss (correct)")
        checks.append(True)
    else:
        print(f"✗ Mel: Expected SF=0 < SF=15, got {mel_low:.4f} vs {mel_high:.4f}")
        checks.append(False)

    # Gradient flow test
    print("\n--- Gradient Flow Test ---")
    coeffs.requires_grad = True
    sf = torch.rand(4, NUM_BANDS) * 15
    sf.requires_grad = True

    quantized = mp3.encode_coeffs(coeffs, sf, thresholds=None)
    audio_orig = mdct.inverse(coeffs)
    audio_recon = mdct.inverse(quantized)

    sc, mag = stft_loss(audio_recon, audio_orig.detach())
    mel = mel_loss(audio_recon, audio_orig.detach())

    total = (sc + mag) + mel
    total.backward()

    if sf.grad is not None and sf.grad.abs().sum() > 0:
        print(f"✓ Gradients flow to scalefactors (grad norm: {sf.grad.norm().item():.4f})")
        checks.append(True)
    else:
        print("✗ No gradients to scalefactors")
        checks.append(False)

    print("\n" + "=" * 60)
    if all(checks):
        print("ALL TESTS PASSED")
        return True
    else:
        print(f"FAILED: {checks.count(False)}/{len(checks)} tests failed")
        return False


def test_training_step():
    """Simulate a full training step to verify everything works."""
    print("\n" + "=" * 60)
    print("Testing Full Training Step Simulation")
    print("=" * 60)

    from notlame_train.model import create_model

    # Load data
    files = sorted(glob.glob('data/processed/*.npy'))[:1]
    if not files:
        print("ERROR: No data files")
        return False

    coeffs = torch.from_numpy(np.load(files[0])[:4]).float()

    # Setup (matching train.py)
    model = create_model('default')
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

    # Weights from config
    mdct_weight = config.LOSS_WEIGHTS["mdct"]
    stft_weight = config.LOSS_WEIGHTS["stft"]
    mel_weight = config.LOSS_WEIGHTS["mel"]
    rate_weight = config.LOSS_WEIGHTS["rate"]
    target_sf = config.TARGET_SCALEFACTOR

    # Forward pass
    output = model(coeffs)
    scalefactors = output['scalefactors']

    print(f"\nInitial SF: mean={scalefactors.mean().item():.2f}, std={scalefactors.std().item():.2f}")

    # Quantize
    quantized = mp3.encode_coeffs(coeffs, scalefactors, thresholds=None)

    # Convert to audio
    orig_audio = mdct.inverse(coeffs)
    recon_audio = mdct.inverse(quantized)

    # Losses
    from notlame_train.losses import MDCTLoss
    mdct_loss_fn = MDCTLoss(use_perceptual_weights=True)
    mdct_l = mdct_loss_fn(quantized, coeffs)

    sc, mag = stft_loss(recon_audio, orig_audio)
    stft_l = sc + mag
    mel_l = mel_loss(recon_audio, orig_audio)

    sf_mean = scalefactors.mean()
    rate_penalty = torch.relu(target_sf - sf_mean)

    total_loss = (
        mdct_weight * mdct_l +
        stft_weight * stft_l +
        mel_weight * mel_l +
        rate_weight * rate_penalty
    )

    print("\nLoss components:")
    print(f"  MDCT:  {mdct_l.item():.4f} (weight={mdct_weight})")
    print(f"  STFT:  {stft_l.item():.4f} (weight={stft_weight})")
    print(f"  Mel:   {mel_l.item():.4f} (weight={mel_weight})")
    print(f"  Rate:  {rate_penalty.item():.4f} (weight={rate_weight})")
    print(f"  Total: {total_loss.item():.4f}")

    # Backward
    total_loss.backward()

    # Check gradients
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\nGradient norm: {total_grad:.6f}")

    checks = []

    # Check loss components are reasonable
    if mdct_l.item() > 0 and mdct_l.item() < 100:
        checks.append(True)
    else:
        print(f"✗ MDCT loss out of range: {mdct_l.item()}")
        checks.append(False)

    if stft_l.item() > 0 and stft_l.item() < 10:
        checks.append(True)
    else:
        print(f"✗ STFT loss out of range: {stft_l.item()}")
        checks.append(False)

    if mel_l.item() > 0 and mel_l.item() < 10:
        checks.append(True)
    else:
        print(f"✗ Mel loss out of range: {mel_l.item()}")
        checks.append(False)

    # Check gradients flow
    if total_grad > 0.001:
        print("✓ Gradients are substantial")
        checks.append(True)
    else:
        print(f"✗ Gradients too small: {total_grad}")
        checks.append(False)

    # Check mel loss is dominant (should be ~10x stft due to weight)
    weighted_mel = mel_weight * mel_l.item()
    weighted_stft = stft_weight * stft_l.item()
    if weighted_mel > weighted_stft:
        print(f"✓ Mel loss dominates (weighted: {weighted_mel:.2f} vs STFT: {weighted_stft:.2f})")
        checks.append(True)
    else:
        print("✗ Mel should dominate but doesn't")
        checks.append(False)

    print("\n" + "=" * 60)
    if all(checks):
        print("TRAINING STEP SIMULATION PASSED")
        return True
    else:
        print(f"FAILED: {checks.count(False)}/{len(checks)} checks failed")
        return False


if __name__ == "__main__":
    success1 = test_losses()
    success2 = test_training_step()
    sys.exit(0 if (success1 and success2) else 1)
