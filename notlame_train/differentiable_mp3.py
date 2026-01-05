"""Differentiable MP3 encoding pipeline.

Makes MDCT and quantization differentiable for end-to-end training.
Uses straight-through estimators and soft quantization.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SCALEFACTOR_BANDS_LONG, NUM_BANDS


class DifferentiableMDCT(nn.Module):
    """Differentiable Modified Discrete Cosine Transform.

    Computes MDCT and inverse MDCT using matrix operations.

    IMPORTANT: MDCT is critically sampled (N samples -> N/2 coefficients).
    Perfect reconstruction requires overlap-add with 50% overlapping frames.
    Single-frame reconstruction will NOT perfectly recover the original signal.

    For training, we work directly on MDCT coefficients (no reconstruction needed).
    """

    def __init__(self, frame_size: int = 1152):
        super().__init__()

        self.frame_size = frame_size
        self.n_coeffs = frame_size // 2  # 576 for MP3

        # Create MDCT basis matrix
        N = frame_size
        M = self.n_coeffs
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(M, dtype=torch.float32)

        # MDCT cosine basis: cos(π/N * (2n + 1 + N/2) * (2k + 1) / 2)
        # Simplified: cos(π/M * (n + n0) * (k + 0.5)) where n0 = (N+1)/2
        mdct_cos = torch.cos(
            math.pi / N * (2 * n[:, None] + 1 + N / 2) * (2 * k[None, :] + 1) / 2
        )

        # Create sine window (Princen-Bradley condition for perfect reconstruction)
        # w[n]^2 + w[n+M]^2 = 1
        window = torch.sin(math.pi / N * (n + 0.5))

        # Analysis basis: windowed cosines
        # Forward MDCT: X[k] = sum_n w[n] * x[n] * cos(...)
        mdct_basis = mdct_cos * window[:, None]

        self.register_buffer("mdct_basis", mdct_basis)
        self.register_buffer("window", window)

        # Inverse MDCT basis
        # IMDCT: y[n] = (2/M) * sum_k X[k] * cos(...)
        # where M = N/2 = n_coeffs = 576
        # Then apply synthesis window: y_out[n] = w[n] * y[n]
        imdct_basis = mdct_cos.T * (2.0 / M)
        self.register_buffer("imdct_basis", imdct_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MDCT.

        Args:
            x: (batch, frame_size) audio frames

        Returns:
            (batch, n_coeffs) MDCT coefficients
        """
        return torch.matmul(x, self.mdct_basis)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute inverse MDCT.

        Note: Returns windowed frames. Perfect reconstruction requires
        overlap-add of consecutive frames with 50% overlap.

        Args:
            coeffs: (batch, n_coeffs) MDCT coefficients

        Returns:
            (batch, frame_size) audio frames (windowed for overlap-add)
        """
        x = torch.matmul(coeffs, self.imdct_basis)
        return x * self.window


class StraightThroughQuantize(torch.autograd.Function):
    """Straight-through estimator for quantization.

    Forward: round to nearest integer
    Backward: pass gradients through unchanged
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class SoftQuantize(nn.Module):
    """Differentiable soft quantization.

    Uses temperature-scaled softmax to approximate hard quantization.
    As temperature → 0, approaches hard quantization.
    """

    def __init__(self, num_levels: int = 8192, temperature: float = 1.0):
        super().__init__()
        self.num_levels = num_levels
        self.temperature = temperature

        # Quantization levels
        levels = torch.arange(num_levels, dtype=torch.float32)
        levels = levels - num_levels // 2  # Center around 0
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft quantize values.

        Args:
            x: Input values to quantize

        Returns:
            Soft-quantized values
        """
        # Compute distance to each level
        x_expanded = x.unsqueeze(-1)  # (..., 1)
        distances = -torch.abs(x_expanded - self.levels)  # (..., num_levels)

        # Softmax weights
        weights = F.softmax(distances / self.temperature, dim=-1)

        # Weighted sum of levels
        return torch.sum(weights * self.levels, dim=-1)


class MP3Quantizer(nn.Module):
    """Linear quantization with scalefactor-controlled step size.

    Uses simple linear quantization which produces much lower spectral
    distortion (MR-STFT) than the traditional x^0.75 power law formula.

    The scalefactor controls quantization coarseness:
    - SF=0: finest quantization (best quality, most bits)
    - SF=15: coarsest quantization (worst quality, fewest bits)

    Step size = base_step * 2^(sf/4), so SF=15 gives ~13x coarser quantization.
    """

    def __init__(self, use_soft: bool = False, temperature: float = 1.0):
        super().__init__()

        self.use_soft = use_soft

        # Base step size for quantization (at SF=0)
        # Chosen to give ~8-bit equivalent precision for typical MDCT coefficients
        # MDCT coeffs of normalized audio typically range [-20, 20]
        # With base_step=0.15, SF=0 gives fine quantization (~0.15 step)
        # SF=15 gives coarse quantization (~2.0 step)
        self.base_step = 0.15

        # Band boundaries
        bands = torch.tensor(SCALEFACTOR_BANDS_LONG, dtype=torch.long)
        self.register_buffer("band_boundaries", bands)

        if use_soft:
            self.soft_quantize = SoftQuantize(temperature=temperature)

    def forward(
        self,
        coeffs: torch.Tensor,
        scalefactors: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quantize MDCT coefficients.

        Args:
            coeffs: (batch, 576) MDCT coefficients
            scalefactors: (batch, 21) scalefactor values [0-15]
            thresholds: (batch, 21) optional masking thresholds

        Returns:
            (batch, 576) quantized coefficients
        """
        quantized = torch.zeros_like(coeffs)

        for i in range(NUM_BANDS):
            start = SCALEFACTOR_BANDS_LONG[i]
            end = SCALEFACTOR_BANDS_LONG[i + 1]

            band_coeffs = coeffs[:, start:end]
            sf = scalefactors[:, i : i + 1]  # (batch, 1)

            # Step size controlled by scalefactor
            # SF=0 -> step=base_step, SF=15 -> step=base_step*13.45
            step = self.base_step * torch.pow(2.0, sf / 4.0)

            # Linear quantization: round(x / step) * step
            if self.use_soft:
                scaled = band_coeffs / step
                quant = self.soft_quantize(scaled)
                dequant = quant * step
            else:
                scaled = band_coeffs / step
                quant = StraightThroughQuantize.apply(scaled)
                dequant = quant * step

            # Apply masking threshold if provided
            if thresholds is not None:
                mask = thresholds[:, i : i + 1]
                # Zero out coefficients below threshold
                dequant = dequant * (band_coeffs.abs() > mask).float()

            quantized[:, start:end] = dequant

        return quantized


class DifferentiableMP3(nn.Module):
    """Complete differentiable MP3 encoding pipeline.

    Simulates MP3 encoding: MDCT → Quantize → Dequantize → IMDCT
    """

    def __init__(
        self,
        frame_size: int = 1152,
        use_soft_quantize: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.frame_size = frame_size
        self.hop_size = frame_size // 2

        self.mdct = DifferentiableMDCT(frame_size)
        self.quantizer = MP3Quantizer(
            use_soft=use_soft_quantize,
            temperature=temperature,
        )

    def forward(
        self,
        audio: torch.Tensor,
        scalefactors: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simulate MP3 encoding and decoding.

        Args:
            audio: (batch, frame_size) audio frames
            scalefactors: (batch, 21) scalefactor values
            thresholds: (batch, 21) optional masking thresholds

        Returns:
            (batch, frame_size) reconstructed audio
        """
        # MDCT
        coeffs = self.mdct(audio)

        # Quantize
        quantized = self.quantizer(coeffs, scalefactors, thresholds)

        # Inverse MDCT
        reconstructed = self.mdct.inverse(quantized)

        return reconstructed

    def encode_coeffs(
        self,
        coeffs: torch.Tensor,
        scalefactors: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quantize pre-computed MDCT coefficients.

        Args:
            coeffs: (batch, 576) MDCT coefficients
            scalefactors: (batch, 21) scalefactor values
            thresholds: (batch, 21) optional masking thresholds

        Returns:
            (batch, 576) quantized coefficients
        """
        return self.quantizer(coeffs, scalefactors, thresholds)


class OverlapAdd(nn.Module):
    """Overlap-add for frame reconstruction.

    MP3 uses 50% overlap between frames.
    """

    def __init__(self, frame_size: int = 1152):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = frame_size // 2

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from overlapping frames.

        Args:
            frames: (batch, num_frames, frame_size) windowed frames

        Returns:
            (batch, output_length) reconstructed audio
        """
        batch_size, num_frames, frame_size = frames.shape

        output_length = (num_frames - 1) * self.hop_size + frame_size
        output = torch.zeros(batch_size, output_length, device=frames.device)

        for i in range(num_frames):
            start = i * self.hop_size
            output[:, start : start + frame_size] += frames[:, i]

        return output


def process_audio_through_model(
    audio: torch.Tensor,
    model: torch.nn.Module,
    mdct: DifferentiableMDCT,
    mp3: "DifferentiableMP3",
    frame_size: int = 1152,
) -> tuple:
    """Process audio through model with proper overlap-add reconstruction.

    This is the canonical pipeline used by both training and evaluation.

    Args:
        audio: (batch, samples) or (samples,) audio tensor
        model: PsychoNet model that outputs scalefactors
        mdct: DifferentiableMDCT instance
        mp3: DifferentiableMP3 instance
        frame_size: MP3 frame size (default 1152)

    Returns:
        tuple of:
            - reconstructed_audio: (batch, samples) properly reconstructed audio
            - original_audio: (batch, samples) original audio (for loss computation)
            - all_scalefactors: (batch, num_frames, 21) predicted scalefactors
            - all_quantized: (batch, num_frames, 576) quantized coefficients
            - all_original: (batch, num_frames, 576) original coefficients
    """
    hop_size = frame_size // 2

    # Ensure batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    batch_size, total_samples = audio.shape
    device = audio.device

    # Pad for proper framing
    start_pad = hop_size
    end_pad = frame_size - ((total_samples + start_pad) % hop_size)
    if end_pad >= frame_size:
        end_pad = 0
    audio_padded = torch.nn.functional.pad(audio, (start_pad, end_pad))

    # Extract frames and process
    all_scalefactors = []
    all_quantized = []
    all_original = []
    all_recon_frames = []
    all_orig_frames = []

    for i in range(0, audio_padded.shape[1] - frame_size + 1, hop_size):
        frame = audio_padded[:, i:i + frame_size]

        # MDCT
        coeffs = mdct(frame)
        all_original.append(coeffs)

        # Model prediction
        output = model(coeffs)
        scalefactors = output["scalefactors"]
        all_scalefactors.append(scalefactors)

        # Quantize
        quantized = mp3.encode_coeffs(coeffs, scalefactors, thresholds=None)
        all_quantized.append(quantized)

        # IMDCT
        all_orig_frames.append(mdct.inverse(coeffs))
        all_recon_frames.append(mdct.inverse(quantized))

    # Stack
    all_scalefactors = torch.stack(all_scalefactors, dim=1)  # (batch, num_frames, 21)
    all_quantized = torch.stack(all_quantized, dim=1)  # (batch, num_frames, 576)
    all_original = torch.stack(all_original, dim=1)  # (batch, num_frames, 576)

    # Overlap-add
    num_frames = len(all_recon_frames)
    output_len = num_frames * hop_size + hop_size
    reconstructed = torch.zeros(batch_size, output_len, device=device)
    original_recon = torch.zeros(batch_size, output_len, device=device)

    for i, (recon_frame, orig_frame) in enumerate(zip(all_recon_frames, all_orig_frames)):
        start = i * hop_size
        reconstructed[:, start:start + frame_size] += recon_frame
        original_recon[:, start:start + frame_size] += orig_frame

    # Trim to original length (remove padding)
    reconstructed = reconstructed[:, start_pad:start_pad + total_samples]
    original_recon = original_recon[:, start_pad:start_pad + total_samples]

    return reconstructed, original_recon, all_scalefactors, all_quantized, all_original


def process_coeffs_through_model(
    coeffs: torch.Tensor,
    model: torch.nn.Module,
    mdct: DifferentiableMDCT,
    mp3: "DifferentiableMP3",
) -> tuple:
    """Process MDCT coefficients through model with proper overlap-add.

    For training on pre-computed MDCT coefficients.

    Args:
        coeffs: (batch, num_frames, 576) MDCT coefficients
        model: PsychoNet model
        mdct: DifferentiableMDCT instance
        mp3: DifferentiableMP3 instance

    Returns:
        tuple of:
            - reconstructed_audio: properly reconstructed audio
            - original_audio: original audio from coefficients
            - all_scalefactors: (batch, num_frames, 21)
            - all_quantized: (batch, num_frames, 576)
    """
    # Handle single-frame input
    if coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(1)

    batch_size, num_frames, n_coeffs = coeffs.shape
    device = coeffs.device
    hop_size = 576  # frame_size // 2

    # Process all frames
    all_scalefactors = []
    all_quantized = []
    all_orig_frames = []
    all_recon_frames = []

    for i in range(num_frames):
        frame = coeffs[:, i, :]

        # Model prediction
        output = model(frame)
        scalefactors = output["scalefactors"]
        all_scalefactors.append(scalefactors)

        # Quantize
        quantized = mp3.encode_coeffs(frame, scalefactors, thresholds=None)
        all_quantized.append(quantized)

        # IMDCT
        all_orig_frames.append(mdct.inverse(frame))
        all_recon_frames.append(mdct.inverse(quantized))

    # Stack
    all_scalefactors = torch.stack(all_scalefactors, dim=1)
    all_quantized = torch.stack(all_quantized, dim=1)

    # Overlap-add
    frame_size = 1152
    output_len = num_frames * hop_size + hop_size
    reconstructed = torch.zeros(batch_size, output_len, device=device)
    original_audio = torch.zeros(batch_size, output_len, device=device)

    for i, (recon_frame, orig_frame) in enumerate(zip(all_recon_frames, all_orig_frames)):
        start = i * hop_size
        reconstructed[:, start:start + frame_size] += recon_frame
        original_audio[:, start:start + frame_size] += orig_frame

    # Return middle section (properly reconstructed)
    # Skip first and last half-frame
    if num_frames > 1:
        reconstructed = reconstructed[:, hop_size:-hop_size]
        original_audio = original_audio[:, hop_size:-hop_size]
    else:
        # Single frame: just return the frame as-is
        reconstructed = all_recon_frames[0]
        original_audio = all_orig_frames[0]

    return reconstructed, original_audio, all_scalefactors, all_quantized


def compute_mdct_energy(coeffs: torch.Tensor) -> torch.Tensor:
    """Compute energy per scalefactor band.

    Args:
        coeffs: (batch, 576) MDCT coefficients

    Returns:
        (batch, 21) band energies
    """
    energies = []
    for i in range(NUM_BANDS):
        start = SCALEFACTOR_BANDS_LONG[i]
        end = SCALEFACTOR_BANDS_LONG[i + 1]
        band = coeffs[:, start:end]
        energy = torch.mean(band ** 2, dim=-1)
        energies.append(energy)

    return torch.stack(energies, dim=-1)


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio in dB.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal

    Returns:
        SNR in dB
    """
    signal_power = torch.mean(original ** 2, dim=-1)
    noise_power = torch.mean((original - reconstructed) ** 2, dim=-1)

    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr


if __name__ == "__main__":
    # Test the differentiable pipeline
    print("Testing differentiable MP3 pipeline...")
    print("=" * 60)

    frame_size = 1152
    hop_size = frame_size // 2

    # Test 1: MDCT with proper overlap-add reconstruction
    print("\n1. Testing MDCT with overlap-add reconstruction:")

    # Create longer test signal (several frames worth)
    num_frames = 10
    signal_length = (num_frames + 1) * hop_size
    test_signal = torch.sin(2 * torch.pi * 440 * torch.arange(signal_length) / 44100)
    test_signal = test_signal + 0.5 * torch.sin(2 * torch.pi * 880 * torch.arange(signal_length) / 44100)

    mdct = DifferentiableMDCT(frame_size)
    overlap_add = OverlapAdd(frame_size)

    # Extract overlapping frames
    frames = []
    for i in range(num_frames):
        start = i * hop_size
        frame = test_signal[start:start + frame_size]
        frames.append(frame)
    frames = torch.stack(frames)  # (num_frames, frame_size)

    # MDCT -> IMDCT for each frame
    coeffs = mdct(frames)
    reconstructed_frames = mdct.inverse(coeffs)

    # Overlap-add reconstruction
    reconstructed = overlap_add(reconstructed_frames.unsqueeze(0)).squeeze(0)

    # Compare middle section (avoid edge effects)
    start_sample = hop_size
    end_sample = (num_frames - 1) * hop_size
    orig_section = test_signal[start_sample:end_sample]
    recon_section = reconstructed[start_sample:end_sample]

    signal_power = torch.mean(orig_section ** 2)
    noise_power = torch.mean((orig_section - recon_section) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

    print(f"   Signal length: {signal_length} samples")
    print(f"   Number of frames: {num_frames}")
    print(f"   MDCT coefficients per frame: {coeffs.shape[1]}")
    print(f"   Overlap-add reconstruction SNR: {snr.item():.1f} dB")
    if snr > 50:
        print("   ✓ MDCT reconstruction is working correctly!")
    else:
        print("   ✗ WARNING: SNR should be >50 dB for perfect reconstruction")

    # Test 2: Energy preservation (Parseval's theorem)
    print("\n2. Testing energy preservation:")
    signal_energy = torch.sum(frames ** 2)
    # With 50% overlap, each sample appears in ~2 frames, so scale by hop_size/frame_size
    coeff_energy = torch.sum(coeffs ** 2) * (frame_size / hop_size)
    energy_ratio = coeff_energy / signal_energy
    print(f"   Signal energy: {signal_energy.item():.4f}")
    print(f"   MDCT energy (scaled): {coeff_energy.item():.4f}")
    print(f"   Energy ratio: {energy_ratio.item():.4f}")

    # Test 3: Quantization pipeline
    print("\n3. Testing quantization pipeline:")
    batch_size = 4
    batch_coeffs = torch.randn(batch_size, 576) * 0.1  # Typical MDCT coefficient range

    # Test different scalefactor values
    for sf_val in [0, 7, 15]:
        scalefactors = torch.ones(batch_size, NUM_BANDS) * sf_val
        mp3 = DifferentiableMP3(frame_size)
        quantized = mp3.quantizer(batch_coeffs, scalefactors)

        # Check non-zero ratio
        nonzero_ratio = (quantized != 0).float().mean().item()

        # Check reconstruction quality (coefficients, not audio)
        coeff_snr = compute_snr(batch_coeffs, quantized).mean().item()

        print(f"   SF={sf_val:2d}: non-zero={nonzero_ratio*100:.1f}%, coeff SNR={coeff_snr:.1f} dB")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow:")
    batch_coeffs = torch.randn(batch_size, 576) * 0.1
    scalefactors = torch.rand(batch_size, NUM_BANDS) * 15
    scalefactors.requires_grad = True

    mp3 = DifferentiableMP3(frame_size)
    quantized = mp3.quantizer(batch_coeffs, scalefactors)
    loss = torch.mean((quantized - batch_coeffs) ** 2)
    loss.backward()

    nonzero_grads = (scalefactors.grad.abs() > 1e-12).sum().item()
    print(f"   Gradient exists: {scalefactors.grad is not None}")
    print(f"   Gradient mean: {scalefactors.grad.mean().item():.2e}")
    print(f"   Gradient abs max: {scalefactors.grad.abs().max().item():.2e}")
    print(f"   Non-zero gradients: {nonzero_grads}/{scalefactors.grad.numel()}")
    if nonzero_grads > 0:
        print("   ✓ Gradient flow is working!")

    print("\n" + "=" * 60)
    print("All tests complete!")
