#!/usr/bin/env python3
"""Extract scalefactors from LAME-encoded MP3s for supervised training.

This creates training pairs: (MDCT coefficients) -> (LAME's scalefactor choices)
Much easier than learning from scratch!
"""

import argparse
import subprocess
import struct
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import soundfile as sf
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from notlame_train.model import SCALEFACTOR_BANDS_LONG, NUM_BANDS


def decode_mp3_frame_header(data: bytes, offset: int) -> Optional[dict]:
    """Decode MP3 frame header."""
    if offset + 4 > len(data):
        return None

    header = struct.unpack('>I', data[offset:offset+4])[0]

    # Check sync word (11 bits of 1s)
    if (header >> 21) != 0x7FF:
        return None

    # Parse header fields
    version = (header >> 19) & 0x3
    layer = (header >> 17) & 0x3
    bitrate_idx = (header >> 12) & 0xF
    sample_rate_idx = (header >> 10) & 0x3
    padding = (header >> 9) & 0x1

    # Bitrate table for MPEG1 Layer 3
    bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    sample_rates = [44100, 48000, 32000, 0]

    if version != 3 or layer != 1:  # MPEG1 Layer 3
        return None

    bitrate = bitrates[bitrate_idx] * 1000
    sample_rate = sample_rates[sample_rate_idx]

    if bitrate == 0 or sample_rate == 0:
        return None

    frame_size = (144 * bitrate // sample_rate) + padding

    return {
        'bitrate': bitrate,
        'sample_rate': sample_rate,
        'frame_size': frame_size,
        'padding': padding,
    }


def extract_scalefactors_from_mp3(mp3_path: Path) -> Optional[np.ndarray]:
    """Extract scalefactor decisions from MP3 file.

    Note: This is a simplified extraction. For full accuracy,
    you'd need to fully parse the MP3 bitstream.

    Returns approximate scalefactors based on bitrate allocation.
    """
    # For now, return None - full MP3 parsing is complex
    # We'll use a different approach: encode with LAME and analyze output quality
    return None


def create_lame_training_data(
    audio_dir: Path,
    output_dir: Path,
    bitrate: int = 192,
    max_files: int = None,
) -> Tuple[int, int]:
    """Create training data by analyzing LAME's encoding decisions.

    Approach: For each audio file:
    1. Encode with LAME at target bitrate
    2. Decode back to WAV
    3. Compute MDCT of original and decoded
    4. Infer "good" scalefactors from the quality difference per band

    This gives us target scalefactors that achieve LAME-like quality.
    """
    import torch
    from notlame_train.differentiable_mp3 import DifferentiableMDCT

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = []
    for ext in ['.wav', '.flac', '.mp3']:
        audio_files.extend(audio_dir.rglob(f'*{ext}'))
    audio_files = [f for f in audio_files if not f.name.startswith('._')]
    audio_files = sorted(set(audio_files))

    if max_files:
        audio_files = audio_files[:max_files]

    print(f"Processing {len(audio_files)} files...")

    mdct = DifferentiableMDCT()
    total_frames = 0
    saved_files = 0

    for audio_path in tqdm(audio_files, desc="Creating LAME training data"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Load original audio
                audio, sr = sf.read(str(audio_path))
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.astype(np.float32)

                # Normalize
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val

                # Save as WAV
                wav_path = tmpdir / "original.wav"
                sf.write(wav_path, audio, sr)

                # Encode with LAME
                mp3_path = tmpdir / "encoded.mp3"
                result = subprocess.run(
                    ["lame", "-b", str(bitrate), "-q", "0", "--quiet", str(wav_path), str(mp3_path)],
                    capture_output=True, timeout=60
                )
                if result.returncode != 0:
                    continue

                # Decode back
                decoded_path = tmpdir / "decoded.wav"
                result = subprocess.run(
                    ["lame", "--decode", "--quiet", str(mp3_path), str(decoded_path)],
                    capture_output=True, timeout=60
                )
                if result.returncode != 0:
                    continue

                # Load decoded
                decoded, _ = sf.read(str(decoded_path))
                if len(decoded.shape) > 1:
                    decoded = np.mean(decoded, axis=1)

                # Align lengths
                min_len = min(len(audio), len(decoded))
                audio = audio[:min_len]
                decoded = decoded[:min_len]

                # Compute MDCT frames
                frame_size = 1152
                hop_size = 576

                mdct_original = []
                target_scalefactors = []

                for i in range(0, len(audio) - frame_size, hop_size):
                    orig_frame = torch.from_numpy(audio[i:i+frame_size]).float().unsqueeze(0)
                    dec_frame = torch.from_numpy(decoded[i:i+frame_size]).float().unsqueeze(0)

                    orig_coeffs = mdct(orig_frame).squeeze(0).numpy()
                    dec_coeffs = mdct(dec_frame).squeeze(0).numpy()

                    # Compute per-band error
                    band_errors = []
                    for b in range(NUM_BANDS):
                        start, end = SCALEFACTOR_BANDS_LONG[b], SCALEFACTOR_BANDS_LONG[b+1]
                        orig_band = orig_coeffs[start:end]
                        dec_band = dec_coeffs[start:end]

                        # Relative error
                        band_energy = np.mean(orig_band ** 2) + 1e-10
                        band_error = np.mean((orig_band - dec_band) ** 2) / band_energy
                        band_errors.append(band_error)

                    band_errors = np.array(band_errors)

                    # Infer scalefactors from error (lower error = lower SF was used)
                    # This is approximate: SF ~ log2(error) mapped to [0, 15]
                    # Low error -> low SF (fine quantization)
                    # High error -> high SF (coarse quantization)
                    log_errors = np.log2(band_errors + 1e-10)
                    # Map to [0, 15]: typical log_errors range from -20 to 0
                    inferred_sf = np.clip((log_errors + 20) / 20 * 15, 0, 15)

                    mdct_original.append(orig_coeffs)
                    target_scalefactors.append(inferred_sf)

                if len(mdct_original) > 0:
                    # Save training data
                    mdct_array = np.stack(mdct_original).astype(np.float32)
                    sf_array = np.stack(target_scalefactors).astype(np.float32)

                    out_name = audio_path.stem
                    np.save(output_dir / f"{out_name}_mdct.npy", mdct_array)
                    np.save(output_dir / f"{out_name}_sf.npy", sf_array)

                    total_frames += len(mdct_original)
                    saved_files += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    print(f"\nCreated {total_frames} training frames from {saved_files} files")
    return saved_files, total_frames


def main():
    parser = argparse.ArgumentParser(description="Create LAME-supervised training data")
    parser.add_argument("--audio-dir", "-i", type=Path, required=True,
                        help="Directory with audio files")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for training data")
    parser.add_argument("--bitrate", "-b", type=int, default=192,
                        help="Target bitrate for LAME encoding")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum files to process")

    args = parser.parse_args()

    create_lame_training_data(
        args.audio_dir,
        args.output_dir,
        bitrate=args.bitrate,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
