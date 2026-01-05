#!/usr/bin/env python3
"""Prepare audio dataset for training.

Converts audio files to MDCT frames and saves as .npy files.
MP3 uses 576 samples per granule with 50% overlap (1152 sample frames).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from notlame_train.model import FRAME_SIZE, HOP_SIZE

TARGET_SR = 44100


def create_mdct_window(n: int) -> np.ndarray:
    """Create MDCT sine window."""
    return np.sin(np.pi / n * (np.arange(n) + 0.5))


def mdct(x: np.ndarray, window: np.ndarray = None) -> np.ndarray:
    """Compute Modified Discrete Cosine Transform.

    Standard MDCT formula matching differentiable_mp3.py:
    X[k] = sum_{n=0}^{N-1} w[n] * x[n] * cos(π/N * (2n + 1 + N/2) * (2k + 1) / 2)

    Args:
        x: Input signal of length N (should be 2*M where M is output length)
        window: Window function of length N

    Returns:
        MDCT coefficients of length N/2 (M = N/2)
    """
    N = len(x)
    M = N // 2

    if window is not None:
        x = x * window

    # Standard MDCT formula: cos(π/N * (2n + 1 + N/2) * (2k + 1) / 2)
    n = np.arange(N)
    k = np.arange(M)

    # Compute transform - NO extra scaling needed for training
    # (scaling only matters for reconstruction, not for learning)
    cos_term = np.cos(np.pi / N * np.outer(2 * n + 1 + N / 2, 2 * k + 1) / 2)
    return np.sum(x[:, None] * cos_term, axis=0)


def process_audio_to_mdct(
    audio: np.ndarray,
    sample_rate: int,
    window: np.ndarray,
) -> np.ndarray:
    """Convert audio to MDCT frames.

    Args:
        audio: Audio samples (mono, float32)
        sample_rate: Sample rate (should be 44100)
        window: MDCT window

    Returns:
        Array of shape (num_frames, 576) containing MDCT coefficients
    """
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sample_rate != TARGET_SR:
        from scipy import signal
        num_samples = int(len(audio) * TARGET_SR / sample_rate)
        audio = signal.resample(audio, num_samples)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Pad to ensure we have complete frames
    pad_length = FRAME_SIZE - (len(audio) % HOP_SIZE)
    if pad_length < FRAME_SIZE:
        audio = np.pad(audio, (0, pad_length))

    # Extract MDCT frames
    num_frames = (len(audio) - FRAME_SIZE) // HOP_SIZE + 1
    frames = []

    for i in range(num_frames):
        start = i * HOP_SIZE
        frame = audio[start : start + FRAME_SIZE]

        if len(frame) < FRAME_SIZE:
            break

        # Apply MDCT (outputs 576 coefficients from 1152 input)
        coeffs = mdct(frame, window)
        frames.append(coeffs)

    if not frames:
        return np.array([])

    return np.array(frames, dtype=np.float32)


def process_file(
    filepath: Path,
    output_dir: Path,
    window: np.ndarray,
    skip_existing: bool = True,
) -> Optional[dict]:
    """Process a single audio file to MDCT frames.

    Returns dict with stats or None on error.
    """
    # Check if already processed
    rel_path = filepath.stem
    output_path = output_dir / f"{rel_path}.npy"

    if skip_existing and output_path.exists():
        # Load existing to get stats
        try:
            existing = np.load(output_path)
            return {
                "input": str(filepath),
                "output": str(output_path),
                "num_frames": len(existing),
                "duration": len(existing) * 576 / 44100,
                "skipped": True,
            }
        except Exception:
            pass  # Re-process if can't load

    try:
        # Read audio
        audio, sr = sf.read(str(filepath), dtype="float32")

        # Convert to MDCT
        frames = process_audio_to_mdct(audio, sr, window)

        if len(frames) == 0:
            return None

        # Handle duplicates (if same name from different dirs)
        if output_path.exists():
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{rel_path}_{counter}.npy"
                counter += 1

        # Save
        np.save(output_path, frames)

        return {
            "input": str(filepath),
            "output": str(output_path),
            "num_frames": len(frames),
            "duration": len(frames) * HOP_SIZE / TARGET_SR,
        }

    except Exception as e:
        return {"input": str(filepath), "error": str(e)}


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    valid_files: list[str] = None,
    max_workers: int = None,
) -> dict:
    """Process all audio files to MDCT format.

    Args:
        input_dir: Directory with audio files
        output_dir: Output directory for .npy files
        valid_files: Optional list of valid file paths (from validation)
        max_workers: Number of parallel workers

    Returns:
        Stats dictionary
    """
    # Create MDCT window
    window = create_mdct_window(FRAME_SIZE)

    # Get files to process
    if valid_files:
        files = [Path(f) for f in valid_files]
    else:
        # Scan for audio files
        extensions = [".wav", ".flac", ".mp3", ".ogg", ".aiff"]
        files = []
        for ext in extensions:
            files.extend(input_dir.rglob(f"*{ext}"))
            files.extend(input_dir.rglob(f"*{ext.upper()}"))
        files = sorted(set(files))

    print(f"Processing {len(files)} audio files")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files in parallel
    max_workers = max_workers or min(os.cpu_count() or 4, 8)
    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, output_dir, window): f
            for f in files
        }

        with tqdm(total=len(files), desc="Processing", unit="files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    if "error" in result:
                        errors.append(result)
                    else:
                        results.append(result)
                pbar.update(1)

    # Calculate stats
    total_frames = sum(r["num_frames"] for r in results)
    total_duration = sum(r["duration"] for r in results)
    skipped_count = sum(1 for r in results if r.get("skipped"))
    new_count = len(results) - skipped_count

    stats = {
        "processed_files": len(results),
        "new_files": new_count,
        "skipped_files": skipped_count,
        "failed_files": len(errors),
        "total_frames": total_frames,
        "total_hours": total_duration / 3600,
        "output_dir": str(output_dir),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    print(f"New files:        {stats['new_files']}")
    print(f"Skipped (exist):  {stats['skipped_files']}")
    print(f"Failed files:     {stats['failed_files']}")
    print(f"Total frames:     {stats['total_frames']:,}")
    print(f"Total hours:      {stats['total_hours']:.1f}")
    print(f"Output directory: {stats['output_dir']}")
    print("=" * 60)

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e['input']}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio dataset for training (WAV → MDCT .npy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all audio in directory
  python scripts/prepare_dataset.py --input data/raw --output data/processed

  # Use validation report to only process valid files
  python scripts/prepare_dataset.py --input data/raw --output data/processed \\
      --validation-report validation_report.json

  # Use more workers
  python scripts/prepare_dataset.py --input data/raw --output data/processed --workers 16
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory with audio files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for .npy files (default: data/processed)",
    )
    parser.add_argument(
        "--validation-report",
        "-v",
        type=Path,
        help="Validation report JSON to use only valid files",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Load validation report if provided
    valid_files = None
    if args.validation_report:
        if not args.validation_report.exists():
            print(f"Error: Validation report not found: {args.validation_report}")
            sys.exit(1)

        with open(args.validation_report) as f:
            report = json.load(f)
            valid_files = report.get("valid", [])
            print(f"Using {len(valid_files)} valid files from validation report")

    stats = prepare_dataset(
        args.input,
        args.output,
        valid_files,
        args.workers,
    )

    # Save stats
    stats_path = args.output / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
