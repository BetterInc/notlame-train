#!/usr/bin/env python3
"""Validate audio dataset for training.

Scans audio files and checks:
- Format (WAV, FLAC, MP3 ≥256kbps)
- Sample rate (≥44.1kHz)
- Bit depth (≥16 bit)
- Duration (>5 sec)
- Clipping detection
- Silence detection (>50% = excluded)
- Corruption (file readability)

Outputs JSON report and terminal summary.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


@dataclass
class ValidationResult:
    """Result of validating a single audio file."""

    path: str
    valid: bool
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    format: str = ""
    bit_depth: Optional[int] = None
    bitrate_kbps: Optional[int] = None  # For MP3

    # Issues
    error: Optional[str] = None
    exclude_reason: Optional[str] = None
    warnings: list = field(default_factory=list)


def get_mp3_bitrate(filepath: Path) -> Optional[int]:
    """Get bitrate of MP3 file in kbps."""
    try:
        # Use pydub if available
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(str(filepath))
        # Estimate bitrate from file size and duration
        file_size_bits = filepath.stat().st_size * 8
        duration_sec = len(audio) / 1000.0
        if duration_sec > 0:
            return int(file_size_bits / duration_sec / 1000)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: read MP3 header
    try:
        with open(filepath, "rb") as f:
            # Find MP3 frame sync
            data = f.read(4096)
            for i in range(len(data) - 4):
                if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                    # Found frame sync
                    header = data[i : i + 4]
                    version = (header[1] >> 3) & 0x03
                    layer = (header[1] >> 1) & 0x03
                    bitrate_idx = (header[2] >> 4) & 0x0F

                    # Bitrate table for MPEG-1 Layer III
                    if version == 3 and layer == 1:  # MPEG-1 Layer 3
                        bitrates = [
                            0, 32, 40, 48, 56, 64, 80, 96,
                            112, 128, 160, 192, 224, 256, 320, 0
                        ]
                        if 0 < bitrate_idx < 15:
                            return bitrates[bitrate_idx]
                    break
    except Exception:
        pass

    return None


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """Detect if audio has clipping (samples at max amplitude)."""
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        max_val = np.max(np.abs(audio))
        return max_val >= threshold
    elif audio.dtype == np.int16:
        max_val = np.max(np.abs(audio))
        return max_val >= 32767 * threshold
    elif audio.dtype == np.int32:
        max_val = np.max(np.abs(audio))
        return max_val >= 2147483647 * threshold
    return False


def detect_silence_ratio(audio: np.ndarray, threshold_db: float = -60) -> float:
    """Calculate ratio of silence in audio."""
    if len(audio) == 0:
        return 1.0

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Calculate RMS in windows
    window_size = 1024
    threshold_linear = 10 ** (threshold_db / 20)

    silent_samples = 0
    for i in range(0, len(audio) - window_size, window_size):
        window = audio[i : i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        if rms < threshold_linear:
            silent_samples += window_size

    return silent_samples / len(audio)


def validate_file(filepath: Path) -> ValidationResult:
    """Validate a single audio file."""
    result = ValidationResult(path=str(filepath), valid=False)

    try:
        # Check file extension
        ext = filepath.suffix.lower()
        if ext not in [".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"]:
            result.error = f"Unsupported format: {ext}"
            result.exclude_reason = "unsupported_format"
            return result

        # For MP3, check bitrate first
        if ext == ".mp3":
            result.format = "MP3"
            bitrate = get_mp3_bitrate(filepath)
            result.bitrate_kbps = bitrate
            if bitrate and bitrate < 256:
                result.exclude_reason = "low_bitrate"
                result.error = f"Bitrate too low: {bitrate} kbps (min 256)"
                return result

        # Read audio file
        try:
            info = sf.info(str(filepath))
        except Exception as e:
            result.error = f"Cannot read file: {e}"
            result.exclude_reason = "corrupt"
            return result

        result.sample_rate = info.samplerate
        result.channels = info.channels
        result.duration = info.duration
        result.format = info.format

        # Get bit depth for PCM formats
        if info.subtype:
            if "PCM_16" in info.subtype:
                result.bit_depth = 16
            elif "PCM_24" in info.subtype:
                result.bit_depth = 24
            elif "PCM_32" in info.subtype:
                result.bit_depth = 32

        # Check sample rate (MP3 supports 22050, 24000, 32000, 44100, 48000 Hz)
        valid_sample_rates = [22050, 24000, 32000, 44100, 48000]
        if info.samplerate < 22050:
            result.exclude_reason = "low_sample_rate"
            result.error = f"Sample rate too low: {info.samplerate} Hz (min 22050)"
            return result

        # Check duration
        if info.duration < 5.0:
            result.exclude_reason = "too_short"
            result.error = f"Duration too short: {info.duration:.1f}s (min 5s)"
            return result

        # Read audio data for content analysis
        try:
            audio, _ = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            result.error = f"Cannot decode audio: {e}"
            result.exclude_reason = "corrupt"
            return result

        # Check for clipping
        if detect_clipping(audio):
            result.warnings.append("clipping_detected")

        # Check for silence
        silence_ratio = detect_silence_ratio(audio)
        if silence_ratio > 0.5:
            result.exclude_reason = "mostly_silent"
            result.error = f"Too much silence: {silence_ratio*100:.1f}% (max 50%)"
            return result

        # All checks passed
        result.valid = True
        return result

    except Exception as e:
        result.error = f"Unexpected error: {e}"
        result.exclude_reason = "error"
        return result


def scan_directory(
    input_dir: Path, extensions: list = None
) -> list[Path]:
    """Recursively find all audio files in directory."""
    if extensions is None:
        extensions = [".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"]

    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext.upper()}"))

    # Filter out macOS resource fork files (._*)
    files = [f for f in files if not f.name.startswith("._")]

    return sorted(set(files))


def validate_dataset(
    input_dir: Path,
    output_report: Path = None,
    max_workers: int = None,
) -> dict:
    """Validate all audio files in a directory."""

    print(f"Scanning: {input_dir}")
    files = scan_directory(input_dir)
    print(f"Found {len(files)} audio files")

    if not files:
        return {"valid": [], "excluded": {}, "warnings": {}, "stats": {}}

    # Validate files in parallel
    results = []
    max_workers = max_workers or min(os.cpu_count() or 4, 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_file, f): f for f in files}

        with tqdm(total=len(files), desc="Validating", unit="files") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    filepath = futures[future]
                    results.append(
                        ValidationResult(
                            path=str(filepath),
                            valid=False,
                            error=str(e),
                            exclude_reason="error",
                        )
                    )
                pbar.update(1)

    # Organize results
    valid_files = []
    excluded = {
        "low_bitrate": [],
        "low_sample_rate": [],
        "too_short": [],
        "mostly_silent": [],
        "corrupt": [],
        "unsupported_format": [],
        "error": [],
    }
    warnings = {
        "clipping_detected": [],
    }

    total_duration = 0.0
    lossless_count = 0

    for r in results:
        if r.valid:
            valid_files.append(r.path)
            total_duration += r.duration
            if r.format in ["WAV", "FLAC", "AIFF"]:
                lossless_count += 1
        elif r.exclude_reason:
            if r.exclude_reason in excluded:
                excluded[r.exclude_reason].append(r.path)
            else:
                excluded["error"].append(r.path)

        for w in r.warnings:
            if w in warnings:
                warnings[w].append(r.path)

    # Calculate stats
    stats = {
        "total_files": len(files),
        "valid_files": len(valid_files),
        "excluded_files": len(files) - len(valid_files),
        "total_hours": total_duration / 3600,
        "lossless_percent": (
            lossless_count / len(valid_files) * 100 if valid_files else 0
        ),
    }

    report = {
        "valid": valid_files,
        "excluded": {k: v for k, v in excluded.items() if v},
        "warnings": {k: v for k, v in warnings.items() if v},
        "stats": stats,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✓ Valid:          {stats['valid_files']:>6} ({stats['valid_files']/len(files)*100:.1f}%)")

    for reason, paths in excluded.items():
        if paths:
            symbol = "⚠" if reason == "clipping_detected" else "✗"
            label = reason.replace("_", " ").title()
            print(f"{symbol} {label}:  {len(paths):>6} ({len(paths)/len(files)*100:.1f}%) → excluded")

    for reason, paths in warnings.items():
        if paths:
            label = reason.replace("_", " ").title()
            print(f"⚠ {label}: {len(paths):>6} → warning only")

    print("-" * 60)
    print(f"Total valid:      {stats['valid_files']} files")
    print(f"Total hours:      {stats['total_hours']:.1f}")
    print(f"Lossless:         {stats['lossless_percent']:.1f}%")
    print("=" * 60)

    # Save report
    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_report}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate audio dataset for notlame training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate audio directory
  python -m notlame_train.validate_dataset --input /path/to/audio

  # Save report to custom location
  python -m notlame_train.validate_dataset --input data/raw --output my_report.json

  # Use more workers for faster processing
  python -m notlame_train.validate_dataset --input data/raw --workers 16
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("validation_report.json"),
        help="Output JSON report path (default: validation_report.json)",
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

    report = validate_dataset(args.input, args.output, args.workers)

    # Exit with error if no valid files
    if not report["valid"]:
        print("\nError: No valid files found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
