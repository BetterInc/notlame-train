#!/usr/bin/env python3
"""Download audio datasets for training.

Downloads LibriSpeech (speech), FMA (music), and other audio datasets.
"""

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


# Dataset definitions
DATASETS = {
    # === SPEECH DATASETS ===
    "librispeech-clean-100": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "size_gb": 6.3,
        "hours": 100,
        "description": "100 hours of clean English speech (FLAC)",
        "md5": "2a93770f6d4c0f19f735242de78c8b35",
        "type": "speech",
        "format": "tar.gz",
    },
    "librispeech-clean-360": {
        "url": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "size_gb": 23.0,
        "hours": 360,
        "description": "360 hours of clean English speech (FLAC)",
        "md5": "3c6bc4a529d21a8e8a42c3a1d91bf5fb",
        "type": "speech",
        "format": "tar.gz",
    },
    "librispeech-dev-clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "size_gb": 0.35,
        "hours": 5,
        "description": "Development set - clean speech (FLAC)",
        "md5": "42e2234ba48799c1f50f24a7926300a1",
        "type": "speech",
        "format": "tar.gz",
    },
    # === MUSIC DATASETS ===
    "fma-small": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "size_gb": 7.2,
        "hours": 66,
        "description": "8000 tracks, 30s each, 8 genres (MP3 320k)",
        "type": "music",
        "format": "zip",
    },
    "fma-medium": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
        "size_gb": 22.0,
        "hours": 208,
        "description": "25000 tracks, 30s each, 16 genres (MP3 320k)",
        "type": "music",
        "format": "zip",
    },
    "fma-large": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
        "size_gb": 93.0,
        "hours": 879,
        "description": "106574 tracks, 30s each, 161 genres (MP3 320k)",
        "type": "music",
        "format": "zip",
    },
    "gtzan": {
        "url": "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz",
        "size_gb": 1.2,
        "hours": 8.3,
        "description": "1000 tracks, 30s each, 10 genres (WAV 22kHz)",
        "type": "music",
        "format": "tar.gz",
    },
    # High-quality 44.1kHz+ datasets
    "musdb-sample": {
        "url": "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1",
        "size_gb": 30.0,
        "hours": 10,
        "description": "150 songs, full stems, 44.1kHz WAV (large!)",
        "type": "music",
        "format": "zip",
    },
    "maestro-v3": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "size_gb": 0.1,
        "hours": 0,
        "description": "MIDI only (need audio separately)",
        "type": "music",
        "format": "zip",
    },
    # === QUICK TEST SAMPLES ===
    "test-samples": {
        "url": "INDIVIDUAL_FILES",
        "size_gb": 0.05,
        "hours": 0.1,
        "description": "Small set of test WAV files for quick testing",
        "type": "test",
        "format": "individual",
    },
}

# Individual test sample URLs (public domain / CC0)
TEST_SAMPLES = [
    # Kozco.com samples (reliable)
    {
        "url": "https://www.kozco.com/tech/organfinale.wav",
        "name": "organ.wav",
        "description": "Organ finale",
    },
    {
        "url": "https://www.kozco.com/tech/LRMonoPhase4.wav",
        "name": "stereo_test.wav",
        "description": "Stereo phase test",
    },
    # SampleLib.com samples
    {
        "url": "https://samplelib.com/lib/preview/wav/sample-3s.wav",
        "name": "sample_3s.wav",
        "description": "Music 3 seconds",
    },
    {
        "url": "https://samplelib.com/lib/preview/wav/sample-6s.wav",
        "name": "sample_6s.wav",
        "description": "Music 6 seconds",
    },
    {
        "url": "https://samplelib.com/lib/preview/wav/sample-9s.wav",
        "name": "sample_9s.wav",
        "description": "Music 9 seconds",
    },
    {
        "url": "https://samplelib.com/lib/preview/wav/sample-12s.wav",
        "name": "sample_12s.wav",
        "description": "Music 12 seconds",
    },
    {
        "url": "https://samplelib.com/lib/preview/wav/sample-15s.wav",
        "name": "sample_15s.wav",
        "description": "Music 15 seconds",
    },
]


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, output_path: Path, expected_md5: str = None) -> bool:
    """Download a file with progress bar and optional MD5 verification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if output_path.exists():
        if expected_md5:
            print(f"Verifying existing file: {output_path.name}")
            if get_md5(output_path) == expected_md5:
                print("  MD5 verified, skipping download")
                return True
            else:
                print("  MD5 mismatch, re-downloading")
                output_path.unlink()
        else:
            print(f"File exists, skipping: {output_path.name}")
            return True

    print(f"Downloading: {url}")
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=output_path.name
    ) as t:
        urlretrieve(url, output_path, reporthook=t.update_to)

    # Verify MD5
    if expected_md5:
        print("Verifying MD5...")
        actual_md5 = get_md5(output_path)
        if actual_md5 != expected_md5:
            print("  ERROR: MD5 mismatch!")
            print(f"  Expected: {expected_md5}")
            print(f"  Got: {actual_md5}")
            return False
        print("  MD5 verified")

    return True


def extract_tar(tar_path: Path, output_dir: Path) -> bool:
    """Extract a tar.gz archive."""
    print(f"Extracting: {tar_path.name}")

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Get total size for progress
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
                for member in members:
                    tar.extract(member, output_dir)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"  ERROR: Failed to extract: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract a zip archive."""
    print(f"Extracting: {zip_path.name}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
                for member in members:
                    zf.extract(member, output_dir)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"  ERROR: Failed to extract: {e}")
        return False


def download_test_samples(output_dir: Path) -> bool:
    """Download individual test sample WAV files."""
    print(f"\n{'='*60}")
    print("Downloading test samples...")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for sample in TEST_SAMPLES:
        output_path = output_dir / sample["name"]

        if output_path.exists():
            print(f"  Exists: {sample['name']}")
            success_count += 1
            continue

        try:
            print(f"  Downloading: {sample['name']} ({sample['description']})")
            urlretrieve(sample["url"], output_path)
            success_count += 1
        except Exception as e:
            print(f"    Failed: {e}")

    print(f"\nDownloaded {success_count}/{len(TEST_SAMPLES)} samples")
    return success_count > 0


def download_dataset(
    name: str, output_dir: Path, keep_archive: bool = False
) -> bool:
    """Download and extract a dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[name]

    # Handle individual test samples separately
    if info.get("format") == "individual":
        return download_test_samples(output_dir)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Description: {info['description']}")
    print(f"Size: {info['size_gb']} GB")
    print(f"Hours: {info['hours']}")
    print(f"Type: {info.get('type', 'unknown')}")
    print(f"{'='*60}\n")

    # Determine archive extension
    fmt = info.get("format", "tar.gz")
    if fmt == "zip":
        archive_path = output_dir / f"{name}.zip"
    else:
        archive_path = output_dir / f"{name}.tar.gz"

    # Download
    if not download_file(info["url"], archive_path, info.get("md5")):
        return False

    # Extract based on format
    if fmt == "zip":
        if not extract_zip(archive_path, output_dir):
            return False
    else:
        if not extract_tar(archive_path, output_dir):
            return False

    # Clean up
    if not keep_archive:
        print(f"Removing archive: {archive_path.name}")
        archive_path.unlink()

    print(f"Done: {name}")
    return True


def list_datasets():
    """Print available datasets."""
    print("\nAvailable datasets:")
    print("-" * 70)

    # Group by type
    types = {"speech": [], "music": [], "test": []}
    for name, info in DATASETS.items():
        t = info.get("type", "other")
        if t in types:
            types[t].append((name, info))

    for dtype, datasets in types.items():
        if datasets:
            print(f"\n  === {dtype.upper()} ===")
            for name, info in datasets:
                print(f"  {name}")
                print(f"    {info['description']}")
                print(f"    Size: {info['size_gb']} GB, Hours: {info['hours']}")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Download audio datasets for notlame training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download quick test samples (small WAV files)
  python download_data.py --datasets test-samples

  # Download GTZAN music dataset (1.2GB WAV, 8 hours)
  python download_data.py --datasets gtzan

  # Download FMA music dataset (7.2GB, 66 hours)
  python download_data.py --datasets fma-small

  # Download LibriSpeech speech (6.3GB, 100 hours)
  python download_data.py --datasets librispeech-clean-100

  # Download multiple datasets
  python download_data.py --datasets gtzan librispeech-dev-clean

  # List all available datasets
  python download_data.py --list

Recommended for testing:
  python download_data.py --datasets test-samples gtzan
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        default=["test-samples"],
        help="Datasets to download (default: test-samples)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep archives after extraction",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Download datasets
    success = True
    for dataset in args.datasets:
        if not download_dataset(dataset, args.output, args.keep_archive):
            success = False

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("All downloads completed successfully!")
        print(f"Data location: {args.output.absolute()}")

        # Count FLAC files
        flac_count = len(list(args.output.rglob("*.flac")))
        if flac_count > 0:
            print(f"Total FLAC files: {flac_count}")
    else:
        print("Some downloads failed. Check the output above.")
        exit(1)


if __name__ == "__main__":
    main()
