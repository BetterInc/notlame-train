"""PyTorch Dataset and DataLoader for training.

Loads pre-computed MDCT frames from .npy files.
"""

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class MDCTDataset(Dataset):
    """Dataset of MDCT frames for training.

    Loads .npy files containing MDCT coefficients.
    Each file contains (num_frames, 576) array.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_files: Optional[int] = None,
        frames_per_sample: int = 1,
        cache_in_memory: bool = False,
        device: str = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npy files
            max_files: Maximum number of files to use (for debugging)
            frames_per_sample: Number of consecutive frames per sample
            cache_in_memory: Load all data into memory (RAM or GPU)
            device: Device to cache on ('cuda' for GPU, None for RAM)
        """
        self.data_dir = Path(data_dir)
        self.frames_per_sample = frames_per_sample
        self.cache_in_memory = cache_in_memory
        self.device = device
        self.gpu_data = None

        # Find all .npy files
        self.files = sorted(self.data_dir.glob("*.npy"))
        if max_files:
            self.files = self.files[:max_files]

        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        # Build index: (file_idx, frame_idx) for each sample
        self.index = []
        self.file_lengths = []

        for file_idx, filepath in enumerate(self.files):
            # Get number of frames without loading full file
            data = np.load(filepath, mmap_mode="r")
            num_frames = len(data)
            self.file_lengths.append(num_frames)

            # Add index entries
            for frame_idx in range(num_frames - frames_per_sample + 1):
                self.index.append((file_idx, frame_idx))

        # Cache if requested
        self.cache = {}
        if cache_in_memory:
            if device and device.startswith('cuda'):
                # Load ALL data into a single GPU tensor
                print(f"Loading dataset into GPU ({device})...")
                all_frames = []
                for filepath in self.files:
                    all_frames.append(np.load(filepath))
                all_data = np.concatenate(all_frames, axis=0)
                self.gpu_data = torch.from_numpy(all_data).float().to(device)
                print(f"Loaded {len(self.gpu_data):,} frames to GPU ({self.gpu_data.element_size() * self.gpu_data.numel() / 1e9:.2f} GB)")

                # Rebuild index as simple offsets
                self.gpu_offsets = []
                offset = 0
                for length in self.file_lengths:
                    for i in range(length - frames_per_sample + 1):
                        self.gpu_offsets.append(offset + i)
                    offset += length
            else:
                print("Loading dataset into memory...")
                for file_idx, filepath in enumerate(self.files):
                    self.cache[file_idx] = np.load(filepath)
                print(f"Loaded {len(self.files)} files")

        print(f"Dataset: {len(self.files)} files, {len(self.index)} samples")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample."""
        # Fast path: data on GPU
        if self.gpu_data is not None:
            offset = self.gpu_offsets[idx]
            return self.gpu_data[offset:offset + self.frames_per_sample]

        # RAM cache path
        file_idx, frame_idx = self.index[idx]

        if file_idx in self.cache:
            data = self.cache[file_idx]
        else:
            data = np.load(self.files[file_idx])

        frames = data[frame_idx : frame_idx + self.frames_per_sample]
        return torch.from_numpy(frames.copy()).float()


class AudioDataset(Dataset):
    """Dataset that loads raw audio files.

    For training with full audio reconstruction.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 44100,
        segment_length: int = 32768,  # ~0.74 seconds
        max_files: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            segment_length: Length of audio segments
            max_files: Maximum files to use
        """
        import soundfile as sf

        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        # Find audio files
        extensions = [".wav", ".flac", ".mp3"]
        self.files = []
        for ext in extensions:
            self.files.extend(self.data_dir.rglob(f"*{ext}"))
            self.files.extend(self.data_dir.rglob(f"*{ext.upper()}"))

        self.files = sorted(set(self.files))
        if max_files:
            self.files = self.files[:max_files]

        if not self.files:
            raise ValueError(f"No audio files found in {data_dir}")

        # Build index with file lengths
        self.index = []
        for file_idx, filepath in enumerate(self.files):
            try:
                info = sf.info(str(filepath))
                duration_samples = int(info.duration * sample_rate)

                # Add segments
                for start in range(0, duration_samples - segment_length, segment_length // 2):
                    self.index.append((file_idx, start))
            except Exception:
                continue

        print(f"AudioDataset: {len(self.files)} files, {len(self.index)} segments")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get an audio segment.

        Args:
            idx: Segment index

        Returns:
            (segment_length,) tensor of audio samples
        """
        import soundfile as sf
        from scipy import signal

        file_idx, start_sample = self.index[idx]
        filepath = self.files[file_idx]

        # Load audio segment
        audio, sr = sf.read(
            str(filepath),
            start=start_sample,
            stop=start_sample + self.segment_length,
            dtype="float32",
        )

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        # Pad if needed
        if len(audio) < self.segment_length:
            audio = np.pad(audio, (0, self.segment_length - len(audio)))

        # Truncate if needed
        audio = audio[: self.segment_length]

        return torch.from_numpy(audio).float()


class InfiniteSampler(Sampler):
    """Infinite random sampler for training."""

    def __init__(self, data_source: Dataset, shuffle: bool = True):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            if self.shuffle:
                yield random.randint(0, len(self.data_source) - 1)
            else:
                for i in range(len(self.data_source)):
                    yield i

    def __len__(self):
        return 2 ** 31  # Effectively infinite


def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    frames_per_sample: int = 1,
    cache_in_memory: bool = False,
    infinite: bool = True,
    max_files: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for training.

    Args:
        data_dir: Directory with .npy MDCT files
        batch_size: Batch size
        num_workers: Number of data loading workers
        frames_per_sample: Frames per sample
        cache_in_memory: Cache data in memory
        infinite: Use infinite sampler
        max_files: Max files to use

    Returns:
        DataLoader instance
    """
    dataset = MDCTDataset(
        data_dir=data_dir,
        max_files=max_files,
        frames_per_sample=frames_per_sample,
        cache_in_memory=cache_in_memory,
    )

    if infinite:
        sampler = InfiniteSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


def create_train_val_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    device: str = None,
    frames_per_sample: int = 1,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        data_dir: Directory with .npy files
        batch_size: Batch size
        val_split: Fraction for validation
        num_workers: Number of workers
        device: Device for GPU caching (None for CPU)
        frames_per_sample: Consecutive frames per sample (for overlap-add training)
        **kwargs: Additional args for MDCTDataset

    Returns:
        (train_loader, val_loader)
    """
    # Create full dataset
    dataset = MDCTDataset(data_dir=data_dir, device=device, frames_per_sample=frames_per_sample, **kwargs)

    # Split
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Don't use pin_memory if data is already on GPU
    use_pin_memory = device is None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_loader, val_loader
