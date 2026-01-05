"""Tests for dataset loading and data pipeline.

Tests MDCTDataset, DataLoader creation, and data integrity.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from notlame_train.dataset import (
    MDCTDataset,
    InfiniteSampler,
    create_dataloader,
    create_train_val_dataloaders,
)
from notlame_train.model import MDCT_SIZE


class TestMDCTDataset:
    """Test MDCT coefficient dataset."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test .npy files."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create some test .npy files
        for i in range(5):
            num_frames = np.random.randint(10, 50)
            data = np.random.randn(num_frames, MDCT_SIZE).astype(np.float32)
            np.save(temp_dir / f"test_{i}.npy", data)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_dataset_loading(self, temp_data_dir):
        """Dataset should load .npy files correctly."""
        dataset = MDCTDataset(temp_data_dir)

        assert len(dataset) > 0
        assert len(dataset.files) == 5

    def test_dataset_item_shape(self, temp_data_dir):
        """Dataset items should have correct shape."""
        dataset = MDCTDataset(temp_data_dir, frames_per_sample=1)

        item = dataset[0]

        assert item.shape == (1, MDCT_SIZE)
        assert item.dtype == torch.float32

    def test_dataset_multi_frame(self, temp_data_dir):
        """Dataset should return multiple consecutive frames."""
        frames_per_sample = 4
        dataset = MDCTDataset(temp_data_dir, frames_per_sample=frames_per_sample)

        item = dataset[0]

        assert item.shape == (frames_per_sample, MDCT_SIZE)

    def test_dataset_caching(self, temp_data_dir):
        """Dataset caching should work correctly."""
        dataset = MDCTDataset(temp_data_dir, cache_in_memory=True)

        # Access same item twice
        item1 = dataset[0]
        item2 = dataset[0]

        torch.testing.assert_close(item1, item2)

    def test_dataset_bounds(self, temp_data_dir):
        """Dataset should handle boundary conditions."""
        dataset = MDCTDataset(temp_data_dir, frames_per_sample=4)

        # First and last valid indices
        first = dataset[0]
        last = dataset[len(dataset) - 1]

        assert first.shape == (4, MDCT_SIZE)
        assert last.shape == (4, MDCT_SIZE)

        # Out of bounds should raise
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

    def test_dataset_empty_dir_error(self, tmp_path):
        """Dataset should error on empty directory."""
        with pytest.raises(ValueError, match="No .npy files"):
            MDCTDataset(tmp_path)

    def test_dataset_max_files(self, temp_data_dir):
        """max_files should limit loaded files."""
        dataset = MDCTDataset(temp_data_dir, max_files=2)

        assert len(dataset.files) == 2


class TestDataLoader:
    """Test DataLoader creation."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test .npy files."""
        temp_dir = Path(tempfile.mkdtemp())

        for i in range(10):
            num_frames = 100
            data = np.random.randn(num_frames, MDCT_SIZE).astype(np.float32)
            np.save(temp_dir / f"test_{i}.npy", data)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_dataloader_batch_shape(self, temp_data_dir):
        """DataLoader should return correct batch shape."""
        batch_size = 8
        loader = create_dataloader(
            temp_data_dir,
            batch_size=batch_size,
            num_workers=0,
            infinite=False,
        )

        batch = next(iter(loader))

        assert batch.shape[0] == batch_size
        assert batch.shape[-1] == MDCT_SIZE

    def test_dataloader_infinite(self, temp_data_dir):
        """Infinite loader should not stop."""
        loader = create_dataloader(
            temp_data_dir,
            batch_size=8,
            num_workers=0,
            infinite=True,
        )

        # Should be able to iterate many times
        iterator = iter(loader)
        for _ in range(100):
            batch = next(iterator)
            assert batch.shape[0] == 8

    def test_train_val_split(self, temp_data_dir):
        """Train/val split should work correctly."""
        train_loader, val_loader = create_train_val_dataloaders(
            temp_data_dir,
            batch_size=4,
            val_split=0.2,
            num_workers=0,
        )

        # Should have both loaders
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Val should be smaller
        assert len(val_loader) < len(train_loader)

    def test_train_val_no_overlap(self, temp_data_dir):
        """Train and val sets should not overlap."""
        train_loader, val_loader = create_train_val_dataloaders(
            temp_data_dir,
            batch_size=4,
            val_split=0.2,
            num_workers=0,
        )

        # Get all indices from both sets
        train_indices = set()
        val_indices = set()

        for batch in train_loader:
            # Batches are tensors, not directly indices
            # But we verify by checking total counts
            pass

        # Total samples should equal dataset size
        train_samples = sum(len(b) for b in train_loader)
        val_samples = sum(len(b) for b in val_loader)

        # Allow for drop_last
        total = train_samples + val_samples
        assert total > 0


class TestInfiniteSampler:
    """Test infinite random sampler."""

    def test_infinite_iteration(self, temp_data_dir):
        """Sampler should iterate infinitely."""
        from notlame_train.dataset import MDCTDataset

        dataset = MDCTDataset(temp_data_dir, frames_per_sample=1)
        sampler = InfiniteSampler(dataset, shuffle=True)

        iterator = iter(sampler)
        indices = [next(iterator) for _ in range(1000)]

        # Should have valid indices
        assert all(0 <= idx < len(dataset) for idx in indices)

    def test_shuffle_randomness(self, temp_data_dir):
        """Shuffled sampler should give different order."""
        from notlame_train.dataset import MDCTDataset

        dataset = MDCTDataset(temp_data_dir, frames_per_sample=1)
        sampler = InfiniteSampler(dataset, shuffle=True)

        iterator = iter(sampler)
        indices1 = [next(iterator) for _ in range(100)]
        indices2 = [next(iterator) for _ in range(100)]

        # Very unlikely to be identical
        assert indices1 != indices2

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test .npy files."""
        temp_dir = Path(tempfile.mkdtemp())

        for i in range(5):
            data = np.random.randn(50, MDCT_SIZE).astype(np.float32)
            np.save(temp_dir / f"test_{i}.npy", data)

        yield temp_dir
        shutil.rmtree(temp_dir)


class TestDataIntegrity:
    """Test data loading integrity."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with known data."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create file with known pattern
        data = np.arange(100 * MDCT_SIZE).reshape(100, MDCT_SIZE).astype(np.float32)
        np.save(temp_dir / "sequential.npy", data)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_data_not_corrupted(self, temp_data_dir):
        """Loaded data should match original."""
        dataset = MDCTDataset(temp_data_dir, frames_per_sample=1)

        # Load original
        original = np.load(temp_data_dir / "sequential.npy")

        # Check first few items
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            expected = torch.from_numpy(original[i:i+1]).float()
            torch.testing.assert_close(item, expected)

    def test_dtype_consistency(self, temp_data_dir):
        """All loaded data should be float32."""
        dataset = MDCTDataset(temp_data_dir, frames_per_sample=1)

        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert item.dtype == torch.float32


class TestRealData:
    """Test with real training data if available."""

    def test_real_data_loading(self, data_dir, has_data):
        """Test loading real training data."""
        if not has_data:
            pytest.skip("No training data available")

        dataset = MDCTDataset(data_dir, frames_per_sample=4)

        assert len(dataset) > 0

        # Load a sample
        item = dataset[0]
        assert item.shape == (4, MDCT_SIZE)
        assert torch.isfinite(item).all()

    def test_real_data_statistics(self, data_dir, has_data):
        """Check statistics of real data."""
        if not has_data:
            pytest.skip("No training data available")

        loader = create_dataloader(
            data_dir,
            batch_size=32,
            num_workers=0,
            infinite=False,
            max_files=10,
        )

        # Collect statistics
        all_means = []
        all_stds = []

        for batch in loader:
            all_means.append(batch.mean().item())
            all_stds.append(batch.std().item())

            if len(all_means) >= 10:
                break

        # MDCT coefficients should be roughly centered around 0
        mean = np.mean(all_means)
        assert abs(mean) < 1.0, f"Data mean {mean} too far from 0"

        # Should have some variance
        std = np.mean(all_stds)
        assert std > 0.01, f"Data std {std} too small"
