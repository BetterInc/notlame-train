#!/usr/bin/env python3
"""Training loop for PsychoNet.

Trains the neural psychoacoustic model with perceptual losses.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import create_model, count_parameters, SCALEFACTOR_BANDS_BY_SR
from .differentiable_mp3 import DifferentiableMP3, DifferentiableMDCT, process_coeffs_through_model
from .losses import RateDistortionLoss, MultiResolutionSTFTLoss, MultiScaleMelLoss
from .dataset import create_train_val_dataloaders
from . import config

# Training sample rate - model architecture is tied to this
TRAIN_SAMPLE_RATE = config.MODEL_SAMPLE_RATE


class Trainer:
    """Training manager for PsychoNet."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        checkpoint_dir: Path = Path("checkpoints"),
        log_dir: Path = Path("runs"),
        experiment_name: Optional[str] = None,
        frames_per_sample: int = 4,
        sample_rate: int = TRAIN_SAMPLE_RATE,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.frames_per_sample = frames_per_sample
        self.sample_rate = sample_rate  # Store for checkpoint

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2,
        )

        # Loss functions
        # MDCT-domain rate-distortion loss
        self.rd_loss = RateDistortionLoss(
            rate_weight=0.0,  # Disable rate penalty - focus on quality
            use_perceptual_weights=True,
        ).to(self.device)

        # Perceptual losses (on audio domain)
        # Configuration from config.py ensures training/eval consistency
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=config.STFT_FFT_SIZES,
            hop_sizes=config.STFT_HOP_SIZES,
            win_sizes=config.STFT_WIN_SIZES,
        ).to(self.device)

        # Multi-scale mel loss (DAC-style)
        # Use training sample rate for mel computation
        self.mel_loss = MultiScaleMelLoss(
            sample_rate=self.sample_rate,
            window_lengths=config.MEL_WINDOW_LENGTHS,
            n_mels=config.MEL_N_MELS,
            use_l2=config.MEL_USE_L2,
        ).to(self.device)

        # Loss weights from config
        self.mdct_weight = config.LOSS_WEIGHTS["mdct"]
        self.stft_weight = config.LOSS_WEIGHTS["stft"]
        self.mel_weight = config.LOSS_WEIGHTS["mel"]

        # Rate penalty
        self.rate_weight = config.LOSS_WEIGHTS["rate"]
        self.target_sf = config.TARGET_SCALEFACTOR

        self.mp3_pipeline = DifferentiableMP3().to(self.device)
        self.mdct = DifferentiableMDCT().to(self.device)

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        print(f"Device: {self.device}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"TensorBoard: {self.log_dir}")

    def train_step(self, batch: torch.Tensor) -> dict:
        """Single training step with proper overlap-add reconstruction.

        Args:
            batch: (batch, num_frames, 576) MDCT coefficients

        Returns:
            dict of losses
        """
        self.model.train()
        batch = batch.to(self.device)

        # Use shared pipeline for proper overlap-add reconstruction
        reconstructed_audio, original_audio, all_scalefactors, all_quantized = \
            process_coeffs_through_model(batch, self.model, self.mdct, self.mp3_pipeline)

        # Handle single-frame for coefficients
        if batch.dim() == 2:
            batch = batch.unsqueeze(1)

        # MDCT-domain loss (average over frames)
        mdct_losses = []
        for i in range(all_quantized.shape[1]):
            rd_losses = self.rd_loss(all_quantized[:, i, :], batch[:, i, :], all_scalefactors[:, i, :])
            mdct_losses.append(rd_losses["distortion"])
        mdct_loss = torch.stack(mdct_losses).mean()

        # MR-STFT loss on properly reconstructed audio
        sc_loss, mag_loss = self.stft_loss(reconstructed_audio, original_audio)
        stft_loss = sc_loss + mag_loss

        # Mel spectrogram loss on properly reconstructed audio
        mel_loss = self.mel_loss(reconstructed_audio, original_audio)

        # Rate penalty
        sf_mean = all_scalefactors.mean()
        rate_penalty = torch.relu(self.target_sf - sf_mean)

        # Combined loss
        total_loss = (
            self.mdct_weight * mdct_loss +
            self.stft_weight * stft_loss +
            self.mel_weight * mel_loss +
            self.rate_weight * rate_penalty
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total": total_loss.item(),
            "mdct": mdct_loss.item(),
            "stft": stft_loss.item(),
            "mel": mel_loss.item(),
            "rate": rate_penalty.item(),
            "sf_mean": sf_mean.item(),
        }

    @torch.no_grad()
    def validate(self, max_batches: int = 100) -> dict:
        """Run validation on a subset of data.

        Args:
            max_batches: Maximum batches to validate (default 100 = 3200 samples)

        Returns:
            dict of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_mdct = 0.0
        total_stft = 0.0
        total_mel = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            # Use shared pipeline for proper overlap-add
            reconstructed_audio, original_audio, all_scalefactors, all_quantized = \
                process_coeffs_through_model(batch, self.model, self.mdct, self.mp3_pipeline)

            # Handle single-frame for coefficients
            if batch.dim() == 2:
                batch = batch.unsqueeze(1)

            # MDCT-domain loss
            mdct_losses = []
            for i in range(all_quantized.shape[1]):
                rd_losses = self.rd_loss(all_quantized[:, i, :], batch[:, i, :], all_scalefactors[:, i, :])
                mdct_losses.append(rd_losses["distortion"])
            mdct_loss = torch.stack(mdct_losses).mean()

            # Perceptual losses on properly reconstructed audio
            sc_loss, mag_loss = self.stft_loss(reconstructed_audio, original_audio)
            stft_loss = sc_loss + mag_loss
            mel_loss = self.mel_loss(reconstructed_audio, original_audio)

            combined_loss = (
                self.mdct_weight * mdct_loss +
                self.stft_weight * stft_loss +
                self.mel_weight * mel_loss
            )

            total_loss += combined_loss.item()
            total_mdct += mdct_loss.item()
            total_stft += stft_loss.item()
            total_mel += mel_loss.item()
            num_batches += 1

            # Early stop after max_batches for faster validation
            if num_batches >= max_batches:
                break

        n = max(num_batches, 1)
        return {
            "val_loss": total_loss / n,
            "val_mdct": total_mdct / n,
            "val_stft": total_stft / n,
            "val_mel": total_mel / n,
        }

    def _unwrap_model(self, model):
        """Unwrap model from DataParallel and torch.compile wrappers."""
        # Handle torch.compile wrapper
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        # Handle DataParallel wrapper
        if hasattr(model, 'module'):
            model = model.module
        # Handle nested case (compile wrapping DataParallel or vice versa)
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        if hasattr(model, 'module'):
            model = model.module
        return model

    def save_checkpoint(self, name: str = "checkpoint.pt"):
        """Save training checkpoint."""
        # Unwrap model from DataParallel/torch.compile for clean state dict
        model_to_save = self._unwrap_model(self.model)

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            # Sample rate metadata - important for inference
            "sample_rate": self.sample_rate,
            "scalefactor_bands": SCALEFACTOR_BANDS_BY_SR.get(self.sample_rate),
        }

        path = self.checkpoint_dir / name
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Unwrap model from DataParallel/torch.compile
        model_to_load = self._unwrap_model(self.model)

        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        # Load sample rate if available (for backwards compatibility)
        if "sample_rate" in checkpoint:
            loaded_sr = checkpoint["sample_rate"]
            if loaded_sr != self.sample_rate:
                print(f"Warning: Checkpoint trained at {loaded_sr}Hz, current training at {self.sample_rate}Hz")

        print(f"Loaded checkpoint: step={self.global_step}, epoch={self.epoch}")

    def train(
        self,
        num_steps: int = 100000,
        log_interval: int = 100,
        save_interval: int = 5000,
        val_interval: int = 1000,
    ):
        """Main training loop.

        Args:
            num_steps: Total training steps
            log_interval: Steps between logging
            save_interval: Steps between checkpoints
            val_interval: Steps between validation
        """
        print(f"\nStarting training for {num_steps} steps...")

        train_iter = iter(self.train_loader)
        running_loss = 0.0
        running_count = 0

        pbar = tqdm(range(self.global_step, num_steps), desc="Training")

        for step in pbar:
            self.global_step = step

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.epoch += 1

            # Train step
            losses = self.train_step(batch)

            running_loss += losses["total"]
            running_count += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total']:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Log to TensorBoard
            if step % log_interval == 0:
                avg_loss = running_loss / running_count
                running_loss = 0.0
                running_count = 0

                self.writer.add_scalar("train/loss", avg_loss, step)
                self.writer.add_scalar("train/mdct", losses["mdct"], step)
                self.writer.add_scalar("train/stft", losses["stft"], step)
                self.writer.add_scalar("train/mel", losses["mel"], step)
                self.writer.add_scalar("train/rate", losses["rate"], step)
                self.writer.add_scalar("train/sf_mean", losses["sf_mean"], step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], step)

            # Validation
            if step % val_interval == 0 and step > 0:
                val_metrics = self.validate()
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, step)

                # Update best checkpoint
                val_loss = val_metrics.get("val_loss", float("inf"))
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best.pt")
                    print(f"\n  New best: {val_loss:.4f}")

            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                self.save_checkpoint(f"step_{step}.pt")
                self.save_checkpoint("latest.pt")

        # Final save
        self.save_checkpoint("final.pt")
        print(f"\nTraining complete. Best loss: {self.best_loss:.4f}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PsychoNet model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python -m notlame_train.train --data-dir data/processed

  # Train with custom settings
  python -m notlame_train.train --data-dir data/processed \\
      --batch-size 64 --lr 3e-4 --steps 200000

  # Resume training
  python -m notlame_train.train --data-dir data/processed \\
      --resume checkpoints/latest.pt
        """,
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with processed .npy files",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )

    # Model
    parser.add_argument(
        "--model",
        choices=["default", "lite", "large"],
        default="default",
        help="Model variant (default: default)",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Training steps (default: 100000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data loading workers (default: 4)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Consecutive frames per sample for proper overlap-add (default: 4)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name for TensorBoard",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache all data in memory (faster, uses ~5GB RAM)",
    )

    # Sample rate
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=TRAIN_SAMPLE_RATE,
        choices=config.SUPPORTED_SAMPLE_RATES,
        help=f"Training sample rate (default: {TRAIN_SAMPLE_RATE}). "
             f"Supported: {config.SUPPORTED_SAMPLE_RATES}",
    )

    args = parser.parse_args()

    # Check data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Run prepare_dataset.py first")
        sys.exit(1)

    # Create data loaders
    print(f"Creating data loaders (frames_per_sample={args.frames})...")
    # When caching to GPU, use workers=0 (no CPU-GPU transfer needed)
    workers = 0 if args.cache else args.workers
    train_loader, val_loader = create_train_val_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=workers,
        cache_in_memory=args.cache,
        device=args.device if args.cache else None,
        frames_per_sample=args.frames,
    )

    # Enable TensorFloat32 for faster matmul on RTX 3090
    torch.set_float32_matmul_precision('high')

    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(args.model)

    # Compile model for faster training (PyTorch 2.0+)
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment,
        sample_rate=args.sample_rate,
    )

    # Resume if specified
    if args.resume:
        if args.resume.exists():
            trainer.load_checkpoint(args.resume)
        else:
            print(f"Warning: Checkpoint not found: {args.resume}")

    # Train
    trainer.train(num_steps=args.steps)


if __name__ == "__main__":
    main()
