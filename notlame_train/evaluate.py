#!/usr/bin/env python3
"""Evaluate trained PsychoNet model.

Research-based evaluation metrics:
- ViSQOL (Google's perceptual metric, MOS 1-5)
- Multi-resolution STFT distance
- Mel spectrogram distance
- SNR (Signal-to-Noise Ratio)
- A/B comparison vs LAME

Based on evaluation methodologies from:
- EnCodec (Meta): https://arxiv.org/abs/2210.13438
- SoundStream (Google): https://arxiv.org/abs/2107.03312
- ViSQOL v3: https://arxiv.org/abs/2004.09584
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .model import create_model
from .differentiable_mp3 import DifferentiableMP3, DifferentiableMDCT, process_audio_through_model
from . import config


# =============================================================================
# Target Metrics (from config)
# =============================================================================
TARGETS = config.EVAL_TARGETS


# =============================================================================
# Basic Metrics
# =============================================================================

def compute_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    # Align lengths
    min_len = min(len(reference), len(degraded))
    ref = reference[:min_len]
    deg = degraded[:min_len]

    signal_power = np.mean(ref ** 2)
    noise_power = np.mean((ref - deg) ** 2)

    if noise_power < 1e-10:
        return 100.0
    return 10 * np.log10(signal_power / (noise_power + 1e-10))


def compute_spectral_convergence(reference: np.ndarray, degraded: np.ndarray,
                                  n_fft: int = 1024, hop: int = 256) -> float:
    """Spectral convergence: ||S_ref - S_deg||_F / ||S_ref||_F"""
    from scipy import signal

    min_len = min(len(reference), len(degraded))
    ref = reference[:min_len]
    deg = degraded[:min_len]

    _, _, S_ref = signal.stft(ref, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, S_deg = signal.stft(deg, nperseg=n_fft, noverlap=n_fft-hop)

    S_ref_mag = np.abs(S_ref)
    S_deg_mag = np.abs(S_deg)

    return np.linalg.norm(S_ref_mag - S_deg_mag) / (np.linalg.norm(S_ref_mag) + 1e-8)


def compute_log_spectral_distance(reference: np.ndarray, degraded: np.ndarray,
                                   n_fft: int = 1024, hop: int = 256) -> float:
    """Log spectral distance (LSD)."""
    from scipy import signal

    min_len = min(len(reference), len(degraded))
    ref = reference[:min_len]
    deg = degraded[:min_len]

    _, _, S_ref = signal.stft(ref, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, S_deg = signal.stft(deg, nperseg=n_fft, noverlap=n_fft-hop)

    # Log magnitude (with floor to avoid log(0))
    log_ref = np.log10(np.abs(S_ref) + 1e-8)
    log_deg = np.log10(np.abs(S_deg) + 1e-8)

    return np.mean(np.abs(log_ref - log_deg))


# =============================================================================
# Multi-Resolution STFT (from EnCodec/SoundStream papers)
# =============================================================================

def compute_multi_resolution_stft(reference: np.ndarray, degraded: np.ndarray,
                                   fft_sizes: List[int] = None,
                                   hop_sizes: List[int] = None) -> Dict[str, float]:
    """Multi-resolution STFT distance.

    Combines spectral convergence and log magnitude loss at multiple resolutions.
    Used by EnCodec, SoundStream, DAC for training and evaluation.
    """
    # Use config defaults if not specified
    if fft_sizes is None:
        fft_sizes = config.STFT_FFT_SIZES
    if hop_sizes is None:
        hop_sizes = config.STFT_HOP_SIZES
    from scipy import signal

    min_len = min(len(reference), len(degraded))
    ref = reference[:min_len]
    deg = degraded[:min_len]

    sc_total = 0.0
    mag_total = 0.0

    for n_fft, hop in zip(fft_sizes, hop_sizes):
        _, _, S_ref = signal.stft(ref, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, S_deg = signal.stft(deg, nperseg=n_fft, noverlap=n_fft-hop)

        S_ref_mag = np.abs(S_ref)
        S_deg_mag = np.abs(S_deg)

        # Spectral convergence
        sc = np.linalg.norm(S_ref_mag - S_deg_mag) / (np.linalg.norm(S_ref_mag) + 1e-8)
        sc_total += sc

        # Log magnitude distance
        log_ref = np.log(S_ref_mag + 1e-8)
        log_deg = np.log(S_deg_mag + 1e-8)
        mag = np.mean(np.abs(log_ref - log_deg))
        mag_total += mag

    n = len(fft_sizes)
    return {
        "sc": sc_total / n,
        "mag": mag_total / n,
        "total": (sc_total + mag_total) / n,
    }


# =============================================================================
# Mel Spectrogram Distance
# =============================================================================

def compute_mel_distance(reference: np.ndarray, degraded: np.ndarray,
                         sr: int = None, n_mels: int = None,
                         n_fft: int = None, hop: int = None) -> float:
    """Mel spectrogram L1 distance.

    Perceptually-weighted metric used in neural audio codec evaluation.
    Uses largest window from config for evaluation (best frequency resolution).
    """
    # Use config defaults
    if sr is None:
        sr = config.MEL_SAMPLE_RATE
    if n_mels is None:
        n_mels = config.MEL_N_MELS
    if n_fft is None:
        n_fft = max(config.MEL_WINDOW_LENGTHS)  # Largest window for eval
    if hop is None:
        hop = n_fft // 4
    try:
        import librosa

        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]

        # Compute mel spectrograms
        mel_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_fft=n_fft,
                                                  hop_length=hop, n_mels=n_mels)
        mel_deg = librosa.feature.melspectrogram(y=deg, sr=sr, n_fft=n_fft,
                                                  hop_length=hop, n_mels=n_mels)

        # Log scale
        log_mel_ref = np.log(mel_ref + 1e-8)
        log_mel_deg = np.log(mel_deg + 1e-8)

        return np.mean(np.abs(log_mel_ref - log_mel_deg))

    except ImportError:
        # Fallback without librosa
        return compute_log_spectral_distance(reference, degraded, n_fft, hop)


# =============================================================================
# ViSQOL (Google's perceptual metric)
# =============================================================================

def compute_visqol(reference_path: str, degraded_path: str,
                   mode: str = "audio") -> Optional[float]:
    """Compute ViSQOL MOS score (1-5).

    Requires: pip install visqol
    Audio mode requires 48kHz, speech mode requires 16kHz.
    """
    try:
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        from visqol.pb2 import similarity_result_pb2

        config = visqol_config_pb2.VisqolConfig()

        if mode == "speech":
            config.audio.sample_rate = 16000
            config.options.use_speech_scoring = True
            svr_model = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        else:
            config.audio.sample_rate = 48000
            config.options.use_speech_scoring = False
            svr_model = "libsvm_nu_svr_model.txt"

        import visqol
        config.options.svr_model_path = os.path.join(
            os.path.dirname(visqol.__file__), "model", svr_model
        )

        api = visqol_lib_py.VisqolApi()
        api.Create(config)

        result = api.Measure(reference_path, degraded_path)
        return result.moslqo

    except ImportError:
        return None
    except Exception as e:
        print(f"  ViSQOL error: {e}")
        return None


def resample_for_visqol(audio: np.ndarray, sr_in: int, sr_out: int = 48000) -> np.ndarray:
    """Resample audio for ViSQOL (requires 48kHz for audio mode)."""
    if sr_in == sr_out:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)
    except ImportError:
        from scipy import signal
        num_samples = int(len(audio) * sr_out / sr_in)
        return signal.resample(audio, num_samples)


# =============================================================================
# LAME Encoding/Decoding
# =============================================================================

def encode_with_lame(input_path: str, output_path: str, bitrate: int = 192) -> bool:
    """Encode audio with LAME MP3 encoder."""
    try:
        result = subprocess.run(
            ["lame", "-b", str(bitrate), "-q", "0", "--quiet", input_path, output_path],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("Warning: LAME not found. Install with: apt install lame")
        return False
    except Exception as e:
        print(f"LAME error: {e}")
        return False


def decode_mp3(input_path: str, output_path: str) -> bool:
    """Decode MP3 to WAV using LAME."""
    try:
        result = subprocess.run(
            ["lame", "--decode", "--quiet", input_path, output_path],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Decode error: {e}")
        return False


# =============================================================================
# Model Evaluator
# =============================================================================

class Evaluator:
    """Neural MP3 model evaluator."""

    def __init__(
        self,
        model_path: Path,
        model_variant: str = "default",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = create_model(model_variant)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle torch.compile() wrapped models - strip _orig_mod. prefix if present
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Pipeline components
        self.mdct = DifferentiableMDCT().to(self.device)
        self.mp3 = DifferentiableMP3().to(self.device)

        print(f"Loaded model from {model_path} (device: {self.device})")

    @torch.no_grad()
    def encode_decode(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Encode audio with neural model and decode back."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Use shared pipeline for proper overlap-add reconstruction
        reconstructed, _, _, _, _ = process_audio_through_model(
            audio_tensor, self.model, self.mdct, self.mp3
        )

        return reconstructed[0].cpu().numpy()

    def evaluate_file(self, audio_path: Path, bitrate: int = 192) -> dict:
        """Evaluate on a single audio file with all metrics."""
        # Load audio
        audio, sr = sf.read(str(audio_path))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono

        # Normalize
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        result = {
            "file": str(audio_path.name),
            "duration": len(audio) / sr,
            "sample_rate": sr,
        }

        # Encode with our model
        try:
            notlame_audio = self.encode_decode(audio, sr)
        except Exception as e:
            result["error"] = f"Model error: {e}"
            return result

        # Basic metrics for notlame
        result["notlame"] = {
            "snr": compute_snr(audio, notlame_audio),
            "mr_stft": compute_multi_resolution_stft(audio, notlame_audio)["total"],
            "mel": compute_mel_distance(audio, notlame_audio, sr),
        }

        # Compare with LAME
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save files
            wav_path = tmpdir / "original.wav"
            mp3_path = tmpdir / "lame.mp3"
            lame_wav_path = tmpdir / "lame_decoded.wav"
            notlame_wav_path = tmpdir / "notlame.wav"

            sf.write(wav_path, audio, sr)
            sf.write(notlame_wav_path, notlame_audio, sr)

            # LAME encode/decode
            if encode_with_lame(str(wav_path), str(mp3_path), bitrate):
                if decode_mp3(str(mp3_path), str(lame_wav_path)):
                    lame_audio, _ = sf.read(str(lame_wav_path))
                    if len(lame_audio.shape) > 1:
                        lame_audio = np.mean(lame_audio, axis=1)

                    # LAME metrics
                    result["lame"] = {
                        "snr": compute_snr(audio, lame_audio),
                        "mr_stft": compute_multi_resolution_stft(audio, lame_audio)["total"],
                        "mel": compute_mel_distance(audio, lame_audio, sr),
                    }

                    # Winners (notlame wins if better)
                    result["winner"] = {
                        "snr": "notlame" if result["notlame"]["snr"] > result["lame"]["snr"] else "lame",
                        "mr_stft": "notlame" if result["notlame"]["mr_stft"] < result["lame"]["mr_stft"] else "lame",
                        "mel": "notlame" if result["notlame"]["mel"] < result["lame"]["mel"] else "lame",
                    }

            # ViSQOL (requires 48kHz resampling)
            try:
                # Resample to 48kHz for ViSQOL
                audio_48k = resample_for_visqol(audio, sr, 48000)
                notlame_48k = resample_for_visqol(notlame_audio, sr, 48000)

                ref_48k_path = tmpdir / "ref_48k.wav"
                notlame_48k_path = tmpdir / "notlame_48k.wav"

                sf.write(ref_48k_path, audio_48k, 48000)
                sf.write(notlame_48k_path, notlame_48k, 48000)

                visqol_notlame = compute_visqol(str(ref_48k_path), str(notlame_48k_path))
                if visqol_notlame is not None:
                    result["notlame"]["visqol"] = visqol_notlame

                # ViSQOL for LAME
                if lame_wav_path.exists() and visqol_notlame is not None:
                    lame_audio_full, _ = sf.read(str(lame_wav_path))
                    if len(lame_audio_full.shape) > 1:
                        lame_audio_full = np.mean(lame_audio_full, axis=1)
                    lame_48k = resample_for_visqol(lame_audio_full, sr, 48000)
                    lame_48k_path = tmpdir / "lame_48k.wav"
                    sf.write(lame_48k_path, lame_48k, 48000)

                    visqol_lame = compute_visqol(str(ref_48k_path), str(lame_48k_path))
                    if visqol_lame is not None:
                        result["lame"]["visqol"] = visqol_lame
                        result["winner"]["visqol"] = "notlame" if visqol_notlame > visqol_lame else "lame"

            except Exception as e:
                pass  # ViSQOL is optional

        return result


def evaluate_directory(evaluator: Evaluator, test_dir: Path,
                       bitrate: int = 192, max_files: int = None) -> dict:
    """Evaluate all audio files in a directory."""
    # Find audio files
    extensions = [".wav", ".flac", ".mp3", ".ogg"]
    files = []
    for ext in extensions:
        files.extend(test_dir.rglob(f"*{ext}"))
        files.extend(test_dir.rglob(f"*{ext.upper()}"))

    # Filter out macOS resource forks
    files = [f for f in files if not f.name.startswith("._")]
    files = sorted(set(files))

    if max_files:
        files = files[:max_files]

    if not files:
        print(f"No audio files found in {test_dir}")
        return {"error": "No files found"}

    print(f"\nEvaluating {len(files)} files at {bitrate} kbps...")

    results = []
    for filepath in tqdm(files, desc="Evaluating"):
        try:
            result = evaluator.evaluate_file(filepath, bitrate)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")

    # Aggregate statistics
    return aggregate_results(results)


def aggregate_results(results: List[dict]) -> dict:
    """Compute aggregate statistics from individual results."""
    def safe_mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    def safe_std(values):
        values = [v for v in values if v is not None]
        return float(np.std(values)) if len(values) > 1 else None

    # Extract metrics
    notlame_snr = [r["notlame"]["snr"] for r in results if "notlame" in r and "snr" in r["notlame"]]
    notlame_stft = [r["notlame"]["mr_stft"] for r in results if "notlame" in r and "mr_stft" in r["notlame"]]
    notlame_mel = [r["notlame"]["mel"] for r in results if "notlame" in r and "mel" in r["notlame"]]
    notlame_visqol = [r["notlame"]["visqol"] for r in results if "notlame" in r and "visqol" in r.get("notlame", {})]

    lame_snr = [r["lame"]["snr"] for r in results if "lame" in r and "snr" in r["lame"]]
    lame_stft = [r["lame"]["mr_stft"] for r in results if "lame" in r and "mr_stft" in r["lame"]]
    lame_mel = [r["lame"]["mel"] for r in results if "lame" in r and "mel" in r["lame"]]
    lame_visqol = [r["lame"]["visqol"] for r in results if "lame" in r and "visqol" in r.get("lame", {})]

    # Win rates
    def win_rate(metric, higher_better=True):
        wins = 0
        total = 0
        for r in results:
            if "winner" in r and metric in r["winner"]:
                total += 1
                if r["winner"][metric] == "notlame":
                    wins += 1
        return wins / total if total > 0 else None

    stats = {
        "num_files": len(results),
        "notlame": {
            "snr_mean": safe_mean(notlame_snr),
            "snr_std": safe_std(notlame_snr),
            "mr_stft_mean": safe_mean(notlame_stft),
            "mel_mean": safe_mean(notlame_mel),
            "visqol_mean": safe_mean(notlame_visqol),
        },
        "lame": {
            "snr_mean": safe_mean(lame_snr),
            "snr_std": safe_std(lame_snr),
            "mr_stft_mean": safe_mean(lame_stft),
            "mel_mean": safe_mean(lame_mel),
            "visqol_mean": safe_mean(lame_visqol),
        },
        "win_rates": {
            "snr": win_rate("snr", higher_better=True),
            "mr_stft": win_rate("mr_stft", higher_better=False),
            "mel": win_rate("mel", higher_better=False),
            "visqol": win_rate("visqol", higher_better=True),
        },
    }

    return {"results": results, "stats": stats}


def print_report(report: dict):
    """Print formatted evaluation report."""
    stats = report.get("stats", {})

    print("\n" + "=" * 70)
    print("                    EVALUATION REPORT")
    print("=" * 70)
    print(f"Files evaluated: {stats.get('num_files', 0)}")

    # Metrics comparison table
    print("\n" + "-" * 70)
    print(f"{'Metric':<20} {'notlame':>12} {'LAME':>12} {'Winner':>12} {'Target':>12}")
    print("-" * 70)

    notlame = stats.get("notlame", {})
    lame = stats.get("lame", {})
    win_rates = stats.get("win_rates", {})

    # SNR (higher is better)
    n_snr = notlame.get("snr_mean")
    l_snr = lame.get("snr_mean")
    w_snr = win_rates.get("snr")
    if n_snr is not None:
        winner = "notlame" if w_snr and w_snr > 0.5 else "LAME"
        target_met = "Y" if n_snr >= TARGETS["snr"] else ""
        print(f"{'SNR (dB)':<20} {n_snr:>12.2f} {l_snr or 0:>12.2f} {winner:>12} {'>'+str(TARGETS['snr'])+' '+target_met:>12}")

    # MR-STFT (lower is better)
    n_stft = notlame.get("mr_stft_mean")
    l_stft = lame.get("mr_stft_mean")
    w_stft = win_rates.get("mr_stft")
    if n_stft is not None:
        winner = "notlame" if w_stft and w_stft > 0.5 else "LAME"
        target_met = "Y" if n_stft <= TARGETS["mr_stft"] else ""
        print(f"{'MR-STFT':<20} {n_stft:>12.4f} {l_stft or 0:>12.4f} {winner:>12} {'<'+str(TARGETS['mr_stft'])+' '+target_met:>12}")

    # Mel (lower is better)
    n_mel = notlame.get("mel_mean")
    l_mel = lame.get("mel_mean")
    w_mel = win_rates.get("mel")
    if n_mel is not None:
        winner = "notlame" if w_mel and w_mel > 0.5 else "LAME"
        target_met = "Y" if n_mel <= TARGETS["mel"] else ""
        print(f"{'Mel Distance':<20} {n_mel:>12.4f} {l_mel or 0:>12.4f} {winner:>12} {'<'+str(TARGETS['mel'])+' '+target_met:>12}")

    # ViSQOL (higher is better)
    n_visqol = notlame.get("visqol_mean")
    l_visqol = lame.get("visqol_mean")
    w_visqol = win_rates.get("visqol")
    if n_visqol is not None:
        winner = "notlame" if w_visqol and w_visqol > 0.5 else "LAME"
        target_met = "Y" if n_visqol >= TARGETS["visqol"] else ""
        print(f"{'ViSQOL (MOS)':<20} {n_visqol:>12.2f} {l_visqol or 0:>12.2f} {winner:>12} {'>'+str(TARGETS['visqol'])+' '+target_met:>12}")

    # Win rates summary
    print("\n" + "-" * 70)
    print("Win Rates vs LAME:")
    print("-" * 70)

    for metric, rate in win_rates.items():
        if rate is not None:
            bar_len = int(rate * 40)
            bar = "#" * bar_len + "-" * (40 - bar_len)
            status = "BETTER" if rate > 0.5 else "WORSE" if rate < 0.5 else "TIE"
            print(f"  {metric:<12} [{bar}] {rate*100:5.1f}% {status}")

    # Overall verdict
    print("\n" + "=" * 70)
    wins = sum(1 for r in win_rates.values() if r is not None and r > 0.5)
    total = sum(1 for r in win_rates.values() if r is not None)

    if total > 0:
        if wins == total:
            print("VERDICT: notlame WINS on all metrics!")
        elif wins > total / 2:
            print(f"VERDICT: notlame WINS on {wins}/{total} metrics")
        elif wins == total / 2:
            print(f"VERDICT: TIE ({wins}/{total} metrics each)")
        else:
            print(f"VERDICT: LAME wins on {total-wins}/{total} metrics (more training needed)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PsychoNet model with research-based metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics computed:
  SNR         Signal-to-Noise Ratio (higher is better)
  MR-STFT     Multi-resolution STFT distance (lower is better)
  Mel         Mel spectrogram distance (lower is better)
  ViSQOL      Google's perceptual MOS metric (higher is better, requires visqol)

Example:
  python -m notlame_train.evaluate -c checkpoints/best.pt -t data/raw --max-files 20
        """,
    )

    parser.add_argument("--checkpoint", "-c", type=Path, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--test-dir", "-t", type=Path, required=True,
                        help="Directory with test audio files")
    parser.add_argument("--output", "-o", type=Path, default=Path("evaluation_report.json"),
                        help="Output JSON report path")
    parser.add_argument("--bitrate", "-b", type=int, default=192,
                        help="Bitrate for LAME comparison (default: 192)")
    parser.add_argument("--model", choices=["default", "lite", "large"], default="default",
                        help="Model variant")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum files to evaluate")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.test_dir.exists():
        print(f"Error: Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Create evaluator
    evaluator = Evaluator(
        model_path=args.checkpoint,
        model_variant=args.model,
        device=args.device,
    )

    # Evaluate
    report = evaluate_directory(
        evaluator,
        args.test_dir,
        bitrate=args.bitrate,
        max_files=args.max_files,
    )

    # Print report
    print_report(report)

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
