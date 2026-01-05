#!/usr/bin/env python3
"""Export trained PsychoNet model to ONNX format.

Exports the model for inference in notlame-lib (Rust).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.onnx
import onnx
from onnx import checker

from .model import create_model, count_parameters


class PsychoNetONNX(torch.nn.Module):
    """Wrapper for ONNX export.

    Simplifies the interface for inference:
    Input: (batch, 576) MDCT coefficients
    Output: (batch, 21) scalefactors [0-15]

    Scalefactor semantics (MP3 standard):
    - Low values (0-5): fine quantization, high quality, more bits
    - High values (10-15): coarse quantization, lower quality, fewer bits
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output["scalefactors"]


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    model_variant: str = "default",
    opset_version: int = 14,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> bool:
    """Export model to ONNX.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Output ONNX file path
        model_variant: Model variant (default, lite, large)
        opset_version: ONNX opset version
        dynamic_batch: Allow dynamic batch size
        verify: Verify exported model

    Returns:
        True if successful
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create model
    model = create_model(model_variant)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle torch.compile() wrapped models - strip _orig_mod. prefix if present
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model parameters: {count_parameters(model):,}")

    # Wrap for ONNX
    export_model = PsychoNetONNX(model)

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 576)

    # Dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export
    print(f"Exporting to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported successfully: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify
    if verify:
        print("\nVerifying ONNX model...")

        # Load and check
        onnx_model = onnx.load(str(output_path))
        try:
            checker.check_model(onnx_model)
            print("  Model structure: OK")
        except Exception as e:
            print(f"  Model check failed: {e}")
            return False

        # Test inference with ONNX Runtime
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(output_path))

            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]

            print(f"  Input: {input_info.name}, shape={input_info.shape}")
            print(f"  Output: {output_info.name}, shape={output_info.shape}")

            # Run inference
            test_input = dummy_input.numpy()
            outputs = session.run(None, {"input": test_input})

            print(f"  Test inference: OK")
            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output range: [{outputs[0].min():.2f}, {outputs[0].max():.2f}]")

            # Compare with PyTorch
            with torch.no_grad():
                torch_output = export_model(dummy_input).numpy()

            diff = abs(outputs[0] - torch_output).max()
            print(f"  Max difference vs PyTorch: {diff:.6f}")

            if diff > 1e-4:
                print("  WARNING: Large difference between PyTorch and ONNX!")
            else:
                print("  Outputs match PyTorch")

        except ImportError:
            print("  ONNX Runtime not installed, skipping inference test")
        except Exception as e:
            print(f"  Inference test failed: {e}")
            return False

    # Print model info
    print("\n--- Model Info ---")
    print(f"Variant: {model_variant}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Input: (batch, 576) MDCT coefficients")
    print(f"Output: (batch, 21) scalefactors")
    print(f"Scalefactor range: [0, 15]")
    print(f"  Low SF (0-5): fine quantization, high quality, more bits")
    print(f"  High SF (10-15): coarse quantization, lower quality, fewer bits")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export PsychoNet model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export best checkpoint
  python -m notlame_train.export_onnx \\
      --checkpoint checkpoints/best.pt \\
      --output models/psycho_v1.onnx

  # Export lite model
  python -m notlame_train.export_onnx \\
      --checkpoint checkpoints/best.pt \\
      --output models/psycho_lite.onnx \\
      --model lite
        """,
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("models/psycho_v1.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--model",
        choices=["default", "lite", "large"],
        default="default",
        help="Model variant",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Use static batch size (1) instead of dynamic",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    success = export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_variant=args.model,
        opset_version=args.opset,
        dynamic_batch=not args.static_batch,
        verify=not args.no_verify,
    )

    if success:
        print(f"\n✓ Export complete: {args.output}")
        print(f"\nCopy to notlame-lib:")
        print(f"  cp {args.output} ../notlame-lib/models/")
    else:
        print("\n✗ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
