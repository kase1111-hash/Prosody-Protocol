#!/usr/bin/env python3
"""Export trained model to a portable format for SDK integration.

Usage:
    python training/scripts/export.py \\
        --checkpoint training/checkpoints/ser_v1 \\
        --output training/exports/ser_v1 \\
        --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training.models.base import BaseModel


def export_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    export_format: str = "json",
) -> dict:
    """Export a trained model checkpoint for SDK use.

    Parameters
    ----------
    checkpoint_path:
        Path to the model checkpoint directory.
    output_path:
        Path to the export output directory.
    export_format:
        Export format ('json' or 'pickle').

    Returns
    -------
    dict
        Export metadata including paths and model info.
    """
    model = BaseModel.load(checkpoint_path)
    output_path = Path(output_path)

    model.export(output_path)

    # Read back config for metadata
    config_file = output_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = model.get_params()

    export_meta = {
        "source_checkpoint": str(checkpoint_path),
        "export_path": str(output_path),
        "export_format": export_format,
        "model_class": config.get("model_class", type(model).__name__),
        "params": config.get("params", {}),
    }

    with open(output_path / "export_metadata.json", "w") as f:
        json.dump(export_meta, f, indent=2)

    return export_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model for SDK integration")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Path to export output directory")
    parser.add_argument("--format", default="json", choices=["json", "pickle"],
                        help="Export format")
    args = parser.parse_args()

    meta = export_model(args.checkpoint, args.output, args.format)

    print(f"Model exported successfully:")
    print(f"  Model class: {meta['model_class']}")
    print(f"  Export path: {meta['export_path']}")
    print(f"  Format: {meta['export_format']}")


if __name__ == "__main__":
    main()
