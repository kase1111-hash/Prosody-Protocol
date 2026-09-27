#!/usr/bin/env python3
"""Export trained model to a portable format for SDK integration.

Formats:

- ``json`` (default): ``model.json`` holds the fitted parameters as plain
  JSON. :class:`training.portable.PortableModel` runs it with numpy alone,
  and :class:`training.inference.TrainedEmotionClassifier` plugs an SER
  export into ``AudioToIML``. Loading it never runs code from the file, so
  this is the format to share.
- ``pickle``: ``model.joblib``, a pickle of the model object for
  ``BaseModel.load``. Loading it runs code from the file: trusted use only.

Both also write ``config.json`` (model class and parameters) and
``export_metadata.json``.

The checkpoint itself is a pickle, so only export checkpoints you trust.

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
import warnings
from pathlib import Path
from typing import Any

# Allow imports from project root (the imports below must follow this)
# ruff: noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from prosody_protocol.exceptions import ProsodyProtocolError
from training.config import load_config
from training.models.base import EXPORT_FORMATS, BaseModel
from training.scripts.train import CHECKPOINT_CONFIG


def export_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    export_format: str = "json",
) -> dict[str, Any]:
    """Export a trained model checkpoint for SDK use.

    Parameters
    ----------
    checkpoint_path:
        Path to the model checkpoint directory. Its ``model.joblib`` is
        unpickled, which runs code from the file: only use trusted
        checkpoints.
    output_path:
        Path to the export output directory.
    export_format:
        ``'json'`` (portable, pickle-free ``model.json``) or ``'pickle'``
        (``model.joblib``).

    Returns
    -------
    dict
        Export metadata including paths and model info.
    """
    model = BaseModel.load(checkpoint_path)
    output_path = Path(output_path)

    files = model.export(output_path, export_format)

    export_meta = {
        "source_checkpoint": str(checkpoint_path),
        "export_path": str(output_path),
        "export_format": export_format,
        "files": files,
        "model_class": type(model).__name__,
        "params": model.get_params(),
    }

    with open(output_path / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return export_meta


def _default_format(checkpoint_path: Path) -> str:
    """``output.export_format`` of the checkpoint's saved config, else ``json``."""
    config_path = checkpoint_path / CHECKPOINT_CONFIG
    if not config_path.is_file():
        return "json"
    with warnings.catch_warnings():  # reported when the model was trained
        warnings.simplefilter("ignore")
        return str(load_config(config_path).output.get("export_format", "json"))


# Errors reported as a one-line message instead of a traceback.
_USER_ERRORS = (ValueError, FileNotFoundError, NotImplementedError, ProsodyProtocolError)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained model for SDK integration",
        epilog="The checkpoint's model.joblib is a pickle: loading it runs code from the file. "
        "Only export checkpoints you trust, and share the json export.",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a model checkpoint directory (trusted files only)")
    parser.add_argument("--output", required=True, help="Path to export output directory")
    parser.add_argument(
        "--format", choices=EXPORT_FORMATS,
        help="json: pickle-free model.json (safe to share); pickle: model.joblib "
        "(trusted use only). Default: output.export_format of the training config, else json",
    )
    args = parser.parse_args()

    try:
        export_format = args.format or _default_format(Path(args.checkpoint))
        meta = export_model(args.checkpoint, args.output, export_format)
    except _USER_ERRORS as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("Model exported successfully:")
    print(f"  Model class: {meta['model_class']}")
    print(f"  Export path: {meta['export_path']}")
    print(f"  Format: {meta['export_format']} ({', '.join(meta['files'])})")


if __name__ == "__main__":
    main()
