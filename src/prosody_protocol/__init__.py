"""Prosody Protocol SDK -- preserving prosodic intent in speech-to-text.

Public API re-exports for convenient access::

    from prosody_protocol import IMLParser, IMLValidator, AudioToIML

The core (parsing, validation, SSML conversion, text prediction, profiles,
datasets) needs only ``lxml``. Classes that need optional dependencies are
loaded lazily on first access, so ``import prosody_protocol`` always works:

- ``AudioToIML``, ``ProsodyAnalyzer``: ``pip install 'prosody-protocol[audio]'``
- ``IMLToAudio``, ``Benchmark``, ``BenchmarkReport``, ``MavisBridge``,
  ``PhonemeEvent``: ``numpy`` (included in the ``audio`` extra)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._types import PauseInterval, SpanFeatures, WordAlignment
from ._version import __version__
from .assembler import IMLAssembler
from .datasets import Dataset, DatasetEntry, DatasetLoader
from .emotion_classifier import EmotionClassifier, RuleBasedEmotionClassifier
from .exceptions import (
    AudioProcessingError,
    ConversionError,
    DatasetError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
    TrainingError,
)
from .iml_to_ssml import IMLToSSML
from .models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from .parser import IMLParser
from .profiles import ProfileApplier, ProfileLoader, ProsodyMapping, ProsodyProfile
from .text_to_iml import TextToIML
from .validator import IMLValidator, ValidationIssue, ValidationResult

if TYPE_CHECKING:
    from .audio_to_iml import AudioToIML
    from .benchmarks import Benchmark, BenchmarkReport
    from .iml_to_audio import IMLToAudio
    from .mavis_bridge import MavisBridge, PhonemeEvent
    from .prosody_analyzer import ProsodyAnalyzer

# Names whose modules need optional third-party packages (numpy, parselmouth).
# They are imported on first attribute access (PEP 562).
_LAZY: dict[str, str] = {
    "AudioToIML": ".audio_to_iml",
    "ProsodyAnalyzer": ".prosody_analyzer",
    "IMLToAudio": ".iml_to_audio",
    "Benchmark": ".benchmarks",
    "BenchmarkReport": ".benchmarks",
    "MavisBridge": ".mavis_bridge",
    "PhonemeEvent": ".mavis_bridge",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    # Core
    "IMLParser",
    "IMLValidator",
    "ValidationIssue",
    "ValidationResult",
    # Models
    "IMLDocument",
    "Utterance",
    "Prosody",
    "Pause",
    "Emphasis",
    "Segment",
    # Conversion
    "AudioToIML",
    "IMLToAudio",
    "IMLToSSML",
    "TextToIML",
    "IMLAssembler",
    # Analysis
    "ProsodyAnalyzer",
    "SpanFeatures",
    "WordAlignment",
    "PauseInterval",
    # Emotion
    "EmotionClassifier",
    "RuleBasedEmotionClassifier",
    # Profiles
    "ProfileLoader",
    "ProfileApplier",
    "ProsodyProfile",
    "ProsodyMapping",
    # Benchmarks
    "Benchmark",
    "BenchmarkReport",
    # Datasets
    "DatasetLoader",
    "DatasetEntry",
    "Dataset",
    # Mavis Bridge
    "MavisBridge",
    "PhonemeEvent",
    # Exceptions
    "ProsodyProtocolError",
    "IMLParseError",
    "IMLValidationError",
    "ProfileError",
    "AudioProcessingError",
    "ConversionError",
    "DatasetError",
    "TrainingError",
]
