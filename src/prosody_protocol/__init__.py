"""Prosody Protocol SDK -- preserving prosodic intent in speech-to-text.

Public API re-exports for convenient access::

    from prosody_protocol import IMLParser, IMLValidator, AudioToIML

The core (parsing, validation, SSML conversion, text prediction, profiles,
datasets) needs only ``lxml``. Classes that need optional dependencies are
loaded lazily on first access, so ``import prosody_protocol`` always works:

- ``AudioToIML``, ``ConversionResult``, ``ProsodyAnalyzer``:
  the ``audio`` extra
- ``IMLToAudio``, ``Benchmark``, ``BenchmarkReport``, ``MavisBridge``,
  ``PhonemeEvent``: ``numpy`` (included in the ``audio`` extra)
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

from ._types import PauseInterval, SpanFeatures, WordAlignment
from ._version import __version__
from .alignment import load_word_timings, parse_word_timings
from .assembler import IMLAssembler, ProfileMatch
from .datasets import Dataset, DatasetEntry, DatasetLoader
from .emotion_classifier import (
    BaselineAwareEmotionClassifier,
    EmotionClassifier,
    RuleBasedEmotionClassifier,
    SpeakerBaseline,
)
from .exceptions import (
    AudioProcessingError,
    ConversionError,
    DatasetError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
    SpeechRecognitionError,
    TrainingError,
)
from .iml_to_ssml import IMLToSSML
from .llm import build_messages, to_llm_context
from .models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from .parser import IMLParser
from .profiles import (
    ProfileApplier,
    ProfileLoader,
    ProsodyMapping,
    ProsodyProfile,
    categorize_features,
)
from .text_to_iml import TextToIML
from .validator import IMLValidator, ValidationIssue, ValidationResult

if TYPE_CHECKING:
    from .audio_to_iml import AudioToIML as AudioToIML
    from .audio_to_iml import ConversionResult as ConversionResult
    from .benchmarks import Benchmark as Benchmark
    from .benchmarks import BenchmarkReport as BenchmarkReport
    from .iml_to_audio import IMLToAudio as IMLToAudio
    from .mavis_bridge import MavisBridge as MavisBridge
    from .mavis_bridge import PhonemeEvent as PhonemeEvent
    from .prosody_analyzer import ProsodyAnalyzer as ProsodyAnalyzer

# Names whose modules need optional third-party packages (numpy, parselmouth).
# They are imported on first attribute access (PEP 562).
_LAZY: dict[str, str] = {
    "AudioToIML": ".audio_to_iml",
    "ConversionResult": ".audio_to_iml",
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
    "IMLToSSML",
    "TextToIML",
    "IMLAssembler",
    # Analysis
    "SpanFeatures",
    "WordAlignment",
    "PauseInterval",
    # Word timings from external speech-to-text services
    "load_word_timings",
    "parse_word_timings",
    # LLM hand-off
    "to_llm_context",
    "build_messages",
    # Emotion
    "EmotionClassifier",
    "BaselineAwareEmotionClassifier",
    "RuleBasedEmotionClassifier",
    "SpeakerBaseline",
    # Profiles
    "ProfileLoader",
    "ProfileApplier",
    "ProsodyProfile",
    "ProsodyMapping",
    "ProfileMatch",
    "categorize_features",
    # Datasets
    "DatasetLoader",
    "DatasetEntry",
    "Dataset",
    # Exceptions
    "ProsodyProtocolError",
    "IMLParseError",
    "IMLValidationError",
    "ProfileError",
    "AudioProcessingError",
    "ConversionError",
    "DatasetError",
    "SpeechRecognitionError",
    "TrainingError",
]

# The lazily loaded names join __all__ only when their dependencies are
# installed, so ``from prosody_protocol import *`` works on the lxml-only core.
_LAZY_REQUIRES: dict[str, tuple[str, ...]] = {
    ".audio_to_iml": ("numpy", "parselmouth"),
    ".prosody_analyzer": ("numpy", "parselmouth"),
    ".iml_to_audio": ("numpy",),
    ".benchmarks": ("numpy",),
    ".mavis_bridge": ("numpy",),
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


__all__ += [
    name
    for name, module in _LAZY.items()
    if all(_installed(dep) for dep in _LAZY_REQUIRES[module])
]
