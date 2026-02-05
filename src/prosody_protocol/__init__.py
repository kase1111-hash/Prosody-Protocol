"""Prosody Protocol SDK -- preserving prosodic intent in speech-to-text.

Public API re-exports for convenient access::

    from prosody_protocol import IMLParser, IMLValidator, AudioToIML
"""

from ._version import __version__
from .assembler import IMLAssembler
from .audio_to_iml import AudioToIML
from .emotion_classifier import EmotionClassifier, RuleBasedEmotionClassifier
from .exceptions import (
    AudioProcessingError,
    ConversionError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
)
from .iml_to_audio import IMLToAudio
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
from .prosody_analyzer import PauseInterval, ProsodyAnalyzer, SpanFeatures, WordAlignment
from .text_to_iml import TextToIML
from .validator import IMLValidator, ValidationIssue, ValidationResult

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
    # Exceptions
    "ProsodyProtocolError",
    "IMLParseError",
    "IMLValidationError",
    "ProfileError",
    "AudioProcessingError",
    "ConversionError",
]
