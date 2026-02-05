"""Data models for IML documents.

Immutable dataclasses mirroring the IML tag set defined in spec.md.
All content nodes use ``children`` tuples containing a mix of strings
(text spans) and other model objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, Union

# Type alias for mixed content children: plain text strings interspersed
# with inline markup elements.
ChildNode: TypeAlias = Union[str, "Prosody", "Pause", "Emphasis", "Segment"]


@dataclass(frozen=True)
class Pause:
    """A ``<pause>`` element -- a semantically significant silence.

    Spec reference: Section 3.3.
    """

    duration: int  # milliseconds


@dataclass(frozen=True)
class Prosody:
    """A ``<prosody>`` element -- a span with prosodic features.

    Spec reference: Sections 3.2, 4.
    """

    children: tuple[ChildNode, ...] = ()
    # Core attributes
    pitch: str | None = None
    pitch_contour: str | None = None
    volume: str | None = None
    rate: str | None = None
    quality: str | None = None
    # Extended attributes (Section 4)
    f0_mean: float | None = None
    f0_range: str | None = None
    f0_contour: str | None = None
    intensity_mean: float | None = None
    intensity_range: float | None = None
    speech_rate: float | None = None
    duration_ms: int | None = None
    jitter: float | None = None
    shimmer: float | None = None
    hnr: float | None = None


@dataclass(frozen=True)
class Emphasis:
    """An ``<emphasis>`` element -- a span with notable stress.

    Spec reference: Section 3.4.
    """

    level: str  # "strong", "moderate", "reduced"
    children: tuple[ChildNode, ...] = ()


@dataclass(frozen=True)
class Segment:
    """A ``<segment>`` element -- a clause-level prosodic grouping.

    Spec reference: Section 3.5.
    """

    children: tuple[ChildNode, ...] = ()
    tempo: str | None = None
    rhythm: str | None = None


@dataclass(frozen=True)
class Utterance:
    """An ``<utterance>`` element -- a complete spoken phrase or sentence.

    Spec reference: Section 3.1.
    """

    children: tuple[ChildNode, ...] = ()
    emotion: str | None = None
    confidence: float | None = None
    speaker_id: str | None = None


@dataclass(frozen=True)
class IMLDocument:
    """A complete IML document.

    Spec reference: Section 2.
    """

    utterances: tuple[Utterance, ...] = ()
    version: str | None = None
    language: str | None = None
    consent: str | None = field(default=None, repr=False)
    processing: str | None = field(default=None, repr=False)
