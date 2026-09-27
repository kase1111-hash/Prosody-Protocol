"""Prosody profile loader and applier for accessibility support.

Loads JSON prosody profiles (spec Section 7) and applies pattern-based
emotion re-mapping to adjust classifications for atypical speakers.

Profile matching logic (Section 8.2 of the execution guide):
  1. For each mapping, check if *all* pattern keys match observed features.
  2. Matching is categorical (exact string match on observed feature labels).
     :func:`categorize_features` turns measured :class:`SpanFeatures` into
     those labels, using the thresholds documented below.
  3. If multiple mappings match, use the most specific (most pattern keys).
  4. Apply ``confidence_boost`` (capped at 1.0).

A profile is valid when :meth:`ProfileLoader.load_json` accepts it and
:meth:`ProfileLoader.validate` reports no errors. Together they enforce
``schemas/prosody-profile.schema.json`` (and reject NaN and Infinity, which
JSON cannot hold but Python's json module reads).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._types import PauseInterval, SpanFeatures
from .exceptions import ProfileError
from .validator import ValidationIssue, ValidationResult

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProsodyMapping:
    """A single pattern-to-interpretation mapping."""

    pattern: dict[str, str]
    interpretation_emotion: str
    confidence_boost: float = 0.0


@dataclass(frozen=True)
class ProsodyProfile:
    """A user's prosody profile."""

    profile_version: str
    user_id: str
    description: str | None
    mappings: tuple[ProsodyMapping, ...]


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?")

# Keys the schema allows (additionalProperties is false at every level).
_PROFILE_KEYS = frozenset({
    "profile_version", "user_id", "description", "prosody_mappings", "metadata",
})
_MAPPING_KEYS = frozenset({"pattern", "interpretation"})
_INTERPRETATION_KEYS = frozenset({"emotion", "confidence_boost"})
# Metadata keys the schema types as strings; others are free-form.
_METADATA_STRING_KEYS = ("created_at", "updated_at", "author", "clinical_context")

_VALID_PATTERN_KEYS = frozenset({
    "pitch", "pitch_contour", "volume", "rate",
    "quality", "pause_frequency", "emphasis_frequency",
})

_PITCH_VALUES = frozenset({"high", "low", "normal"})
_PITCH_CONTOUR_VALUES = frozenset({
    "rise", "fall", "rise-fall", "fall-rise",
    "fall-sharp", "rise-sharp", "flat",
})
_VOLUME_VALUES = frozenset({"loud", "quiet", "normal", "spike"})
_RATE_VALUES = frozenset({"fast", "slow", "normal"})
_QUALITY_VALUES = frozenset({
    "modal", "breathy", "tense", "creaky", "whispery", "harsh",
})
_FREQUENCY_VALUES = frozenset({"high", "low", "normal"})

_PATTERN_VALUE_MAP: dict[str, frozenset[str]] = {
    "pitch": _PITCH_VALUES,
    "pitch_contour": _PITCH_CONTOUR_VALUES,
    "volume": _VOLUME_VALUES,
    "rate": _RATE_VALUES,
    "quality": _QUALITY_VALUES,
    "pause_frequency": _FREQUENCY_VALUES,
    "emphasis_frequency": _FREQUENCY_VALUES,
}


# ---------------------------------------------------------------------------
# ProfileLoader
# ---------------------------------------------------------------------------


class ProfileLoader:
    """Load and validate prosody profile JSON files."""

    def load(self, path: str | Path) -> ProsodyProfile:
        """Load a profile from a JSON file.

        Raises :class:`~prosody_protocol.exceptions.ProfileError`
        if the file cannot be read or the JSON structure is invalid.
        """
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfileError(f"Cannot read profile file: {exc}") from exc

        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Invalid JSON in profile file: {exc}") from exc

        return self.load_json(data)

    def load_json(self, data: dict[str, object]) -> ProsodyProfile:
        """Load a profile from a parsed JSON dict.

        Raises :class:`~prosody_protocol.exceptions.ProfileError`
        if required fields are missing, have wrong types, or a key is not
        one the schema allows. Value rules (version format, pattern
        vocabulary, boost range) are checked by :meth:`validate`.
        """
        if not isinstance(data, dict):
            raise ProfileError("Profile must be a JSON object")
        _reject_unknown_keys(data, _PROFILE_KEYS, "profile")

        # Required fields.
        profile_version = data.get("profile_version")
        if not isinstance(profile_version, str) or not profile_version:
            raise ProfileError("Missing or invalid 'profile_version' (must be a non-empty string)")

        user_id = data.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ProfileError("Missing or invalid 'user_id' (must be a non-empty string)")

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ProfileError("'description' must be a string or null")

        if "metadata" in data:
            metadata = data["metadata"]
            if not isinstance(metadata, dict):
                raise ProfileError("'metadata' must be an object")
            for key in _METADATA_STRING_KEYS:
                if key in metadata and not isinstance(metadata[key], str):
                    raise ProfileError(f"'metadata.{key}' must be a string")

        raw_mappings = data.get("prosody_mappings")
        if not isinstance(raw_mappings, list):
            raise ProfileError("Missing or invalid 'prosody_mappings' (must be an array)")

        mappings: list[ProsodyMapping] = []
        for i, raw in enumerate(raw_mappings):
            if not isinstance(raw, dict):
                raise ProfileError(f"prosody_mappings[{i}] must be an object")
            mappings.append(self._parse_mapping(raw, i))

        return ProsodyProfile(
            profile_version=profile_version,
            user_id=user_id,
            description=description,
            mappings=tuple(mappings),
        )

    def validate(self, profile: ProsodyProfile) -> ValidationResult:
        """Validate a loaded profile against spec rules.

        Returns a :class:`ValidationResult` with ``valid=True`` when
        no errors are found.
        """
        issues: list[ValidationIssue] = []

        # Version format.
        if not _VERSION_RE.fullmatch(profile.profile_version):
            issues.append(ValidationIssue(
                severity="error",
                rule="P1",
                message=(
                    f"profile_version '{profile.profile_version}' "
                    "does not match semver format (X.Y.Z)"
                ),
            ))

        # User ID non-empty.
        if not profile.user_id:
            issues.append(ValidationIssue(
                severity="error",
                rule="P2",
                message="user_id must be a non-empty string",
            ))

        # At least one mapping.
        if not profile.mappings:
            issues.append(ValidationIssue(
                severity="error",
                rule="P3",
                message="prosody_mappings must contain at least one mapping",
            ))

        # Validate each mapping.
        for i, m in enumerate(profile.mappings):
            if not m.pattern:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="P4",
                    message=f"prosody_mappings[{i}].pattern must have at least one key",
                ))

            # Unknown keys or values could never match categorize_features()
            # output, so the mapping would silently never apply.
            for key, value in m.pattern.items():
                if key not in _VALID_PATTERN_KEYS:
                    issues.append(ValidationIssue(
                        severity="error",
                        rule="P5",
                        message=(
                            f"prosody_mappings[{i}].pattern has unknown key '{key}' "
                            f"(expected one of {sorted(_VALID_PATTERN_KEYS)})"
                        ),
                    ))
                elif value not in _PATTERN_VALUE_MAP[key]:
                    issues.append(ValidationIssue(
                        severity="error",
                        rule="P6",
                        message=(
                            f"prosody_mappings[{i}].pattern.{key}='{value}' "
                            f"is not one of {sorted(_PATTERN_VALUE_MAP[key])}"
                        ),
                    ))

            if not m.interpretation_emotion:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="P7",
                    message=f"prosody_mappings[{i}].interpretation.emotion must be non-empty",
                ))

            if not 0.0 <= m.confidence_boost <= 1.0:  # also rejects NaN
                issues.append(ValidationIssue(
                    severity="error",
                    rule="P8",
                    message=(
                        f"prosody_mappings[{i}].interpretation.confidence_boost="
                        f"{m.confidence_boost} must be between 0.0 and 1.0"
                    ),
                ))

        valid = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(valid=valid, issues=issues)

    @staticmethod
    def _parse_mapping(raw: dict[str, object], index: int) -> ProsodyMapping:
        """Parse a single mapping entry from a JSON dict."""
        _reject_unknown_keys(raw, _MAPPING_KEYS, f"prosody_mappings[{index}]")
        pattern_raw = raw.get("pattern")
        if not isinstance(pattern_raw, dict):
            raise ProfileError(
                f"prosody_mappings[{index}].pattern must be an object"
            )

        pattern: dict[str, str] = {}
        for k, v in pattern_raw.items():
            if not isinstance(v, str):
                raise ProfileError(
                    f"prosody_mappings[{index}].pattern.{k} must be a string"
                )
            pattern[k] = v

        interpretation = raw.get("interpretation")
        if not isinstance(interpretation, dict):
            raise ProfileError(
                f"prosody_mappings[{index}].interpretation must be an object"
            )

        _reject_unknown_keys(
            interpretation, _INTERPRETATION_KEYS, f"prosody_mappings[{index}].interpretation"
        )
        emotion = interpretation.get("emotion")
        if not isinstance(emotion, str) or not emotion:
            raise ProfileError(
                f"prosody_mappings[{index}].interpretation.emotion must be a non-empty string"
            )

        confidence_boost = interpretation.get("confidence_boost", 0.0)
        if (
            isinstance(confidence_boost, bool)
            or not isinstance(confidence_boost, (int, float))
            or not math.isfinite(confidence_boost)
        ):
            raise ProfileError(
                f"prosody_mappings[{index}].interpretation.confidence_boost must be a "
                f"finite number, got {confidence_boost!r}"
            )

        return ProsodyMapping(
            pattern=pattern,
            interpretation_emotion=emotion,
            confidence_boost=float(confidence_boost),
        )


def _reject_constant(name: str) -> float:
    raise ProfileError(f"Invalid JSON in profile file: {name} is not a JSON number")


def _reject_unknown_keys(data: dict[str, object], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise ProfileError(
            f"{where} has unknown key(s) {unknown}; allowed: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# ProfileApplier
# ---------------------------------------------------------------------------


class ProfileApplier:
    """Apply prosody profiles to adjust emotion classification.

    Matching logic:
      1. Check each mapping's pattern against observed features.
      2. A mapping matches only if *all* its pattern keys are present
         in the features dict and their values match.
      3. Among all matching mappings, select the most specific one
         (most pattern keys).  Ties are broken by order in the profile
         (first match wins).
      4. Apply ``confidence_boost`` (capped at 1.0).
    """

    def apply(
        self,
        profile: ProsodyProfile,
        features: dict[str, str],
        base_emotion: str,
        base_confidence: float,
    ) -> tuple[str, float]:
        """Return ``(adjusted_emotion, adjusted_confidence)``.

        If no mapping matches, returns the base values unchanged.
        """
        best_match: ProsodyMapping | None = None
        best_specificity = 0

        for mapping in profile.mappings:
            if self._matches(mapping.pattern, features):
                specificity = len(mapping.pattern)
                if specificity > best_specificity:
                    best_match = mapping
                    best_specificity = specificity

        if best_match is None:
            return (base_emotion, base_confidence)

        adjusted_emotion = best_match.interpretation_emotion
        adjusted_confidence = min(1.0, base_confidence + best_match.confidence_boost)

        return (adjusted_emotion, adjusted_confidence)

    @staticmethod
    def _matches(pattern: dict[str, str], features: dict[str, str]) -> bool:
        """Check if all pattern entries match observed features."""
        return all(features.get(key) == expected for key, expected in pattern.items())


# ---------------------------------------------------------------------------
# Feature categorisation
# ---------------------------------------------------------------------------

# Thresholds that turn measured features (SpanFeatures units, spec Section 4)
# into the categorical values profile patterns match on. Absolute pitch and
# loudness depend on the speaker and the recording, so "pitch" and "volume"
# levels are only judged against a baseline of the speaker's own speech.

# rate: articulation rate in syllables per second of speaking time.
# Conversational English runs at about 4-5.5; "fast" and "slow" are about
# one standard deviation of speaker variation beyond that.
RATE_FAST_SYLLABLES_PER_S = 6.0
RATE_SLOW_SYLLABLES_PER_S = 3.5
# With a baseline, rate is judged against the speaker's own rate instead:
# at least this factor faster is "fast", this factor slower is "slow".
RATE_RELATIVE_FACTOR = 1.25
# Without a baseline, a rate at or below RATE_SLOW_SYLLABLES_PER_S is left
# undecided instead of "slow". The analyzer counts voiced intensity peaks,
# and syllables it misses (in very low, creaky or whispered voices) lower
# the estimate while nothing raises it: espeak-ng at its default 175 words
# per minute measures 4.8 syllables/s in a 100 Hz voice but 3.0 in an 80 Hz
# one. A baseline in the same voice has the same bias, so relative judgments
# hold. Rate is only judged when at least MIN_RATE_SPAN_SHARE of the spans
# (and of the baseline's) have an estimate.
MIN_RATE_SPAN_SHARE = 0.5

# pitch (baseline only): the utterance's median F0 at least this many
# semitones above the baseline median is "high", below it "low".
PITCH_LEVEL_ST = 2.0

# pitch_contour: "flat" (monotone) when the utterance's F0 spans less than
# this many semitones from its 10th to its 90th percentile; typical speech
# spans 4-8. Otherwise the medians of the first, middle and last thirds of
# the F0 track are compared: a rise or fall of at least CONTOUR_MOVEMENT_ST
# semitones is a contour, "-sharp" when it is at least SHARP_CONTOUR_ST
# semitones at SHARP_CONTOUR_ST_PER_S or faster. Varied pitch with no
# overall shape gets no pitch_contour.
FLAT_PITCH_RANGE_ST = 3.0
CONTOUR_MOVEMENT_ST = 2.0
SHARP_CONTOUR_ST = 5.0
SHARP_CONTOUR_ST_PER_S = 20.0
# Contours need at least this many voiced F0 samples; samples this far
# (semitones) from the median are pitch-tracking octave errors.
MIN_CONTOUR_SAMPLES = 6
_OCTAVE_ERROR_ST = 10.0

# volume: "spike" when one span is at least VOLUME_SPIKE_DB louder than the
# median of the other spans (10 dB sounds about twice as loud). Otherwise,
# with a baseline, a median intensity at least VOLUME_LEVEL_DB above the
# baseline's is "loud", as far below it "quiet".
VOLUME_SPIKE_DB = 10.0
VOLUME_LEVEL_DB = 4.0

# pause_frequency: the share of word boundaries with a pause of at least
# MIN_PAUSE_MS, either a detected silence or a gap between the spans. Fluent
# speech pauses every 5-10 words. "high" from PAUSE_FREQUENCY_HIGH; "low"
# below PAUSE_FREQUENCY_LOW, judged only over at least MIN_BOUNDARIES_FOR_LOW
# boundaries (a short phrase without a pause is ordinary).
MIN_PAUSE_MS = 200
PAUSE_FREQUENCY_HIGH = 0.25
PAUSE_FREQUENCY_LOW = 0.05
MIN_BOUNDARIES_FOR_LOW = 10

# emphasis_frequency: the share of spans at least EMPHASIS_VOLUME_DB louder
# or EMPHASIS_PITCH_ST higher than the utterance's median span. "high" from
# EMPHASIS_FREQUENCY_HIGH; "low" below EMPHASIS_FREQUENCY_LOW over at least
# MIN_SPANS_FOR_LOW spans.
EMPHASIS_VOLUME_DB = 6.0
EMPHASIS_PITCH_ST = 3.5
EMPHASIS_FREQUENCY_HIGH = 0.3
EMPHASIS_FREQUENCY_LOW = 0.05
MIN_SPANS_FOR_LOW = 10

# pause_frequency and emphasis_frequency need at least this many spans.
MIN_SPANS_FOR_FREQUENCY = 4

# quality: the analyzer's label covering at least this share of the duration
# of the spans that have one.
QUALITY_MIN_SHARE = 0.5


def categorize_features(
    features: Sequence[SpanFeatures],
    pauses: Sequence[PauseInterval] = (),
    *,
    baseline: Sequence[SpanFeatures] | None = None,
) -> dict[str, str]:
    """Describe an utterance with the categorical values profiles match on.

    *features* are the utterance's spans (typically one per word, as
    :meth:`ProsodyAnalyzer.analyze` returns them) and *pauses* the silences
    detected in the same audio (:meth:`ProsodyAnalyzer.detect_pauses`);
    pauses outside the utterance are ignored. *baseline* is the speaker's
    ordinary speech (e.g. features of a calibration recording); without it
    the result has no ``pitch`` key, ``volume`` is only ever ``"spike"``,
    and ``rate`` is never ``"slow"`` (a low reading may be the analyzer
    missing syllables of a low voice). Pass a baseline whenever one exists.

    The result maps pattern keys of spec Section 7 (``pitch``,
    ``pitch_contour``, ``volume``, ``rate``, ``quality``,
    ``pause_frequency``, ``emphasis_frequency``) to values of the profile
    schema's vocabulary. A key is left out when the measurements cannot
    decide it, so it never matches a pattern by default. The thresholds are
    the module constants documented above.

    Example::

        analyzer = ProsodyAnalyzer()
        baseline = analyzer.analyze("calibration.wav", calibration_alignments)
        spans = analyzer.analyze("clip.wav", alignments)
        observed = categorize_features(
            spans, analyzer.detect_pauses("clip.wav"), baseline=baseline
        )
        emotion, confidence = ProfileApplier().apply(profile, observed, "neutral", 0.5)
    """
    spans = sorted(features, key=lambda f: f.start_ms)
    reference = sorted(baseline, key=lambda f: f.start_ms) if baseline else []
    result: dict[str, str] = {}

    f0 = _f0_samples(spans)
    base_f0 = _f0_samples(reference)
    if f0 and base_f0:
        offset = _semitones(statistics.median(f0), statistics.median(base_f0))
        result["pitch"] = _level(offset, PITCH_LEVEL_ST, "high", "low")

    contour = _utterance_contour(f0, spans)
    if contour is not None:
        result["pitch_contour"] = contour

    volume = _volume(spans, reference)
    if volume is not None:
        result["volume"] = volume

    rate = _rate(spans)
    base_rate = _rate(reference)
    if rate is not None and base_rate:
        if rate >= base_rate * RATE_RELATIVE_FACTOR:
            result["rate"] = "fast"
        elif rate <= base_rate / RATE_RELATIVE_FACTOR:
            result["rate"] = "slow"
        else:
            result["rate"] = "normal"
    elif rate is not None and rate > RATE_SLOW_SYLLABLES_PER_S:
        result["rate"] = "fast" if rate >= RATE_FAST_SYLLABLES_PER_S else "normal"

    quality = _dominant_quality(spans)
    if quality is not None:
        result["quality"] = quality

    if len(spans) >= MIN_SPANS_FOR_FREQUENCY:
        paused = _paused_boundaries(spans, pauses)
        result["pause_frequency"] = _frequency(
            paused / (len(spans) - 1),
            len(spans) - 1,
            PAUSE_FREQUENCY_HIGH,
            PAUSE_FREQUENCY_LOW,
            MIN_BOUNDARIES_FOR_LOW,
        )
        emphasized = _emphasized_spans(spans)
        if emphasized is not None:
            result["emphasis_frequency"] = _frequency(
                emphasized / len(spans),
                len(spans),
                EMPHASIS_FREQUENCY_HIGH,
                EMPHASIS_FREQUENCY_LOW,
                MIN_SPANS_FOR_LOW,
            )
    return result


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _rate(spans: Sequence[SpanFeatures]) -> float | None:
    """Median speech rate of *spans*, or ``None`` when fewer than
    MIN_RATE_SPAN_SHARE of them have an estimate."""
    rates = [r for r in (_finite(f.speech_rate) for f in spans) if r is not None and r >= 0.0]
    if not rates or len(rates) < MIN_RATE_SPAN_SHARE * len(spans):
        return None
    return statistics.median(rates)


def _semitones(hz: float, reference_hz: float) -> float:
    return 12.0 * math.log2(hz / reference_hz)


def _level(offset: float, threshold: float, above: str, below: str) -> str:
    if offset >= threshold:
        return above
    if offset <= -threshold:
        return below
    return "normal"


def _frequency(share: float, count: int, high: float, low: float, min_count: int) -> str:
    if share >= high:
        return "high"
    if share < low and count >= min_count:
        return "low"
    return "normal"


def _f0_samples(spans: Sequence[SpanFeatures]) -> list[float]:
    """Voiced F0 values (Hz) in time order: each span's contour, else its mean."""
    samples: list[float] = []
    for span in spans:
        values = span.f0_contour if span.f0_contour else [span.f0_mean]
        samples.extend(v for v in (_finite(x) for x in values) if v is not None and v > 0.0)
    if not samples:
        return []
    center = statistics.median(samples)
    return [v for v in samples if abs(_semitones(v, center)) <= _OCTAVE_ERROR_ST]


def _utterance_contour(f0: list[float], spans: Sequence[SpanFeatures]) -> str | None:
    """The spec 3.2 contour of a whole utterance's F0 track, if it has one."""
    if len(f0) < MIN_CONTOUR_SAMPLES:
        return None
    center = statistics.median(f0)
    st = [_semitones(v, center) for v in f0]
    deciles = statistics.quantiles(st, n=10)
    if deciles[-1] - deciles[0] < FLAT_PITCH_RANGE_ST:
        return "flat"

    third = len(st) // 3
    start = statistics.median(st[:third])
    middle = statistics.median(st[third:-third])
    end = statistics.median(st[-third:])
    if middle - max(start, end) >= CONTOUR_MOVEMENT_ST:
        return "rise-fall"
    if min(start, end) - middle >= CONTOUR_MOVEMENT_ST:
        return "fall-rise"
    movement = end - start
    if abs(movement) < CONTOUR_MOVEMENT_ST:
        return None
    duration_s = (spans[-1].end_ms - spans[0].start_ms) / 1000.0
    sharp = (
        abs(movement) >= SHARP_CONTOUR_ST
        and duration_s > 0.0
        and abs(movement) / duration_s >= SHARP_CONTOUR_ST_PER_S
    )
    if movement > 0:
        return "rise-sharp" if sharp else "rise"
    return "fall-sharp" if sharp else "fall"


def _volume(spans: Sequence[SpanFeatures], reference: Sequence[SpanFeatures]) -> str | None:
    levels = [v for v in (_finite(f.intensity_mean) for f in spans) if v is not None]
    for i, level in enumerate(levels):
        others = levels[:i] + levels[i + 1:]
        if others and level - statistics.median(others) >= VOLUME_SPIKE_DB:
            return "spike"
    base = [v for v in (_finite(f.intensity_mean) for f in reference) if v is not None]
    if not levels or not base:
        return None
    offset = statistics.median(levels) - statistics.median(base)
    return _level(offset, VOLUME_LEVEL_DB, "loud", "quiet")


def _dominant_quality(spans: Sequence[SpanFeatures]) -> str | None:
    durations: dict[str, int] = {}
    for span in spans:
        if span.quality in _QUALITY_VALUES:
            durations[span.quality] = (
                durations.get(span.quality, 0) + max(span.end_ms - span.start_ms, 1)
            )
    if not durations:
        return None
    label, duration = max(durations.items(), key=lambda item: item[1])
    return label if duration >= QUALITY_MIN_SHARE * sum(durations.values()) else None


def _paused_boundaries(spans: Sequence[SpanFeatures], pauses: Sequence[PauseInterval]) -> int:
    """Word boundaries with a gap between the spans, or a detected pause,
    of at least MIN_PAUSE_MS. A detected pause belongs to the boundary
    between the midpoints of the spans around it."""
    middles = [(s.start_ms + s.end_ms) / 2 for s in spans]
    paused = [
        spans[i + 1].start_ms - spans[i].end_ms >= MIN_PAUSE_MS for i in range(len(spans) - 1)
    ]
    for pause in pauses:
        if pause.end_ms - pause.start_ms < MIN_PAUSE_MS:
            continue
        middle = (pause.start_ms + pause.end_ms) / 2
        for i in range(len(spans) - 1):
            if middles[i] < middle < middles[i + 1]:
                paused[i] = True
                break
    return sum(paused)


def _emphasized_spans(spans: Sequence[SpanFeatures]) -> int | None:
    """Spans standing out from the utterance's median loudness or pitch;
    ``None`` when neither was measured on enough spans."""
    levels = [_finite(f.intensity_mean) for f in spans]
    pitches = [_finite(f.f0_mean) for f in spans]
    measured_levels = [v for v in levels if v is not None]
    measured_pitches = [v for v in pitches if v is not None and v > 0.0]
    if max(len(measured_levels), len(measured_pitches)) < MIN_SPANS_FOR_FREQUENCY:
        return None
    level_median = statistics.median(measured_levels) if measured_levels else None
    pitch_median = statistics.median(measured_pitches) if measured_pitches else None
    count = 0
    for level, pitch in zip(levels, pitches, strict=True):
        loud = (
            level is not None
            and level_median is not None
            and level - level_median >= EMPHASIS_VOLUME_DB
        )
        high = (
            pitch is not None
            and pitch > 0.0
            and pitch_median is not None
            and _semitones(pitch, pitch_median) >= EMPHASIS_PITCH_ST
        )
        count += loud or high
    return count
