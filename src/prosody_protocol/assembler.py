"""IML Assembler -- constructs IMLDocument objects from pipeline outputs.

Takes STT word alignments, prosody features, emotion classification,
and pause information, and assembles a structured IML document.

Steps:
  1. Group words into utterances (by sentence boundaries or long pauses)
  2. Classify emotion per utterance
  3. Determine prosodic deviation from speaker baseline
  4. Wrap deviating spans in <prosody> tags with relative values
  5. Detect emphasis from intensity/F0 spikes
  6. Insert <pause> elements for significant gaps (> 200ms)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .emotion_classifier import EmotionClassifier, RuleBasedEmotionClassifier
from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Utterance,
)
from .prosody_analyzer import PauseInterval, SpanFeatures, WordAlignment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pauses longer than this (ms) split utterances.
UTTERANCE_SPLIT_PAUSE_MS = 1000

# Pauses shorter than utterance split but >=200ms become <pause> elements.
MIN_PAUSE_MS = 200

# Thresholds for wrapping a word in <prosody> -- relative to utterance baseline.
F0_DEVIATION_PCT = 15.0
INTENSITY_DEVIATION_DB = 5.0

# Threshold for <emphasis>: intensity spike above utterance mean.
EMPHASIS_INTENSITY_DB = 6.0
EMPHASIS_F0_PCT = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _WordWithFeatures:
    alignment: WordAlignment
    features: SpanFeatures | None


def _is_sentence_boundary(word: str) -> bool:
    """Check if a word ends with sentence-terminal punctuation."""
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def _group_into_utterances(
    words: list[_WordWithFeatures],
    pauses: list[PauseInterval],
) -> list[list[_WordWithFeatures]]:
    """Split words into utterance groups based on pauses and sentence endings."""
    if not words:
        return []

    pause_lookup: set[int] = set()
    for p in pauses:
        if p.duration_ms >= UTTERANCE_SPLIT_PAUSE_MS:
            pause_lookup.add(p.start_ms)

    groups: list[list[_WordWithFeatures]] = []
    current: list[_WordWithFeatures] = []

    for i, wf in enumerate(words):
        current.append(wf)

        should_split = False

        # Split on sentence-ending punctuation.
        if _is_sentence_boundary(wf.alignment.word):
            should_split = True

        # Split if there's a long pause after this word.
        if i < len(words) - 1:
            gap_start = wf.alignment.end_ms
            gap_end = words[i + 1].alignment.start_ms
            if gap_end - gap_start >= UTTERANCE_SPLIT_PAUSE_MS:
                should_split = True

        if should_split and current:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _compute_baseline(features_list: list[SpanFeatures]) -> tuple[float, float]:
    """Compute utterance-level baseline F0 and intensity."""
    f0s = [f.f0_mean for f in features_list if f.f0_mean is not None]
    intensities = [f.intensity_mean for f in features_list if f.intensity_mean is not None]

    baseline_f0 = sum(f0s) / len(f0s) if f0s else 180.0
    baseline_intensity = sum(intensities) / len(intensities) if intensities else 65.0

    return baseline_f0, baseline_intensity


def _relative_pitch(f0: float, baseline: float) -> str:
    """Express pitch as a relative percentage from baseline."""
    pct = ((f0 - baseline) / baseline) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _relative_volume(intensity: float, baseline: float) -> str:
    """Express volume as relative dB from baseline."""
    diff = intensity - baseline
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.0f}dB"


def _build_children(
    words: list[_WordWithFeatures],
    pauses: list[PauseInterval],
    baseline_f0: float,
    baseline_intensity: float,
    include_extended: bool,
) -> tuple[ChildNode, ...]:
    """Build a mixed-content children tuple for an utterance.

    Inserts <prosody>, <emphasis>, and <pause> elements as needed.
    """
    children: list[ChildNode] = []
    pause_map: dict[int, PauseInterval] = {p.start_ms: p for p in pauses}

    for i, wf in enumerate(words):
        feat = wf.features

        # Check for a pause before this word.
        if i > 0:
            prev_end = words[i - 1].alignment.end_ms
            gap_start = prev_end
            gap_end = wf.alignment.start_ms
            gap_ms = gap_end - gap_start

            # Insert pause if gap is significant.
            if gap_ms >= MIN_PAUSE_MS:
                children.append(Pause(duration=gap_ms))
            elif gap_ms > 0:
                children.append(" ")

        text = wf.alignment.word

        # Determine if this word needs prosody or emphasis wrapping.
        needs_prosody = False
        needs_emphasis = False
        pitch_str: str | None = None
        volume_str: str | None = None

        if feat is not None:
            if feat.f0_mean is not None:
                deviation_pct = abs((feat.f0_mean - baseline_f0) / baseline_f0) * 100
                if deviation_pct > F0_DEVIATION_PCT:
                    needs_prosody = True
                    pitch_str = _relative_pitch(feat.f0_mean, baseline_f0)
                if deviation_pct > EMPHASIS_F0_PCT:
                    needs_emphasis = True

            if feat.intensity_mean is not None:
                deviation_db = feat.intensity_mean - baseline_intensity
                if abs(deviation_db) > INTENSITY_DEVIATION_DB:
                    needs_prosody = True
                    volume_str = _relative_volume(feat.intensity_mean, baseline_intensity)
                if deviation_db > EMPHASIS_INTENSITY_DB:
                    needs_emphasis = True

        if needs_emphasis:
            inner: tuple[ChildNode, ...] = (text,)
            if needs_prosody:
                kwargs: dict[str, object] = {
                    "children": (text,),
                    "pitch": pitch_str,
                    "volume": volume_str,
                }
                if include_extended and feat is not None:
                    kwargs.update(_extended_attrs(feat))
                inner = (Prosody(**kwargs),)  # type: ignore[arg-type]
            children.append(Emphasis(level="strong", children=inner))
        elif needs_prosody:
            kwargs = {
                "children": (text,),
                "pitch": pitch_str,
                "volume": volume_str,
            }
            if include_extended and feat is not None:
                kwargs.update(_extended_attrs(feat))
            children.append(Prosody(**kwargs))  # type: ignore[arg-type]
        else:
            # Plain text -- add spacing between words.
            if children and isinstance(children[-1], str) and not children[-1].endswith(" "):
                children.append(" " + text)
            else:
                children.append(text)

    return tuple(children)


def _extended_attrs(feat: SpanFeatures) -> dict[str, object]:
    """Build extended attribute dict from SpanFeatures."""
    attrs: dict[str, object] = {}
    if feat.f0_mean is not None:
        attrs["f0_mean"] = round(feat.f0_mean, 1)
    if feat.f0_range is not None:
        attrs["f0_range"] = f"{feat.f0_range[0]:.0f}-{feat.f0_range[1]:.0f}"
    if feat.intensity_mean is not None:
        attrs["intensity_mean"] = round(feat.intensity_mean, 1)
    if feat.intensity_range is not None:
        attrs["intensity_range"] = round(feat.intensity_range, 1)
    if feat.speech_rate is not None:
        attrs["speech_rate"] = round(feat.speech_rate, 1)
    if feat.jitter is not None:
        attrs["jitter"] = round(feat.jitter, 4)
    if feat.shimmer is not None:
        attrs["shimmer"] = round(feat.shimmer, 4)
    if feat.hnr is not None:
        attrs["hnr"] = round(feat.hnr, 1)
    return attrs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLAssembler:
    """Assemble an IMLDocument from pipeline components."""

    def __init__(
        self,
        emotion_classifier: EmotionClassifier | None = None,
        include_extended: bool = False,
    ) -> None:
        self._classifier: EmotionClassifier = (
            emotion_classifier or RuleBasedEmotionClassifier()
        )
        self._include_extended = include_extended

    def assemble(
        self,
        alignments: list[WordAlignment],
        features: list[SpanFeatures],
        pauses: list[PauseInterval],
        language: str | None = None,
    ) -> IMLDocument:
        """Build an IMLDocument from pipeline outputs.

        Parameters
        ----------
        alignments:
            Word-level time boundaries from STT.
        features:
            Prosodic features per word span (same length as *alignments*).
        pauses:
            Detected silence intervals.
        language:
            Optional BCP-47 language tag.

        Returns
        -------
        IMLDocument
        """
        # Pair words with their features using (start_ms, end_ms) to avoid
        # collisions when two words share the same start timestamp.
        feat_map: dict[tuple[int, int], SpanFeatures] = {
            (f.start_ms, f.end_ms): f for f in features
        }
        words = [
            _WordWithFeatures(
                alignment=a,
                features=feat_map.get((a.start_ms, a.end_ms)),
            )
            for a in alignments
        ]

        groups = _group_into_utterances(words, pauses)

        utterances: list[Utterance] = []
        for group in groups:
            group_features = [wf.features for wf in group if wf.features is not None]
            emotion, confidence = self._classifier.classify(group_features)

            baseline_f0, baseline_intensity = _compute_baseline(group_features)

            children = _build_children(
                group,
                pauses,
                baseline_f0,
                baseline_intensity,
                self._include_extended,
            )

            utterances.append(Utterance(
                children=children,
                emotion=emotion,
                confidence=confidence,
            ))

        return IMLDocument(
            utterances=tuple(utterances),
            version="0.1.0",
            language=language,
        )
