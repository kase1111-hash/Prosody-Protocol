"""Mavis Data Bridge -- convert Mavis game data to prosody_protocol datasets.

Mavis is a vocal typing instrument that produces PhonemeEvent data with
prosody parameters (pitch_hz, volume, breathiness, vibrato, duration_ms).
This module converts that data into the prosody_protocol Dataset format
and extracts feature vectors for sklearn training.

Usage::

    from prosody_protocol.mavis_bridge import MavisBridge

    bridge = MavisBridge()

    # Convert a Mavis session to a dataset entry
    entry = bridge.phoneme_events_to_entry(
        events=[...],
        transcript="the SUN is RISING",
        session_id="session_001",
        emotion_label="joyful",
        consent=True,  # the player agreed to their data being stored
    )

    # Extract feature vectors for training
    features = bridge.extract_training_features(events)
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._install import install_hint

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    raise ImportError(
        "MavisBridge requires numpy. Install with: " + install_hint("audio")
    ) from exc

from .datasets import (
    _VALID_ANNOTATORS,
    Dataset,
    DatasetEntry,
    DatasetLoader,
    resolve_audio_path,
)
from .exceptions import DatasetError
from .models import ChildNode, IMLDocument, Prosody, Utterance
from .parser import _XML_INVALID_CHAR_RE, IMLParser

# ---------------------------------------------------------------------------
# Mavis data types (mirrors mavis.llm_processor.PhonemeEvent)
# ---------------------------------------------------------------------------


@dataclass
class PhonemeEvent:
    """A single phoneme event from a Mavis session.

    This mirrors the ``mavis.llm_processor.PhonemeEvent`` dataclass so that
    prosody_protocol can work with Mavis data without importing Mavis.
    """

    phoneme: str
    start_ms: int = 0
    duration_ms: int = 100
    volume: float = 0.5
    pitch_hz: float = 220.0
    vibrato: bool = False
    breathiness: float = 0.0
    harmony_intervals: list[int] | None = None


_EVENT_FIELDS = tuple(f.name for f in fields(PhonemeEvent))

# Feature names in canonical order for sklearn training
MAVIS_FEATURE_NAMES = [
    "mean_pitch_hz",
    "pitch_range_hz",
    "mean_volume",
    "volume_range",
    "mean_breathiness",
    "speech_rate",  # phonemes per second
    "vibrato_ratio",
]

# Session ids become file names (entries/mavis_<id>.json, audio/<id>.wav).
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# A word is marked up when its pitch differs from the session mean by more
# than this percentage, or its volume by more than this ratio.
_WORD_PITCH_DEVIATION_PCT = 10.0
_WORD_VOLUME_DEVIATION = 0.3


# ---------------------------------------------------------------------------
# Emotion mapping from Mavis emphasis levels
# ---------------------------------------------------------------------------


_EMPHASIS_EMOTION_MAP = {
    "shout": "angry",
    "loud": "joyful",
    "soft": "sad",
    "none": "neutral",
}


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------


class MavisBridge:
    """Convert Mavis PhonemeEvent streams to prosody_protocol datasets.

    The bridge produces:
    - Dataset entries (JSON) for the dataset infrastructure
    - Feature vectors (numpy) for sklearn training
    - IML markup from phoneme prosody parameters
    """

    def __init__(self, language: str = "en-US") -> None:
        self.language = language
        self._parser = IMLParser()

    def phoneme_events_to_entry(
        self,
        events: list[PhonemeEvent],
        transcript: str,
        session_id: str,
        emotion_label: str | None = None,
        speaker_id: str | None = None,
        *,
        consent: bool = False,
        annotator: str | None = None,
        phonemes_per_word: Sequence[int] | None = None,
    ) -> DatasetEntry:
        """Convert a sequence of PhonemeEvents into a DatasetEntry.

        Parameters
        ----------
        events:
            Phoneme events from a Mavis session (ordered by ``start_ms``).
        transcript:
            Plain text transcript of the vocal typing session. Characters
            that XML does not allow (control characters other than tab and
            line breaks, such as a terminal's escape character, and lone
            surrogates) are dropped, with a warning once the entry has been
            made; the entry's transcript and IML hold the cleaned text. A
            word made only of such characters disappears with them.
        session_id:
            Unique identifier for this recording session: letters, digits,
            ``.``, ``_`` and ``-`` (it becomes part of file names).
        emotion_label:
            Ground-truth emotion label. If None, inferred from prosody.
        speaker_id:
            Optional speaker identifier.
        consent:
            Whether the speaker explicitly agreed to this data being stored
            (spec 8.1). Recorded on the entry; :meth:`export_dataset` and
            :class:`DatasetLoader` refuse entries without it.
        annotator:
            Who produced the annotation (``human``, ``model`` or
            ``hybrid``). By default ``model`` when the emotion is inferred
            and ``hybrid`` when *emotion_label* is given (the label is the
            caller's, the IML the bridge's).
        phonemes_per_word:
            How many events belong to each word of *transcript*, when known
            (Mavis knows which keystrokes form a word). Otherwise words are
            told apart by the silent gaps between events; see
            :meth:`_group_events_by_word`. Count the words of the cleaned
            transcript: a word made only of characters XML does not allow
            has been removed and must not be counted.

        The events are kept in the entry's metadata as plain JSON values
        (numpy scalars become Python numbers).

        Raises
        ------
        DatasetError
            If *events* or *transcript* is empty, *session_id* is not a safe
            file name, an event's timing or prosody value is not a finite
            number, *emotion_label* contains a character XML does not allow,
            or an argument is out of its vocabulary.

        Warns
        -----
        UserWarning
            Characters that XML does not allow were dropped from
            *transcript* (given only when the entry is made).
        """
        if not events:
            raise DatasetError("Cannot create entry from empty event list")
        if not isinstance(transcript, str) or not transcript.strip():
            raise DatasetError("Cannot create entry with an empty transcript")
        _check_session_id(session_id)
        transcript, dropped_warning = _xml_safe_transcript(transcript, session_id)
        for event in events:
            values = (
                event.start_ms, event.duration_ms, event.volume, event.pitch_hz, event.breathiness
            )
            if not all(isinstance(v, numbers.Real) and math.isfinite(v) for v in values):
                raise DatasetError(
                    f"Phoneme event {event.phoneme!r} has a missing or non-finite value"
                )

        if emotion_label is None:
            emotion_label = self._infer_emotion(events)
            default_annotator = "model"
        else:
            default_annotator = "hybrid"
        if not isinstance(emotion_label, str) or not emotion_label.strip():
            raise DatasetError("emotion_label must be a non-empty string")
        bad = _XML_INVALID_CHAR_RE.search(emotion_label)
        if bad:
            raise DatasetError(
                f"emotion_label {emotion_label!r} contains the character "
                f"U+{ord(bad.group()):04X}, which XML does not allow"
            )
        annotator = default_annotator if annotator is None else annotator
        if annotator not in _VALID_ANNOTATORS:
            raise DatasetError(
                f"annotator must be one of {sorted(_VALID_ANNOTATORS)}, got {annotator!r}"
            )

        iml = self._events_to_iml(events, transcript, emotion_label, phonemes_per_word)
        features = self.extract_training_features(events)
        if dropped_warning is not None:  # only once the input has been accepted
            warnings.warn(dropped_warning, UserWarning, stacklevel=2)

        return DatasetEntry(
            id=f"mavis_{session_id}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="mavis",
            language=self.language,
            audio_file=f"audio/{session_id}.wav",
            transcript=transcript,
            iml=iml,
            emotion_label=emotion_label,
            annotator=annotator,
            consent=consent,
            speaker_id=speaker_id,
            metadata={
                "mavis_events": len(events),
                "phoneme_events": [_event_to_json(event) for event in events],
                "mavis_features": dict(zip(MAVIS_FEATURE_NAMES, features.tolist(), strict=True)),
            },
        )

    def extract_training_features(
        self, events: list[PhonemeEvent]
    ) -> np.ndarray:
        """Extract a 7-dimensional feature vector from a PhonemeEvent sequence.

        Returns a 1D array of shape (7,) with features:
        [mean_pitch_hz, pitch_range_hz, mean_volume, volume_range,
         mean_breathiness, speech_rate, vibrato_ratio]
        """
        if not events:
            return np.zeros(len(MAVIS_FEATURE_NAMES))

        pitches = [e.pitch_hz for e in events]
        volumes = [e.volume for e in events]
        breathiness = [e.breathiness for e in events]
        total_duration_s = sum(e.duration_ms for e in events) / 1000.0
        vibrato_count = sum(1 for e in events if e.vibrato)

        return np.array([
            float(np.mean(pitches)),
            float(max(pitches) - min(pitches)),
            float(np.mean(volumes)),
            float(max(volumes) - min(volumes)),
            float(np.mean(breathiness)),
            len(events) / max(total_duration_s, 0.001),
            vibrato_count / len(events),
        ], dtype=np.float64)

    def batch_extract_features(
        self, sessions: list[list[PhonemeEvent]]
    ) -> np.ndarray:
        """Extract features from multiple sessions into a feature matrix.

        Returns array of shape (n_sessions, 7).
        """
        return np.vstack([
            self.extract_training_features(session) for session in sessions
        ])

    @staticmethod
    def events_from_entry(entry: DatasetEntry) -> list[PhonemeEvent]:
        """Return the phoneme events an entry was made from.

        Entries made by this bridge keep their events in
        ``metadata["phoneme_events"]``, so a loaded dataset can be turned
        back into feature vectors with :meth:`extract_training_features`.
        Returns an empty list for entries without events.
        """
        raw = entry.metadata.get("phoneme_events")
        if not isinstance(raw, list):
            return []
        try:
            return [PhonemeEvent(**item) for item in raw]
        except TypeError as exc:
            raise DatasetError(f"Entry {entry.id} has malformed phoneme_events: {exc}") from exc

    def export_dataset(
        self,
        sessions: list[dict[str, Any]],
        output_dir: str | Path,
        *,
        consent: bool | None = None,
        overwrite: bool = False,
    ) -> Dataset:
        """Export multiple Mavis sessions as a prosody_protocol dataset.

        Every session is checked, and every entry serialised, before anything
        is written. The files are then written to a staging directory inside
        *output_dir* and only moved into place once all of them are written,
        so an invalid session or a failure while writing (a full disk, say)
        leaves an earlier export intact. The result loads back with
        :class:`DatasetLoader` and conforms to
        ``schemas/dataset-entry.schema.json``.

        Parameters
        ----------
        sessions:
            List of dicts with keys 'events' (list[PhonemeEvent]),
            'transcript' (str), 'session_id' (str), and optional
            'emotion_label' (str), 'speaker_id' (str), 'consent' (bool),
            'annotator' (str), 'phonemes_per_word' (list[int]) and
            'audio_path' (a recording of the session, copied to
            ``audio/<session_id>.wav``). Session ids must be unique.
        output_dir:
            Directory to write the dataset to.
        consent:
            Consent for sessions that do not state their own. Every session
            must end up with ``True`` -- explicit consent is required before
            emotional annotations are stored (spec 8.1).
        overwrite:
            Replace an earlier export in *output_dir*: once the new files are
            in place, the earlier entry files and the audio files under
            ``audio/`` that they reference are removed (so a session left
            out of the new export leaves nothing behind). Without it,
            existing entries are an error.

        Entries of sessions without 'audio_path' reference an audio file
        that does not exist; load them without ``check_audio``. Characters
        that XML does not allow are dropped from transcripts as in
        :meth:`phoneme_events_to_entry`, with one warning per session once
        the export has been written.

        Raises
        ------
        DatasetError
            If a session lacks consent, has a missing key, a duplicate or
            unsafe session id, a missing audio file, an invalid entry or a
            value that cannot be stored as JSON, if the output directory
            already holds entries, or if writing fails.

        Returns the dataset as :meth:`DatasetLoader.load` reads it back.
        """
        output_path = Path(output_dir)
        loader = DatasetLoader()

        prepared: list[_PreparedEntry] = []
        dropped_warnings: list[str] = []
        seen: set[str] = set()
        for index, session in enumerate(sessions):
            missing = [k for k in ("events", "transcript", "session_id") if k not in session]
            if missing:
                raise DatasetError(f"Session {index} is missing {', '.join(missing)}")
            session_id = session["session_id"]
            _check_session_id(session_id)
            if session_id in seen:
                raise DatasetError(f"Duplicate session_id {session_id!r}")
            seen.add(session_id)

            session_consent = session.get("consent", consent)
            if session_consent is not True:
                raise DatasetError(
                    f"Session {session_id!r} has no recorded consent: pass consent=True "
                    "(or a 'consent' key) only if the speaker explicitly agreed to their "
                    "data being stored"
                )
            audio_source = session.get("audio_path")
            if audio_source is not None and not Path(audio_source).is_file():
                raise DatasetError(f"Audio for session {session_id!r} not found: {audio_source}")

            transcript = session["transcript"]
            if isinstance(transcript, str):  # else phoneme_events_to_entry rejects it
                # Cleaned here, so that the warning is given once and points
                # at the caller of export_dataset.
                transcript, dropped_warning = _xml_safe_transcript(transcript, session_id)
                if dropped_warning is not None:
                    dropped_warnings.append(dropped_warning)
            entry = self.phoneme_events_to_entry(
                events=session["events"],
                transcript=transcript,
                session_id=session_id,
                emotion_label=session.get("emotion_label"),
                speaker_id=session.get("speaker_id"),
                consent=True,
                annotator=session.get("annotator"),
                phonemes_per_word=session.get("phonemes_per_word"),
            )
            record = _entry_to_dict(entry)
            result = loader.validate_entry(record)
            if not result.valid:
                problems = "; ".join(f"{i.rule} {i.message}" for i in result.errors)
                raise DatasetError(f"Session {session_id!r} makes an invalid entry: {problems}")
            try:
                text = json.dumps(record, indent=2, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"Session {session_id!r} cannot be stored as JSON: {exc}"
                ) from exc
            prepared.append(_PreparedEntry(
                entry, text, None if audio_source is None else Path(audio_source)
            ))

        entries_dir = output_path / "entries"
        existing = sorted(entries_dir.glob("*.json")) if entries_dir.is_dir() else []
        if existing and not overwrite:
            raise DatasetError(
                f"{entries_dir} already contains {len(existing)} entries; "
                "export to an empty directory or pass overwrite=True"
            )

        meta = {
            "name": output_path.name,
            "version": "0.1.0",
            "size": len(prepared),
            "source": "mavis",
            "language": self.language,
        }
        _write_export(output_path, prepared, json.dumps(meta, indent=2), existing)
        for message in dropped_warnings:  # only for an export that succeeded
            warnings.warn(message, UserWarning, stacklevel=2)
        return loader.load(output_path)

    # -- Private helpers ----------------------------------------------------

    def _infer_emotion(self, events: list[PhonemeEvent]) -> str:
        """Infer a simple emotion label from aggregate prosody features."""
        mean_volume = sum(e.volume for e in events) / len(events)
        mean_breathiness = sum(e.breathiness for e in events) / len(events)
        mean_pitch = sum(e.pitch_hz for e in events) / len(events)

        if mean_volume > 0.8:
            return "angry" if mean_pitch > 300 else "joyful"
        if mean_breathiness > 0.5:
            return "sad"
        if mean_volume < 0.3:
            return "calm"
        return "neutral"

    def _events_to_iml(
        self,
        events: list[PhonemeEvent],
        transcript: str,
        emotion: str,
        phonemes_per_word: Sequence[int] | None = None,
    ) -> str:
        """Build IML markup from phoneme events and transcript.

        A word whose events' mean pitch differs from the session mean by more
        than 10 %, or whose mean volume differs by more than 30 %, is wrapped
        in ``<prosody>`` with its relative pitch and volume.
        """
        # Compute overall prosody stats for the utterance
        mean_pitch = sum(e.pitch_hz for e in events) / max(len(events), 1)
        mean_volume = sum(e.volume for e in events) / max(len(events), 1)

        confidence = self._compute_confidence(events)

        words = transcript.split()
        groups = self._group_events_by_word(events, len(words), phonemes_per_word)
        parts: list[ChildNode] = []
        for i, (word, word_events) in enumerate(zip(words, groups, strict=True)):
            if i:
                parts.append(" ")
            parts.append(self._word_node(word, word_events, mean_pitch, mean_volume))
        children: list[ChildNode] = []
        for part in parts:
            last = children[-1] if children else None
            if isinstance(part, str) and isinstance(last, str):
                children[-1] = last + part
            else:
                children.append(part)

        doc = IMLDocument(utterances=(Utterance(
            children=tuple(children),
            emotion=emotion,
            confidence=round(confidence, 2),
        ),))
        return self._parser.to_iml_string(doc)

    @staticmethod
    def _word_node(
        word: str,
        word_events: list[PhonemeEvent],
        mean_pitch: float,
        mean_volume: float,
    ) -> ChildNode:
        """The word as plain text, or wrapped in <prosody> when it stands out."""
        if not word_events:
            return word
        word_pitch = sum(e.pitch_hz for e in word_events) / len(word_events)
        word_vol = sum(e.volume for e in word_events) / len(word_events)

        pitch_dev = (word_pitch - mean_pitch) / max(mean_pitch, 1) * 100
        vol_ratio = word_vol / max(mean_volume, 0.001)
        if (
            abs(pitch_dev) <= _WORD_PITCH_DEVIATION_PCT
            and abs(vol_ratio - 1.0) <= _WORD_VOLUME_DEVIATION
        ):
            return word
        vol_db = 20.0 * math.log10(max(vol_ratio, 0.001))
        return Prosody(children=(word,), pitch=f"{pitch_dev:+.0f}%", volume=f"{vol_db:+.0f}dB")

    @staticmethod
    def _group_events_by_word(
        events: list[PhonemeEvent],
        n_words: int,
        phonemes_per_word: Sequence[int] | None = None,
    ) -> list[list[PhonemeEvent]]:
        """Assign the events, in time order, to the *n_words* words.

        With *phonemes_per_word* the counts are used as given. Otherwise the
        words are separated at the ``n_words - 1`` longest silent gaps
        between consecutive events (words are typed with a gap between
        them). Without enough gaps, the events are shared out in order, word
        *i* getting events ``i*n//n_words`` up to ``(i+1)*n//n_words``. Every
        event is assigned; a word gets none only when there are fewer events
        than words.
        """
        ordered = sorted(events, key=lambda e: e.start_ms)
        if phonemes_per_word is not None:
            counts = list(phonemes_per_word)
            if len(counts) != n_words or any(c < 0 for c in counts) or sum(counts) != len(events):
                raise DatasetError(
                    f"phonemes_per_word {counts} must give a non-negative count for each "
                    f"of the {n_words} words, adding up to the {len(events)} events"
                )
            cuts = [sum(counts[:i]) for i in range(1, n_words)]
        else:
            gaps = [
                (ordered[i + 1].start_ms - (ordered[i].start_ms + ordered[i].duration_ms), i)
                for i in range(len(ordered) - 1)
            ]
            silent = sorted((g for g in gaps if g[0] > 0), key=lambda g: (-g[0], g[1]))
            if len(silent) >= n_words - 1:
                cuts = sorted(i + 1 for _, i in silent[: n_words - 1])
            else:
                cuts = [i * len(ordered) // n_words for i in range(1, n_words)]
        bounds = [0, *cuts, len(ordered)]
        return [ordered[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]

    def _compute_confidence(self, events: list[PhonemeEvent]) -> float:
        """Compute confidence score based on prosodic distinctiveness."""
        if len(events) < 2:
            return 0.5

        volumes = [e.volume for e in events]
        pitches = [e.pitch_hz for e in events]

        vol_range = max(volumes) - min(volumes)
        pitch_range = max(pitches) - min(pitches)

        # More dynamic range → higher confidence
        confidence = 0.5 + min(vol_range * 0.3 + pitch_range / 500, 0.4)
        return float(min(confidence, 0.95))


@dataclass(frozen=True)
class _PreparedEntry:
    """An entry ready to be written: its JSON text and the audio to copy."""

    entry: DatasetEntry
    text: str
    audio_source: Path | None


def _write_export(
    output_path: Path,
    prepared: list[_PreparedEntry],
    metadata_text: str,
    existing: list[Path],
) -> None:
    """Write an export through a staging directory inside *output_path*.

    Files are only moved into ``entries/``, ``audio/`` and ``metadata.json``
    once all of them were written; then the *existing* entry files that the
    new export did not replace, and the audio under ``audio/`` they
    referenced, are removed.
    """
    created = not output_path.exists()
    stale_audio = _exported_audio(output_path, existing)
    staging: Path | None = None
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".export-", dir=output_path))
        for item in prepared:
            (staging / f"{item.entry.id}.json").write_text(item.text, encoding="utf-8")
            if item.audio_source is not None:
                shutil.copyfile(item.audio_source, staging / f"{item.entry.id}.audio")
        (staging / "metadata.json").write_text(metadata_text, encoding="utf-8")

        entries_dir = output_path / "entries"
        entries_dir.mkdir(exist_ok=True)
        written: set[str] = set()
        kept_audio: set[Path] = set()
        for item in prepared:
            name = f"{item.entry.id}.json"
            (staging / name).replace(entries_dir / name)
            written.add(name)
            if item.audio_source is not None:
                target = output_path / item.entry.audio_file
                target.parent.mkdir(parents=True, exist_ok=True)
                (staging / f"{item.entry.id}.audio").replace(target)
                kept_audio.add(Path(os.path.normpath(target)))
        for stale in existing:
            if stale.name not in written:
                stale.unlink(missing_ok=True)
        for stale_file in stale_audio - kept_audio:
            stale_file.unlink(missing_ok=True)
        (staging / "metadata.json").replace(output_path / "metadata.json")
    except OSError as exc:
        if created:
            shutil.rmtree(output_path, ignore_errors=True)
        raise DatasetError(f"Cannot write dataset to {output_path}: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _exported_audio(output_path: Path, entry_files: list[Path]) -> set[Path]:
    """Audio files under ``output_path/audio`` that *entry_files* reference.

    Unreadable entries and paths leaving the dataset are ignored, so only
    files an export could have written are ever removed.
    """
    audio_dir = Path(os.path.normpath(output_path / "audio"))
    found: set[Path] = set()
    for entry_file in entry_files:
        try:
            raw = json.loads(entry_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        audio_file = raw.get("audio_file") if isinstance(raw, dict) else None
        if not isinstance(audio_file, str):
            continue
        try:
            path = Path(os.path.normpath(resolve_audio_path(output_path, audio_file)))
        except DatasetError:
            continue
        if path.is_relative_to(audio_dir) and path.is_file():
            found.add(path)
    return found


def _event_to_json(event: PhonemeEvent) -> dict[str, Any]:
    """The event's fields as JSON values.

    Audio code often produces numpy scalars (``np.int64`` start times,
    ``np.float32`` pitches), which :mod:`json` cannot write.
    """
    return {name: _json_value(getattr(event, name)) for name in _EVENT_FIELDS}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    return value


def _xml_safe_transcript(transcript: str, session_id: str) -> tuple[str, str | None]:
    """*transcript* without the characters XML does not allow, and the
    warning to give about them (None when nothing was dropped).

    Mavis text is typed, so a stray control character (a terminal escape
    sequence, say) is dropped with a warning rather than failing the
    session. Those that :meth:`str.split` treats as word separators
    (vertical tab, form feed, U+001C-U+001F) become spaces, so that the
    words they separate stay apart; the others are removed. The caller
    gives the warning once the rest of its input has been accepted.

    Raises DatasetError if nothing but whitespace is left.
    """
    found = _XML_INVALID_CHAR_RE.findall(transcript)
    if not found:
        return transcript, None
    cleaned = _XML_INVALID_CHAR_RE.sub(lambda m: " " if m.group().isspace() else "", transcript)
    if not cleaned.strip():
        raise DatasetError(
            f"Cannot create entry with an empty transcript: the transcript of session "
            f"{session_id!r} holds only characters that XML does not allow"
        )
    distinct = list(dict.fromkeys(found))
    codes = ", ".join(f"U+{ord(char):04X}" for char in distinct[:3])
    if len(distinct) > 3:
        codes += ", ..."
    count = f"{len(found)} character{'s' if len(found) > 1 else ''}"
    return cleaned, (
        f"Dropped {count} that XML does not allow ({codes}) from the transcript of "
        f"session {session_id!r}"
    )


def _check_session_id(session_id: object) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise DatasetError(
            f"session_id {session_id!r} must start with a letter or digit and contain "
            "only letters, digits, '.', '_' and '-'"
        )


def _entry_to_dict(entry: DatasetEntry) -> dict[str, object]:
    """An entry as the JSON object stored in ``entries/``."""
    return {
        "id": entry.id,
        "timestamp": entry.timestamp,
        "source": entry.source,
        "language": entry.language,
        "audio_file": entry.audio_file,
        "transcript": entry.transcript,
        "iml": entry.iml,
        "emotion_label": entry.emotion_label,
        "annotator": entry.annotator,
        "consent": entry.consent,
        "speaker_id": entry.speaker_id,
        "metadata": entry.metadata,
    }
