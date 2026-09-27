"""AudioToIML -- convert audio files to IML-annotated transcripts.

Orchestrates STT, prosody analysis, emotion classification, and IML assembly.

Pipeline:
  1. Words with timings: supplied by the caller, or from Whisper
  2. ProsodyAnalyzer extracts acoustic features per word span
  3. Pause detection finds silence gaps
  4. IMLAssembler constructs the document with emotion/prosody/emphasis/pause tags

Without word timings the output is coarser: a caller transcript gets
utterance-level prosody only, and with no transcript at all each stretch
of speech becomes a ``[speech]`` placeholder. Such output is flagged in
:attr:`ConversionResult.warnings`.

Spec reference: Sections 3-4.
"""

from __future__ import annotations

import dataclasses
import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ._types import PauseInterval, SpanFeatures, WordAlignment
from .assembler import DEFAULT_MIN_EMOTION_CONFIDENCE, IMLAssembler
from .emotion_classifier import EmotionClassifier
from .exceptions import AudioProcessingError
from .models import IMLDocument, Utterance
from .parser import IMLParser
from .prosody_analyzer import _AudioAnalysis, _checked_max_duration

__all__ = ["PLACEHOLDER_TOKEN", "AudioToIML", "ConversionResult"]

#: Token standing in for each stretch of speech when no transcript exists.
PLACEHOLDER_TOKEN = "[speech]"

# Whisper models expect 16 kHz mono float32 audio.
_WHISPER_SAMPLE_RATE = 16_000

_STT_MODES = ("auto", "whisper", "none")

# Calibration speech is measured in spans of about one word.
_CALIBRATION_SPAN_MS = 300

# Characters outside the XML 1.0 ``Char`` production cannot appear in IML.
_XML_INVALID_CHARS = re.compile("[^\t\n\r\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


@dataclass(frozen=True)
class ConversionResult:
    """The result of :meth:`AudioToIML.convert_detailed`.

    Attributes
    ----------
    document:
        The assembled IML document.
    iml:
        *document* serialised as an IML XML string.
    transcript_source:
        Where the words came from: ``"words"`` (caller-supplied word
        timings), ``"transcript"`` (caller text without timings, so prosody
        is utterance-level only), ``"whisper"`` (built-in speech
        recognition) or ``"none"`` (no transcript; each stretch of speech
        is a ``[speech]`` placeholder).
    warnings:
        Human-readable notes on anything that degraded the output, such as
        placeholder text or audio without voiced speech.
    """

    document: IMLDocument
    iml: str
    transcript_source: Literal["words", "transcript", "whisper", "none"]
    warnings: tuple[str, ...] = ()


def _whisper_language(tag: str) -> str:
    """Map a BCP-47 tag (``en-US``) to the ISO 639 code Whisper takes (``en``)."""
    return tag.replace("_", "-").split("-")[0].lower()


def _checked_text(text: object, what: str) -> str:
    """Check that caller text is a string that can appear in XML."""
    if not isinstance(text, str):
        raise TypeError(f"{what} must be a str, not {type(text).__name__}")
    bad = _XML_INVALID_CHARS.search(text)
    if bad:
        raise ValueError(
            f"{what} contains the character {bad.group()!r}, which XML does not allow"
        )
    return text


def _checked_words(words: Sequence[WordAlignment]) -> list[WordAlignment]:
    """Validate caller-supplied word timings and return them in time order."""
    words = list(words)  # an iterator can be read only once
    for i, word in enumerate(words):
        if not isinstance(word, WordAlignment):
            raise TypeError(f"words[{i}] must be a WordAlignment, not {type(word).__name__}")
        _checked_text(word.word, f"words[{i}].word")
        for name in ("start_ms", "end_ms"):
            value = getattr(word, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"words[{i}].{name} must be a number, not {type(value).__name__}"
                )
        if not (
            math.isfinite(word.start_ms) and math.isfinite(word.end_ms)
            and 0 <= word.start_ms <= word.end_ms
        ):
            raise ValueError(
                f"words[{i}] ({word.word!r}) has invalid timings "
                f"{word.start_ms}-{word.end_ms} ms"
            )
    return sorted(words, key=lambda w: (w.start_ms, w.end_ms))


class AudioToIML:
    """Convert audio files to IML markup.

    Parameters
    ----------
    stt_model:
        Whisper model size (``"tiny"``, ``"base"``, ``"small"``, etc.).
        The model is loaded on first use and reused by later conversions.
    emotion_classifier:
        Optional custom emotion classifier.  Defaults to the rule-based
        baseline.
    include_extended:
        When ``True``, include extended prosodic attributes (f0_mean,
        jitter, etc.) on ``<prosody>`` elements.
    language:
        Optional BCP-47 language tag. It labels the output document and is
        passed to Whisper (as ``en`` for ``en-US``). When it is ``None``,
        the language Whisper detects labels the document.
    min_emotion_confidence:
        Utterances whose emotion is classified with lower confidence carry
        no ``emotion`` or ``confidence`` attribute.
    calibration_audio:
        Optional recording of the same speaker talking neutrally. Its pitch
        and loudness become the baseline that relative ``pitch``/``volume``
        values and emphasis are measured against, instead of the converted
        file's own average.
    stt:
        Where words come from when a conversion gets neither ``words`` nor
        ``transcript``: ``"whisper"`` requires openai-whisper, ``"none"``
        never transcribes and emits ``[speech]`` placeholders, and
        ``"auto"`` (default) uses Whisper when it is installed and
        placeholders otherwise.
    max_duration_s:
        Optional limit on the length of the audio (and of
        ``calibration_audio``), in seconds. Longer audio is rejected with
        :class:`~prosody_protocol.exceptions.AudioProcessingError` before it
        is loaded, since a small compressed upload can decode to hours of
        audio. ``None`` (default) means no limit.
    """

    def __init__(
        self,
        stt_model: str = "base",
        emotion_classifier: EmotionClassifier | None = None,
        include_extended: bool = False,
        language: str | None = None,
        *,
        min_emotion_confidence: float = DEFAULT_MIN_EMOTION_CONFIDENCE,
        calibration_audio: str | Path | None = None,
        stt: Literal["auto", "whisper", "none"] = "auto",
        max_duration_s: float | None = None,
    ) -> None:
        if stt not in _STT_MODES:
            raise ValueError(f"stt must be one of {', '.join(_STT_MODES)}; got {stt!r}")
        self.max_duration_s = _checked_max_duration(max_duration_s)
        self.stt_model = stt_model
        self.include_extended = include_extended
        self.language = language
        self.min_emotion_confidence = min_emotion_confidence
        self.calibration_audio = None if calibration_audio is None else Path(calibration_audio)
        self.stt = stt
        self._assembler = IMLAssembler(
            emotion_classifier=emotion_classifier,
            include_extended=include_extended,
            min_emotion_confidence=min_emotion_confidence,
        )
        self._parser = IMLParser()
        self._whisper_models: dict[str, Any] = {}
        self._calibration: tuple[Path, list[SpanFeatures]] | None = None

    # -- Public API ---------------------------------------------------------

    def convert(
        self,
        audio_path: str | Path,
        *,
        words: Sequence[WordAlignment] | None = None,
        transcript: str | None = None,
    ) -> str:
        """Convert an audio file to an IML XML string.

        See :meth:`convert_detailed` for the parameters. Warnings about
        degraded output are issued as :class:`UserWarning`.

        Raises :class:`~prosody_protocol.exceptions.AudioProcessingError`
        if the audio file cannot be read or processed.
        """
        result = self._convert(audio_path, words, transcript)
        for message in result.warnings:
            warnings.warn(message, UserWarning, stacklevel=2)
        return result.iml

    def convert_to_doc(
        self,
        audio_path: str | Path,
        *,
        words: Sequence[WordAlignment] | None = None,
        transcript: str | None = None,
    ) -> IMLDocument:
        """Convert an audio file to a parsed :class:`IMLDocument`.

        See :meth:`convert_detailed` for the parameters. Warnings about
        degraded output are issued as :class:`UserWarning`.

        Raises :class:`~prosody_protocol.exceptions.AudioProcessingError`
        if the audio file cannot be read or processed.
        """
        result = self._convert(audio_path, words, transcript)
        for message in result.warnings:
            warnings.warn(message, UserWarning, stacklevel=2)
        return result.document

    def convert_detailed(
        self,
        audio_path: str | Path,
        *,
        words: Sequence[WordAlignment] | None = None,
        transcript: str | None = None,
    ) -> ConversionResult:
        """Convert an audio file and report how the transcript was obtained.

        Steps:
          1. Get words: *words* as given, else *transcript*, else speech
             recognition according to ``stt``
          2. Extract prosodic features per word
          3. Detect pauses
          4. Assemble IML document

        Parameters
        ----------
        audio_path:
            Audio file. WAV, AIFF, FLAC and MP3 are read directly; other
            formats (OGG/Opus, WebM, M4A, ...) need ffmpeg on ``PATH``.
        words:
            Word timings from any speech recogniser. No speech recognition
            runs. Each ``word`` is used as-is (surrounding whitespace is
            allowed); timings beyond the end of the audio are clamped.
        transcript:
            The spoken text without timings. No speech recognition runs;
            the text becomes one utterance whose emotion and prosody are
            measured over the whole stretch of speech. No word-level tags
            or pauses are placed, because nothing says where the words are.

        Raises
        ------
        AudioProcessingError
            The audio cannot be read, decoded or analysed, is longer than
            ``max_duration_s``, Whisper is required but not installed, or
            Whisper fails.
        ValueError
            Both *words* and *transcript* are given, a word has negative,
            reversed or non-finite timings, or the text contains characters
            that XML does not allow (such as control characters).
        TypeError
            An item of *words* is not a :class:`WordAlignment`, or a word,
            timing or *transcript* has the wrong type.
        """
        return self._convert(audio_path, words, transcript)

    # -- Pipeline -----------------------------------------------------------

    def _convert(
        self,
        audio_path: str | Path,
        words: Sequence[WordAlignment] | None,
        transcript: str | None,
    ) -> ConversionResult:
        if words is not None and transcript is not None:
            raise ValueError("Pass either words= or transcript=, not both.")
        checked = None if words is None else _checked_words(words)

        if transcript is not None:
            transcript = _checked_text(transcript, "transcript")

        analysis = _AudioAnalysis.from_path(audio_path, self.max_duration_s)
        notes: list[str] = []
        language = self.language
        pauses = analysis.pauses()
        source: Literal["words", "transcript", "whisper", "none"]

        if checked is not None:
            source = "words"
            alignments = checked
            if not alignments:
                notes.append("No words were supplied, so the document has no text.")
            outside = sum(1 for w in alignments if w.start_ms >= analysis.duration_ms)
            if outside:
                notes.append(
                    f"{outside} word(s) start after the end of the audio "
                    f"({analysis.duration_ms} ms) and carry no acoustic features."
                )
        elif transcript is not None:
            source = "transcript"
            alignments = self._transcript_alignment(analysis, transcript, notes)
            # Without word timings there is nowhere to put pauses in the text.
            pauses = []
        elif self.stt == "none":
            source = "none"
            alignments = self._placeholder_words(analysis, notes, "stt='none'")
        else:
            whisper = self._import_whisper()
            if whisper is None:
                source = "none"
                alignments = self._placeholder_words(
                    analysis, notes, "openai-whisper is not installed"
                )
            else:
                source = "whisper"
                alignments, detected = self._transcribe(whisper, analysis)
                language = language or detected
                if not alignments:
                    notes.append("Speech recognition found no words in the audio.")

        document = self._assemble(analysis, alignments, pauses, language)
        if not analysis.has_speech:
            # Silence and noise are never given an emotion.
            notes.append("No voiced speech was detected in the audio; no emotion is reported.")
            document = dataclasses.replace(document, utterances=tuple(
                dataclasses.replace(u, emotion=None, confidence=None)
                for u in document.utterances
            ))

        return ConversionResult(
            document=document,
            iml=self._parser.to_iml_string(document),
            transcript_source=source,
            warnings=tuple(notes),
        )

    def _assemble(
        self,
        analysis: _AudioAnalysis,
        alignments: list[WordAlignment],
        pauses: list[PauseInterval],
        language: str | None,
    ) -> IMLDocument:
        if not alignments:
            # Nothing was said: an empty utterance with no emotion (don't
            # fabricate a classification from silence).
            return IMLDocument(
                utterances=(Utterance(children=("",)),),
                version="0.1.0",
                language=language,
            )
        return self._assembler.assemble(
            alignments=alignments,
            features=analysis.features(alignments),
            pauses=pauses,
            language=language,
            reference_features=self._reference_features(),
        )

    def _transcript_alignment(
        self, analysis: _AudioAnalysis, transcript: str, notes: list[str]
    ) -> list[WordAlignment]:
        """One token holding the whole transcript, spanning the speech."""
        text = " ".join(transcript.split())
        if not text:
            notes.append("The transcript is empty, so the document has no text.")
            return []
        regions = analysis.voiced_regions()
        if regions:
            start_ms, end_ms = regions[0][0], regions[-1][1]
        else:
            start_ms, end_ms = 0, analysis.duration_ms
        return [WordAlignment(word=text, start_ms=start_ms, end_ms=end_ms)]

    @staticmethod
    def _placeholder_words(
        analysis: _AudioAnalysis, notes: list[str], reason: str
    ) -> list[WordAlignment]:
        """One ``[speech]`` token per stretch of voiced sound between pauses."""
        notes.append(
            f"No transcript ({reason}): each stretch of speech is a '{PLACEHOLDER_TOKEN}' "
            "placeholder. Pass words= or transcript=, or install "
            "'prosody-protocol[whisper]' for real words."
        )
        return [
            WordAlignment(word=PLACEHOLDER_TOKEN, start_ms=start_ms, end_ms=end_ms)
            for start_ms, end_ms in analysis.voiced_regions()
        ]

    def _reference_features(self) -> list[SpanFeatures] | None:
        """Features of the calibration recording, measured once per path.

        The speech is cut into word-sized spans so that the baseline is
        comparable with the per-word features it is applied to.
        """
        path = self.calibration_audio
        if path is None:
            return None
        if self._calibration is None or self._calibration[0] != path:
            analysis = _AudioAnalysis.from_path(path, self.max_duration_s)
            spans: list[WordAlignment] = []
            for start_ms, end_ms in analysis.voiced_regions():
                edges = list(range(start_ms, end_ms, _CALIBRATION_SPAN_MS))
                if len(edges) > 1 and end_ms - edges[-1] < _CALIBRATION_SPAN_MS // 2:
                    edges.pop()  # fold a short remainder into the previous span
                spans.extend(
                    WordAlignment(word="[calibration]", start_ms=a, end_ms=b)
                    for a, b in zip(edges, [*edges[1:], end_ms], strict=True)
                )
            features = [f for f in analysis.features(spans) if f.f0_mean is not None]
            if not features:
                raise AudioProcessingError(f"Calibration audio contains no voiced speech: {path}")
            self._calibration = (path, features)
        return self._calibration[1]

    # -- Whisper ----------------------------------------------------------------

    def _import_whisper(self) -> Any:
        """The whisper module, ``None`` in ``auto`` mode when it is missing."""
        try:
            import whisper
        except ImportError as exc:
            if self.stt == "whisper":
                raise AudioProcessingError(
                    "openai-whisper is required for stt='whisper'. Install with: "
                    "pip install 'prosody-protocol[whisper]', or pass words= or "
                    "transcript= instead."
                ) from exc
            return None
        return whisper

    def _transcribe(
        self, whisper: Any, analysis: _AudioAnalysis
    ) -> tuple[list[WordAlignment], str | None]:
        """Run Whisper; return word alignments and the detected language."""
        model = self._whisper_models.get(self.stt_model)
        if model is None:
            try:
                model = whisper.load_model(self.stt_model)
            except Exception as exc:
                raise AudioProcessingError(
                    f"Cannot load Whisper model {self.stt_model!r}: {exc}"
                ) from exc
            self._whisper_models[self.stt_model] = model

        options: dict[str, Any] = {"word_timestamps": True}
        if self.language:
            options["language"] = _whisper_language(self.language)
        # Whisper gets the audio already decoded, so it needs no ffmpeg and
        # its timestamps refer to exactly the samples that are analysed.
        audio = analysis.samples(_WHISPER_SAMPLE_RATE)
        try:
            result = model.transcribe(audio, **options)
            alignments: list[WordAlignment] = []
            for segment in result.get("segments", []):
                for info in segment.get("words", []):
                    token = _XML_INVALID_CHARS.sub("", str(info["word"]))
                    if not token.strip():
                        continue
                    start_ms = int(round(float(info["start"]) * 1000))
                    end_ms = max(start_ms, int(round(float(info["end"]) * 1000)))
                    # Whisper's leading space is kept; tokens are opaque to the assembler.
                    alignments.append(WordAlignment(word=token, start_ms=start_ms, end_ms=end_ms))
        except Exception as exc:
            raise AudioProcessingError(f"Whisper transcription failed: {exc}") from exc

        detected = result.get("language")
        return alignments, detected if isinstance(detected, str) and detected else None
