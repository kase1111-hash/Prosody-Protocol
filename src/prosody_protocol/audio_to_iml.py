"""AudioToIML -- convert audio files to IML-annotated transcripts.

Orchestrates STT, prosody analysis, emotion classification, and IML assembly.

Pipeline:
  1. STT (Whisper) produces word-level timestamps
  2. ProsodyAnalyzer extracts acoustic features per word span
  3. Pause detection finds silence gaps
  4. IMLAssembler constructs the document with emotion/prosody/emphasis/pause tags

Spec reference: Sections 3-4.
"""

from __future__ import annotations

from pathlib import Path

from .assembler import IMLAssembler
from .emotion_classifier import EmotionClassifier
from .exceptions import AudioProcessingError
from .models import IMLDocument
from .parser import IMLParser
from .prosody_analyzer import ProsodyAnalyzer, WordAlignment


def _run_whisper(audio_path: Path, model_name: str) -> list[WordAlignment]:
    """Run Whisper STT and return word-level alignments.

    Imports whisper lazily so the rest of the package works without
    the ``audio`` extra installed.
    """
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError as exc:
        raise AudioProcessingError(
            "openai-whisper is required for audio-to-IML conversion. "
            "Install with: pip install prosody-protocol[audio]"
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), word_timestamps=True)

    alignments: list[WordAlignment] = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            alignments.append(WordAlignment(
                word=word_info["word"].strip(),
                start_ms=int(word_info["start"] * 1000),
                end_ms=int(word_info["end"] * 1000),
            ))

    return alignments


def _run_whisper_or_stub(audio_path: Path, model_name: str) -> list[WordAlignment]:
    """Try Whisper; if unavailable fall back to a single-word stub for the
    full audio duration.  This allows the pipeline to produce *some* output
    even when only parselmouth is installed.
    """
    try:
        return _run_whisper(audio_path, model_name)
    except AudioProcessingError:
        import warnings

        import parselmouth

        sound = parselmouth.Sound(str(audio_path))
        duration_ms = int(sound.duration * 1000)
        warnings.warn(
            "Whisper is not installed; transcript will contain a placeholder "
            "'[speech]' token instead of real words. Install with: "
            "pip install prosody-protocol[audio]",
            stacklevel=2,
        )
        return [WordAlignment(word="[speech]", start_ms=0, end_ms=duration_ms)]


class AudioToIML:
    """Convert audio files to IML markup.

    Parameters
    ----------
    stt_model:
        Whisper model size (``"tiny"``, ``"base"``, ``"small"``, etc.).
    emotion_classifier:
        Optional custom emotion classifier.  Defaults to the rule-based
        baseline.
    include_extended:
        When ``True``, include extended prosodic attributes (f0_mean,
        jitter, etc.) on ``<prosody>`` elements.
    language:
        Optional BCP-47 language tag for the output IML document.
    """

    def __init__(
        self,
        stt_model: str = "base",
        emotion_classifier: EmotionClassifier | None = None,
        include_extended: bool = False,
        language: str | None = None,
    ) -> None:
        self.stt_model = stt_model
        self.include_extended = include_extended
        self.language = language
        self._analyzer = ProsodyAnalyzer()
        self._assembler = IMLAssembler(
            emotion_classifier=emotion_classifier,
            include_extended=include_extended,
        )
        self._parser = IMLParser()

    def convert(self, audio_path: str | Path) -> str:
        """Convert an audio file to an IML XML string.

        Raises :class:`~prosody_protocol.exceptions.AudioProcessingError`
        if the audio file cannot be read or processed.
        """
        doc = self.convert_to_doc(audio_path)
        return self._parser.to_iml_string(doc)

    def convert_to_doc(self, audio_path: str | Path) -> IMLDocument:
        """Convert an audio file to a parsed :class:`IMLDocument`.

        Steps:
          1. Run STT to get word-level timestamps
          2. Extract prosodic features per word
          3. Detect pauses
          4. Assemble IML document

        Raises :class:`~prosody_protocol.exceptions.AudioProcessingError`
        if the audio file cannot be read or processed.
        """
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        # Step 1: Speech-to-text with word timestamps.
        alignments = _run_whisper_or_stub(path, self.stt_model)

        if not alignments:
            # Edge case: no speech detected -- return empty utterance with
            # no emotion (don't fabricate a classification from silence).
            from .models import Utterance

            return IMLDocument(
                utterances=(Utterance(children=("",)),),
                version="0.1.0",
                language=self.language,
            )

        # Step 2: Prosody analysis.
        features = self._analyzer.analyze(path, alignments)

        # Step 3: Pause detection.
        pauses = self._analyzer.detect_pauses(path)

        # Step 4: Assemble.
        return self._assembler.assemble(
            alignments=alignments,
            features=features,
            pauses=pauses,
            language=self.language,
        )
