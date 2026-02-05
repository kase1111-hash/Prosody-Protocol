"""TextToIML -- predict prosody for plain text (no audio input).

Supports a rule-based baseline and a pluggable ML model backend.
Spec reference: Section 3 (output conforms to IML tag set).

Phase 6a implements rule-based prediction using punctuation, capitalisation,
and lexical cues.  Zero external dependencies beyond ``lxml``.

Phase 6b (ML model) is a placeholder pending dataset infrastructure (Phase 10).
"""

from __future__ import annotations

import re

from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Utterance,
)
from .parser import IMLParser
from .validator import IMLValidator

# ---------------------------------------------------------------------------
# Sentiment lexicon (kept intentionally small -- rule-based baseline)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: frozenset[str] = frozenset({
    "love", "happy", "wonderful", "fantastic", "amazing", "great",
    "excellent", "beautiful", "brilliant", "awesome", "delightful",
    "pleased", "glad", "thankful", "grateful", "excited", "thrilled",
    "cheerful", "joyful", "superb", "terrific", "magnificent",
})

_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "hate", "angry", "terrible", "horrible", "awful", "disgusting",
    "miserable", "furious", "annoyed", "frustrated", "disappointed",
    "upset", "sad", "depressed", "worried", "anxious", "dreadful",
    "pathetic", "ridiculous", "stupid", "worse", "worst",
})

_UNCERTAIN_WORDS: frozenset[str] = frozenset({
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "unsure", "uncertain", "wonder", "suppose", "guess", "somehow",
    "apparently", "seemingly",
})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches a word that is ALL CAPS (at least 2 letters, no lowercase).
_ALL_CAPS_RE = re.compile(r"^[A-Z]{2,}$")

# Sentence splitter: split on .!? followed by whitespace or end-of-string,
# but keep the delimiter attached to the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Matches quoted speech: "..." or \u201c...\u201d (smart double quotes).
_QUOTED_SPEECH_RE = re.compile(
    r'"([^"]+)"|'
    r"\u201c([^\u201d]+)\u201d"
)

# Ellipsis: three or more dots or the unicode ellipsis character.
_ELLIPSIS_RE = re.compile(r"\.{3,}|\u2026")


def _strip_punctuation(word: str) -> str:
    """Strip surrounding punctuation from a word for lexicon lookup."""
    return word.strip(".,!?;:\"'()\u201c\u201d\u2018\u2019\u2026")


def _detect_sentence_emotion(sentence: str) -> tuple[str | None, float]:
    """Detect emotion from lexical cues in a sentence.

    Returns (emotion, confidence) or (None, 0.0) if no cues found.
    """
    words_lower = [_strip_punctuation(w).lower() for w in sentence.split()]
    stripped = sentence.rstrip()

    pos_count = sum(1 for w in words_lower if w in _POSITIVE_WORDS)
    neg_count = sum(1 for w in words_lower if w in _NEGATIVE_WORDS)
    unc_count = sum(1 for w in words_lower if w in _UNCERTAIN_WORDS)

    # Punctuation-based emotion (higher priority).
    if stripped.endswith("!"):
        if neg_count > pos_count:
            return ("frustrated", 0.55)
        return ("joyful", 0.55)

    # Lexicon-based emotion (lower confidence).
    if neg_count > pos_count and neg_count > unc_count:
        return ("frustrated", 0.45)
    if pos_count > neg_count and pos_count > unc_count:
        return ("joyful", 0.45)
    if unc_count > 0 and unc_count >= pos_count and unc_count >= neg_count:
        return ("uncertain", 0.40)

    return (None, 0.0)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving terminal punctuation."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_quoted_spans(sentence: str) -> tuple[list[str], str]:
    """Extract quoted speech from a sentence.

    Returns a list of quoted strings and the sentence with quotes removed.
    """
    quoted: list[str] = []
    for m in _QUOTED_SPEECH_RE.finditer(sentence):
        # One of the capture groups will match.
        quoted.append(m.group(1) or m.group(2))

    remainder = _QUOTED_SPEECH_RE.sub("", sentence).strip()
    # Clean up double/leading/trailing spaces left by removal.
    remainder = re.sub(r"\s{2,}", " ", remainder)
    return quoted, remainder


def _build_children_for_sentence(sentence: str) -> tuple[tuple[ChildNode, ...], bool]:
    """Build IML children for a single sentence.

    Returns (children_tuple, has_question_mark).
    """
    stripped = sentence.rstrip()
    is_question = stripped.endswith("?") or (
        _ELLIPSIS_RE.search(stripped) is not None and "?" in stripped
    )

    # Split around ellipsis markers.
    parts = _ELLIPSIS_RE.split(sentence)
    has_ellipsis = len(parts) > 1

    children: list[ChildNode] = []

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            # Only insert pause if there are already children.
            if has_ellipsis and i < len(parts) - 1 and children:
                children.append(Pause(duration=500))
            continue

        # Process each word in this part.
        words = part.split()
        for word in words:
            clean = _strip_punctuation(word)

            if clean and _ALL_CAPS_RE.match(clean):
                # ALL CAPS -> emphasis.
                if children:
                    children.append(" ")
                children.append(Emphasis(level="strong", children=(word,)))
            else:
                # Regular word -- merge with preceding text or start new text.
                if children and isinstance(children[-1], str):
                    children[-1] = children[-1] + " " + word
                elif children:
                    children.append(" " + word)
                else:
                    children.append(word)

        # Insert pause for ellipsis between parts.
        if has_ellipsis and i < len(parts) - 1:
            children.append(Pause(duration=500))

    return tuple(children), is_question


def _build_utterance(
    sentence: str,
    default_confidence: float,
) -> Utterance:
    """Build a single Utterance from a sentence string."""
    children, is_question = _build_children_for_sentence(sentence)

    # Wrap in <prosody> if the sentence is a question (rising pitch contour).
    if is_question and children:
        children = (Prosody(children=children, pitch_contour="rise"),)

    emotion, emotion_conf = _detect_sentence_emotion(sentence)

    # Per spec: confidence MUST be present when emotion is set.
    confidence: float | None = None
    if emotion is not None:
        confidence = max(emotion_conf, default_confidence)

    stripped = sentence.rstrip()
    # Exclamation -> pitch boost on the whole utterance.
    if stripped.endswith("!") and not is_question:
        # If children are not already wrapped in prosody, wrap them.
        if children and not (len(children) == 1 and isinstance(children[0], Prosody)):
            children = (Prosody(children=children, pitch="+5%"),)
        elif children and isinstance(children[0], Prosody):
            # Add pitch to existing prosody wrapper.
            p = children[0]
            children = (Prosody(
                children=p.children,
                pitch="+5%",
                pitch_contour=p.pitch_contour,
                volume=p.volume,
                rate=p.rate,
                quality=p.quality,
            ),)

    return Utterance(
        children=children,
        emotion=emotion,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TextToIML:
    """Predict prosodic markup for plain text.

    Parameters
    ----------
    model:
        Backend to use.  ``"rule-based"`` (default) uses punctuation,
        capitalisation, and lexical cues.  Other values are reserved for
        future ML backends (Phase 6b).
    default_confidence:
        Default confidence floor for rule-based emotion predictions.
    """

    def __init__(
        self,
        model: str = "rule-based",
        default_confidence: float = 0.6,
    ) -> None:
        if model != "rule-based":
            raise NotImplementedError(
                f"Model {model!r} is not yet supported. "
                "Only 'rule-based' is available (Phase 6a)."
            )
        self.model = model
        self.default_confidence = default_confidence
        self._parser = IMLParser()
        self._validator = IMLValidator()

    def predict(self, text: str, context: str | None = None) -> str:
        """Predict prosody and return an IML XML string.

        Parameters
        ----------
        text:
            Plain text input.  May contain multiple sentences.
        context:
            Optional surrounding context to inform predictions.
            Currently unused by the rule-based model but reserved for
            ML backends.

        Returns
        -------
        str
            A valid IML XML string.
        """
        if not text or not text.strip():
            doc = IMLDocument(
                utterances=(Utterance(children=("",)),),
                version="0.1.0",
            )
            return self._parser.to_iml_string(doc)

        utterances: list[Utterance] = []
        sentences = _split_sentences(text)

        for sentence in sentences:
            quoted_spans, remainder = _extract_quoted_spans(sentence)

            # Build utterance for the non-quoted remainder.
            if remainder.strip():
                utterances.append(
                    _build_utterance(remainder, self.default_confidence)
                )

            # Each quoted span becomes its own utterance.
            for q in quoted_spans:
                utterances.append(
                    _build_utterance(q, self.default_confidence)
                )

        if not utterances:
            utterances.append(Utterance(children=(text.strip(),)))

        doc = IMLDocument(
            utterances=tuple(utterances),
            version="0.1.0",
        )
        return self._parser.to_iml_string(doc)

    def predict_document(self, text: str, context: str | None = None) -> IMLDocument:
        """Predict prosody and return a structured IMLDocument.

        Convenience method that parses the XML output of :meth:`predict`
        back into a document object.
        """
        xml = self.predict(text, context=context)
        return self._parser.parse(xml)
