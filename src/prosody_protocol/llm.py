"""Format IML for large language models.

An LLM reads the words of an IML document well but not its numbers:
``pitch="+15%"`` or ``jitter="1.2"`` means little to it, and an ``emotion``
label with ``confidence="0.41"`` reads like a fact. :func:`to_llm_context`
turns a document into a compact annotated transcript instead::

    agent: How can I help you today?
    Delivery: sounds neutral (estimated, 95%).
    caller: I've been on hold for {thirty minutes} (slightly higher pitch, louder)
    and my **account is still locked**.
    Delivery: sounds frustrated (estimated, 89%).

(Each utterance is one line; the caller's is wrapped here.) The notation,
which :data:`SYSTEM_PROMPT` explains to the model:

- ``*words*`` and ``**words**``: ``<emphasis>`` (moderate and strong).
  Reduced emphasis is the note ``de-emphasized``.
- ``word (notes)`` or ``{several words} (notes)``: ``<prosody>`` and
  ``<segment>`` attributes in words -- higher/lower pitch, louder/quieter,
  faster/slower, rising/falling, breathy/tense/..., rushed/clipped/...
  Extended attributes (spec Section 4) appear only with *include_numbers*.
- ``[pause 0.8s]``: a ``<pause>`` of at least 300 ms. Shorter pauses are
  ordinary speech rhythm (spec 3.3) and are left out.
- ``speaker_id: ...``: the utterance's speaker (quoted unless it is a
  plain name).
- ``Delivery: ...``: the utterance as a whole -- markup that spans all of it,
  and its emotion. The emotion is stated only when its confidence is at
  least *min_confidence* (spec 6.2.3 treats confidence below 0.5 as low),
  and always as an estimate; otherwise the note says it was not reliably
  detected. An emotion that a prosody profile set (the utterance carries
  ``x-profile``, see :data:`~prosody_protocol.assembler.PROFILE_ATTRIBUTE`)
  is marked ``interpreted with the speaker's prosody profile``: spec 7.2
  asks that profile usage be reported downstream. The ``x-profile`` value
  itself is not shown.

:func:`build_messages` pairs the annotated transcript with
:data:`SYSTEM_PROMPT` as chat messages for any chat-completion API. Like the
rest of the core, this module needs only ``lxml``.

Spec reference: Sections 3, 6.2, 8.2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from .assembler import PROFILE_ATTRIBUTE
from .models import ChildNode, Emphasis, IMLDocument, Pause, Prosody, Segment, Utterance
from .parser import IMLParser

#: Emotion estimates with lower confidence are reported as "not reliably
#: detected" (spec 6.2.3: consumers SHOULD treat confidence below 0.5 as low).
DEFAULT_MIN_CONFIDENCE = 0.5

#: Pauses shorter than this (ms) are ordinary speech rhythm and are not shown.
DEFAULT_MIN_PAUSE_MS = 300

#: System prompt that explains the notation of :func:`to_llm_context` and
#: how far prosodic cues can be trusted (spec 8.2).
SYSTEM_PROMPT = """\
Some of the input you receive is transcribed speech, given inside <transcript> tags. \
The transcript is annotated with how the words were spoken (prosody):

- *word* marks a stressed word and **word** a strongly stressed one.
- A note in parentheses describes how the word just before it was said, or the whole \
{braced phrase} or *stressed phrase* just before it: higher or lower pitch, louder or \
quieter, faster or slower, the pitch movement (rising, falling, sharply falling, ...), \
the voice quality (breathy, tense, creaky, whispery, harsh) or the pace and rhythm of a \
phrase (rushed, drawn out, clipped, ...). "de-emphasized" marks words said with less \
stress than those around them. Notes may include measurements, such as a pitch change \
in percent or semitones (st) or a loudness change in dB.
- [pause 0.8s] marks a silence of about that length. Pauses can signal hesitation, \
uncertainty, reluctance or a change of topic.
- A line that starts with a name and a colon was spoken by that speaker.
- A "Delivery:" line describes the utterance on the line above it as a whole: its \
overall pitch, loudness or pace, and the speaker's apparent emotion. Emotions are \
automatic estimates, shown with the estimator's confidence; "emotion not reliably \
detected" means the estimate was too uncertain to report. An emotion "interpreted with \
the speaker's prosody profile" was read with a description of how this particular \
speaker expresses themselves (for example, someone whose excited speech is flat and \
fast), so prefer it to a general reading of the other notes on that utterance.

Pitch, loudness and pace are relative to the speaker's usual voice. Words without notes \
were not marked as unusual.

Use these cues to understand what the speaker means, for example to tell sarcasm from \
sincerity or urgency from calm, and respond to their intent rather than only to the \
literal words. Do not quote the notation back to the speaker.

Prosodic cues are probabilistic evidence, not facts. They vary between people, \
languages, cultures and recording conditions, and some speakers (for example autistic \
people or people with speech impairments) express intent differently. When the cues and \
the words point different ways, or when getting it wrong would matter, ask rather than \
assume. Never make a consequential decision on the basis of these cues alone, never \
treat them as evidence of whether someone is telling the truth, and never use them to \
judge or profile a person, for example in hiring, lending or law enforcement."""

# Core emotion vocabulary (spec 3.1): adjectives that read as "sounds <emotion>".
_CORE_EMOTIONS = frozenset({
    "neutral", "sincere", "sarcastic", "frustrated", "joyful", "uncertain", "angry",
    "sad", "fearful", "surprised", "disgusted", "calm", "empathetic",
})
# Labels outside the core set are shown, quoted, only when they read as a
# label: up to four words of letters joined by spaces or hyphens.
_LABEL_RE = re.compile(r"[^\W\d_]+(?:[ -][^\W\d_]+){0,3}")
_MAX_LABEL_LENGTH = 40

# Spec 3.2 / 3.5 vocabularies in words. "modal" voice and "medium" rate are
# the speaker's usual and get no note.
_CONTOURS = {
    "rise": "rising",
    "fall": "falling",
    "rise-fall": "rising then falling",
    "fall-rise": "falling then rising",
    "rise-sharp": "sharply rising",
    "fall-sharp": "sharply falling",
    "flat": "flat pitch",
}
_QUALITIES = {q: q for q in ("breathy", "tense", "creaky", "whispery", "harsh")}
_NAMED_RATES = {"fast": "faster", "slow": "slower"}
_TEMPOS = {"rushed": "rushed", "steady": "steady pace", "drawn-out": "drawn out"}
_RHYTHMS = {"staccato": "clipped", "legato": "flowing", "syncopated": "irregular rhythm"}

# Emphasis levels as (mark, notes); a missing or unknown level reads as moderate.
_EMPHASIS = {
    "strong": ("**", ()),
    "moderate": ("*", ()),
    "reduced": ("", ("de-emphasized",)),
}

# Lexical forms of spec 3.2 values (as in the validator).
_RELATIVE_PITCH_RE = re.compile(r"([+-][0-9]+(?:\.[0-9]+)?)(%|st)")
_ABSOLUTE_PITCH_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)Hz")
_VOLUME_RE = re.compile(r"([+-][0-9]+(?:\.[0-9]+)?)dB")
_RATE_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
_F0_RANGE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?")

# Degrees of change. Smaller changes than the first threshold get no note,
# changes below the second are "slight", and from the third on "much".
_PITCH_ST = (0.5, 1.5, 6.0)  # semitones; 1.5 st is about 9 %, 6 st about 41 %
_VOLUME_DB = (1.0, 3.0, 10.0)
_RATE_LOG2 = (0.07, 0.2, 0.58)  # |log2(rate ratio)|: 5 %, 15 %, 1.5x

# Marks a word boundary made by markup rather than by the text: rendered as a
# space, except before closing punctuation ("GREAT\n</prosody>." reads
# "GREAT."). XML text cannot contain NUL, so it never clashes with content.
_SOFT_SPACE = "\x00"
# XML whitespace, and the other characters that break lines
# (str.splitlines): each utterance stays on one line.
_SPACE_CHARS = " \t\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029" + _SOFT_SPACE
_SENTENCE_PUNCTUATION = ".,;:!?\u2026"
_CLOSING_PUNCTUATION = _SENTENCE_PUNCTUATION + ")]}"
_SOFT_SPACE_BEFORE_PUNCTUATION = re.compile(
    f"{_SOFT_SPACE}+(?=[{re.escape(_CLOSING_PUNCTUATION)}])"
)
_SPACE_RUN = re.compile(f"[{re.escape(_SPACE_CHARS)}]+")

# Speaker names shown as they are; others are quoted.
_PLAIN_SPEAKER = re.compile(r"[\w.'@#+-]+(?: [\w.'@#+-]+)*")
# The start of a line that reads as a "Delivery:" line.
_DELIVERY_PREFIX = re.compile(r"\W*delivery\W*[:\uff1a]", re.IGNORECASE)
# The "<" of a <transcript> or </transcript> tag (build_messages' delimiter).
_TRANSCRIPT_TAG = re.compile(r"<(?=\s*/?\s*transcript)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Attribute descriptions
# ---------------------------------------------------------------------------


def _number(value: float) -> str:
    return f"{value:g}"


def _degree(magnitude: float, thresholds: tuple[float, float, float]) -> str | None:
    """The adverb for a change of *magnitude*, or ``None`` if it is negligible."""
    negligible, slight, much = thresholds
    if magnitude < negligible:
        return None
    if magnitude < slight:
        return "slightly "
    return "much " if magnitude >= much else ""


def _pitch_note(pitch: str | None, include_numbers: bool) -> str | None:
    if pitch is None:
        return None
    relative = _RELATIVE_PITCH_RE.fullmatch(pitch)
    if relative is not None:
        amount = float(relative[1])
        if relative[2] == "%":
            if amount <= -100.0:
                return None
            semitones = 12.0 * math.log2(1.0 + amount / 100.0)
        else:
            semitones = amount
        degree = _degree(abs(semitones), _PITCH_ST)
        if degree is None:
            return None
        note = f"{degree}{'higher' if semitones > 0 else 'lower'} pitch"
        return f"{note} {pitch}" if include_numbers else note
    absolute = _ABSOLUTE_PITCH_RE.fullmatch(pitch)
    # An absolute pitch says nothing without the speaker's baseline.
    if absolute is not None and include_numbers:
        return f"pitch {absolute[1]} Hz"
    return None


def _volume_note(volume: str | None, include_numbers: bool) -> str | None:
    match = _VOLUME_RE.fullmatch(volume) if volume is not None else None
    if match is None:
        return None
    decibels = float(match[1])
    degree = _degree(abs(decibels), _VOLUME_DB)
    if degree is None:
        return None
    note = f"{degree}{'louder' if decibels > 0 else 'quieter'}"
    return f"{note} {volume}" if include_numbers else note


def _rate_note(rate: str | None, include_numbers: bool) -> str | None:
    if rate is None:
        return None
    if rate in _NAMED_RATES:
        return _NAMED_RATES[rate]
    match = _RATE_PERCENT_RE.fullmatch(rate)
    if match is None or float(match[1]) <= 0.0:
        return None
    log_ratio = math.log2(float(match[1]) / 100.0)
    degree = _degree(abs(log_ratio), _RATE_LOG2)
    if degree is None:
        return None
    note = f"{degree}{'faster' if log_ratio > 0 else 'slower'}"
    return f"{note} at {rate} pace" if include_numbers else note


def _measurements(p: Prosody) -> list[str]:
    """Extended attributes (spec Section 4) as notes."""
    notes: list[str] = []
    if p.f0_mean is not None:
        notes.append(f"mean pitch {_number(p.f0_mean)} Hz")
    if p.f0_range is not None and _F0_RANGE_RE.fullmatch(p.f0_range):
        notes.append(f"pitch range {p.f0_range} Hz")
    if p.intensity_mean is not None:
        notes.append(f"intensity {_number(p.intensity_mean)} dB")
    if p.intensity_range is not None:
        notes.append(f"intensity range {_number(p.intensity_range)} dB")
    if p.speech_rate is not None:
        notes.append(f"{_number(p.speech_rate)} syllables/s")
    if p.duration_ms is not None:
        notes.append(f"{p.duration_ms} ms")
    if p.jitter is not None:
        notes.append(f"jitter {_number(p.jitter)}%")
    if p.shimmer is not None:
        notes.append(f"shimmer {_number(p.shimmer)}%")
    if p.hnr is not None:
        notes.append(f"HNR {_number(p.hnr)} dB")
    return notes


def _own_notes(node: Prosody | Emphasis | Segment, include_numbers: bool) -> list[str]:
    """How *node* says its content was spoken, not counting nested markup."""
    if isinstance(node, Emphasis):
        return list(_EMPHASIS.get(node.level, _EMPHASIS["moderate"])[1])
    if isinstance(node, Segment):
        tempo = _TEMPOS.get(node.tempo or "")
        rhythm = _RHYTHMS.get(node.rhythm or "")
        return [note for note in (tempo, rhythm) if note is not None]
    notes = [
        _pitch_note(node.pitch, include_numbers),
        _volume_note(node.volume, include_numbers),
        _rate_note(node.rate, include_numbers),
        _CONTOURS.get(node.pitch_contour or ""),
        _QUALITIES.get(node.quality or ""),
    ]
    described = [note for note in notes if note is not None]
    if include_numbers:
        described.extend(_measurements(node))
    return described


def _mark(node: Prosody | Emphasis | Segment) -> str:
    if isinstance(node, Emphasis):
        return _EMPHASIS.get(node.level, _EMPHASIS["moderate"])[0]
    return ""


def _merge(notes: list[str], more: Sequence[str]) -> list[str]:
    return notes + [note for note in more if note not in notes]


def _emotion_note(utt: Utterance, min_confidence: float, include_numbers: bool) -> str | None:
    """The delivery note on the utterance's emotion (spec 3.1, 6.2.3)."""
    label = " ".join((utt.emotion or "").replace("_", " ").split())
    if not label:
        return None
    if label not in _CORE_EMOTIONS and (
        len(label) > _MAX_LABEL_LENGTH or not _LABEL_RE.fullmatch(label)
    ):
        # Not a label that can be shown: an unknown value, treated as no
        # marked emotion (spec 3.1 treats unknown values as neutral).
        return None
    confidence = utt.confidence
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        confidence = None  # NaN or out of range, in a document built in code
    if confidence is None or confidence < min_confidence:
        if include_numbers and confidence is not None:
            return f"emotion not reliably detected (confidence {_number(confidence)})"
        return "emotion not reliably detected"
    estimate = f"estimated, {round(confidence * 100)}%"
    if _uses_profile(utt):
        estimate += "; interpreted with the speaker's prosody profile"
    if label in _CORE_EMOTIONS:
        return f"sounds {label} ({estimate})"
    return f'emotion labelled "{label}" ({estimate})'


def _uses_profile(utt: Utterance) -> bool:
    """Whether a prosody profile set the utterance's emotion (spec 7.2).

    The assembler marks such utterances with :data:`PROFILE_ATTRIBUTE`
    holding the pattern that matched; any non-blank value counts.
    """
    return any(
        name == PROFILE_ATTRIBUTE and str(value).strip() for name, value in utt.extra_attributes
    )


def _pause_text(duration: int, include_numbers: bool) -> str:
    if duration <= 0:  # missing or invalid duration
        return "[pause]"
    if include_numbers:
        return f"[pause {_number(duration / 1000)}s]"
    tenths = max(1, (duration + 50) // 100)
    return f"[pause {tenths // 10}.{tenths % 10}s]"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _is_blank(text: str) -> bool:
    return not text.strip(_SPACE_CHARS)


def _sole_element(children: Sequence[ChildNode]) -> ChildNode | None:
    """The one element among *children* if everything else is whitespace."""
    elements = [child for child in children if not isinstance(child, str)]
    if len(elements) != 1:
        return None
    if any(isinstance(child, str) and not _is_blank(child) for child in children):
        return None
    return elements[0]


def _whole_span(children: Sequence[ChildNode]) -> tuple[int, Prosody | Segment] | None:
    """A ``<prosody>`` or ``<segment>`` that spans all the words, with its index."""
    found: tuple[int, Prosody | Segment] | None = None
    for index, child in enumerate(children):
        if isinstance(child, str):
            if not _is_blank(child):
                return None
        elif isinstance(child, (Prosody, Segment)) and found is None:
            found = (index, child)
        elif not isinstance(child, Pause):
            return None
    return found


def _merges(node: Prosody | Emphasis | Segment, inner: Prosody | Emphasis) -> bool:
    """Whether *node* and its sole element *inner* read as one span.

    ``<segment>`` notes (tempo, rhythm) stay on their own span, and nested
    ``<prosody>`` keeps both scopes because their values may add up.
    """
    if isinstance(node, Segment):
        return False
    return not (isinstance(node, Prosody) and isinstance(inner, Prosody))


def _finish(text: str) -> str:
    """Resolve soft spaces and collapse whitespace into single spaces."""
    text = _SOFT_SPACE_BEFORE_PUNCTUATION.sub("", text)
    return _SPACE_RUN.sub(" ", text).strip(" ")


def _utterance_line(speaker_id: str | None, text: str) -> str:
    """The line for an utterance's *text*, prefixed by its speaker.

    The line cannot pass for notation: a speaker name other than a plain
    one is quoted, and so are words that would read as a ``Delivery:`` line.
    ``<transcript>`` tags in the text lose their ``<``, so they cannot end
    the block that :func:`build_messages` puts the context in.
    """
    speaker = " ".join((speaker_id or "").split())
    if speaker:
        if not _PLAIN_SPEAKER.fullmatch(speaker) or _DELIVERY_PREFIX.match(f"{speaker}:"):
            speaker = '"{}"'.format(speaker.replace('"', "'"))
        line = f"{speaker}: {text}"
    elif _DELIVERY_PREFIX.match(text):
        line = f'"{text}"'
    else:
        line = text
    return _TRANSCRIPT_TAG.sub("\u2039", line)


class _Renderer:
    """Renders utterance content in the annotation notation."""

    def __init__(self, include_numbers: bool, min_pause_ms: int) -> None:
        self.include_numbers = include_numbers
        self.min_pause_ms = min_pause_ms

    def children(self, children: Sequence[ChildNode]) -> str:
        parts: list[str] = []
        for child in children:
            if isinstance(child, str):
                parts.append(child)
            elif isinstance(child, Pause):
                parts.append(self.pause(child))
            else:
                parts.append(self.element(child))
        return "".join(parts)

    def pause(self, pause: Pause) -> str:
        if 0 < pause.duration < self.min_pause_ms:
            return _SOFT_SPACE
        return f"{_SOFT_SPACE}{_pause_text(pause.duration, self.include_numbers)}{_SOFT_SPACE}"

    def element(self, node: Prosody | Emphasis | Segment) -> str:
        return self.span(*self.parts(node))

    def span(self, raw: str, notes: list[str], mark: str) -> str:
        """Rendered content *raw* with its emphasis *mark* and *notes*."""
        content = raw.strip(_SPACE_CHARS)
        if not content:
            return _SOFT_SPACE if raw else ""
        # Punctuation that ends the span is not spoken: the notes go before it
        # ("yesterday (falling)." rather than "yesterday. (falling)").
        text = content.rstrip(_SENTENCE_PUNCTUATION).rstrip(_SPACE_CHARS) or content
        punctuation = content[len(text):]
        if mark:
            text = f"{mark}{text}{mark}"
        elif notes and any(char in _SPACE_CHARS for char in text):
            text = f"{{{text}}}"  # the notes describe the whole phrase
        if notes:
            text = f"{text} ({', '.join(notes)})"
        lead = _SOFT_SPACE if raw[0] in _SPACE_CHARS else ""
        trail = _SOFT_SPACE if raw[-1] in _SPACE_CHARS else ""
        return f"{lead}{text}{punctuation}{trail}"

    def parts(self, node: Prosody | Emphasis | Segment) -> tuple[str, list[str], str]:
        """The rendered content, notes and emphasis mark of *node*.

        ``<emphasis>`` around a sole ``<prosody>`` (or the reverse) reads as
        one span, ``**word** (louder)``: the notes describe the stressed words.
        Emphasis inside emphasis keeps the stronger mark. Each node is
        rendered once, so the time is linear in the size of the document.
        """
        notes = _own_notes(node, self.include_numbers)
        mark = _mark(node)
        inner = _sole_element(node.children)
        if not isinstance(inner, (Prosody, Emphasis)) or not _merges(node, inner):
            return self.children(node.children), notes, mark
        inner_raw, inner_notes, inner_mark = self.parts(inner)
        # The other children are whitespace; keep it around the merged span.
        raw = "".join(inner_raw if child is inner else str(child) for child in node.children)
        return raw, _merge(notes, inner_notes), max(mark, inner_mark, key=len)

    def utterance(self, utt: Utterance, min_confidence: float) -> list[str]:
        """The utterance line and, when there is something to say, its delivery line."""
        children: Sequence[ChildNode] = utt.children
        overall: list[str] = []
        # Markup around the whole utterance describes it as a whole.
        while (whole := _whole_span(children)) is not None:
            index, node = whole
            overall = _merge(overall, _own_notes(node, self.include_numbers))
            children = [*children[:index], *node.children, *children[index + 1:]]

        text = _finish(self.children(children)) or "[no words]"
        lines = [_utterance_line(utt.speaker_id, text)]

        delivery: list[str] = []
        if overall:
            delivery.append(f"overall {', '.join(overall)}")
        emotion = _emotion_note(utt, min_confidence, self.include_numbers)
        if emotion is not None:
            delivery.append(emotion)
        if delivery:
            lines.append(f"Delivery: {'; '.join(delivery)}.")
        return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_llm_context(
    doc_or_iml: IMLDocument | str,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    include_numbers: bool = False,
    min_pause_ms: int = DEFAULT_MIN_PAUSE_MS,
) -> str:
    """Describe an IML document as an annotated transcript for an LLM.

    See the module docstring for the notation, and use :data:`SYSTEM_PROMPT`
    (or :func:`build_messages`) to explain it to the model.

    Parameters
    ----------
    doc_or_iml:
        A parsed :class:`~prosody_protocol.models.IMLDocument` or an IML
        string (parsed with :class:`~prosody_protocol.parser.IMLParser`).
    min_confidence:
        Emotions with lower confidence (or none) are reported as "emotion not
        reliably detected" rather than named.
    include_numbers:
        Also give the measured values: pitch and volume offsets, rate
        percentages, absolute pitch, extended attributes (spec Section 4),
        exact pause lengths and the confidence of unreliable emotions.
    min_pause_ms:
        Shorter pauses are left out.

    Returns
    -------
    str
        One line per utterance, each followed by a ``Delivery:`` line when
        there is something to say about the utterance as a whole. An empty
        document gives an empty string.

    Raises
    ------
    IMLParseError
        If *doc_or_iml* is a string that is not IML.
    ValueError
        If *min_confidence* is outside [0, 1] or *min_pause_ms* is negative.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence!r}")
    if min_pause_ms < 0:
        raise ValueError(f"min_pause_ms must not be negative, got {min_pause_ms!r}")
    doc = IMLParser().parse(doc_or_iml) if isinstance(doc_or_iml, str) else doc_or_iml
    renderer = _Renderer(include_numbers, min_pause_ms)
    lines: list[str] = []
    for utt in doc.utterances:
        lines.extend(renderer.utterance(utt, min_confidence))
    return "\n".join(lines)


def build_messages(
    iml: IMLDocument | str,
    user_instruction: str | None = None,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    include_numbers: bool = False,
    min_pause_ms: int = DEFAULT_MIN_PAUSE_MS,
) -> list[dict[str, str]]:
    """Build chat messages that give an LLM the annotated transcript of *iml*.

    Returns ``[{"role": "system", ...}, {"role": "user", ...}]``: the system
    message is :data:`SYSTEM_PROMPT`; the user message holds the output of
    :func:`to_llm_context` inside ``<transcript>`` tags, followed by
    *user_instruction* if given (for example ``"Summarize the caller's
    problem."``). Without an instruction the model answers the speaker.
    Text in the document cannot end the ``<transcript>`` block early: a
    transcript tag in it is rendered with ``\u2039`` instead of ``<``.

    The list works as-is with chat-completion APIs that take a system role.
    For APIs that take the system prompt separately, such as Anthropic's
    Messages API, pass ``system=messages[0]["content"]`` and
    ``messages=messages[1:]``.

    Keyword arguments are passed to :func:`to_llm_context`.
    """
    context = to_llm_context(
        iml,
        min_confidence=min_confidence,
        include_numbers=include_numbers,
        min_pause_ms=min_pause_ms,
    )
    content = f"<transcript>\n{context}\n</transcript>"
    if user_instruction is not None and user_instruction.strip():
        content += f"\n\n{user_instruction.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
