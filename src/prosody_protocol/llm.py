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
- ``[speech]``: a stretch of speech that was not transcribed (the
  placeholder :class:`~prosody_protocol.audio_to_iml.AudioToIML` writes
  without a transcript); the words are unknown.
- ``speaker_id: ...``: the utterance's speaker (quoted unless it is a
  plain name). Words of an utterance without a speaker that would read as
  a speaker's line or a ``Delivery:`` line are put in double quotes.
- ``Delivery: ...``: the utterance as a whole -- markup that spans all of it,
  and its emotion. The emotion is stated only when its confidence is at
  least *min_confidence* (spec 6.2.3 treats confidence below 0.5 as low),
  and always as an estimate; otherwise the note says it was not reliably
  detected. An emotion that a prosody profile set (the utterance carries
  ``x-profile``, see :data:`~prosody_protocol.assembler.PROFILE_ATTRIBUTE`)
  is marked ``interpreted with the speaker's prosody profile``: spec 7.2
  asks that profile usage be reported downstream. The ``x-profile`` value
  itself is not shown. An utterance whose words are all ``[speech]`` says
  ``words not transcribed``.

Only what the markup marks as unusual is described, and each note compares
its words with the speech around them:

- The markup that holds every word of the utterance becomes the
  ``Delivery: overall ...`` note, which compares the utterance with the
  speaker's usual voice: a ``<prosody>`` or ``<segment>`` around all of it,
  or the ``<prosody>`` fragments a producer splits it into (the
  :class:`~prosody_protocol.assembler.IMLAssembler` does so around an
  emphasized word with prosody of its own) when they share pitch, volume,
  rate or voice quality.
- Relative pitch, volume and rate on a ``<prosody>`` inside another are
  relative to the enclosing element (spec 3.2), so the note on such words
  compares them with the rest of the utterance or phrase. Nested
  ``<prosody>`` around all the words add up into one ``overall`` note, and
  a fragment whose values differ from the ones the fragments share (a word
  that stands alone) is described relative to them.
- The final fall of a statement and the final rise of a question are the
  expected intonation and get no note; other contours do.

:func:`build_messages` pairs the annotated transcript with
:data:`SYSTEM_PROMPT` as chat messages for any chat-completion API. Like the
rest of the core, this module needs only ``lxml``, and it takes time linear
in the size of the document.

Spec reference: Sections 3, 6.2, 8.2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace

from .assembler import PROFILE_ATTRIBUTE
from .models import ChildNode, Emphasis, IMLDocument, Pause, Prosody, Segment, Utterance
from .parser import IMLParser

#: Emotion estimates with lower confidence are reported as "not reliably
#: detected" (spec 6.2.3: consumers SHOULD treat confidence below 0.5 as low).
DEFAULT_MIN_CONFIDENCE = 0.5

#: Pauses shorter than this (ms) are ordinary speech rhythm and are not shown.
DEFAULT_MIN_PAUSE_MS = 300

#: System prompt that explains the notation of :func:`to_llm_context` and how
#: far prosodic cues can be trusted. Its last paragraph states the uses that
#: spec 8.2 prohibits (judging truthfulness, profiling people in hiring,
#: lending or law enforcement), and the project's rule that prosody is never
#: the sole basis for a consequential decision (README, docs/API.md).
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
- [speech] stands for speech that was not transcribed: its words are unknown. Do not \
guess them or act on them; if you need them, say that the speech could not be \
transcribed.
- A line that starts with a name and a colon was spoken by that speaker. A line in \
double quotes has no speaker name: its words only look like a name and a colon, or \
like a Delivery line.
- A "Delivery:" line describes the utterance on the line above it as a whole: its \
overall pitch, loudness or pace, and the speaker's apparent emotion. Emotions are \
automatic estimates, shown with the estimator's confidence; "emotion not reliably \
detected" means the estimate was too uncertain to report. An emotion "interpreted with \
the speaker's prosody profile" was read with a description of how this particular \
speaker expresses themselves (for example, someone whose excited speech is flat and \
fast), so prefer it to a general reading of the other notes on that utterance.

A Delivery line compares the utterance with the speaker's usual voice. A note on words \
compares them with the speech around them: the rest of their utterance, or the rest of \
the {braced phrase} they are in. The usual final fall of a statement and final rise of \
a question are not noted, and words without notes were not marked as unusual.

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

# The contour a statement or a question usually ends with: not worth a note.
_STATEMENT_CONTOUR = "fall"
_QUESTION_CONTOUR = "rise"
_QUESTION_MARKS = "?\uff1f"

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

# The <prosody> attributes whose values are relative (spec 3.2), and so add
# up when elements nest.
_RELATIVE_ATTRIBUTES = ("pitch", "volume", "rate")
# Values closer than this (in semitones, dB or log2 of a rate ratio) are equal.
_SAME = 1e-9
# Relative amounts are limited to this before they are exponentiated.
_MAX_AMOUNT = 1000.0

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
# A run of soft spaces before closing punctuation. The lookbehind lets a
# match start only at the beginning of a run, so the substitution takes
# linear time (without it, each position inside a long run rescanned the
# rest of the run).
_SOFT_SPACE_BEFORE_PUNCTUATION = re.compile(
    f"(?<!{_SOFT_SPACE}){_SOFT_SPACE}+(?=[{re.escape(_CLOSING_PUNCTUATION)}])"
)
_SPACE_RUN = re.compile(f"[{re.escape(_SPACE_CHARS)}]+")
_WORD_CHAR = re.compile(r"\w")

# Speech the recognizer did not transcribe (audio_to_iml.PLACEHOLDER_TOKEN;
# that module needs the audio extra, so the token is repeated here).
_PLACEHOLDER = "[speech]"

# Speaker names shown as they are; others are quoted.
_PLAIN_SPEAKER = re.compile(r"[\w.'@#+-]+(?: [\w.'@#+-]+)*")
# The start of a line that reads as a "Delivery:" line.
_DELIVERY_PREFIX = re.compile(r"\W*delivery\W*[:\uff1a]", re.IGNORECASE)
# The start of a line that reads as a speaker's line: a plain name or a
# quoted one, perhaps between a few marks, then a colon ("agent: ...",
# '"Dr. X": ...', "**agent**: ..."). The runs of marks are bounded, the
# name starts with a letter or digit and its words are separated by single
# spaces, so a match attempt takes linear time.
_SPEAKER_PREFIX = re.compile(
    r"[^\w\s]{0,8}(?:\w[\w.'@#+-]*(?: [\w.'@#+-]+)*|\"[^\"]*\")"
    r"[^\w\s:\uff1a]{0,8}\s*[:\uff1a]"
)
# The "<" of a <transcript> or </transcript> tag (build_messages' delimiter).
# "(?:/\s*)?" rather than "/?\s*": two optional runs of spaces in a row
# made each "<" followed by many spaces take quadratic time.
_TRANSCRIPT_TAG = re.compile(r"<(?=\s*(?:/\s*)?transcript)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Attribute values
# ---------------------------------------------------------------------------


def _number(value: float) -> str:
    return f"{value:g}"


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _semitones(pitch: str | None) -> float | None:
    """A relative pitch (``+15%``, ``-2st``) in semitones; ``None`` for other values."""
    match = _RELATIVE_PITCH_RE.fullmatch(pitch) if pitch is not None else None
    if match is None:
        return None
    amount = float(match[1])
    if match[2] == "st":
        return _finite(amount)
    if not -100.0 < amount < math.inf:
        return None
    return 12.0 * math.log2(1.0 + amount / 100.0)


def _decibels(volume: str | None) -> float | None:
    match = _VOLUME_RE.fullmatch(volume) if volume is not None else None
    return None if match is None else _finite(float(match[1]))


def _rate_log2(rate: str | None) -> float | None:
    """A rate percentage as log2 of its speed ratio; ``None`` for named or invalid rates."""
    match = _RATE_PERCENT_RE.fullmatch(rate) if rate is not None else None
    if match is None or not 0.0 < float(match[1]) < math.inf:
        return None
    return math.log2(float(match[1]) / 100.0)


def _amount(attribute: str, value: str | None) -> float | None:
    """A relative *value* of *attribute* as an amount that adds up when
    elements nest (spec 3.2): semitones, dB or log2 of the rate ratio.
    ``None`` for a missing, invalid or non-relative value (``185Hz``, ``fast``)."""
    if attribute == "pitch":
        return _semitones(value)
    if attribute == "volume":
        return _decibels(value)
    return _rate_log2(value)


def _signed(amount: float, unit: str) -> str:
    """*amount* written as a signed IML value (``+28%``, ``-4.3st``, ``+3dB``)."""
    rounded = round(amount, 0 if unit == "%" else 1) + 0.0  # + 0.0: no "-0"
    return f"{rounded:+g}{unit}"


def _value(attribute: str, amount: float, sources: Sequence[str | None]) -> str:
    """The *attribute* value for an *amount* (see :func:`_amount`), in the
    unit of the *sources* it was computed from."""
    amount = max(-_MAX_AMOUNT, min(_MAX_AMOUNT, amount))
    if attribute == "pitch":
        if all(source is None or source.endswith("%") for source in sources):
            return _signed((2.0 ** (amount / 12.0) - 1.0) * 100.0, "%")
        return _signed(amount, "st")
    if attribute == "volume":
        return _signed(amount, "dB")
    return f"{round(100.0 * 2.0**amount):d}%"


def _composed(level: Prosody, node: Prosody) -> tuple[Prosody, Prosody]:
    """Add *node*'s relative values to the utterance *level* around it.

    Spec 3.2: a relative value inside another ``<prosody>`` is relative to
    it, so the two add up. Returns the new level and *node* without the
    values it gave up; values that are not relative (``185Hz``, ``fast``)
    stay on *node*, whose notes describe them.
    """
    changes: dict[str, str] = {}
    for attribute in _RELATIVE_ATTRIBUTES:
        inner = getattr(node, attribute)
        amount = _amount(attribute, inner)
        if amount is None:
            continue
        outer = getattr(level, attribute)  # the level holds relative values only
        outer_amount = _amount(attribute, outer)
        if outer_amount is None:
            changes[attribute] = inner
        else:
            changes[attribute] = _value(attribute, outer_amount + amount, (outer, inner))
    return _with_values(level, changes), _with_values(node, dict.fromkeys(changes))


def _with_values(node: Prosody, values: Mapping[str, str | None]) -> Prosody:
    """*node* with the pitch, volume, rate and quality in *values* instead of its own."""
    return replace(
        node,
        pitch=values.get("pitch", node.pitch),
        volume=values.get("volume", node.volume),
        rate=values.get("rate", node.rate),
        quality=values.get("quality", node.quality),
    )


def _residual(attribute: str, value: str | None, shared: str) -> str | None:
    """*value*, which is relative to the speaker's baseline, relative to the
    *shared* value of the fragments around it instead (``None`` when equal).

    A missing relative value is at the baseline. Values that cannot be
    compared (voice qualities, ``fast``, ``185Hz``) are kept unless equal.
    """
    if value == shared:
        return None
    shared_amount = _amount(attribute, shared) if attribute in _RELATIVE_ATTRIBUTES else None
    if shared_amount is None:
        return value
    amount = 0.0 if value is None else _amount(attribute, value)
    if amount is None:
        return value
    difference = amount - shared_amount
    return None if abs(difference) < _SAME else _value(attribute, difference, (value, shared))


# ---------------------------------------------------------------------------
# Attribute descriptions
# ---------------------------------------------------------------------------


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
    semitones = _semitones(pitch)
    if semitones is not None:
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
    decibels = _decibels(volume)
    if volume is None or decibels is None:
        return None
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
    log_ratio = _rate_log2(rate)
    if log_ratio is None:
        return None
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


def _own_notes(
    node: Prosody | Emphasis | Segment,
    include_numbers: bool,
    expected_contour: str | None = None,
) -> list[str]:
    """How *node* says its content was spoken, not counting nested markup.

    A ``pitch_contour`` equal to *expected_contour* (the usual end of the
    sentence the node ends) is left out.
    """
    if isinstance(node, Emphasis):
        return list(_EMPHASIS.get(node.level, _EMPHASIS["moderate"])[1])
    if isinstance(node, Segment):
        tempo = _TEMPOS.get(node.tempo or "")
        rhythm = _RHYTHMS.get(node.rhythm or "")
        return [note for note in (tempo, rhythm) if note is not None]
    contour = node.pitch_contour if node.pitch_contour != expected_contour else None
    notes = [
        _pitch_note(node.pitch, include_numbers),
        _volume_note(node.volume, include_numbers),
        _rate_note(node.rate, include_numbers),
        _CONTOURS.get(contour or ""),
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
    return f'emotion labeled "{label}" ({estimate})'


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
# Document structure
# ---------------------------------------------------------------------------


def _is_blank(text: str) -> bool:
    return not text.strip(_SPACE_CHARS)


def _texts(children: Sequence[ChildNode]) -> list[str]:
    """The text of *children*, in document order (iteratively: a recursive
    generator would pass each text through every level of nesting)."""
    texts: list[str] = []
    stack: list[Iterator[ChildNode]] = [iter(children)]
    while stack:
        for child in stack[-1]:
            if isinstance(child, str):
                texts.append(child)
            elif not isinstance(child, Pause):
                stack.append(iter(child.children))
                break
        else:
            stack.pop()
    return texts


class _Words:
    """Whether a node holds a word (rather than only spaces, punctuation or
    pauses), remembered so that asking again about a node, or about one
    inside it, takes constant time."""

    def __init__(self) -> None:
        # id -> (node, answer); holding the node keeps its id from being reused.
        self._known: dict[int, tuple[ChildNode, bool]] = {}

    def __call__(self, node: ChildNode) -> bool:
        known = self._known.get(id(node))
        if known is not None:
            return known[1]
        if isinstance(node, str):
            answer = _WORD_CHAR.search(node) is not None
        elif isinstance(node, Pause):
            answer = False
        else:
            answer = any([self(child) for child in node.children])  # every child is remembered
        self._known[id(node)] = (node, answer)
        return answer


def _sole_element(children: Sequence[ChildNode]) -> ChildNode | None:
    """The one element among *children* if everything else is whitespace."""
    elements = [child for child in children if not isinstance(child, str)]
    if len(elements) != 1:
        return None
    if any(isinstance(child, str) and not _is_blank(child) for child in children):
        return None
    return elements[0]


def _whole_span(
    children: Sequence[ChildNode], has_words: _Words
) -> tuple[int, Prosody | Segment] | None:
    """A ``<prosody>`` or ``<segment>`` that holds all the words, with its index.

    Pauses, spaces and punctuation may be outside it.
    """
    found: tuple[int, Prosody | Segment] | None = None
    for index, child in enumerate(children):
        if isinstance(child, (Prosody, Segment)) and found is None:
            found = (index, child)
        elif has_words(child):
            return None
    return found


def _fragments(
    children: Sequence[ChildNode], has_words: _Words
) -> list[tuple[int, Prosody]] | None:
    """The ``<prosody>`` elements that between them hold all the words.

    Each is a child, or the sole element of an ``<emphasis>`` child (a word
    that stands alone). ``None`` if some words are outside them.
    """
    found: list[tuple[int, Prosody]] = []
    for index, child in enumerate(children):
        if not has_words(child):
            continue
        prosody = _sole_element(child.children) if isinstance(child, Emphasis) else child
        if not isinstance(prosody, Prosody):
            return None
        found.append((index, prosody))
    return found


def _shared_values(prosodies: Sequence[Prosody]) -> dict[str, str]:
    """The pitch, volume, rate and voice quality that all *prosodies* have."""
    shared: dict[str, str] = {}
    for attribute in (*_RELATIVE_ATTRIBUTES, "quality"):
        values: list[str | None] = [getattr(p, attribute) for p in prosodies]
        first = values[0]
        if first is None:
            continue
        amount = _amount(attribute, first) if attribute in _RELATIVE_ATTRIBUTES else None
        if amount is None:
            same = all(value == first for value in values)
        else:
            amounts = [_amount(attribute, value) for value in values]
            same = all(a is not None and abs(a - amount) < _SAME for a in amounts)
        if same:
            shared[attribute] = first
    return shared


def _lift_fragments(
    children: Sequence[ChildNode], has_words: _Words
) -> tuple[Prosody, list[ChildNode]]:
    """Utterance-level prosody that was split into fragments, and the
    children with the fragments described relative to it.

    A producer that cannot put the whole utterance in one ``<prosody>`` (the
    assembler, around an emphasized word with prosody of its own, which would
    otherwise be nested three deep) writes the utterance's values on
    several: the words stay in sibling ``<prosody>`` fragments with the same
    values, and the standalone word's ``<prosody>`` carries the sum of the
    utterance's values and its own, both relative to the speaker's baseline.
    The values that the fragments share (all of them, or all but the
    standalone words) are the utterance's; each fragment keeps only what
    differs from them. Returns an empty ``Prosody`` and *children* unchanged
    when the words are not all in fragments that share a value.
    """
    fragments = _fragments(children, has_words)
    if fragments is None or len(fragments) < 2:
        return Prosody(), list(children)
    plain = [prosody for index, prosody in fragments if children[index] is prosody]
    shared = _shared_values(plain or [prosody for _, prosody in fragments])
    if not shared:
        return Prosody(), list(children)
    out = list(children)
    for index, prosody in fragments:
        rest = _with_values(prosody, {
            attribute: _residual(attribute, getattr(prosody, attribute), value)
            for attribute, value in shared.items()
        })
        child = children[index]
        if child is prosody:
            out[index] = rest
        elif isinstance(child, Emphasis):
            out[index] = replace(
                child, children=tuple(rest if node is prosody else node for node in child.children)
            )
    return _with_values(Prosody(), shared), out


# The last word character of a text: "\W*" cannot start inside the final
# run of non-word characters, so a search takes linear time.
_LAST_WORD_CHAR = re.compile(r"\w\W*\Z")


def _final_word(children: Sequence[ChildNode]) -> tuple[list[ChildNode], bool] | None:
    """Where the last word of *children* is: the elements around it
    (outermost first), and whether a question mark follows it. ``None`` if
    there are no words. Looks at each node at most once.
    """
    question = False
    for child in reversed(children):
        if isinstance(child, str):
            match = _LAST_WORD_CHAR.search(child)
            after = child if match is None else child[match.start() + 1:]
            question = question or any(mark in after for mark in _QUESTION_MARKS)
            if match is not None:
                return [], question
        elif not isinstance(child, Pause):
            found = _final_word(child.children)
            if found is not None:
                path, asks = found
                return [child, *path], asks or question
            after = "".join(_texts(child.children))
            question = question or any(mark in after for mark in _QUESTION_MARKS)
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _merges(node: Prosody | Emphasis | Segment, inner: Prosody | Emphasis) -> bool:
    """Whether *node* and its sole element *inner* read as one span.

    ``<segment>`` notes (tempo, rhythm) stay on their own span, and nested
    ``<prosody>`` keeps both scopes: the inner one is relative to the outer.
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

    The line cannot pass for notation or for another speaker's line: a
    speaker name other than a plain one is quoted, and so are the words of
    an utterance without a speaker that would read as a ``Delivery:`` line
    or as a speaker's line (``agent: ...``). ``<transcript>`` tags in the
    text lose their ``<``, so they cannot end the block that
    :func:`build_messages` puts the context in.
    """
    speaker = " ".join((speaker_id or "").split())
    if speaker:
        if not _PLAIN_SPEAKER.fullmatch(speaker) or _DELIVERY_PREFIX.match(f"{speaker}:"):
            speaker = '"{}"'.format(speaker.replace('"', "'"))
        line = f"{speaker}: {text}"
    elif _DELIVERY_PREFIX.match(text) or _SPEAKER_PREFIX.match(text):
        line = f'"{text}"'
    else:
        line = text
    return _TRANSCRIPT_TAG.sub("\u2039", line)


def _is_placeholder_text(text: str) -> bool:
    """Whether the words of *text* are all ``[speech]`` placeholders."""
    return _PLACEHOLDER in text and not _WORD_CHAR.search(text.replace(_PLACEHOLDER, " "))


class _Renderer:
    """Renders utterance content in the annotation notation."""

    def __init__(self, include_numbers: bool, min_pause_ms: int) -> None:
        self.include_numbers = include_numbers
        self.min_pause_ms = min_pause_ms
        # The elements that end the utterance being rendered, and the
        # contour its last sentence usually ends with (see utterance()).
        self._final: set[int] = set()
        self._expected_contour: str | None = None

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
        # ("yesterday (rising)." rather than "yesterday. (rising)").
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

    def notes(self, node: Prosody | Emphasis | Segment, *, final: bool) -> list[str]:
        """The notes on *node*; *final* if it holds the utterance's last word."""
        expected = self._expected_contour if final else None
        return _own_notes(node, self.include_numbers, expected)

    def parts(self, node: Prosody | Emphasis | Segment) -> tuple[str, list[str], str]:
        """The rendered content, notes and emphasis mark of *node*.

        ``<emphasis>`` around a sole ``<prosody>`` (or the reverse) reads as
        one span, ``**word** (louder)``: the notes describe the stressed words.
        Emphasis inside emphasis keeps the stronger mark. Each node is
        rendered once, so the time is linear in the size of the document.
        """
        notes = self.notes(node, final=id(node) in self._final)
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
        has_words = _Words()
        children: Sequence[ChildNode] = utt.children
        # Markup around all the words describes the utterance as a whole:
        # nested relative values add up into one level (spec 3.2), and so do
        # the values that fragments of the utterance share.
        level = Prosody()
        wholes: list[Prosody | Segment] = []
        while (whole := _whole_span(children, has_words)) is not None:
            index, node = whole
            if isinstance(node, Prosody):
                level, node = _composed(level, node)
            wholes.append(node)
            children = [*children[:index], *node.children, *children[index + 1:]]
        shared, children = _lift_fragments(children, has_words)
        level, rest = _composed(level, shared)

        # The contour the utterance's last sentence usually ends with is not
        # worth a note on the markup that ends it.
        final = _final_word(children)
        self._final = set() if final is None else {id(node) for node in final[0]}
        question = final is not None and final[1]
        self._expected_contour = _QUESTION_CONTOUR if question else _STATEMENT_CONTOUR

        overall = self.notes(level, final=False)
        for node in wholes:
            overall = _merge(overall, self.notes(node, final=final is not None))
        overall = _merge(overall, self.notes(rest, final=False))

        text = _finish(self.children(children)) or "[no words]"
        lines = [_utterance_line(utt.speaker_id, text)]

        delivery: list[str] = []
        if _is_placeholder_text("".join(_texts(utt.children))):
            delivery.append("words not transcribed")
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
        Offsets are relative to what the note compares with (see the
        module docstring), so a value can differ from the one in the markup.
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
