"""TextToIML -- predict prosody for plain text (no audio input).

Supports a rule-based baseline and a pluggable ML model backend.
Spec reference: Section 3 (output conforms to IML tag set).

Phase 6a implements rule-based prediction using punctuation, capitalization,
and lexical cues.  Zero external dependencies beyond ``lxml``.

The rule-based model:

- keeps the text exactly: stripping the tags from the output (or
  :meth:`IMLParser.to_plain_text <prosody_protocol.parser.IMLParser.to_plain_text>`)
  gives the input with whitespace runs collapsed to one space and
  characters XML cannot carry (C0 controls) replaced by spaces;
- makes one ``<utterance>`` per sentence. Sentences end at ``.``, ``!`` or
  ``?``, but not after an ellipsis, a title such as "Dr." or "Mr.", a
  single initial, or an abbreviation such as "e.g." or "U.S." unless the next
  word is a typical sentence opener ("The", "I", "Then", ...). Quoted speech
  ("..." or “...”) becomes its own utterance(s), in place and with its
  quote marks;
- marks questions with ``pitch_contour="rise"`` ("rise-sharp" for "?!"),
  sarcasm with "fall-rise", exclamations with ``pitch="+5%" volume="+3dB"``,
  an all-caps sentence with ``volume="+6dB"``, an ellipsis between words
  ("Well... I", "and ...then"; the dots stay in the text) with
  ``<pause duration="500"/>``, and an ALL-CAPS word in a mixed-case
  sentence with ``<emphasis level="strong">``, except likely acronyms
  (NASA, API, OK);
- labels emotion from lexical cues (emotion words with simple negation
  handling, hedges, "?!", and sarcasm cues: idioms such as "yeah right" or
  "just what I needed", and a positive word ending a clause after "oh",
  "ah" or "just" -- "Oh great.", "Oh, that's GREAT." -- in a sentence
  that is not an exclamation, so "Oh, that's great news!" stays joyful).
  Confidence reflects how strong the cues are (0.5 for a single cue word,
  higher for agreeing cues or an exclamation; sarcasm cues give at most
  0.6) and predictions below ``min_confidence`` are left out, so a
  sentence without clear cues carries no emotion. An exclamation alone is
  not an emotion. The emotion that cue words in ``context`` point to
  counts as one more cue in every sentence.

Phase 6b (ML model) is a placeholder pending dataset infrastructure (Phase 10).
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable

from .exceptions import ConversionError
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
# Emotion lexicon (kept intentionally small -- rule-based baseline)
# ---------------------------------------------------------------------------

# Cue word -> emotion from the spec 3.1 core vocabulary. Words are matched in
# lower case with typographic apostrophes normalized to "'".
_EMOTION_WORDS: dict[str, str] = {
    **dict.fromkeys(
        (
            "love", "loved", "lovely", "happy", "glad", "wonderful", "fantastic",
            "amazing", "great", "excellent", "beautiful", "brilliant", "awesome",
            "delightful", "delighted", "pleased", "thankful", "grateful", "excited",
            "thrilled", "cheerful", "joyful", "superb", "terrific", "magnificent",
            "yay", "hooray", "enjoy", "enjoyed",
        ),
        "joyful",
    ),
    **dict.fromkeys(
        (
            "frustrated", "frustrating", "annoyed", "annoying", "ridiculous",
            "stupid", "useless", "terrible", "horrible", "awful", "dreadful",
            "pathetic", "worse", "worst", "disappointed", "disappointing",
            "unacceptable", "ugh", "argh",
        ),
        "frustrated",
    ),
    **dict.fromkeys(
        (
            "angry", "furious", "hate", "hated", "outraged", "outrageous", "livid",
            "infuriating", "furiously",
        ),
        "angry",
    ),
    **dict.fromkeys(
        (
            "sad", "unhappy", "depressed", "miserable", "heartbroken", "lonely",
            "crying", "grief", "grieving", "devastated", "hopeless", "upset",
        ),
        "sad",
    ),
    **dict.fromkeys(
        (
            "scared", "afraid", "terrified", "frightened", "worried", "anxious",
            "nervous", "panic", "panicking",
        ),
        "fearful",
    ),
    **dict.fromkeys(
        ("disgusting", "disgusted", "gross", "revolting", "yuck", "eww", "repulsive", "vile"),
        "disgusted",
    ),
    **dict.fromkeys(
        ("wow", "whoa", "omg", "astonishing", "astonished", "shocked", "surprised"),
        "surprised",
    ),
    **dict.fromkeys(
        (
            "maybe", "perhaps", "possibly", "probably", "unsure", "uncertain",
            "somehow", "apparently", "seemingly", "hmm", "um", "uh",
        ),
        "uncertain",
    ),
}

# Multi-word cues, matched on the lower-case word sequence.
_EMOTION_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:not sure|i guess|i suppose|i wonder|no idea|who knows)\b"), "uncertain"),
    (re.compile(r"\b(?:i don't know|i do not know)\b"), "uncertain"),
    (re.compile(r"\b(?:watch out|look out)\b|^help(?: me| us)?$"), "fearful"),
    (re.compile(r"\bno way\b"), "surprised"),
    (re.compile(r"\b(?:i'm so sorry|i am so sorry|i understand how you feel)\b"), "empathetic"),
)

# Sarcasm cues. Words inside a match do not also count as joyful.
#
# Idioms that are sarcastic in almost any frame.
_SARCASM_IDIOM_RE = re.compile(
    r"\byeah,? right\b|\bjust what i (?:needed|wanted)\b|\b(?:what a|big) surprise\b",
    re.IGNORECASE,
)
# A positive word after "oh", "ah" or "just" that ends its clause: "Oh
# great.", "Oh, that's GREAT.", "That's just perfect, thanks." It counts only
# in a sentence that is not an exclamation, since "Oh, that's great news!"
# and "Oh great, you made it!" are sincere; and it is a weaker cue than an
# idiom unless the positive word is in capitals, the clause trails off
# ("Oh great..."), or a follow-up such as "another" comes after it ("Oh
# great, another meeting.").
#
# The words between the opener and the positive word are never openers
# themselves, and there are at most _MAX_FRAME_FILLERS of them, so each
# match attempt looks at a bounded number of words and a scan of the whole
# text takes linear time. (With "just" among them, as it once was, every
# "just" in a run of them rescanned the rest of the run: 100 KB of "just
# just ..." took about a minute.) "Oh, that's just great" still matches,
# from its "just".
_POSITIVE_FOR_SARCASM = (
    r"(?:great|wonderful|perfect|fantastic|brilliant|lovely|marvell?ous|terrific|"
    r"super|fabulous|splendid|nice|awesome|joy|goody)"
)
_MAX_FRAME_FILLERS = 6
_POSITIVE_FRAME_RE = re.compile(
    r"\b(?:oh|ah|just),?\s+"
    r"(?:(?:that's|thats|that is|it's|this is|how|so|really|yeah),?\s+)"
    f"{{0,{_MAX_FRAME_FILLERS}}}"
    r"(?P<word>" + _POSITIVE_FOR_SARCASM + r")"
    r"(?=[\"'\u201d)]*\s*(?:[,;:.!?\u2026\u2014\u2013]|$))",
    re.IGNORECASE,
)
_SARCASM_FOLLOW_UP_RE = re.compile(r"\b(?:another|again|more)\b", re.IGNORECASE)

_NEGATORS = frozenset({"not", "no", "never", "cannot", "hardly", "nor", "without"})
_NEGATION_WINDOW = 3

# Every core emotion name is also a cue (e.g. context="frustrated").
_CORE_EMOTIONS = (
    "neutral", "sincere", "sarcastic", "frustrated", "joyful", "uncertain", "angry",
    "sad", "fearful", "surprised", "disgusted", "calm", "empathetic",
)
_CONTEXT_WORDS: dict[str, str] = {**_EMOTION_WORDS, **{e: e for e in _CORE_EMOTIONS}}

# Emotions an exclamation mark makes more likely to be meant. (Sarcasm is
# not one of them: an exclamation argues for sincerity.)
_AROUSED_EMOTIONS = frozenset(
    {"joyful", "angry", "frustrated", "surprised", "fearful", "disgusted"}
)

# Cue weights and the confidence they translate to.
_SARCASM_WEIGHT = 1.5  # an idiom or a reinforced positive frame -> 0.6
_SARCASM_FRAME_WEIGHT = 1.0  # a bare positive frame ("Oh great.") -> 0.5
_INTERROBANG_WEIGHT = 1.0  # "?!" -> surprised
_ELLIPSIS_WEIGHT = 0.5  # hesitation -> uncertain
_CONTEXT_WEIGHT = 1.0  # the emotion of the caller's context counts as one cue word
_BASE_CONFIDENCE = 0.3
_CONFIDENCE_PER_WEIGHT = 0.2  # a single cue word -> 0.5
_MAX_CUE_CONFIDENCE = 0.7
_EXCLAMATION_BONUS = 0.1

# ---------------------------------------------------------------------------
# Emphasis
# ---------------------------------------------------------------------------

# A word written in capitals: letters with optional inner apostrophes or
# hyphens (DON'T, SUPER-FAST, ÉTÉ), no digits.
_CAPS_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*")

# Short capitalized words (2-4 letters) are emphasized only if they are
# ordinary words; any other short all-caps token is taken to be an acronym
# (NASA, FBI, API, USA, OK).
_SHOUTED_SHORT_WORDS = frozenset({
    "no", "yes", "not", "now", "stop", "help", "why", "what", "how", "who", "you",
    "me", "my", "all", "bad", "big", "go", "so", "too", "very", "real", "sure",
    "good", "best", "love", "hate", "wow", "oh", "hey", "out", "ever", "must",
    "did", "do", "done", "can", "will", "was", "is", "are", "am", "he", "she",
    "we", "they", "it", "this", "that", "here", "late", "fast", "hot", "cold",
    "huge", "mine", "your", "just", "only", "one", "two", "get", "off", "up",
    "on", "in", "at", "to", "the", "and", "but", "or", "if", "of", "for", "his",
    "her", "our", "them", "him", "us", "then", "than", "more", "less", "most",
    "much", "many", "each", "any", "some", "same", "own", "keep", "stay", "wait",
    "look", "come", "move", "run", "hard", "easy", "fun", "true", "damn", "dead",
    "died", "die", "kill", "hurt", "pain", "fire", "cool", "nice", "fine", "mad",
    "sad", "yet", "ugh", "omg",
})
# Capitalized words of five or more letters that are usually acronyms.
_LONG_ACRONYMS = frozenset({
    "nasdaq", "unicef", "unesco", "naacp", "asean", "https", "ascii", "scuba",
    "laser", "radar", "sonar",
})

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Titles that precede a name: a following period never ends the sentence.
_TITLES = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "prof", "st", "mt", "sgt", "capt", "col", "gen",
    "lt", "rev", "hon", "fr", "gov", "pres", "sen", "rep", "messrs", "mme", "mlle",
})
# Other abbreviations: the period ends the sentence only before a typical
# sentence opener.
_ABBREVIATIONS = frozenset({
    "etc", "inc", "ltd", "co", "corp", "jr", "sr", "vs", "cf", "fig", "vol",
    "approx", "dept", "est", "al", "e.g", "i.e", "a.m", "p.m", "ph.d", "u.s",
    "u.k", "u.s.a", "e.u",
})
# Abbreviations that are also words ("No. 5" but "I said no. Nobody
# listened."): the period does not end the sentence only before a number.
_NUMBER_ABBREVIATIONS = frozenset({"no", "nos"})
_INITIALISM_RE = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")
_SENTENCE_OPENERS = frozenset({
    "the", "a", "an", "i", "i'm", "i've", "i'll", "i'd", "he", "she", "it", "we",
    "they", "you", "this", "that", "these", "those", "there", "then", "but", "and",
    "so", "however", "now", "my", "our", "his", "her", "their", "its", "what", "why",
    "how", "when", "where", "who", "if", "in", "on", "at", "after", "before", "yes",
    "no", "oh", "well", "it's", "that's", "there's", "let's",
})

# Sentence-final punctuation, which closing quotes or brackets may follow.
_TERMINAL_CHARS = ".!?\u2026"
_OPENING_PUNCT = "\"'\u201c\u2018([\u00ab"
_CLOSING_QUOTES = "\"'\u201d\u2019)]\u00bb"
_CLOSING_PUNCT = ".,;:!?\"'\u201d\u2019)]}\u00bb\u2026"

# Quoted speech: opening mark -> the marks that close it (straight or
# typographic double quotes, and guillemets).
_QUOTE_CLOSERS: dict[str, str] = {
    '"': '"',
    "\u201c": "\u201d",
    "\u201e": "\u201c\u201d",
    "\u00ab": "\u00bb",
}

# Ellipsis: three or more dots or the unicode ellipsis character.
_ELLIPSIS_RE = re.compile(r"\.{3,}|\u2026")
_ELLIPSIS_PAUSE_MS = 500

# Characters XML 1.0 cannot carry (C0 controls other than tab/LF/CR, lone
# surrogates, U+FFFE and U+FFFF); replaced by spaces.
_XML_INVALID_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_XML_WHITESPACE_RUN_RE = re.compile(r"[ \t\r\n]+")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")

_IML_VERSION = "0.1.0"


def _normalize_text(text: str) -> str:
    """Return the text an IML document for ``text`` carries (see module docstring)."""
    text = _XML_INVALID_CHARS_RE.sub(" ", text)
    return _XML_WHITESPACE_RUN_RE.sub(" ", text).strip(" ")


def _strip_punctuation(word: str) -> str:
    """Strip surrounding punctuation from a token."""
    return word.strip(_OPENING_PUNCT + _CLOSING_PUNCT + "-\u2014\u2013")


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------


def _sarcasm_weight(text: str, claim: Callable[[int, int], None]) -> float:
    """Return the weight of the sarcasm cues in ``text`` (0.0 if none).

    ``claim`` is called with the character span of each cue, so that its
    positive word does not also count as joyful.
    """
    weight = 0.0
    for m in _SARCASM_IDIOM_RE.finditer(text):
        weight = _SARCASM_WEIGHT
        claim(m.start(), m.end())
    if "!" in _terminal_punctuation(text):
        return weight
    follow_ups: list[int] | None = None
    for m in _POSITIVE_FRAME_RE.finditer(text):
        word, after = m.group("word"), m.end()
        while after < len(text) and text[after] in _CLOSING_QUOTES + " ":
            after += 1
        if follow_ups is None:
            follow_ups = [f.start() for f in _SARCASM_FOLLOW_UP_RE.finditer(text)]
        reinforced = (
            (len(word) > 1 and word.isupper())
            or _ELLIPSIS_RE.match(text, after) is not None
            or bisect_left(follow_ups, m.end()) < len(follow_ups)
        )
        weight = max(weight, _SARCASM_WEIGHT if reinforced else _SARCASM_FRAME_WEIGHT)
        claim(m.start(), m.end())
    return weight


def _cue_scores(text: str, lexicon: dict[str, str]) -> dict[str, float]:
    """Score emotions by sarcasm cues, cue phrases and cue words in ``text``.

    Negated cue words are ignored, and words that are part of a sarcasm cue
    or a cue phrase do not count again on their own. Runs in time linear in
    the length of ``text`` (see ``_POSITIVE_FRAME_RE``).
    """
    text = text.replace("\u2019", "'")
    matches = list(_WORD_RE.finditer(text))
    starts = [m.start() for m in matches]
    words = [m.group().lower() for m in matches]
    scores: dict[str, float] = {}
    claimed: set[int] = set()

    def claim(start: int, end: int) -> None:
        claimed.update(range(bisect_left(starts, start), bisect_left(starts, end)))

    sarcasm = _sarcasm_weight(text, claim)
    if sarcasm:
        scores["sarcastic"] = sarcasm

    # Multi-word cues are matched on the lower-case word sequence.
    joined = " ".join(words)
    offsets: list[int] = []
    position = 0
    for word in words:
        offsets.append(position)
        position += len(word) + 1
    for pattern, emotion in _EMOTION_PHRASES:
        for m in pattern.finditer(joined):
            scores[emotion] = scores.get(emotion, 0.0) + 1.0
            first = bisect_right(offsets, m.start()) - 1
            claimed.update(range(first, bisect_left(offsets, m.end())))
    for i, word in enumerate(words):
        cue = lexicon.get(word)
        if cue is None or i in claimed:
            continue
        window = words[max(0, i - _NEGATION_WINDOW) : i]
        if any(w in _NEGATORS or w.endswith("n't") for w in window):
            continue  # "not happy", "I'm not angry" -- no reliable label
        scores[cue] = scores.get(cue, 0.0) + 1.0
    return scores


def _context_emotion(context: str | None) -> str | None:
    """Return the emotion that cue words in ``context`` point to, if any."""
    if not context:
        return None
    scores = _cue_scores(context, _CONTEXT_WORDS)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def _detect_sentence_emotion(
    sentence: str, context_emotion: str | None = None
) -> tuple[str | None, float]:
    """Detect emotion from lexical cues in a sentence.

    Returns (emotion, confidence) or (None, 0.0) if the cues are absent or
    tied. ``context_emotion`` counts like one more cue word for that emotion.
    """
    scores = _cue_scores(sentence, _EMOTION_WORDS)
    terminal = _terminal_punctuation(sentence)
    if "?" in terminal and "!" in terminal:
        scores["surprised"] = scores.get("surprised", 0.0) + _INTERROBANG_WEIGHT
    if _ELLIPSIS_RE.search(sentence) and "sarcastic" not in scores:
        # Trailing off after a sarcasm cue ("Yeah right...") is deadpan
        # delivery, not hesitation.
        scores["uncertain"] = scores.get("uncertain", 0.0) + _ELLIPSIS_WEIGHT
    if context_emotion is not None:
        scores[context_emotion] = scores.get(context_emotion, 0.0) + _CONTEXT_WEIGHT

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return (None, 0.0)
    emotion, best = ranked[0]
    margin = best - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if margin <= 0:
        return (None, 0.0)
    confidence = min(_MAX_CUE_CONFIDENCE, _BASE_CONFIDENCE + _CONFIDENCE_PER_WEIGHT * margin)
    if "!" in terminal and emotion in _AROUSED_EMOTIONS:
        confidence += _EXCLAMATION_BONUS
    return (emotion, round(confidence, 2))


# ---------------------------------------------------------------------------
# Sentences and quotes
# ---------------------------------------------------------------------------


def _terminal_run(token: str) -> tuple[int, int]:
    """Return the span of the final run of ``_TERMINAL_CHARS`` in ``token``.

    Closing quotes and brackets may follow the run. The span is empty if
    ``token`` does not end in sentence-final punctuation.
    """
    end = len(token.rstrip(_CLOSING_QUOTES))
    start = end
    while start > 0 and token[start - 1] in _TERMINAL_CHARS:
        start -= 1
    return start, end


def _terminal_punctuation(sentence: str) -> str:
    # A quoted exclamation can be followed by a comma: '"Stop!",'.
    body = sentence.rstrip(" ,;:")
    start, end = _terminal_run(body)
    return body[start:end]


def _ends_with_ellipsis(token: str) -> bool:
    return bool(_ELLIPSIS_RE.search(token.rstrip(_CLOSING_QUOTES)[-3:]))


def _ends_sentence(token: str, next_token: str) -> bool:
    """Whether a sentence boundary falls between ``token`` and ``next_token``."""
    start, end = _terminal_run(token)
    if start == end:
        return False
    punct = token[start:end]
    if punct.endswith("...") or punct.endswith("\u2026"):
        return False  # An ellipsis is a hesitation, not a boundary.
    if "!" in punct or "?" in punct:
        return True
    core = token[:start].lstrip(_OPENING_PUNCT)
    following = next_token.lstrip(_OPENING_PUNCT)
    if core.lower() in _TITLES or re.fullmatch(r"[A-Za-z]", core):
        return False  # "Dr. Smith", "J. R. R. Tolkien"
    if following[:1].islower():
        return False  # "5 p.m. and", "e.g. far"
    if core.lower() in _NUMBER_ABBREVIATIONS and following[:1].isdigit():
        return False  # "No. 5"
    if core.lower() in _ABBREVIATIONS or _INITIALISM_RE.fullmatch(core):
        return following.lower().replace("\u2019", "'") in _SENTENCE_OPENERS
    return True


def _split_sentences(text: str) -> list[str]:
    """Split normalized text into sentences at single spaces."""
    tokens = text.split(" ")
    sentences: list[str] = []
    current: list[str] = []
    for i, token in enumerate(tokens):
        current.append(token)
        if i + 1 < len(tokens) and _ends_sentence(token, tokens[i + 1]):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return [s for s in sentences if s]


def _quote_spans(text: str) -> list[tuple[int, int]]:
    """Return the spans of quoted speech in ``text``, marks included.

    A quote is an opening mark, at least one character, and the first mark
    that closes it. Runs in linear time: the search for each kind of closing
    mark resumes where the previous one ended.
    """
    spans: list[tuple[int, int]] = []
    next_closer: dict[str, int] = {}  # closing marks -> next position (-1: none left)
    i = 0
    while i < len(text):
        closers = _QUOTE_CLOSERS.get(text[i])
        if closers is None:
            i += 1
            continue
        j = next_closer.get(closers, 0)
        if 0 <= j <= i:
            found = [k for k in (text.find(c, i + 1) for c in closers) if k >= 0]
            j = next_closer[closers] = min(found, default=-1)
        if j > i + 1:
            spans.append((i, j + 1))
            i = j + 1
        else:
            i += 1  # Unclosed, or an empty quote.
    return spans


def _split_quotes(text: str) -> list[str]:
    """Split normalized text into runs, each quoted span in a run of its own.

    Runs break only at spaces, so a quote keeps its marks and any
    punctuation attached to them, and joining the runs with single spaces
    gives ``text`` back.
    """
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    bounds: list[tuple[int, int]] = []
    for quote_start, quote_end in _quote_spans(text):
        before = bisect_left(spaces, quote_start)
        start = spaces[before - 1] + 1 if before else 0
        after = bisect_left(spaces, quote_end)
        end = spaces[after] if after < len(spaces) else len(text)
        if bounds and start <= bounds[-1][1]:
            bounds[-1] = (bounds[-1][0], max(end, bounds[-1][1]))
        else:
            bounds.append((start, end))
    runs: list[str] = []
    pos = 0
    for start, end in bounds:
        runs.append(text[pos:start].strip(" "))
        runs.append(text[start:end])
        pos = end
    runs.append(text[pos:].strip(" "))
    return [r for r in runs if r]


# ---------------------------------------------------------------------------
# Utterance construction
# ---------------------------------------------------------------------------


def _is_emphasized(core: str, shouting: bool) -> bool:
    """Whether a token's letters are an ALL-CAPS word to emphasize."""
    if shouting or not _CAPS_WORD_RE.fullmatch(core) or core.upper() != core:
        return False
    letters = [ch for ch in core if ch.isalpha()]
    if len(letters) < 2 or not any(ch.isupper() for ch in letters):
        return False
    if any(ch in core for ch in "'\u2019-"):
        return True  # DON'T, I'M, SUPER-FAST: words, not acronyms
    if len(letters) <= 4:
        return core.lower() in _SHOUTED_SHORT_WORDS
    return core.lower() not in _LONG_ACRONYMS


def _is_shouting(tokens: list[str]) -> bool:
    """Whether a sentence is written entirely in capitals (at least two words)."""
    letters = [ch for ch in " ".join(tokens) if ch.isalpha()]
    words = [t for t in tokens if sum(ch.isalpha() for ch in t) >= 2]
    return len(words) >= 2 and all(not ch.islower() for ch in letters)


def _build_children_for_sentence(sentence: str, shouting: bool) -> tuple[ChildNode, ...]:
    """Build IML children for a single sentence, keeping its text exactly."""
    tokens = sentence.split(" ")
    children: list[ChildNode] = []
    pending: list[str] = []  # Text since the last element (joined once: linear time).

    def add_text(text: str) -> None:
        if text:
            pending.append(text)

    def add_node(node: ChildNode) -> None:
        if pending:
            children.append("".join(pending))
            pending.clear()
        children.append(node)

    def add_pause() -> None:
        if pending or not (children and isinstance(children[-1], Pause)):
            add_node(Pause(duration=_ELLIPSIS_PAUSE_MS))

    for i, token in enumerate(tokens):
        if i > 0:
            add_text(" ")
        trailing_ellipsis = _ends_with_ellipsis(token)
        if i > 0 and not trailing_ellipsis and _ELLIPSIS_RE.match(token.lstrip(_OPENING_PUNCT)):
            add_pause()  # "well ...and": the pause goes before the ellipsis
        core = _strip_punctuation(token)
        start = token.find(core) if core else -1
        if start >= 0 and _is_emphasized(core, shouting):
            add_text(token[:start])
            add_node(Emphasis(level="strong", children=(core,)))
            add_text(token[start + len(core) :])
        else:
            add_text(token)
        if trailing_ellipsis:
            add_pause()  # "Well... I": the pause follows the ellipsis
    if pending:
        children.append("".join(pending))
    return tuple(children)


def _build_utterance(
    sentence: str,
    context_emotion: str | None,
    min_confidence: float,
    confidence_floor: float | None,
) -> Utterance:
    """Build a single Utterance from a sentence string."""
    shouting = _is_shouting(sentence.split(" "))
    children = _build_children_for_sentence(sentence, shouting)
    terminal = _terminal_punctuation(sentence)
    is_question = "?" in terminal
    is_exclamation = "!" in terminal

    emotion, confidence = _detect_sentence_emotion(sentence, context_emotion)
    if emotion is not None and confidence < min_confidence:
        emotion = None
    if emotion is not None and confidence_floor is not None:
        confidence = max(confidence, confidence_floor)

    # One wrapper carries the sentence-level prosody, so that nesting stays
    # within two levels (spec 5.2) around word-level emphasis.
    contour: str | None = None
    pitch: str | None = None
    volume: str | None = None
    if emotion == "sarcastic":
        contour = "fall-rise"  # spec 3.2: sarcasm
    elif is_question:
        contour = "rise-sharp" if is_exclamation else "rise"
    if is_exclamation and not is_question:
        pitch, volume = "+5%", "+3dB"
    if shouting:
        volume = "+6dB"
    if children and (contour or pitch or volume):
        children = (
            Prosody(children=children, pitch=pitch, pitch_contour=contour, volume=volume),
        )

    # Per spec: confidence MUST be present when emotion is set.
    return Utterance(
        children=children,
        emotion=emotion,
        confidence=confidence if emotion is not None else None,
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
        capitalization, and lexical cues.  Other values are reserved for
        future ML backends (Phase 6b).
    default_confidence:
        Optional confidence floor for emitted emotions (0.0-1.0). By default
        (``None``) the rule-based model reports its own confidence. When set,
        emitted emotions report at least this confidence; this does not
        make weaker predictions pass ``min_confidence``.
    min_confidence:
        Emotion predictions below this confidence are left out, and the
        utterance then carries neither ``emotion`` nor ``confidence``.
        Defaults to 0.5, the level below which spec 6.2 treats a confidence
        as low. Use 0.0 to keep every prediction.
    """

    def __init__(
        self,
        model: str = "rule-based",
        default_confidence: float | None = None,
        *,
        min_confidence: float = 0.5,
    ) -> None:
        if model != "rule-based":
            raise NotImplementedError(
                f"Model {model!r} is not yet supported. "
                "Only 'rule-based' is available (Phase 6a)."
            )
        for name, value in (
            ("default_confidence", default_confidence),
            ("min_confidence", min_confidence),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value!r}")
        self.model = model
        self.default_confidence = default_confidence
        self.min_confidence = min_confidence
        self._parser = IMLParser()
        self._validator = IMLValidator()

    def predict(self, text: str, context: str | None = None) -> str:
        """Predict prosody and return an IML XML string.

        Parameters
        ----------
        text:
            Plain text input.  May contain multiple sentences.
        context:
            Optional surrounding text, such as the previous turn of a
            conversation or a description like ``"the user is frustrated"``.
            The emotion its cue words point to (the same lexicon, plus the
            spec's core emotion names, e.g. ``"frustrated"`` or
            ``"user_frustrated"``) counts as one more cue word in every
            sentence: it labels a sentence without cues of its own at
            confidence 0.5, raises the confidence of a sentence that agrees,
            breaks ties, and lowers the confidence of a sentence whose cues
            point elsewhere. A context without cue words (or with tied cues)
            has no effect.

        Returns
        -------
        str
            A valid IML XML string whose plain text is the normalized input
            (see the module docstring).
        """
        normalized = _normalize_text(text or "")
        if not normalized:
            doc = IMLDocument(utterances=(Utterance(children=("",)),), version=_IML_VERSION)
            return self._parser.to_iml_string(doc)

        context_emotion = _context_emotion(context)
        utterances = tuple(
            _build_utterance(
                sentence, context_emotion, self.min_confidence, self.default_confidence
            )
            for run in _split_quotes(normalized)
            for sentence in _split_sentences(run)
        )
        # The serializer separates utterances by a space, the one the text
        # had between them, so stripping the tags gives the text back (spec 6.2).
        xml = self._parser.to_iml_string(IMLDocument(utterances=utterances, version=_IML_VERSION))
        result = self._validator.validate(xml)
        if not result.valid:  # pragma: no cover - guards against model bugs
            raise ConversionError(
                "TextToIML produced invalid IML (please report this bug): "
                + "; ".join(f"{i.rule}: {i.message}" for i in result.errors)
            )
        return xml

    def predict_document(self, text: str, context: str | None = None) -> IMLDocument:
        """Predict prosody and return a structured IMLDocument.

        Convenience method that parses the XML output of :meth:`predict`
        back into a document object.
        """
        xml = self.predict(text, context=context)
        return self._parser.parse(xml)
