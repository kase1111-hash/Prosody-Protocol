"""Word timings from speech-to-text services, as :class:`WordAlignment` lists.

The prosody pipeline (:class:`~prosody_protocol.prosody_analyzer.ProsodyAnalyzer`,
:class:`~prosody_protocol.assembler.IMLAssembler`, ``AudioToIML.convert(...,
words=...)``) takes the words of a recording with their times as
:class:`~prosody_protocol._types.WordAlignment` objects, in whole
milliseconds. These adapters build them from the output of common
speech-to-text services, so any of them can supply the words:

- :func:`from_whisper` -- openai-whisper ``transcribe(..., word_timestamps=True)``
  (``segments[].words[]``), faster-whisper and WhisperX segments, and the
  OpenAI transcription API's ``verbose_json`` with word timestamps
  (top-level ``words[]``). Times in seconds.
- :func:`from_deepgram` -- ``results.channels[].alternatives[].words[]``,
  preferring ``punctuated_word``. Times in seconds.
- :func:`from_assemblyai` -- a completed transcript's ``words[]`` (``text``).
  Times in milliseconds.
- :func:`from_google` -- Google Cloud Speech-to-Text
  ``results[].alternatives[0].words[]``, with times as ``"1.300s"``
  durations (``startTime``/``endTime`` in v1, ``startOffset``/``endOffset``
  in v2).
- :func:`from_records` -- any list of records with configurable keys and unit.
- :func:`load_word_timings` -- detects which of these shapes a JSON file (or
  parsed object) has.
- :func:`parse_word_timings` -- the same for JSON text, such as a request
  body; it never reads a file, so it is the one to use on untrusted input.

Each adapter accepts the parsed JSON (dicts and lists) or the SDK's response
object with the same field names. It strips whitespace from words (Whisper's
leading space), drops empty tokens, checks that times are finite,
non-negative, end no earlier than they start and in time order, and rounds
them to whole milliseconds. Bad input raises
:class:`~prosody_protocol.exceptions.ConversionError` naming the word.

Speaker labels from speaker diarization are kept, as strings, in
:attr:`WordAlignment.speaker <prosody_protocol._types.WordAlignment>`:
Deepgram's ``words[].speaker`` (``diarize=true``; ``0`` becomes ``"0"``),
AssemblyAI's ``words[].speaker`` (``speaker_labels``; ``"A"``), Google's
``speakerTag`` (v1) or ``speakerLabel`` (v2), WhisperX's ``speaker``
(``"SPEAKER_00"``, from the word or else its segment) and a ``speaker``
field of other records. The assembler then measures each speaker against
their own baseline.

This module needs only the standard library.
"""

from __future__ import annotations

import difflib
import json
import math
import numbers
import os
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from ._types import WordAlignment
from .exceptions import ConversionError

TimeUnit = Literal["s", "ms"]

_MS_PER_UNIT: dict[str, float] = {"s": 1000.0, "ms": 1.0}

# A median word longer than this (ms) means the times were not in the
# assumed unit -- milliseconds read as seconds; one shorter than
# _MIN_MEDIAN_WORD_MS means seconds read as milliseconds.
_MAX_MEDIAN_WORD_MS = 10_000
_MIN_MEDIAN_WORD_MS = 5
# The unit check needs a few words to be meaningful.
_UNIT_CHECK_MIN_WORDS = 3

# (word, start, end, speaker) as the source gave them, before conversion.
_RawWord = tuple[Any, Any, Any, Any]

# A token ending in one of these ends a sentence.
_SENTENCE_END = (".", "!", "?", "\u2026", "\u3002", "\uff01", "\uff1f")

# Punctuation is restored by matching the words with the tokens of the text.
# difflib finds the best matching, but its time grows with the product of
# their numbers, so beyond this product a linear-time matching is used: it
# follows the two sequences and, where they differ, looks up to
# _RESYNC_LOOKAHEAD positions ahead in either for _RESYNC_RUN equal keys in a
# row (a run, so that a common word such as "the" does not resynchronise
# them at the wrong place).
_DIFFLIB_BUDGET = 1_000_000
_RESYNC_LOOKAHEAD = 8
_RESYNC_RUN = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, key: str) -> Any:
    """``obj[key]`` for a mapping, else the attribute (SDK response objects)."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_field(obj: Any, *keys: str) -> Any:
    for key in keys:
        value = _field(obj, key)
        if value is not None:
            return value
    return None


def _items(value: Any, what: str) -> list[Any]:
    """*value* as a list, for a JSON array or any other non-string iterable."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ConversionError(f"{what} must be a list, got {type(value).__name__}")
    return list(value)


def _number(value: Any, what: str) -> float:
    """A finite number from a JSON number, numeric string (CSV rows) or a
    number type such as numpy's."""
    if isinstance(value, bool) or value is None:
        raise ConversionError(f"{what} must be a number, got {value!r}")
    if isinstance(value, numbers.Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ConversionError(f"{what} must be a number, got {value!r}") from None
    else:
        raise ConversionError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise ConversionError(f"{what} must be a finite number, got {value!r}")
    return number


def _speaker(value: Any, what: str) -> str | None:
    """A speaker label as a string (``0`` becomes ``"0"``); ``None`` for none or a blank one."""
    if value is None:
        return None
    value = getattr(value, "value", value)  # an SDK enum
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    raise ConversionError(f"{what} speaker must be a string or an integer, got {value!r}")


def _check_unit(alignments: Sequence[WordAlignment], source: str) -> None:
    """Reject timings whose median word length shows they are in the wrong unit."""
    if len(alignments) < _UNIT_CHECK_MIN_WORDS:
        return
    median = statistics.median(a.end_ms - a.start_ms for a in alignments)
    if median > _MAX_MEDIAN_WORD_MS:
        raise ConversionError(
            f"{source}: the median word lasts {median / 1000:g} s, so the times are "
            "probably in milliseconds rather than seconds"
        )
    if median < _MIN_MEDIAN_WORD_MS and max(a.end_ms for a in alignments) > 0:
        raise ConversionError(
            f"{source}: the median word lasts {median:g} ms, so the times are "
            "probably in seconds rather than milliseconds"
        )


def _to_alignments(
    raw_words: Iterable[_RawWord], source: str, *, unit: TimeUnit = "s"
) -> list[WordAlignment]:
    """Validate ``(word, start, end, speaker)`` tuples and convert them to milliseconds.

    Empty words are dropped first; the others must have finite, non-negative
    times with ``end >= start``, and start no earlier than the word before.
    """
    scale = _MS_PER_UNIT[unit]
    alignments: list[WordAlignment] = []
    previous_start = -math.inf
    for index, (word, start, end, speaker) in enumerate(raw_words):
        if not isinstance(word, str):
            raise ConversionError(f"{source}: word {index} must be a string, got {word!r}")
        text = word.strip()
        if not text:
            continue
        what = f"{source}: word {index} ({text!r})"
        start_ms = _number(start, f"{what} start time") * scale
        end_ms = _number(end, f"{what} end time") * scale
        if start_ms < 0:
            raise ConversionError(f"{what} starts at a negative time ({start}{unit})")
        if end_ms < start_ms:
            raise ConversionError(f"{what} ends ({end}{unit}) before it starts ({start}{unit})")
        if start_ms < previous_start:
            raise ConversionError(
                f"{what} starts ({start}{unit}) before the word before it; "
                "word timings must be in time order"
            )
        previous_start = start_ms
        alignments.append(
            WordAlignment(text, round(start_ms), round(end_ms), _speaker(speaker, what))
        )
    _check_unit(alignments, source)
    return alignments


def _match_key(token: str) -> str:
    return "".join(char for char in token.casefold() if char.isalnum())


def _linear_matches(a: Sequence[str], b: Sequence[str]) -> Iterator[tuple[int, int]]:
    """Pairs ``(i, j)`` of equal keys ``a[i] == b[j]``, in order, in linear time.

    The sequences are followed together; where they differ, the nearest
    place up to :data:`_RESYNC_LOOKAHEAD` positions ahead where
    :data:`_RESYNC_RUN` keys agree again (after keys missing from ``a``,
    missing from ``b``, or replaced) is where matching resumes. Keys that
    never agree are skipped one by one.
    """
    n, m = len(a), len(b)

    def agree(i: int, j: int) -> bool:
        run = min(_RESYNC_RUN, n - i, m - j)
        return run > 0 and a[i: i + run] == b[j: j + run]

    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            yield i, j
            i, j = i + 1, j + 1
            continue
        for skip in range(1, _RESYNC_LOOKAHEAD + 1):
            if agree(i, j + skip):
                j += skip
                break
            if agree(i + skip, j):
                i += skip
                break
            if agree(i + skip, j + skip):
                i, j = i + skip, j + skip
                break
        else:
            i, j = i + 1, j + 1


def _restore_punctuation(words: list[Any], text: str) -> list[Any]:
    """Replace bare words with their written form, punctuation included, from *text*.

    The OpenAI transcription API gives word timestamps without punctuation
    but the full ``text`` with it; sentence punctuation is what lets the
    assembler split utterances. Words are matched to the whitespace-separated
    tokens of *text* in order; words that do not match keep their form. The
    time is linear in the size of the input where the two differ a lot
    (see :data:`_DIFFLIB_BUDGET`), so untrusted input cannot make it slow.
    """
    tokens = text.split()
    token_keys = [_match_key(token) for token in tokens]
    word_keys = [_match_key(word) if isinstance(word, str) else "" for word in words]
    restored = list(words)
    # Usually the words are exactly the text's words: pair them up directly.
    keyed_words = [index for index, key in enumerate(word_keys) if key]
    keyed_tokens = [index for index, key in enumerate(token_keys) if key]
    if [word_keys[i] for i in keyed_words] == [token_keys[j] for j in keyed_tokens]:
        for word_index, token_index in zip(keyed_words, keyed_tokens, strict=True):
            restored[word_index] = tokens[token_index]
        return restored
    pairs: Iterable[tuple[int, int]]
    if len(word_keys) * len(token_keys) <= _DIFFLIB_BUDGET:
        matcher = difflib.SequenceMatcher(None, word_keys, token_keys, autojunk=False)
        pairs = (
            (word_index + offset, token_index + offset)
            for word_index, token_index, size in matcher.get_matching_blocks()
            for offset in range(size)
        )
    else:
        pairs = (
            (keyed_words[i], keyed_tokens[j])
            for i, j in _linear_matches(
                [word_keys[i] for i in keyed_words], [token_keys[j] for j in keyed_tokens]
            )
        )
    for word_index, token_index in pairs:
        if token_keys[token_index]:
            restored[word_index] = tokens[token_index]
    return restored


def _merge_untimed(segments: Iterable[Iterable[_RawWord]], source: str) -> list[_RawWord]:
    """Attach words that have no times to a neighbouring timed word.

    WhisperX leaves out ``start``/``end`` for words it cannot align (often
    numbers). Such a word joins the timed word before it in the same
    segment, unless that word ends a sentence; at the start of a segment or
    sentence it joins the timed word after it instead, so the sentence
    break stays where it was. No words are lost.
    """
    merged: list[list[Any]] = []
    pending: list[str] = []
    for segment in segments:
        previous: list[Any] | None = None  # the last timed word of this segment
        for word, start, end, speaker in segment:
            if start is None and end is None:
                if not isinstance(word, str) or not word.strip():
                    continue
                if previous is not None and not str(previous[0]).rstrip().endswith(
                    _SENTENCE_END
                ):
                    previous[0] = f"{str(previous[0]).rstrip()} {word.strip()}"
                else:
                    pending.append(word.strip())
                continue
            if pending and isinstance(word, str):
                word = " ".join([*pending, word.strip()])
                pending = []
            previous = [word, start, end, speaker]
            merged.append(previous)
    if pending:
        if not merged:
            raise ConversionError(f"{source}: no word has start and end times")
        # Untimed words at the very end, after a sentence end.
        merged[-1][0] = " ".join([str(merged[-1][0]).rstrip(), *pending])
    return [(word, start, end, speaker) for word, start, end, speaker in merged]


def _parse_json(text: str | bytes | bytearray, what: str) -> Any:
    """Parse JSON text; bytes may be UTF-8 (with or without a BOM), UTF-16 or UTF-32."""
    if isinstance(text, str):
        text = text.removeprefix("\ufeff")
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError, UnicodeDecodeError
        raise ConversionError(f"{what} are not valid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def from_seconds(word: str, start: float, end: float) -> WordAlignment:
    """One word with its start and end times in seconds.

    >>> from_seconds(" country.", 5.21, 5.74)
    WordAlignment(word='country.', start_ms=5210, end_ms=5740)
    """
    alignments = _to_alignments([(word, start, end, None)], "from_seconds")
    if not alignments:
        raise ConversionError("from_seconds: the word is empty")
    return alignments[0]


def from_records(
    records: Iterable[Any],
    *,
    word_key: str = "word",
    start_key: str = "start",
    end_key: str = "end",
    unit: TimeUnit = "s",
    speaker_key: str | None = "speaker",
) -> list[WordAlignment]:
    """Word timings from records with the given keys (dicts or objects).

    *unit* is the unit of the start and end times: ``"s"`` (seconds, the
    default) or ``"ms"``. Times may be numbers or numeric strings, so rows
    from :class:`csv.DictReader` work. A record's *speaker_key* field, when
    it has one, is its speaker label (a string or an integer); ``None``
    reads no speaker labels.
    """
    if unit not in _MS_PER_UNIT:
        raise ValueError(f"unit must be 's' or 'ms', got {unit!r}")
    raw_words: list[_RawWord] = []
    for index, record in enumerate(_items(records, "records")):
        values = [_field(record, key) for key in (word_key, start_key, end_key)]
        for key, value in zip((word_key, start_key, end_key), values, strict=True):
            if value is None:
                raise ConversionError(f"records: record {index} has no {key!r}")
        speaker = None if speaker_key is None else _field(record, speaker_key)
        raw_words.append((values[0], values[1], values[2], speaker))
    return _to_alignments(raw_words, "records", unit=unit)


def from_whisper(result: Any) -> list[WordAlignment]:
    """Word timings from Whisper output (times in seconds).

    Accepts:

    - openai-whisper's ``model.transcribe(..., word_timestamps=True)`` result
      (``segments[].words[]`` with ``word``, ``start``, ``end``);
    - WhisperX output of the same shape (words it could not align, which
      have no times, are attached to the neighbouring word; the ``speaker``
      that ``assign_word_speakers`` gives a word, or else its segment, is
      kept);
    - a list of segments, such as faster-whisper's
      ``list(model.transcribe(..., word_timestamps=True)[0])``;
    - the OpenAI transcription API's ``verbose_json`` response with
      ``timestamp_granularities=["word"]`` (top-level ``words[]``). Its words
      have no punctuation, so each takes its written form from ``text``.

    Raises :class:`ConversionError` if the result has no word timestamps.
    """
    if isinstance(result, Mapping) or _first_field(result, "words", "segments") is not None:
        words, segments = _field(result, "words"), _field(result, "segments")
    else:  # an iterable of segments
        words, segments = None, result
    if words is not None:
        raw_words = [
            (_field(w, "word"), _field(w, "start"), _field(w, "end"), _field(w, "speaker"))
            for w in _items(words, "whisper: words")
        ]
        text = _field(result, "text")
        if isinstance(text, str):
            restored = _restore_punctuation([word for word, _, _, _ in raw_words], text)
            raw_words = [
                (r, s, e, speaker)
                for r, (_, s, e, speaker) in zip(restored, raw_words, strict=True)
            ]
        return _to_alignments(_merge_untimed([raw_words], "whisper"), "whisper")

    if segments is None:
        raise ConversionError("whisper: the result has no 'segments' or 'words'")
    per_segment: list[list[_RawWord]] = []
    for segment in _items(segments, "whisper: segments"):
        segment_words = _field(segment, "words")
        if segment_words is None:
            raise ConversionError(
                "whisper: the segments have no word timestamps; transcribe with "
                "word_timestamps=True (or timestamp_granularities=['word'] in the API)"
            )
        segment_speaker = _field(segment, "speaker")
        per_segment.append(
            [
                (
                    _field(w, "word"),
                    _field(w, "start"),
                    _field(w, "end"),
                    segment_speaker if _field(w, "speaker") is None else _field(w, "speaker"),
                )
                for w in _items(segment_words, "whisper: segment words")
            ]
        )
    return _to_alignments(_merge_untimed(per_segment, "whisper"), "whisper")


def _deepgram_words(words: Any) -> list[WordAlignment]:
    return _to_alignments(
        (
            (
                _first_field(w, "punctuated_word", "word"),
                _field(w, "start"),
                _field(w, "end"),
                _field(w, "speaker"),
            )
            for w in _items(words, "deepgram: words")
        ),
        "deepgram",
    )


def from_deepgram(response: Any, *, channel: int = 0, alternative: int = 0) -> list[WordAlignment]:
    """Word timings from a Deepgram pre-recorded transcription response.

    Reads ``results.channels[channel].alternatives[alternative].words[]``,
    using ``punctuated_word`` (present with ``punctuate`` or
    ``smart_format``) and falling back to ``word``. Times are in seconds.
    With ``diarize=true`` each word's ``speaker`` (``0``, ``1``, ...) is kept
    as its speaker label (``"0"``, ``"1"``, ...).
    """
    channels = _field(_field(response, "results"), "channels")
    if channels is None:
        raise ConversionError("deepgram: the response has no 'results.channels'")
    try:
        alternatives = _field(_items(channels, "deepgram: channels")[channel], "alternatives")
        words = _field(_items(alternatives, "deepgram: alternatives")[alternative], "words")
    except IndexError:
        raise ConversionError(
            f"deepgram: the response has no channel {channel}, alternative {alternative}"
        ) from None
    if words is None:
        raise ConversionError("deepgram: the alternative has no 'words'")
    return _deepgram_words(words)


def _assemblyai_words(words: Any) -> list[WordAlignment]:
    return _to_alignments(
        (
            (_field(w, "text"), _field(w, "start"), _field(w, "end"), _field(w, "speaker"))
            for w in _items(words, "assemblyai: words")
        ),
        "assemblyai",
        unit="ms",
    )


def from_assemblyai(transcript: Any) -> list[WordAlignment]:
    """Word timings from a completed AssemblyAI transcript (times in milliseconds).

    Accepts the JSON of ``GET /v2/transcript/{id}`` or the SDK's
    ``Transcript`` object, and reads ``words[]`` (``text``, ``start``,
    ``end``, and with ``speaker_labels`` the ``speaker``, such as ``"A"``).
    Raises :class:`ConversionError` if the transcript is not completed yet
    or failed.
    """
    words = _field(transcript, "words")
    status = _field(transcript, "status")
    status = getattr(status, "value", status)  # the SDK's TranscriptStatus enum
    if status is not None and status != "completed":
        error = _field(transcript, "error")
        detail = f": {error}" if error else ""
        raise ConversionError(
            f"assemblyai: the transcript is not completed (status {status!r}){detail}"
        )
    if words is None:
        raise ConversionError("assemblyai: the transcript has no 'words'")
    return _assemblyai_words(words)


def _google_seconds(value: Any, what: str) -> float:
    """A Google duration: ``"1.300s"``, seconds, a timedelta or ``{seconds, nanos}``."""
    if value is None:
        return 0.0  # proto3 JSON leaves out zero durations
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, str) and value.strip().endswith("s"):
        return _number(value.strip()[:-1], what)
    if isinstance(value, Mapping):
        seconds = _number(value.get("seconds", 0), what)
        return seconds + _number(value.get("nanos", 0), what) / 1e9
    return _number(value, what)


def _google_words(words: Any) -> list[_RawWord]:
    raw_words: list[_RawWord] = []
    for index, w in enumerate(_items(words, "google: words")):
        what = f"google: word {index}"
        start = _first_field(w, "startTime", "startOffset", "start_time", "start_offset")
        end = _first_field(w, "endTime", "endOffset", "end_time", "end_offset")
        raw_words.append(
            (
                _field(w, "word"),
                _google_seconds(start, f"{what} start time"),
                _google_seconds(end, f"{what} end time"),
                _google_speaker(w),
            )
        )
    return raw_words


def _google_speaker(word: Any) -> Any:
    """A word's speaker label: ``speakerLabel`` (v2), or ``speakerTag`` (v1) unless 0 (unset)."""
    label = _first_field(word, "speakerLabel", "speaker_label")
    if label is not None and label != "":
        return label
    tag = _first_field(word, "speakerTag", "speaker_tag")
    return None if tag == 0 else tag


def _google_speaker_tagged(words: Any) -> bool:
    """Whether the words carry speaker labels (``speakerTag`` in v1, ``speakerLabel`` in v2)."""
    keys = ("speakerTag", "speaker_tag", "speakerLabel", "speaker_label")
    return any(_field(w, key) for w in words for key in keys)


def from_google(response: Any, *, channel_tag: int | None = None) -> list[WordAlignment]:
    """Word timings from a Google Cloud Speech-to-Text recognize response.

    Reads ``results[].alternatives[0].words[]``, which needs word time
    offsets (``enableWordTimeOffsets``). Times are durations such as
    ``"1.300s"`` (JSON), :class:`~datetime.timedelta` (Python client) or
    seconds.

    With ``enableSeparateRecognitionPerChannel`` each result carries a
    ``channelTag``: pass *channel_tag* to choose the channel. With speaker
    diarization the last result repeats every word of the earlier ones with
    speaker labels; only that result is then used, and each word's
    ``speakerTag`` (v1) or ``speakerLabel`` (v2) becomes its speaker label.

    Raises :class:`ConversionError` if the results have transcripts but no
    word time offsets, or come from several channels and *channel_tag* is
    not given. A response without speech (no ``results``) gives ``[]``.
    """
    results = _field(response, "results")
    if results is None:
        # proto3 JSON leaves out an empty list: a response for silence has
        # only its billing fields.
        if _first_field(response, "totalBilledTime", "requestId", "metadata") is not None:
            return []
        raise ConversionError("google: the response has no 'results'")
    results = _items(results, "google: results")
    tags = [
        int(_number(_first_field(result, "channelTag", "channel_tag") or 0, "google: channelTag"))
        for result in results
    ]
    channels = sorted(set(tags))
    if channel_tag is not None:
        results = [result for result, tag in zip(results, tags, strict=True) if tag == channel_tag]
        if not results and channels:
            raise ConversionError(
                f"google: the response has no results for channel_tag {channel_tag}; "
                f"its channels are {channels}"
            )
    elif len(channels) > 1:
        raise ConversionError(
            f"google: the response has results for channels {channels}; "
            "choose one with channel_tag="
        )

    per_result: list[list[_RawWord]] = []
    last_tagged = False
    for result in results:
        alternatives = _field(result, "alternatives")
        best = _items(alternatives, "google: alternatives")[0] if alternatives else None
        words = _field(best, "words")
        if not words:
            transcript = _field(best, "transcript")
            if isinstance(transcript, str) and transcript.strip():
                raise ConversionError(
                    "google: the response has transcripts but no word time offsets; "
                    "request enableWordTimeOffsets=true (v1) or "
                    "features.enableWordTimeOffsets (v2)"
                )
            continue
        per_result.append(_google_words(words))
        last_tagged = _google_speaker_tagged(words)
    earlier = [word for words in per_result[:-1] for word in words]
    if (
        earlier
        and last_tagged
        and len(per_result[-1]) >= len(earlier)
        and per_result[-1][0][1] <= min(w[1] for w in earlier)
    ):
        per_result = per_result[-1:]  # the speaker-labelled summary of all the words
    return _to_alignments((word for words in per_result for word in words), "google")


def _load_data(data: Any) -> list[WordAlignment]:
    if isinstance(data, Mapping):
        return _load_mapping(data)
    if isinstance(data, (list, tuple)):
        return _load_list(data)
    raise ConversionError(
        f"word timings must be a JSON object or array, got {type(data).__name__}"
    )


def load_word_timings(
    source: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
) -> list[WordAlignment]:
    """Word timings from a JSON file, or parsed data, of any supported shape.

    *source* is the path of a JSON file (a :class:`str` or path object; UTF-8
    with or without a BOM, UTF-16 or UTF-32), or already-parsed data such as
    a dict, a list or an SDK response object's fields. To parse JSON *text*,
    for example from a request, use :func:`parse_word_timings`, which never
    reads files. The shape is detected:

    - ``{"results": {"channels": ...}}``: Deepgram;
    - ``{"results": [...]}``: Google Cloud Speech-to-Text;
    - an object with ``status``: an AssemblyAI transcript;
    - ``{"segments": ...}`` or ``{"words": [{"word", "start", "end"}]}``:
      Whisper or the OpenAI transcription API;
    - any other ``{"words": [...]}``: the list of words, as below;
    - a list of segments with ``words`` (faster-whisper, WhisperX), or a list
      of words: ``text`` (AssemblyAI, ms), ``punctuated_word`` (Deepgram),
      ``startTime``/``startOffset`` (Google), ``word`` with
      ``start_ms``/``end_ms`` (ms, as :func:`dataclasses.asdict` writes a
      :class:`WordAlignment`) or ``start``/``end`` (seconds), each with an
      optional ``speaker``, or :class:`WordAlignment` objects.

    Use :func:`from_records` for other keys or units, and the ``from_*``
    functions for their options (such as a Deepgram channel).

    Raises :class:`ConversionError` for unrecognised or invalid data and
    :class:`OSError` if the file cannot be read.
    """
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        return _load_data(_parse_json(path.read_bytes(), f"the word timings in {path}"))
    return _load_data(source)


def parse_word_timings(
    data: str | bytes | bytearray | Mapping[str, Any] | Sequence[Any],
) -> list[WordAlignment]:
    """Word timings from JSON text, or parsed data, of any supported shape.

    Like :func:`load_word_timings`, but a :class:`str` or :class:`bytes` is
    JSON text, never a file name: this function does not touch the file
    system, so it is safe on untrusted input such as a form field (limit
    its size first; the time taken grows linearly with it). The shapes are
    those of :func:`load_word_timings`.

    Raises :class:`ConversionError` if the text is not JSON (including an
    empty string) or the data is unrecognised or invalid.
    """
    if isinstance(data, (str, bytes, bytearray)):
        return _load_data(_parse_json(data, "word timings"))
    return _load_data(data)


def _load_mapping(data: Mapping[str, Any]) -> list[WordAlignment]:
    results = data.get("results")
    words = data.get("words")
    if isinstance(results, Mapping):
        # Deepgram; other services with a results object are not supported.
        if "channels" in results:
            return from_deepgram(data)
    elif isinstance(results, list) or "totalBilledTime" in data:
        return from_google(data)
    elif "status" in data:
        return from_assemblyai(data)
    elif isinstance(words, list):
        first = words[0] if words else None
        if isinstance(first, Mapping) and "word" in first and "start" in first:
            return from_whisper(data)  # the OpenAI API, whose text restores punctuation
        return _load_list(words)
    elif "segments" in data:
        return from_whisper(data)
    raise ConversionError(
        "unrecognised word timing format: expected Whisper, OpenAI, Deepgram, AssemblyAI "
        f"or Google output, got an object with keys {sorted(data)[:8]}"
    )


def _load_list(data: Sequence[Any]) -> list[WordAlignment]:
    if not data:
        return []
    first = data[0]
    if isinstance(first, WordAlignment):
        if not all(isinstance(item, WordAlignment) for item in data):
            raise ConversionError("word timings mix WordAlignment objects with other items")
        return _to_alignments(
            ((a.word, a.start_ms, a.end_ms, a.speaker) for a in data), "word timings", unit="ms"
        )
    if not isinstance(first, Mapping):
        raise ConversionError(
            f"word timings must be a list of objects, got a list of {type(first).__name__}"
        )
    if "words" in first:
        return from_whisper(list(data))
    if "text" in first:
        return _assemblyai_words(data)
    if "punctuated_word" in first:
        return _deepgram_words(data)
    if any(key in first for key in ("startTime", "startOffset", "endTime", "endOffset")):
        return _to_alignments(_google_words(data), "google")
    if "start_ms" in first or "end_ms" in first:
        return from_records(data, start_key="start_ms", end_key="end_ms", unit="ms")
    if "word" in first:
        return from_records(data)
    raise ConversionError(
        f"unrecognised word timing records with keys {sorted(first)[:8]}; "
        "use from_records() with the right keys"
    )
