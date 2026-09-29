"""Command-line interface: ``prosody-protocol``.

Subcommands::

    validate FILE...      check IML documents against the spec (exit 1 if invalid)
    to-text FILE          plain text of an IML document
    to-ssml FILE          SSML 1.1 for text-to-speech engines
    to-prompt FILE        annotated transcript for a large language model
    from-text TEXT        predict IML markup for plain text
    from-audio AUDIO      IML from a recording (with word timings or a transcript)
    synthesize FILE       speak an IML document to a WAV file
    benchmark DIR         score AudioToIML against a labeled dataset
    serve                 run the REST API
    doctor                list the optional capabilities that are installed

A FILE, TEXT or AUDIO argument of ``-`` is read from standard input (as is
``-`` for ``--words``, ``--transcript-file``, ``--profile`` or
``--calibration``; only one input can come from stdin). Results go to
standard output, or to the file given with ``-o``; warnings go to standard
error as they happen.

Exit status: 0 on success; 1 when the input was processed but rejected
(invalid IML, a failed conversion, a benchmark regression); 2 for usage
errors, files that cannot be read or written, and missing optional
dependencies (the message names the extra to install).

Only the standard library is imported here. Each subcommand imports what it
needs when it runs, so ``validate`` and the other text commands work on the
core install (lxml only).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import platform
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ._install import install_hint
from ._types import normalize_language_tag
from ._version import __version__
from .exceptions import ProsodyProtocolError

if TYPE_CHECKING:
    from .audio_to_iml import ConversionResult
    from .models import IMLDocument
    from .profiles import ProsodyProfile
    from .validator import ValidationIssue, ValidationResult

__all__ = ["main", "serve_main"]

PROG = "prosody-protocol"

# Exit statuses.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

STDIN = "-"

# A BCP 47 language tag, as validator rule V29 checks it.

_MAX_PORT = 65_535


class CLIError(Exception):
    """An expected failure: printed as ``error: <message>``, exit status *code*."""

    def __init__(self, message: str, code: int = EXIT_FAILED) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ---------------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------------


def _stdin_bytes() -> bytes:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        data: bytes = buffer.read()
        return data
    return sys.stdin.read().encode("utf-8")


def _read_bytes(path: str) -> bytes:
    """The contents of *path*, or of stdin for ``-``."""
    if path == STDIN:
        return _stdin_bytes()
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise CLIError(f"cannot read {path}: {exc.strerror or exc}", EXIT_USAGE) from None


def _display(path: str) -> str:
    return "<stdin>" if path == STDIN else path


def _read_text(path: str) -> str:
    """UTF-8 text (a byte order mark is allowed) from *path* or stdin."""
    data = _read_bytes(path)
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CLIError(
            f"{_display(path)} is not UTF-8 text (byte 0x{data[exc.start]:02x} at offset "
            f"{exc.start})"
        ) from None


def _read_iml(path: str) -> str:
    try:
        return _read_text(path)
    except CLIError as exc:
        if exc.code == EXIT_USAGE:
            raise
        raise CLIError(f"{exc.message}; IML documents must be UTF-8 (spec 2.2)") from None


def _write_text(text: str, output: str | None) -> None:
    """Write *text* and a newline to *output*, or to stdout."""
    if not text.endswith("\n"):
        text += "\n"
    if output is None or output == STDIN:
        sys.stdout.write(text)
        return
    try:
        Path(output).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CLIError(f"cannot write {output}: {exc.strerror or exc}", EXIT_USAGE) from None


def _write_bytes(data: bytes, output: str) -> None:
    if output == STDIN:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    try:
        Path(output).write_bytes(data)
    except OSError as exc:
        raise CLIError(f"cannot write {output}: {exc.strerror or exc}", EXIT_USAGE) from None


def _require_file(path: str) -> None:
    """Fail with exit status 2 unless *path* is a file that can be opened."""
    try:
        with open(path, "rb"):
            pass
    except OSError as exc:
        raise CLIError(f"cannot read {path}: {exc.strerror or exc}", EXIT_USAGE) from None


def _check_single_stdin(*paths: str | None) -> None:
    if sum(path == STDIN for path in paths) > 1:
        raise CLIError("only one input can be read from stdin ('-')", EXIT_USAGE)


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _require(module: str, command: str, extra: str) -> ModuleType:
    """Import *module* for *command*, or fail with a hint naming the *extra*."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        # The SDK re-raises a missing numpy or parselmouth with an install hint.
        cause = exc.__cause__
        name = exc.name or (cause.name if isinstance(cause, ImportError) else None)
        missing = f" ({name} is not installed)" if name and name != module else ""
        raise CLIError(
            f"{command} needs the '{extra}' extra{missing}: "
            f"{install_hint(extra)}",
            EXIT_USAGE,
        ) from None


@contextlib.contextmanager
def _warnings_to_stderr() -> Iterator[None]:
    """Print each distinct warning raised inside as ``warning: <message>``, at once.

    The interpreter's warning filters still decide which warnings are shown,
    so ``-W`` and ``PYTHONWARNINGS`` apply and other packages'
    DeprecationWarnings stay hidden as usual.
    """
    seen: set[str] = set()

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        if text not in seen:
            seen.add(text)
            print(f"warning: {text}", file=sys.stderr, flush=True)

    with warnings.catch_warnings():
        warnings.showwarning = show
        yield


# ---------------------------------------------------------------------------
# Argument types
# ---------------------------------------------------------------------------


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value >= 0.0):
        raise argparse.ArgumentTypeError(f"{text} is not a non-negative number")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if not 0 <= value <= _MAX_PORT:
        raise argparse.ArgumentTypeError(f"{text} is not a port number (0-{_MAX_PORT})")
    return value


def _language(text: str) -> str:
    # POSIX locale style ("en_US", as in $LANG) names the same language.
    try:
        return normalize_language_tag(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{text!r} is not a BCP 47 language tag (e.g. "en-US")'
        ) from None


def _threshold(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"{text!r} is not METRIC=VALUE")
    try:
        return name.strip(), float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _issue_location(name: str, issue: ValidationIssue) -> str:
    if issue.line is None:
        return name
    if issue.column is None:
        return f"{name}:{issue.line}"
    return f"{name}:{issue.line}:{issue.column}"


def _validate_one(path: str) -> ValidationResult:
    from .parser import _utf8_error
    from .validator import IMLValidator, ValidationIssue, ValidationResult

    validator = IMLValidator()
    if path != STDIN:
        try:
            return validator.validate_file(path)
        except OSError as exc:
            raise CLIError(f"cannot read {path}: {exc.strerror or exc}", EXIT_USAGE) from None
    data = _stdin_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        message, line, column = _utf8_error(data, exc)
        issue = ValidationIssue("error", "V30", message, line, column)
        return ValidationResult(valid=False, issues=[issue])
    return validator.validate(text)


def _cmd_validate(args: argparse.Namespace) -> int:
    _check_single_stdin(*args.files)
    reports: list[dict[str, Any]] = []
    status = EXIT_OK
    for path in args.files:
        name = _display(path)
        try:
            result = _validate_one(path)
        except CLIError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            status = EXIT_USAGE
            continue
        errors, warns = len(result.errors), len(result.warnings)
        passed = result.valid and not (args.strict and warns)
        if not passed and status == EXIT_OK:
            status = EXIT_FAILED
        if args.json:
            reports.append({
                "file": name,
                "valid": result.valid,
                "errors": errors,
                "warnings": warns,
                "issues": [
                    {
                        "severity": i.severity,
                        "rule": i.rule,
                        "message": i.message,
                        "line": i.line,
                        "column": i.column,
                    }
                    for i in result.issues
                ],
            })
            continue
        for issue in result.issues:
            print(f"{_issue_location(name, issue)}: {issue.severity} {issue.rule} {issue.message}")
        counts = [f"{n} {word}{'s' if n != 1 else ''}" for n, word in
                  ((errors, "error"), (warns, "warning")) if n]
        detail = f" ({', '.join(counts)})" if counts else ""
        if result.valid and not passed:
            print(f"{name}: fails --strict{detail}")
        else:
            print(f"{name}: {'valid' if result.valid else 'invalid'}{detail}")
    if args.json:
        _write_text(_json(reports), None)
    return status


# ---------------------------------------------------------------------------
# to-text, to-ssml, to-prompt, from-text
# ---------------------------------------------------------------------------


def _parse(path: str) -> IMLDocument:
    from .parser import IMLParser

    return IMLParser().parse(_read_iml(path))


def _cmd_to_text(args: argparse.Namespace) -> int:
    from .parser import IMLParser

    _write_text(IMLParser().to_plain_text(_parse(args.file)), args.output)
    return EXIT_OK


def _cmd_to_ssml(args: argparse.Namespace) -> int:
    from .iml_to_ssml import IMLToSSML

    converter = IMLToSSML(vendor=args.vendor, strict=not args.lenient)
    _write_text(converter.convert(_read_iml(args.file)), args.output)
    return EXIT_OK


def _cmd_to_prompt(args: argparse.Namespace) -> int:
    from .llm import build_messages, to_llm_context

    if args.instruction is not None and not args.messages:
        raise CLIError("--instruction needs --messages", EXIT_USAGE)
    doc = _parse(args.file)
    options = {"min_confidence": args.min_confidence, "include_numbers": args.numbers}
    if args.messages:
        text = _json(build_messages(doc, args.instruction, **options))
    else:
        text = to_llm_context(doc, **options)
    _write_text(text, args.output)
    return EXIT_OK


def _cmd_from_text(args: argparse.Namespace) -> int:
    from .text_to_iml import TextToIML

    text = _read_text(STDIN) if args.text == STDIN else args.text
    _write_text(TextToIML().predict(text, context=args.context), args.output)
    return EXIT_OK


# ---------------------------------------------------------------------------
# from-audio
# ---------------------------------------------------------------------------


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not a JSON number")


def _load_profile(path: str) -> ProsodyProfile:
    from .exceptions import ProfileError
    from .profiles import ProfileLoader

    text = _read_text(path)
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError, or nested too deeply
        raise ProfileError(f"invalid JSON in profile {_display(path)}: {exc}") from None
    return ProfileLoader().load_json(data)


def _profile_notes(result: ConversionResult, user_id: str, min_confidence: float) -> list[str]:
    """What the prosody profile did, for stderr (spec 7.2: report profile usage)."""
    if not result.profile_matches:
        return [f"prosody profile {user_id!r} matched no utterance"]
    notes = []
    for match in result.profile_matches:
        pattern = ", ".join(f"{k}={v}" for k, v in match.pattern.items())
        if match.applied:
            notes.append(
                f"prosody profile {user_id!r} set utterance {match.utterance + 1} to "
                f"{match.emotion!r} (confidence {match.confidence:.2f}; matched {pattern})"
            )
        else:
            notes.append(
                f"prosody profile {user_id!r} matched utterance {match.utterance + 1} "
                f"({pattern}) as {match.emotion!r}, but its confidence {match.confidence:.2f} "
                f"is below {min_confidence:g}, so no emotion is reported"
            )
    return notes


def _require_whisper(command: str, alternatives: str) -> None:
    """Fail with the install hint unless openai-whisper is installed."""
    if not _installed("whisper"):
        raise CLIError(
            f"{command} --stt whisper needs the 'whisper' extra (openai-whisper is not "
            f"installed): {install_hint('whisper')}{alternatives}",
            EXIT_USAGE,
        )


def _cmd_from_audio(args: argparse.Namespace) -> int:
    calibrations: list[str] = args.calibration or []
    _check_single_stdin(
        args.audio, args.words, args.transcript_file, args.profile, *calibrations
    )
    audio_to_iml = _require("prosody_protocol.audio_to_iml", "from-audio", "audio")
    from .alignment import load_word_timings, parse_word_timings

    for path in (args.audio, *calibrations):
        if path is not None and path != STDIN:
            _require_file(path)
    words = None
    if args.words is not None:
        if args.words == STDIN:
            words = parse_word_timings(_stdin_bytes())
        else:
            try:
                words = load_word_timings(args.words)
            except OSError as exc:
                raise CLIError(
                    f"cannot read {args.words}: {exc.strerror or exc}", EXIT_USAGE
                ) from None
    transcript = args.transcript
    if args.transcript_file is not None:
        transcript = _read_text(args.transcript_file)
    if args.stt == "whisper" and words is None and transcript is None:
        _require_whisper("from-audio", ", or give the words with --words or --transcript")
    profile = None if args.profile is None else _load_profile(args.profile)

    with contextlib.ExitStack() as stack:
        audio = stack.enter_context(_audio_path(args.audio))
        calibration_paths = [stack.enter_context(_audio_path(c)) for c in calibrations]
        calibration: str | list[str] | None = (
            calibration_paths[0] if len(calibration_paths) == 1 else calibration_paths or None
        )
        converter = audio_to_iml.AudioToIML(
            stt_model=args.whisper_model,
            include_extended=args.extended,
            language=args.language,
            min_emotion_confidence=args.min_confidence,
            calibration_audio=calibration,
            stt=args.stt,
            profile=profile,
        )
        stdin_copy = audio if args.audio == STDIN else next(
            (copy for copy, c in zip(calibration_paths, calibrations, strict=True) if c == STDIN),
            None,
        )
        try:
            result = converter.convert_detailed(audio, words=words, transcript=transcript)
        except ProsodyProtocolError as exc:
            if stdin_copy is None:
                raise
            # Name stdin, not the temporary file that holds it.
            message = str(exc).replace(str(Path(stdin_copy).resolve()), "<stdin>")
            raise CLIError(message.replace(stdin_copy, "<stdin>")) from None

    for message in result.warnings:
        print(f"warning: {message}", file=sys.stderr)
    if args.json:
        from .parser import IMLParser

        text = _json({
            "iml": result.iml,
            "plain_text": IMLParser().to_plain_text(result.document),
            "transcript_source": result.transcript_source,
            "warnings": list(result.warnings),
            "profile_matches": [
                {
                    "utterance": m.utterance,
                    "observed": m.observed,
                    "pattern": m.pattern,
                    "emotion": m.emotion,
                    "confidence": m.confidence,
                    "applied": m.applied,
                }
                for m in result.profile_matches
            ],
        })
    else:
        if profile is not None:
            for note in _profile_notes(result, profile.user_id, args.min_confidence):
                print(f"note: {note}", file=sys.stderr)
        if args.prompt:
            from .llm import to_llm_context

            # Emotions the IML carries are at or above --min-confidence: name them.
            text = to_llm_context(result.document, min_confidence=args.min_confidence)
        else:
            text = result.iml
    _write_text(text, args.output)
    return EXIT_OK


@contextlib.contextmanager
def _audio_path(path: str) -> Iterator[str]:
    """*path*, or for ``-`` a temporary file holding stdin."""
    if path != STDIN:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="prosody-protocol-") as tmp:
        audio = Path(tmp) / "stdin-audio"
        audio.write_bytes(_stdin_bytes())
        yield str(audio)


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


def _cmd_synthesize(args: argparse.Namespace) -> int:
    iml_to_audio = _require("prosody_protocol.iml_to_audio", "synthesize", "audio")
    iml = _read_iml(args.file)
    synth = iml_to_audio.IMLToAudio(voice=args.voice, engine=args.engine, strict=not args.lenient)
    wav = synth.synthesize(iml)
    _write_bytes(wav, args.output)
    if args.output != STDIN:
        kind = "speech (espeak-ng)" if synth.backend == "espeak" else "tone preview, not speech"
        print(f"wrote {args.output}: {kind}", file=sys.stderr)
    return EXIT_OK


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


class _LogToStderr(logging.Handler):
    """Benchmark log records as one-line warnings (never a traceback)."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            message += f": {record.exc_info[1]}"
        print(f"warning: {message}", file=sys.stderr)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# How BenchmarkReport.word_sources names what the converter was given.
_WORD_SOURCE_NAMES = {
    "timings": "word timings",
    "transcript": "transcripts",
    "stt": "none (speech recognition)",
}


def _word_sources(sources: dict[str, int]) -> str:
    """``"transcripts 3"``: how many entries got their words from each source."""
    counts = [f"{_WORD_SOURCE_NAMES.get(k, k)} {n}" for k, n in sources.items() if n]
    return ", ".join(counts) or "n/a"


def _cmd_benchmark(args: argparse.Namespace) -> int:
    benchmarks = _require("prosody_protocol.benchmarks", "benchmark", "audio")
    audio_to_iml = _require("prosody_protocol.audio_to_iml", "benchmark", "audio")
    from .datasets import DatasetLoader

    thresholds = dict(args.threshold)
    # Unknown metrics or bad values are rejected before the (slow) run.
    empty = benchmarks.BenchmarkReport(0.0, {}, None, None, None, 0.0, 0, 0, 0.0)
    try:
        empty.check_regression(thresholds=thresholds)
    except ValueError as exc:
        raise CLIError(f"--threshold: {exc}", EXIT_USAGE) from None

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        reason = "not a directory" if dataset_dir.exists() else "no such directory"
        raise CLIError(f"cannot read {args.dataset_dir}: {reason}", EXIT_USAGE)
    if args.save is not None and not Path(args.save).resolve().parent.is_dir():
        # Checked before the (slow) run, which would otherwise be lost.
        raise CLIError(
            f"cannot write {args.save}: no such directory {Path(args.save).parent}", EXIT_USAGE
        )
    if args.stt == "whisper":
        _require_whisper("benchmark", "")

    baseline = None
    if args.baseline is not None:
        try:
            baseline = benchmarks.BenchmarkReport.load(args.baseline)
        except OSError as exc:
            raise CLIError(
                f"cannot read {args.baseline}: {exc.strerror or exc}", EXIT_USAGE
            ) from None
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            raise CLIError(
                f"{args.baseline} is not a benchmark report ({type(exc).__name__}: {exc})",
                EXIT_USAGE,
            ) from None

    dataset = DatasetLoader().load(args.dataset_dir)
    calibrations: list[str] = args.calibration or []
    for path in calibrations:
        _require_file(path)
    converter = audio_to_iml.AudioToIML(
        language=args.language,
        stt=args.stt,
        calibration_audio=calibrations or None,
    )
    logger = logging.getLogger("prosody_protocol.benchmarks")
    handler = _LogToStderr(logging.WARNING)
    logger.addHandler(handler)
    propagate, logger.propagate = logger.propagate, False
    try:
        benchmark = benchmarks.Benchmark(
            dataset,
            converter,
            words_from=args.words_from,
            abstention_label=args.abstention_label,
        )
        report = benchmark.run(max_samples=args.max_samples)
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate

    data = report.to_dict()
    lines = [
        f"Benchmark of {dataset.name}: {report.num_entries} entries, "
        f"{report.num_failures} failed conversions, {data['duration_seconds']:.1f} s",
        f"  words from        {_word_sources(report.word_sources)}",
    ]
    for metric in (
        "emotion_accuracy", "emotion_coverage", "emotion_f1_macro", "confidence_ece",
        "pitch_accuracy", "pitch_coverage", "pause_f1", "validity_rate", "failure_rate",
    ):
        lines.append(f"  {metric:<17} {_format_metric(data[metric])}")
    if report.emotion_f1:
        per_class = ", ".join(f"{k} {v:.2f}" for k, v in sorted(report.emotion_f1.items()))
        lines.append(f"  emotion_f1        {per_class}")
    if report.abstention_label is not None:
        lines.append(f"  (an output without an emotion counts as {report.abstention_label!r})")
    if report.num_unaligned:
        # The benchmark has warned why; the summary says how many.
        lines.append(
            f"  ({report.num_unaligned} outputs were only [speech] placeholders, without "
            "words: pause_f1 and the pitch metrics leave them out)"
        )
    if args.save is not None:
        try:
            report.save(args.save)
        except OSError as exc:
            raise CLIError(f"cannot write {args.save}: {exc.strerror or exc}", EXIT_USAGE) from None
        lines.append(f"Saved the report to {args.save}")

    failures = report.check_regression(baseline, thresholds, tolerance=args.tolerance)
    if failures:
        lines.append("FAILED:")
        lines.extend(f"  {failure}" for failure in failures)
    else:
        lines.append("Passed" + (f" (baseline {args.baseline})" if baseline else ""))
    _write_text("\n".join(lines), None)
    return EXIT_FAILED if failures else EXIT_OK


# ---------------------------------------------------------------------------
# serve, doctor
# ---------------------------------------------------------------------------


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _missing_api_modules() -> list[str]:
    missing = [name for name in ("fastapi", "uvicorn") if not _installed(name)]
    if not (_installed("python_multipart") or _installed("multipart")):
        missing.append("python-multipart")
    return missing


def _cmd_serve(args: argparse.Namespace) -> int:
    missing = _missing_api_modules()
    if missing:
        raise CLIError(
            f"serve needs the 'api' extra ({', '.join(missing)} not installed): "
            f"{install_hint('api')}",
            EXIT_USAGE,
        )
    server = _require("prosody_protocol.server", "serve", "api")
    try:
        server.run(host=args.host, port=args.port)
    except ValueError as exc:  # invalid PP_* settings
        raise CLIError(str(exc), EXIT_USAGE) from None
    except SystemExit as exc:
        if exc.code in (None, 0):
            return EXIT_OK
        # uvicorn has logged why (e.g. the port is in use).
        raise CLIError("the server could not start (see the log above)") from None
    return EXIT_OK


def _version(distribution: str) -> str:
    try:
        return f" {importlib.metadata.version(distribution)}"
    except importlib.metadata.PackageNotFoundError:
        return ""


def _capabilities() -> list[tuple[bool, str, str, str]]:
    """(available, name with versions, what it enables, how to get it)."""
    audio = _installed("numpy") and _installed("parselmouth")
    espeak = shutil.which("espeak-ng")
    ffmpeg = shutil.which("ffmpeg")
    api_missing = _missing_api_modules()
    return [
        (
            _installed("lxml"),
            f"core (lxml{_version('lxml')})",
            "validate, to-text, to-ssml, to-prompt, from-text",
            install_hint(),
        ),
        (
            audio,
            f"audio analysis (numpy{_version('numpy')}, "
            f"praat-parselmouth{_version('praat-parselmouth')})",
            "from-audio, benchmark, synthesize",
            install_hint("audio"),
        ),
        (
            _installed("whisper"),
            f"speech recognition (openai-whisper{_version('openai-whisper')})",
            "from-audio transcribes audio given without --words or --transcript "
            "(otherwise each stretch of speech is a [speech] placeholder)",
            f"{install_hint('whisper')} (pulls in PyTorch)",
        ),
        (
            espeak is not None,
            f"speech synthesis (espeak-ng{f' at {espeak}' if espeak else ''})",
            "synthesize speaks IML (otherwise it renders a tone preview, not speech)",
            "install espeak-ng with your package manager, e.g. apt install espeak-ng",
        ),
        (
            ffmpeg is not None,
            f"audio decoding (ffmpeg{f' at {ffmpeg}' if ffmpeg else ''})",
            "from-audio reads OGG/Opus, WebM, M4A and other formats besides WAV, AIFF, "
            "FLAC and MP3",
            "install ffmpeg with your package manager, e.g. apt install ffmpeg",
        ),
        (
            not api_missing,
            f"REST API (fastapi{_version('fastapi')}, uvicorn{_version('uvicorn')}, "
            f"python-multipart{_version('python-multipart')})",
            "serve",
            install_hint("api"),
        ),
        (
            _installed("sklearn"),
            f"training baselines (scikit-learn{_version('scikit-learn')})",
            "the training/ scripts of a source checkout",
            install_hint("ml"),
        ),
    ]


def _cmd_doctor(args: argparse.Namespace) -> int:
    lines = [f"{PROG} {__version__} (Python {platform.python_version()})"]
    for available, name, enables, install in _capabilities():
        lines.append(f"[{'ok' if available else 'missing':<7}] {name}")
        lines.append(f"          enables: {enables}")
        if not available:
            lines.append(f"          install: {install}")
    _write_text("\n".join(lines), None)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--output", metavar="OUT", help="write the result to OUT instead of stdout"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Intent Markup Language (IML): prosody-preserving transcripts. "
            "Use '-' to read an input from stdin."
        ),
        epilog=(
            "Exit status: 0 success, 1 invalid input or failed conversion, "
            "2 usage error, unreadable file or missing optional dependency."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def command(
        name: str, handler: Callable[[argparse.Namespace], int], help_text: str
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    sub = command("validate", _cmd_validate, "Check IML documents against the specification.")
    sub.add_argument("files", nargs="+", metavar="FILE", help="IML file ('-' for stdin)")
    sub.add_argument("--json", action="store_true", help="print the results as JSON")
    sub.add_argument(
        "--strict", action="store_true", help="fail on warnings (SHOULD rules) too"
    )

    sub = command("to-text", _cmd_to_text, "Print the plain text of an IML document.")
    sub.add_argument("file", metavar="FILE", help="IML file ('-' for stdin)")
    _add_output(sub)

    sub = command("to-ssml", _cmd_to_ssml, "Convert IML to SSML 1.1 for speech synthesizers.")
    sub.add_argument("file", metavar="FILE", help="IML file ('-' for stdin)")
    sub.add_argument(
        "--vendor", choices=["espeak-ng"], help="adapt the SSML to this synthesizer"
    )
    sub.add_argument(
        "--lenient",
        action="store_true",
        help="convert documents with validation errors, ignoring invalid values (spec 6.2)",
    )
    _add_output(sub)

    sub = command(
        "to-prompt", _cmd_to_prompt, "Describe IML as an annotated transcript for an LLM."
    )
    sub.add_argument("file", metavar="FILE", help="IML file ('-' for stdin)")
    sub.add_argument(
        "--min-confidence",
        type=_unit_float,
        default=0.5,
        metavar="F",
        help="name emotions only at this confidence or above (default 0.5)",
    )
    sub.add_argument(
        "--numbers", action="store_true", help="include measured values (pitch %%, dB, ...)"
    )
    sub.add_argument(
        "--messages",
        action="store_true",
        help="print chat messages (system prompt + transcript) as JSON",
    )
    sub.add_argument(
        "--instruction",
        metavar="TEXT",
        help="with --messages: a task for the model, after the transcript",
    )
    _add_output(sub)

    sub = command("from-text", _cmd_from_text, "Predict IML markup for plain text.")
    sub.add_argument("text", metavar="TEXT", help="the text ('-' for stdin)")
    sub.add_argument(
        "--context", metavar="TEXT", help='surrounding text, e.g. "the user is frustrated"'
    )
    _add_output(sub)

    sub = command(
        "from-audio",
        _cmd_from_audio,
        "Convert a recording to IML. Give the words with --words (from any speech "
        "recognizer) or --transcript; otherwise Whisper transcribes the audio if installed.",
    )
    sub.add_argument("audio", metavar="AUDIO", help="audio file ('-' for stdin)")
    source = sub.add_mutually_exclusive_group()
    source.add_argument(
        "--words",
        metavar="FILE",
        help="word timings JSON from Whisper, the OpenAI API, Deepgram, AssemblyAI, "
        "Google or a list of {word, start_ms, end_ms} records",
    )
    source.add_argument("--transcript", metavar="TEXT", help="the spoken text, without timings")
    source.add_argument(
        "--transcript-file", metavar="FILE", help="read the spoken text from FILE"
    )
    sub.add_argument(
        "--language", type=_language, metavar="TAG", help='BCP 47 language tag, e.g. "en-US"'
    )
    sub.add_argument(
        "--profile", metavar="FILE", help="the speaker's prosody profile (JSON, spec Section 7)"
    )
    sub.add_argument(
        "--calibration",
        metavar="AUDIO",
        action="append",
        help="a recording of the same speaker talking as usual (such as an earlier turn), "
        "as the baseline; repeat for several ('-' for stdin)",
    )
    sub.add_argument(
        "--stt",
        choices=["auto", "whisper", "none"],
        default="auto",
        help="speech recognition when neither --words nor a transcript is given "
        "(default auto: Whisper if installed, else [speech] placeholders)",
    )
    sub.add_argument(
        "--whisper-model", default="base", metavar="NAME", help="Whisper model (default base)"
    )
    sub.add_argument(
        "--extended", action="store_true", help="add measured values (f0_mean, jitter, ...)"
    )
    sub.add_argument(
        "--min-confidence",
        type=_unit_float,
        default=0.5,
        metavar="F",
        help="leave out emotions below this confidence (default 0.5)",
    )
    output = sub.add_mutually_exclusive_group()
    output.add_argument(
        "--json",
        action="store_true",
        help="print {iml, plain_text, transcript_source, warnings, profile_matches}",
    )
    output.add_argument(
        "--prompt", action="store_true", help="print the annotated transcript for an LLM"
    )
    _add_output(sub)

    sub = command("synthesize", _cmd_synthesize, "Speak an IML document to a WAV file.")
    sub.add_argument("file", metavar="FILE", help="IML file ('-' for stdin)")
    sub.add_argument(
        "-o", "--output", required=True, metavar="OUT.wav", help="WAV file to write ('-': stdout)"
    )
    sub.add_argument(
        "--engine",
        choices=["auto", "espeak", "tones"],
        default="auto",
        help="espeak: speech from espeak-ng; tones: a prosody preview, not speech; "
        "auto (default): espeak-ng if installed",
    )
    sub.add_argument(
        "--voice", metavar="V", help='e.g. "en-US", "en_US-female-medium" or "en-us+f3"'
    )
    sub.add_argument(
        "--lenient",
        action="store_true",
        help="speak documents with validation errors, ignoring invalid values (spec 6.2)",
    )

    sub = command(
        "benchmark",
        _cmd_benchmark,
        "Score AudioToIML against a labeled dataset; exit 1 on failed conversions or "
        "a regression.",
    )
    sub.add_argument("dataset_dir", metavar="DATASET_DIR", help="dataset directory")
    sub.add_argument("--save", metavar="REPORT.json", help="save the report as JSON")
    sub.add_argument(
        "--baseline", metavar="REPORT.json", help="fail on regressions against this report"
    )
    sub.add_argument(
        "--tolerance",
        type=_non_negative_float,
        default=0.01,
        metavar="F",
        help="allowed drop against --baseline (default 0.01)",
    )
    sub.add_argument(
        "--threshold",
        type=_threshold,
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="a limit such as emotion_accuracy=0.7 or failure_rate=0.1 (repeatable)",
    )
    sub.add_argument(
        "--max-samples", type=_positive_int, metavar="N", help="evaluate the first N entries"
    )
    sub.add_argument(
        "--calibration",
        metavar="AUDIO",
        action="append",
        help="a recording of the speakers talking as usual, as the baseline for every entry; "
        "repeat for several (without one, a single-utterance entry gets no emotion)",
    )
    sub.add_argument(
        "--words-from",
        choices=["auto", "timings", "transcript", "stt"],
        default="auto",
        help="what the converter gets besides the audio: auto (default): each entry's word "
        "timings (metadata.word_timings) when it has them, else nothing when Whisper is "
        "installed and --stt is not none (Whisper finds the words, with timings), else its "
        "transcript; timings: word timings only; transcript: the transcript; stt: nothing "
        "(speech recognition, see --stt)",
    )
    sub.add_argument(
        "--stt",
        choices=["auto", "whisper", "none"],
        default="auto",
        help="speech recognition for entries given no words (default auto: Whisper if "
        "installed, else [speech] placeholders)",
    )
    sub.add_argument(
        "--abstention-label",
        metavar="LABEL",
        help="score an output without an emotion as this label (e.g. neutral) instead of "
        "as an abstention, which lowers emotion_coverage but not emotion_accuracy",
    )
    sub.add_argument("--language", type=_language, metavar="TAG", help="BCP 47 language tag")

    sub = command("serve", _cmd_serve, _SERVE_HELP)
    _add_serve_arguments(sub)

    command("doctor", _cmd_doctor, "Show which optional capabilities are installed.")
    return parser


_SERVE_HELP = "Run the REST API (needs the 'api' extra)."


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host", help="bind address (default: $PP_HOST or 127.0.0.1; empty means the default)"
    )
    parser.add_argument(
        "--port", type=_port, help="port, 0-65535 (default: $PP_PORT or 8000; 0: any free port)"
    )


def _parse_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace | int:
    """The parsed arguments, or the exit status for --help, --version and usage errors."""
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    parser = _build_parser()
    args = _parse_args(parser, argv)
    if isinstance(args, int):
        return args
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    return _run(args.handler, args)


def serve_main(argv: Sequence[str] | None = None, prog: str = f"{PROG} serve") -> int:
    """``serve`` as a program of its own (``python -m prosody_protocol.server``).

    Returns the exit status, as :func:`main` does.
    """
    parser = argparse.ArgumentParser(prog=prog, description=_SERVE_HELP)
    _add_serve_arguments(parser)
    args = _parse_args(parser, argv)
    if isinstance(args, int):
        return args
    return _run(_cmd_serve, args)


def _run(handler: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """Run a subcommand's *handler*, printing expected failures as one line."""
    try:
        if handler is _cmd_serve:
            # A long-running server: its warnings are printed as usual.
            return handler(args)
        with _warnings_to_stderr():
            return handler(args)
    except CLIError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return exc.code
    except (ProsodyProtocolError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
