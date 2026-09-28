"""Tests for the ``prosody-protocol`` command line (prosody_protocol.cli).

Every subcommand is run through ``main(argv)``, for success and for the
expected errors, which must print ``error: ...`` (never a traceback) and
exit 1 (rejected input) or 2 (usage, unreadable files, missing extras).
Commands that need an optional extra, espeak-ng or ffmpeg skip without it,
so this module also runs on the core install. The commands in
examples/README.md are run too, so the examples cannot rot.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shlex
import shutil
import subprocess
import sys
import sysconfig
import warnings
import wave
from pathlib import Path
from typing import Any

import pytest

from prosody_protocol import __version__, cli
from prosody_protocol.cli import main
from prosody_protocol.llm import build_messages, to_llm_context
from prosody_protocol.parser import IMLParser
from prosody_protocol.validator import IMLValidator

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
FIXTURES = Path(__file__).parent / "fixtures"
AUDIO = FIXTURES / "audio"
SPEECH = AUDIO / "speech_pauses.wav"
SARCASM = EXAMPLES / "sarcasm.iml"

HAS_AUDIO = all(importlib.util.find_spec(m) is not None for m in ("numpy", "parselmouth"))
needs_audio = pytest.mark.skipif(not HAS_AUDIO, reason="needs the audio extra")
needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="needs espeak-ng")

MISSING_CONFIDENCE = '<utterance emotion="angry">No confidence.</utterance>\n'
# Valid, with one warning (V32: an unprefixed unknown attribute).
WITH_WARNING = '<utterance mood="odd">Hello.</utterance>\n'


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def set_stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))


def speech_words(tmp_path: Path) -> str:
    """The exact word timings of SPEECH as a {word, start_ms, end_ms} JSON file."""
    words = json.loads((AUDIO / "speech_pauses.json").read_text())["words"]
    return write(tmp_path, "words.json", json.dumps(words))


def pause_profile(tmp_path: Path, boost: float) -> str:
    """A profile whose one mapping matches SPEECH (2 pauses in 6 word boundaries)."""
    return write(tmp_path, "profile.json", json.dumps({
        "profile_version": "0.1.0",
        "user_id": "caller_7",
        "prosody_mappings": [{
            "pattern": {"pause_frequency": "high"},
            "interpretation": {"emotion": "uncertain", "confidence_boost": boost},
        }],
    }))


def assert_no_traceback(err: str) -> None:
    assert "Traceback" not in err
    assert err.startswith("error: ") or "\nerror: " in err, err


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


class TestGeneral:
    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "--version")
        assert (code, out) == (0, f"prosody-protocol {__version__}\n")

    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = run(capsys)
        assert code == 2
        assert "validate" in err and "from-audio" in err and out == ""

    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "from-audio", "--help")
        assert code == 0
        assert "--words" in out and "--profile" in out

    def test_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = run(capsys, "frobnicate")
        assert code == 2
        assert "invalid choice" in err

    def test_console_script(self) -> None:
        """The installed ``prosody-protocol`` entry point runs main()."""
        script = Path(sysconfig.get_path("scripts")) / "prosody-protocol"
        found = str(script) if script.exists() else shutil.which("prosody-protocol")
        if found is None:
            pytest.skip("the prosody-protocol console script is not installed")
        done = subprocess.run(
            [found, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout == f"prosody-protocol {__version__}\n"

    def test_warnings_are_printed_as_they_happen(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each distinct warning is printed at once (it used to wait for the
        command to finish), and the warning filters still apply."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with cli._warnings_to_stderr():
                warnings.warn("first", UserWarning, stacklevel=1)
                assert capsys.readouterr().err == "warning: first\n"
                warnings.warn("first", UserWarning, stacklevel=1)
                warnings.warn("old API", DeprecationWarning, stacklevel=1)
                warnings.warn("second", UserWarning, stacklevel=1)
                assert capsys.readouterr().err == "warning: second\n"

    def test_cli_imports_only_the_standard_library(self) -> None:
        code = (
            "import sys; import prosody_protocol.cli; "
            "heavy = {'numpy', 'parselmouth', 'fastapi', 'uvicorn', 'whisper'} & set(sys.modules); "
            "print(sorted(heavy))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
        )
        assert done.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = run(capsys, "validate", str(SARCASM))
        assert (code, out, err) == (0, f"{SARCASM}: valid\n", "")

    def test_invalid_file_lists_issues_with_location(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        path = write(tmp_path, "bad.iml", MISSING_CONFIDENCE)
        code, out, _ = run(capsys, "validate", path)
        assert code == 1
        assert out.splitlines() == [
            f"{path}:1: error V3 <utterance> has emotion=\"angry\" but no confidence attribute",
            f"{path}: invalid (1 error)",
        ]

    def test_column_is_shown_when_known(self, capsys: pytest.CaptureFixture[str]) -> None:
        path = FIXTURES / "invalid" / "encoding_latin1.xml"
        code, out, _ = run(capsys, "validate", str(path))
        assert code == 1
        assert out.startswith(f"{path}:4:9: error V30 ")

    def test_warnings_fail_only_with_strict(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        path = write(tmp_path, "warn.iml", WITH_WARNING)
        code, out, _ = run(capsys, "validate", path)
        assert code == 0
        assert out.endswith(f"{path}: valid (1 warning)\n")
        assert f"{path}:1: warning V32 " in out
        code, out, _ = run(capsys, "validate", "--strict", path)
        assert code == 1
        assert out.endswith(f"{path}: fails --strict (1 warning)\n")

    def test_several_files_and_json(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        bad = write(tmp_path, "bad.iml", MISSING_CONFIDENCE)
        code, out, _ = run(capsys, "validate", "--json", str(SARCASM), bad)
        assert code == 1
        report = json.loads(out)
        assert [(r["file"], r["valid"], r["errors"]) for r in report] == [
            (str(SARCASM), True, 0), (bad, False, 1)
        ]
        assert report[1]["issues"] == [{
            "severity": "error",
            "rule": "V3",
            "message": '<utterance> has emotion="angry" but no confidence attribute',
            "line": 1,
            "column": None,
        }]

    def test_unreadable_file_is_exit_2_and_others_are_still_checked(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        missing = str(tmp_path / "missing.iml")
        code, out, err = run(capsys, "validate", missing, str(SARCASM))
        assert code == 2
        assert err == f"error: cannot read {missing}: No such file or directory\n"
        assert out == f"{SARCASM}: valid\n"

    def test_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_stdin(monkeypatch, MISSING_CONFIDENCE.encode())
        code, out, _ = run(capsys, "validate", "-")
        assert code == 1
        assert out.startswith("<stdin>:1: error V3 ")

    def test_stdin_that_is_not_utf8(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_stdin(monkeypatch, "<utterance>Café</utterance>".encode("latin-1"))
        code, out, _ = run(capsys, "validate", "-")
        assert code == 1
        assert out.startswith("<stdin>:1:15: error V30 ")

    def test_stdin_only_once(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = run(capsys, "validate", "-", "-")
        assert code == 2
        assert "only one input" in err


# ---------------------------------------------------------------------------
# to-text, to-ssml, to-prompt, from-text
# ---------------------------------------------------------------------------


class TestTextCommands:
    def test_to_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "to-text", str(SARCASM))
        assert code == 0
        assert out == (
            "I've been on hold for forty minutes. Oh, that's just wonderful. "
            "Really great service.\n"
        )

    def test_to_text_from_stdin_to_a_file(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        set_stdin(monkeypatch, SARCASM.read_bytes())
        out_file = tmp_path / "out.txt"
        code, out, _ = run(capsys, "to-text", "-", "-o", str(out_file))
        assert (code, out) == (0, "")
        assert out_file.read_text().startswith("I've been on hold")

    def test_malformed_iml(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        path = write(tmp_path, "broken.iml", "<utterance>oops")
        for command in ("to-text", "to-ssml", "to-prompt"):
            code, out, err = run(capsys, command, path)
            assert (code, out) == (1, ""), command
            assert_no_traceback(err)

    def test_missing_file_is_exit_2(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code, _, err = run(capsys, "to-text", str(tmp_path / "nope.iml"))
        assert code == 2
        assert err.startswith("error: cannot read ")

    def test_unwritable_output_is_exit_2(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code, _, err = run(capsys, "to-text", str(SARCASM), "-o", str(tmp_path / "no" / "x.txt"))
        assert code == 2
        assert err.startswith("error: cannot write ")

    def test_not_utf8(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        path = tmp_path / "latin1.iml"
        path.write_bytes("<utterance>Café</utterance>".encode("latin-1"))
        code, _, err = run(capsys, "to-text", str(path))
        assert code == 1
        assert "not UTF-8" in err

    def test_to_ssml(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "to-ssml", str(SARCASM))
        assert code == 0
        assert out.startswith('<speak version="1.1"')
        assert '<break time="600ms"/>' in out

    def test_to_ssml_espeak_vendor(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "to-ssml", "--vendor", "espeak-ng", str(SARCASM))
        assert code == 0
        assert out.startswith("<speak")

    def test_to_ssml_rejects_invalid_iml_unless_lenient(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        path = write(tmp_path, "bad.iml", MISSING_CONFIDENCE)
        code, _, err = run(capsys, "to-ssml", path)
        assert code == 1
        assert_no_traceback(err)
        assert "V3" in err
        code, out, _ = run(capsys, "to-ssml", "--lenient", path)
        assert code == 0
        assert "No confidence." in out

    def test_to_prompt(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "to-prompt", str(SARCASM))
        assert code == 0
        assert out == to_llm_context(SARCASM.read_text()) + "\n"
        assert "Really **great** service." in out

    def test_to_prompt_options(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(
            capsys, "to-prompt", str(SARCASM), "--numbers", "--min-confidence", "0.8"
        )
        assert code == 0
        expected = to_llm_context(SARCASM.read_text(), min_confidence=0.8, include_numbers=True)
        assert out == expected + "\n"
        assert "+15%" in out and "not reliably detected" in out

    def test_to_prompt_messages(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(
            capsys, "to-prompt", str(SARCASM), "--messages", "--instruction", "Summarize."
        )
        assert code == 0
        assert json.loads(out) == build_messages(SARCASM.read_text(), "Summarize.")

    def test_to_prompt_usage_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = run(capsys, "to-prompt", str(SARCASM), "--instruction", "Hi.")
        assert code == 2
        assert "--instruction needs --messages" in err
        code, _, err = run(capsys, "to-prompt", str(SARCASM), "--min-confidence", "1.5")
        assert code == 2
        assert "not between 0 and 1" in err

    def test_from_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "from-text", "Oh great, another meeting.")
        assert code == 0
        doc = IMLParser().parse(out)
        assert (doc.utterances[0].emotion, IMLParser().to_plain_text(doc)) == (
            "sarcastic", "Oh great, another meeting."
        )
        assert IMLValidator().validate(out).valid

    def test_from_text_stdin_and_context(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_stdin(monkeypatch, b"The app crashed again.\n")
        code, out, _ = run(capsys, "from-text", "-", "--context", "the user is frustrated")
        assert code == 0
        assert 'emotion="frustrated"' in out


# ---------------------------------------------------------------------------
# from-audio
# ---------------------------------------------------------------------------


@needs_audio
class TestFromAudio:
    def test_words_give_emphasis_and_pauses(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code, out, err = run(
            capsys, "from-audio", str(SPEECH), "--words", speech_words(tmp_path),
            "--language", "en-US",
        )
        assert (code, err) == (0, "")
        doc = IMLParser().parse(out)
        assert doc.language == "en-US"
        assert IMLParser().to_plain_text(doc) == "I told you to call me yesterday."
        assert '<emphasis level="strong">' in out and "told</prosody></emphasis>" in out
        assert '<pause duration="6' in out
        assert IMLValidator().validate(out).valid

    def test_json(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        code, out, _ = run(
            capsys, "from-audio", str(SPEECH), "--words", speech_words(tmp_path), "--json"
        )
        assert code == 0
        data = json.loads(out)
        assert set(data) == {
            "iml", "plain_text", "transcript_source", "warnings", "profile_matches"
        }
        assert data["transcript_source"] == "words"
        assert data["warnings"] == [] and data["profile_matches"] == []

    def test_prompt(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        code, out, _ = run(
            capsys, "from-audio", str(SPEECH), "--words", speech_words(tmp_path), "--prompt"
        )
        assert code == 0
        assert out.startswith("I **told** (")
        assert "you [pause 0.6s] to call me [pause 0.3s] yesterday" in out

    def test_transcript(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        code, out, _ = run(
            capsys, "from-audio", str(SPEECH), "--transcript", "I told you to call me.", "--json"
        )
        assert code == 0
        assert json.loads(out)["transcript_source"] == "transcript"
        text_file = write(tmp_path, "t.txt", "I told you to call me.\n")
        code, out, _ = run(capsys, "from-audio", str(SPEECH), "--transcript-file", text_file)
        assert code == 0
        assert IMLParser().to_plain_text(IMLParser().parse(out)) == "I told you to call me."

    def test_words_from_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_stdin(monkeypatch, (EXAMPLES / "speech.whisper.json").read_bytes())
        code, out, _ = run(
            capsys, "from-audio", str(EXAMPLES / "speech.wav"), "--words", "-", "--json"
        )
        assert code == 0
        assert json.loads(out)["plain_text"] == "I never said she stole my money."

    def test_audio_from_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        set_stdin(monkeypatch, SPEECH.read_bytes())
        code, out, _ = run(capsys, "from-audio", "-", "--words", speech_words(tmp_path))
        assert code == 0
        assert "told" in out

    def test_placeholders_warn_on_stderr(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, err = run(capsys, "from-audio", str(SPEECH), "--stt", "none")
        assert code == 0
        assert "[speech]" in out
        assert err.startswith("warning: No transcript")

    @pytest.mark.parametrize(("boost", "applied"), [(0.3, False), (0.6, True)])
    def test_profile_use_is_reported(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, boost: float, applied: bool
    ) -> None:
        words, profile = speech_words(tmp_path), pause_profile(tmp_path, boost)
        code, out, err = run(capsys, "from-audio", str(SPEECH), "--words", words,
                             "--profile", profile)
        assert code == 0
        assert ('x-profile="pause_frequency=high"' in out) is applied
        assert "caller_7" not in out
        if applied:
            assert err.startswith("note: prosody profile 'caller_7' set utterance 1 to 'uncertain'")
        else:
            assert "is below 0.5, so no emotion is reported" in err

        code, out, _ = run(capsys, "from-audio", str(SPEECH), "--words", words,
                           "--profile", profile, "--json")
        [match] = json.loads(out)["profile_matches"]
        assert (match["pattern"], match["emotion"], match["applied"]) == (
            {"pause_frequency": "high"}, "uncertain", applied
        )
        assert match["confidence"] == pytest.approx(boost)

    def test_prompt_names_emotions_at_min_confidence(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """--prompt used to describe emotions below 0.5 as "not reliably
        detected" even when --min-confidence had let them into the IML."""
        words, profile = speech_words(tmp_path), pause_profile(tmp_path, 0.3)
        code, out, err = run(capsys, "from-audio", str(SPEECH), "--words", words,
                             "--profile", profile, "--min-confidence", "0.2", "--prompt")
        assert code == 0
        assert "set utterance 1 to 'uncertain' (confidence 0.30" in err
        assert "Delivery: sounds uncertain (estimated, 30%)." in out
        assert "not reliably detected" not in out

    def test_profile_that_matches_nothing(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = run(
            capsys, "from-audio", str(EXAMPLES / "speech.wav"),
            "--words", str(EXAMPLES / "speech.whisper.json"),
            "--profile", str(EXAMPLES / "profile.json"),
        )
        assert code == 0
        assert err == "note: prosody profile 'example_user' matched no utterance\n"

    @pytest.mark.parametrize(
        ("profile", "message"),
        [
            ("{not json", "invalid JSON in profile"),
            ('{"profile_version": "0.1.0", "user_id": "u", "prosody_mappings": [], "x": 1}',
             "unknown key"),
            ('{"profile_version": "0.1.0", "user_id": "u", "prosody_mappings": [{"pattern": '
             '{"pitch": "shrill"}, "interpretation": {"emotion": "angry"}}]}', "P6"),
            ('{"profile_version": "0.1.0", "user_id": "u", "prosody_mappings": [{"pattern": '
             '{"pitch": "high"}, "interpretation": {"emotion": "angry", '
             '"confidence_boost": NaN}}]}', "not a JSON number"),
            ("[" * 50_000, "invalid JSON in profile"),
            # Used to be accepted, and to fail only after the analysis when it applied.
            ('{"profile_version": "0.1.0", "user_id": "u", "prosody_mappings": [{"pattern": '
             '{"pitch": "high"}, "interpretation": {"emotion": "calm\\u0001"}}]}',
             "XML does not allow"),
        ],
    )
    def test_invalid_profile(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        profile: str,
        message: str,
    ) -> None:
        from prosody_protocol import prosody_analyzer

        # Rejected before any audio is analysed.
        monkeypatch.setattr(prosody_analyzer._AudioAnalysis, "from_path", None)
        path = write(tmp_path, "profile.json", profile)
        code, out, err = run(capsys, "from-audio", str(SPEECH), "--profile", path)
        assert (code, out) == (1, "")
        assert_no_traceback(err)
        assert message in err

    @pytest.mark.parametrize(
        "words",
        ['[{"word": "hi", "start": 2.0, "end": 1.0}]', "not json", '{"unknown": 1}', "[" * 50_000],
    )
    def test_invalid_words(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, words: str
    ) -> None:
        path = write(tmp_path, "words.json", words)
        code, _, err = run(capsys, "from-audio", str(SPEECH), "--words", path)
        assert code == 1
        assert_no_traceback(err)

    def test_unreadable_inputs_are_exit_2(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        missing = str(tmp_path / "missing.json")
        for option in ("--words", "--profile", "--transcript-file", "--calibration"):
            code, _, err = run(capsys, "from-audio", str(SPEECH), option, missing)
            assert code == 2, option
            assert err.startswith(f"error: cannot read {missing}"), (option, err)
        # The audio itself (it used to be exit 1, "Audio file not found").
        for audio in (str(tmp_path / "missing.wav"), str(tmp_path)):
            code, _, err = run(capsys, "from-audio", audio, "--stt", "none")
            assert code == 2, audio
            assert err.startswith(f"error: cannot read {audio}: "), err

    def test_stt_whisper_without_whisper_is_a_hint(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """It used to be exit 1 with the SDK's advice to "pass words=", a
        Python argument."""
        monkeypatch.setattr(cli, "_installed", lambda module: module != "whisper")
        code, out, err = run(capsys, "from-audio", str(SPEECH), "--stt", "whisper")
        assert (code, out) == (2, "")
        assert err == (
            "error: from-audio --stt whisper needs the 'whisper' extra (openai-whisper is not "
            "installed): pip install 'prosody-protocol[whisper]', or give the words with "
            "--words or --transcript\n"
        )
        # With the words given, Whisper is not needed.
        code, _, _ = run(capsys, "from-audio", str(SPEECH), "--stt", "whisper",
                         "--words", speech_words(tmp_path))
        assert code == 0

    def test_stdin_is_named_in_errors(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_stdin(monkeypatch, b"")
        code, _, err = run(capsys, "from-audio", "-", "--stt", "none")
        assert code == 1
        assert err == "error: Audio file is empty: <stdin>\n"

    def test_calibration_from_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        words = speech_words(tmp_path)
        calibration = AUDIO / "speech_calibration.wav"
        code, from_file, _ = run(capsys, "from-audio", str(SPEECH), "--words", words,
                                 "--calibration", str(calibration))
        assert code == 0
        set_stdin(monkeypatch, calibration.read_bytes())
        code, from_stdin, _ = run(capsys, "from-audio", str(SPEECH), "--words", words,
                                  "--calibration", "-")
        assert (code, from_stdin) == (0, from_file)
        code, _, err = run(capsys, "from-audio", "-", "--calibration", "-")
        assert code == 2
        assert "only one input" in err

    def test_overlapping_words_are_rejected(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        path = write(tmp_path, "words.json", json.dumps([
            {"word": "a", "start_ms": 0, "end_ms": 4000},
            {"word": "b", "start_ms": 1, "end_ms": 4000},
        ]))
        code, out, err = run(capsys, "from-audio", str(SPEECH), "--words", path)
        assert (code, out) == (1, "")
        assert_no_traceback(err)
        assert "may overlap by at most 500 ms" in err

    def test_posix_language_tag(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        code, out, _ = run(capsys, "from-audio", str(SPEECH), "--words", speech_words(tmp_path),
                           "--language", "en_US")
        assert code == 0
        assert out.startswith('<iml version="0.1.0" language="en-US">')

    def test_unreadable_audio(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        not_audio = write(tmp_path, "notes.wav", "not audio")
        code, _, err = run(capsys, "from-audio", not_audio, "--stt", "none")
        assert code == 1
        assert_no_traceback(err)

    def test_usage_errors(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        words = speech_words(tmp_path)
        for argv in (
            ["--language", "not a tag"],
            ["--words", words, "--transcript", "Hi."],
            ["--json", "--prompt"],
            ["--min-confidence", "-1"],
            ["--stt", "google"],
        ):
            code, _, err = run(capsys, "from-audio", str(SPEECH), *argv)
            assert code == 2, argv
            assert "Traceback" not in err
        code, _, err = run(capsys, "from-audio", "-", "--words", "-")
        assert code == 2
        assert "only one input" in err

    def test_calibration(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        code, out, _ = run(
            capsys, "from-audio", str(SPEECH), "--words", speech_words(tmp_path),
            "--calibration", str(AUDIO / "speech_calibration.wav"), "--extended",
        )
        assert code == 0
        assert "f0_mean=" in out


class TestMissingExtras:
    """Commands that need an extra print a one-line hint naming it."""

    @pytest.mark.parametrize(
        ("argv", "module", "extra"),
        [
            (["from-audio", str(SPEECH)], "prosody_protocol.audio_to_iml", "audio"),
            (["synthesize", str(SARCASM), "-o", "x.wav"], "prosody_protocol.iml_to_audio",
             "audio"),
            (["benchmark", str(FIXTURES / "datasets" / "sample")], "prosody_protocol.benchmarks",
             "audio"),
        ],
    )
    def test_hint_names_the_extra(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        module: str,
        extra: str,
    ) -> None:
        monkeypatch.setitem(sys.modules, module, None)  # makes the import fail
        code, out, err = run(capsys, *argv)
        assert (code, out) == (2, "")
        assert err.endswith(f"pip install 'prosody-protocol[{extra}]'\n")
        assert err.count("\n") == 1

    def test_serve_without_the_api_extra(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "_installed", lambda module: module != "uvicorn")
        code, _, err = run(capsys, "serve")
        assert code == 2
        assert err == (
            "error: serve needs the 'api' extra (uvicorn not installed): "
            "pip install 'prosody-protocol[api]'\n"
        )


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


@needs_audio
class TestSynthesize:
    def test_tones(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        out_wav = tmp_path / "out.wav"
        code, out, err = run(
            capsys, "synthesize", str(SARCASM), "-o", str(out_wav), "--engine", "tones"
        )
        assert (code, out) == (0, "")
        assert err == f"wrote {out_wav}: tone preview, not speech\n"
        with wave.open(str(out_wav)) as wav:
            assert wav.getnframes() > 0

    @needs_espeak
    def test_espeak_speech(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        out_wav = tmp_path / "out.wav"
        code, _, err = run(
            capsys, "synthesize", str(SARCASM), "-o", str(out_wav), "--voice", "en-US"
        )
        assert code == 0
        assert err == f"wrote {out_wav}: speech (espeak-ng)\n"
        assert out_wav.read_bytes()[:4] == b"RIFF"

    def test_to_stdout(self, capfdbinary: pytest.CaptureFixture[bytes]) -> None:
        code = main(["synthesize", str(SARCASM), "-o", "-", "--engine", "tones"])
        assert code == 0
        assert capfdbinary.readouterr().out[:4] == b"RIFF"

    def test_errors(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        bad = write(tmp_path, "bad.iml", MISSING_CONFIDENCE)
        out_wav = str(tmp_path / "out.wav")
        code, _, err = run(capsys, "synthesize", bad, "-o", out_wav, "--engine", "tones")
        assert code == 1
        assert_no_traceback(err)
        code, _, _ = run(
            capsys, "synthesize", bad, "-o", out_wav, "--engine", "tones", "--lenient"
        )
        assert code == 0
        code, _, err = run(capsys, "synthesize", str(SARCASM))
        assert code == 2
        assert "-o/--output" in err

    @needs_espeak
    def test_unknown_voice(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        code, _, err = run(
            capsys, "synthesize", str(SARCASM), "-o", str(tmp_path / "x.wav"),
            "--engine", "espeak", "--voice", "xx-nonexistent",
        )
        assert code == 1
        assert_no_traceback(err)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


SAMPLE_DATASET = FIXTURES / "datasets" / "sample"


@needs_audio
class TestBenchmark:
    def test_summary_and_saved_report(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        report = tmp_path / "report.json"
        code, out, err = run(
            capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "none", "--save", str(report)
        )
        assert code == 0, out + err
        lines = out.splitlines()
        assert lines[0].startswith("Benchmark of sample: 3 entries, 0 failed conversions")
        assert "  validity_rate     1.0000" in lines
        assert "  pause_f1          n/a" in lines
        assert lines[-1] == "Passed"
        assert json.loads(report.read_text())["num_samples"] == 3
        # The placeholder warning is printed once, not once per entry.
        assert err.count("warning: No transcript") == 1

    def test_regression_against_a_baseline(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        report = tmp_path / "report.json"
        run(capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "none", "--save", str(report))
        code, out, _ = run(
            capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "none", "--baseline", str(report)
        )
        assert code == 0
        assert out.splitlines()[-1] == f"Passed (baseline {report})"

        better = json.loads(report.read_text())
        better["validity_rate"] = 1.0
        better["emotion_accuracy"] = 0.9
        report.write_text(json.dumps(better))
        code, out, _ = run(
            capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "none", "--baseline", str(report)
        )
        assert code == 1
        assert "FAILED:" in out
        assert "emotion_accuracy regressed" in out
        code, _, _ = run(
            capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "none", "--baseline", str(report),
            "--tolerance", "0.9",
        )
        assert code == 0

    def test_thresholds(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(
            capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "none",
            "--threshold", "validity_rate=1.0", "--threshold", "emotion_accuracy=0.99",
        )
        assert code == 1
        assert "emotion_accuracy = " in out and "< threshold 0.9900" in out
        assert "validity_rate" not in out.split("FAILED:")[1]

    def test_failed_conversions_fail(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        dataset = tmp_path / "broken"
        shutil.copytree(SAMPLE_DATASET, dataset)
        (dataset / "audio" / "sample_001.wav").write_bytes(b"not audio")
        code, out, err = run(capsys, "benchmark", str(dataset), "--stt", "none")
        assert code == 1
        assert "1 failed conversions" in out
        assert "failure_rate = 0.3333 > threshold 0.0000" in out
        assert "Traceback" not in err
        assert "warning: Conversion failed for entry entry_001" in err

    def test_usage_and_input_errors(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code, _, err = run(capsys, "benchmark", str(SAMPLE_DATASET), "--threshold", "speed=1")
        assert code == 2
        assert "Unknown threshold metric" in err
        not_a_report = write(tmp_path, "report.json", '{"emotion_accuracy": 1}')
        code, _, err = run(capsys, "benchmark", str(SAMPLE_DATASET), "--baseline", not_a_report)
        assert code == 2
        assert "is not a benchmark report" in err
        # Unreadable inputs are exit 2 (the missing directory used to be exit 1).
        code, _, err = run(capsys, "benchmark", str(tmp_path / "nowhere"))
        assert code == 2
        assert err == f"error: cannot read {tmp_path / 'nowhere'}: no such directory\n"
        code, _, err = run(capsys, "benchmark", not_a_report)
        assert code == 2
        assert err == f"error: cannot read {not_a_report}: not a directory\n"

    def test_save_to_a_missing_directory_fails_before_the_run(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The report used to be written after the whole run, creating the
        missing directories (or failing then, losing the run)."""
        from prosody_protocol import benchmarks

        monkeypatch.setattr(benchmarks.Benchmark, "run", None)  # must not be reached
        target = tmp_path / "no" / "such" / "report.json"
        code, out, err = run(capsys, "benchmark", str(SAMPLE_DATASET), "--save", str(target))
        assert (code, out) == (2, "")
        assert err == f"error: cannot write {target}: no such directory {target.parent}\n"
        assert not (tmp_path / "no").exists()

    def test_stt_whisper_without_whisper_is_a_hint(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "_installed", lambda module: module != "whisper")
        code, _, err = run(capsys, "benchmark", str(SAMPLE_DATASET), "--stt", "whisper")
        assert code == 2
        assert err == (
            "error: benchmark --stt whisper needs the 'whisper' extra (openai-whisper is not "
            "installed): pip install 'prosody-protocol[whisper]'\n"
        )


# ---------------------------------------------------------------------------
# serve, doctor
# ---------------------------------------------------------------------------


class TestServeAndDoctor:
    def test_serve_delegates_to_the_server(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prosody_protocol.server

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(cli, "_missing_api_modules", list)
        monkeypatch.setattr(prosody_protocol.server, "run", lambda **kw: calls.append(kw))
        code, _, _ = run(capsys, "serve", "--host", "0.0.0.0", "--port", "9001")
        assert code == 0
        assert calls == [{"host": "0.0.0.0", "port": 9001}]
        run(capsys, "serve")
        assert calls[-1] == {"host": None, "port": None}
        run(capsys, "serve", "--port", "0")
        assert calls[-1] == {"host": None, "port": 0}

    @pytest.mark.parametrize("port", ["99999", "-5", "http"])
    def test_serve_port_out_of_range_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str], port: str
    ) -> None:
        """--port 99999 used to print two tracebacks (uvicorn's OverflowError)."""
        code, out, err = run(capsys, "serve", "--port", port)
        assert (code, out) == (2, "")
        assert "argument --port: " in err and "Traceback" not in err

    def test_serve_failures_are_one_line(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prosody_protocol.server

        monkeypatch.setattr(cli, "_missing_api_modules", list)

        def cannot_bind(**kwargs: Any) -> None:
            raise SystemExit(3)  # what uvicorn does when the port is in use

        monkeypatch.setattr(prosody_protocol.server, "run", cannot_bind)
        code, _, err = run(capsys, "serve")
        assert (code, err) == (1, "error: the server could not start (see the log above)\n")

    def test_serve_with_invalid_settings(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uvicorn = pytest.importorskip("uvicorn")
        monkeypatch.setattr(uvicorn, "run", None)  # must not be reached
        monkeypatch.setenv("PP_PORT", "70000")
        code, _, err = run(capsys, "serve")
        assert (code, err) == (2, "error: port (PP_PORT) must be at most 65535, got 70000\n")

    def test_serve_warnings_are_not_held_back(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """serve's warnings used to be collected for the server's lifetime and
        printed at shutdown; they now go through the normal warning machinery."""
        import prosody_protocol.server

        monkeypatch.setattr(cli, "_missing_api_modules", list)

        def warn(**kwargs: Any) -> None:
            warnings.warn("during serve", UserWarning, stacklevel=1)

        monkeypatch.setattr(prosody_protocol.server, "run", warn)
        with pytest.warns(UserWarning, match="during serve"):
            code, _, err = run(capsys, "serve")
        assert (code, err) == (0, "")

    def test_doctor(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(capsys, "doctor")
        assert code == 0
        assert out.startswith(f"prosody-protocol {__version__} (Python ")
        assert "[ok     ] core (lxml" in out
        for name in ("audio analysis", "speech recognition", "speech synthesis",
                     "audio decoding", "REST API", "training baselines"):
            assert f"] {name} (" in out
        assert ("[ok     ] audio analysis" in out) is HAS_AUDIO

    def test_doctor_names_what_is_missing(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(cli, "_installed", lambda module: module == "lxml")
        code, out, _ = run(capsys, "doctor")
        assert code == 0
        assert out.count("[missing]") == 6
        assert "install: install espeak-ng with your package manager" in out
        assert "install: pip install 'prosody-protocol[whisper]'" in out
        assert "install: pip install 'prosody-protocol[api]'" in out


# ---------------------------------------------------------------------------
# examples/README.md
# ---------------------------------------------------------------------------


def _readme_blocks(language: str) -> list[str]:
    text = (EXAMPLES / "README.md").read_text()
    blocks = text.split(f"```{language}\n")[1:]
    return [block.split("```", 1)[0] for block in blocks]


def _readme_commands() -> list[list[str]]:
    return [
        shlex.split(line)
        for block in _readme_blocks("sh")
        for line in block.splitlines()
        if line.startswith("prosody-protocol ")
    ]


_AUDIO_COMMANDS = {"from-audio", "synthesize", "benchmark"}


class TestExamples:
    def test_readme_has_the_key_commands(self) -> None:
        commands = {argv[1] for argv in _readme_commands()}
        assert {"validate", "to-text", "to-ssml", "to-prompt", "from-text", "from-audio",
                "synthesize"} <= commands

    @pytest.mark.parametrize(
        "argv", _readme_commands(), ids=lambda argv: " ".join(argv[1:4])
    )
    def test_readme_command_works(
        self,
        argv: list[str],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        if argv[1] in _AUDIO_COMMANDS and not HAS_AUDIO:
            pytest.skip("needs the audio extra")
        args = argv[1:]
        if "-o" in args:  # write outputs to the temporary directory
            index = args.index("-o") + 1
            args[index] = str(tmp_path / args[index])
        monkeypatch.chdir(ROOT)
        code, out, err = run(capsys, *args)
        assert code == 0, err
        assert "Traceback" not in err

    def test_readme_outputs_are_current(self, capsys: pytest.CaptureFixture[str],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        """The to-prompt output shown in the README is what the command prints."""
        monkeypatch.chdir(ROOT)
        _, out, _ = run(capsys, "to-prompt", "examples/sarcasm.iml")
        assert out in _readme_blocks("text")

    @needs_audio
    def test_readme_annotation_marks_stress_and_pause(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(ROOT)
        _, out, _ = run(
            capsys, "from-audio", "examples/speech.wav", "--words",
            "examples/speech.whisper.json", "--prompt",
        )
        assert out.startswith("I never said [pause 0.")
        assert "she **stole** (" in out
        assert out in _readme_blocks("text"), "update the output shown in examples/README.md"
        _, out, _ = run(
            capsys, "from-audio", "examples/speech.wav", "--words",
            "examples/speech.whisper.json", "--language", "en-US",
        )
        assert out in _readme_blocks("xml"), "update the IML shown in examples/README.md"
        _, out, _ = run(
            capsys, "from-audio", "examples/speech.wav", "--words",
            "examples/speech.whisper.json", "--json",
        )
        data = json.loads(out)
        assert data["transcript_source"] == "words"
        assert data["plain_text"] == (EXAMPLES / "speech.txt").read_text().strip()

    @needs_audio
    def test_readme_profile_example_applies_the_profile(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The README shows a profile at work: without it no emotion is
        reported, with it every utterance gets one, marked with x-profile."""
        monkeypatch.chdir(ROOT)
        argv = ["from-audio", "examples/monotone.wav", "--words",
                "examples/monotone.deepgram.json"]
        code, plain, err = run(capsys, *argv)
        assert (code, err) == (0, "")
        assert "emotion=" not in plain
        code, profiled, notes = run(capsys, *argv, "--profile", "examples/profile.json")
        assert code == 0
        doc = IMLParser().parse(profiled)
        assert [(u.emotion, dict(u.extra_attributes).get("x-profile")) for u in doc.utterances] == [
            ("calm", "pitch_contour=flat"), ("calm", "pitch_contour=flat"),
            ("calm", "pitch_contour=flat"), ("joyful", "pitch_contour=flat rate=fast"),
        ]
        assert "example_user" not in profiled
        assert IMLValidator().validate(profiled).valid
        blocks = _readme_blocks("xml")
        assert plain in blocks and profiled in blocks, "update the IML shown in examples/README.md"
        assert notes in _readme_blocks("text"), "update the notes shown in examples/README.md"

    @needs_audio
    def test_readme_python_example(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(ROOT)
        [code] = _readme_blocks("python")
        exec(compile(code, "examples/README.md", "exec"), {})  # noqa: S102
        out = capsys.readouterr().out
        assert "<emphasis" in out and "**stole**" in out
        assert "3 {'pitch_contour': 'flat', 'rate': 'fast'} joyful 0.6 True" in out

    def test_example_files_are_valid(self) -> None:
        from prosody_protocol.alignment import load_word_timings
        from prosody_protocol.profiles import ProfileLoader

        result = IMLValidator().validate_file(SARCASM)
        assert result.valid and not result.issues
        loader = ProfileLoader()
        assert loader.validate(loader.load(EXAMPLES / "profile.json")).valid
        whisper = json.loads((EXAMPLES / "speech.whisper.json").read_text())
        words = [w["word"] for w in whisper["segments"][0]["words"]]
        assert "".join(words).strip() == (EXAMPLES / "speech.txt").read_text().strip()
        monotone = load_word_timings(EXAMPLES / "monotone.deepgram.json")
        assert " ".join(w.word for w in monotone) == (
            "I read the list. The room is booked. I have the slides. And we got the grant!"
        )

    @needs_audio
    def test_speech_example_matches_its_word_timings(self) -> None:
        with wave.open(str(EXAMPLES / "speech.wav")) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            duration = wav.getnframes() / wav.getframerate()
        assert 3.0 <= duration <= 5.0
        assert (EXAMPLES / "speech.wav").stat().st_size < 200_000
        whisper = json.loads((EXAMPLES / "speech.whisper.json").read_text())
        words = whisper["segments"][0]["words"]
        assert words[-1]["end"] < duration
        gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:], strict=False)]
        assert max(gaps) == pytest.approx(0.6, abs=0.001)

    @needs_audio
    def test_monotone_example_matches_its_word_timings(self) -> None:
        with wave.open(str(EXAMPLES / "monotone.wav")) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            duration = wav.getnframes() / wav.getframerate()
        assert 5.0 <= duration <= 8.0
        assert (EXAMPLES / "monotone.wav").stat().st_size < 250_000
        deepgram = json.loads((EXAMPLES / "monotone.deepgram.json").read_text())
        words = deepgram["results"]["channels"][0]["alternatives"][0]["words"]
        assert words[-1]["end"] < duration
        assert all(a["end"] <= b["start"] for a, b in zip(words, words[1:], strict=False))
