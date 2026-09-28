"""What the README promises a new user, checked against the package.

``tests/test_docs_examples.py`` runs every documentation example and checks
that it does not fail. This module checks what that cannot:

- the output the README shows is what the code prints;
- the install instructions name extras that exist, and the extras hold what
  the README says they hold;
- the version agrees in the package, the CLI, the README and the changelog;
- a core-only install (lxml, no numpy or parselmouth) works, and optional
  features fail with an install hint;
- the README's example check for risky actions fails closed;
- the README's REST endpoint table matches the server's routes;
- datasets/README.md shows the errors and rules the loader has;
- relative links in the top-level docs point at files and headings that exist.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    import tomli as tomllib

import prosody_protocol
import prosody_protocol.datasets as datasets
from prosody_protocol import Dataset, DatasetError, __version__
from prosody_protocol.cli import main

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

HAS_AUDIO = all(importlib.util.find_spec(m) is not None for m in ("numpy", "parselmouth"))
needs_audio = pytest.mark.skipif(not HAS_AUDIO, reason="needs the audio extra")

_FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_SKIP = "<!-- docs-test: skip -->"


def _blocks(path: Path = README) -> list[tuple[str, str, bool]]:
    """(language, code, skipped) for each fenced block, in order."""
    text = path.read_text(encoding="utf-8")
    blocks = []
    for match in _FENCE.finditer(text):
        before = text[: match.start()].rstrip("\n").rsplit("\n", 1)[-1]
        blocks.append((match.group(1), match.group(2), before.strip() == _SKIP))
    return blocks


def _block_with(needle: str, language: str = "python", path: Path = README) -> str:
    """The first *language* block of *path* containing *needle*."""
    for lang, code, _ in _blocks(path):
        if lang == language and needle in code:
            return code
    raise AssertionError(f"{path.name} has no {language} block containing {needle!r}")


def _block_after(needle: str, language: str, path: Path = README) -> str:
    """The first *language* block after the block of *path* containing *needle*."""
    blocks = _blocks(path)
    for index, (_, code, _) in enumerate(blocks):
        if needle in code:
            for lang, shown, _ in blocks[index + 1 :]:
                if lang == language:
                    return shown
    raise AssertionError(f"{path.name} shows no {language} block after {needle!r}")


def _run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture()
def examples_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch working directory where ``examples/`` is the repository's."""
    (tmp_path / "examples").symlink_to(ROOT / "examples", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Entry points and version
# ---------------------------------------------------------------------------


class TestEntryPoints:
    def test_cli_version(self) -> None:
        code, out, _ = _run_cli("--version")
        assert code == 0
        assert out.strip() == f"prosody-protocol {__version__}"

    def test_console_script_is_the_cli(self) -> None:
        assert PYPROJECT["project"]["scripts"] == {"prosody-protocol": "prosody_protocol.cli:main"}

    def test_doctor_always_succeeds(self) -> None:
        code, out, _ = _run_cli("doctor")
        assert code == 0
        assert out.startswith(f"prosody-protocol {__version__} ")
        assert "[ok     ] core (lxml" in out


class TestVersion:
    def test_version_is_pep440_prerelease_or_release(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", __version__), __version__

    def test_changelog_leads_with_this_version(self) -> None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        first = re.search(r"^## \[([^\]]+)\]", changelog, re.MULTILINE)
        assert first is not None and first.group(1) == __version__

    def test_readme_names_this_version(self) -> None:
        text = README.read_text(encoding="utf-8")
        badge = __version__.replace("-", "--")
        assert f"badge/version-{badge}-" in text
        assert f"alpha ({__version__})" in text


# ---------------------------------------------------------------------------
# Install instructions
# ---------------------------------------------------------------------------


def _readme_extras_table() -> set[str]:
    text = README.read_text(encoding="utf-8")
    section = text.split("| Extra | Adds | Enables |", 1)[1].split("\n\n", 1)[0]
    return set(re.findall(r"^\| `([\w-]+)` \|", section, re.MULTILINE))


class TestInstallInstructions:
    def test_readme_documents_every_extra(self) -> None:
        extras = set(PYPROJECT["project"]["optional-dependencies"])
        assert _readme_extras_table() == extras

    def test_readme_install_commands_use_real_extras_and_repository(self) -> None:
        extras = set(PYPROJECT["project"]["optional-dependencies"])
        repository = PYPROJECT["project"]["urls"]["Repository"]
        commands = [
            line.strip()
            for lang, code, _ in _blocks()
            if lang == "bash"
            for line in code.splitlines()
            if line.strip().startswith(("pip install", "git clone"))
        ]
        assert any(repository in c for c in commands), "README must say where to install from"
        for command in commands:
            for group in re.findall(r"\[([\w,-]+)\]", command):
                assert set(group.split(",")) <= extras, command
            for url in re.findall(r"https://\S+", command):
                assert url.rstrip('"') == repository, command

    def test_core_needs_only_lxml(self) -> None:
        deps = PYPROJECT["project"]["dependencies"]
        assert [re.split(r"[<>=!~ ]", d, maxsplit=1)[0] for d in deps] == ["lxml"]

    def test_audio_extra_has_no_deep_learning_stack(self) -> None:
        """The README says the audio extra is numpy + parselmouth, and that
        only the whisper extra pulls in PyTorch."""
        extras = PYPROJECT["project"]["optional-dependencies"]

        def names(extra: str) -> set[str]:
            return {re.split(r"[<>=!~ \[]", d, maxsplit=1)[0] for d in extras[extra]}

        assert names("audio") == {"numpy", "praat-parselmouth"}
        assert "openai-whisper" in names("whisper")
        for extra in ("audio", "api", "ml"):
            assert not names(extra) & {"openai-whisper", "torch", "transformers"}, extra


class TestCoreOnlyInstall:
    """In a process where the optional packages cannot be imported, the core
    works and the optional classes explain how to get them."""

    SCRIPT = textwrap.dedent(
        """
        import importlib.abc, sys

        BLOCKED = {"numpy", "parselmouth", "fastapi", "uvicorn", "sklearn", "whisper", "torch"}

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None

        sys.meta_path.insert(0, Block())

        import prosody_protocol as pp
        from prosody_protocol.cli import main

        iml = ('<utterance emotion="sarcastic" confidence="0.87">'
               'Oh, <emphasis level="strong">great</emphasis>.</utterance>')
        assert pp.IMLValidator().validate(iml).valid
        assert pp.IMLParser().to_plain_text(pp.IMLParser().parse(iml)) == "Oh, great."
        assert "**great**" in pp.to_llm_context(iml)
        assert pp.IMLToSSML().convert(iml).startswith("<speak")
        assert "<utterance" in pp.TextToIML().predict("Oh, that's GREAT.")
        assert main(["doctor"]) == 0
        for name in sorted(pp._LAZY):
            try:
                getattr(pp, name)
            except ImportError as exc:
                assert "prosody-protocol[audio]" in str(exc), (name, str(exc))
            else:
                raise AssertionError(f"{name} imported without numpy/parselmouth")
        print("core ok")
        """
    )

    def test_core_works_without_optional_packages(self, tmp_path: Path) -> None:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        proc = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().endswith("core ok")

    def test_every_public_name_resolves(self) -> None:
        """With the optional packages present, every exported name imports."""
        pytest.importorskip("numpy")
        pytest.importorskip("parselmouth")
        for name in prosody_protocol.__all__:
            assert getattr(prosody_protocol, name) is not None, name


# ---------------------------------------------------------------------------
# The README's shown output is real
# ---------------------------------------------------------------------------


QUICKSTART = ("from-audio", "examples/speech.wav", "--words", "examples/speech.whisper.json")


@needs_audio
class TestReadmeOutput:
    def test_quickstart_iml(self, examples_cwd: Path) -> None:
        code, out, err = _run_cli(*QUICKSTART)
        assert code == 0, err
        shown = _block_after("prosody-protocol " + " ".join(QUICKSTART) + "\n", "xml")
        assert out == shown, "update the quickstart IML shown in README.md"

    def test_quickstart_prompt(self, examples_cwd: Path) -> None:
        code, out, err = _run_cli(*QUICKSTART, "--prompt")
        assert code == 0, err
        shown = _block_after("prosody-protocol " + " ".join(QUICKSTART) + " --prompt", "text")
        assert out == shown, "update the quickstart transcript shown in README.md"

    def test_python_examples_print_what_the_readme_shows(self, examples_cwd: Path) -> None:
        """Each Python example followed by a ``text`` block prints that text.

        The examples run in order in one namespace, as a reader would run them.
        """
        blocks = _blocks()
        namespace: dict[str, object] = {"__name__": "__readme__"}
        checked = 0
        for index, (lang, code, skipped) in enumerate(blocks):
            if lang != "python" or skipped:
                continue
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(code, f"README.md block {index}", "exec"), namespace)  # noqa: S102
            following = blocks[index + 1] if index + 1 < len(blocks) else None
            if following is not None and following[0] == "text":
                assert out.getvalue() == following[1], (
                    f"README.md block {index} prints something else than the README shows:\n"
                    f"{out.getvalue()}"
                )
                checked += 1
        assert checked >= 5


@pytest.fixture(scope="module")
def needs_confirmation() -> Callable[..., bool]:
    """The README's ``needs_confirmation`` function."""
    namespace: dict[str, object] = {"__name__": "__readme__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(_block_with("def needs_confirmation"), namespace)  # noqa: S102
    function = namespace["needs_confirmation"]
    assert callable(function)
    return function


class TestConfirmationExample:
    """The README's ``needs_confirmation`` is copied into agents as a gate on
    actions, so it must keep failing closed: prosody may add caution but
    never waive the confirmation of a destructive action."""

    SETTLED = '<utterance emotion="calm" confidence="0.9">Archive the reports.</utterance>'

    def test_settled_request_for_a_reversible_action_goes_ahead(
        self, needs_confirmation: Callable[..., bool]
    ) -> None:
        assert needs_confirmation(self.SETTLED, destructive=False) is False

    def test_destructive_actions_are_always_confirmed(
        self, needs_confirmation: Callable[..., bool]
    ) -> None:
        assert needs_confirmation(self.SETTLED, destructive=True) is True
        assert needs_confirmation(
            '<utterance emotion="calm" confidence="1.0">Delete everything.</utterance>',
            destructive=True,
            min_confidence=0.0,
        ) is True

    @pytest.mark.parametrize(
        "iml",
        [
            "<utterance></utterance>",  # no speech
            '<utterance><pause duration="800"/></utterance>',  # silence only
            "<utterance>Archive the reports.</utterance>",  # no label
            '<utterance emotion="calm" confidence="0.69">Archive the reports.</utterance>',
            '<utterance emotion="calm" confidence="0.0">Archive the reports.</utterance>',
            '<utterance emotion="calm" confidence="high">Archive the reports.</utterance>',
            '<utterance emotion="sarcastic" confidence="0.9">Archive the reports.</utterance>',
            '<iml version="0.1.0"><utterance emotion="calm" confidence="0.9">Archive</utterance>'
            "<utterance>the reports.</utterance></iml>",  # one utterance unlabeled
        ],
    )
    def test_missing_or_weak_evidence_asks(
        self, needs_confirmation: Callable[..., bool], iml: str
    ) -> None:
        assert needs_confirmation(iml, destructive=False) is True


# ---------------------------------------------------------------------------
# REST API and links
# ---------------------------------------------------------------------------


class TestReadmeRestTable:
    def test_endpoint_table_matches_the_server(self) -> None:
        pytest.importorskip("fastapi")
        from prosody_protocol.server.app import create_app
        from prosody_protocol.server.config import Settings

        paths = create_app(Settings()).openapi()["paths"]
        served = {
            (method.upper(), path)
            for path, operations in paths.items()
            if path.startswith("/v1/")
            for method in operations
        }
        text = README.read_text(encoding="utf-8")
        documented = set(re.findall(r"^\| `(GET|POST) (/v1/[\w/-]+)` \|", text, re.MULTILINE))
        assert documented == served


DATASETS_README = ROOT / "datasets" / "README.md"


class TestDatasetsReadme:
    def test_shown_validation_error_is_what_the_loader_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The README's example entry, plus the broken second entry it describes."""
        first = json.loads(_block_with('"id": "utt_0001"', "json", DATASETS_README))
        second = dict(
            first,
            id="utt_0002",
            consent=False,
            iml=re.sub(r' confidence="[^"]*"', "", first["iml"]),
            audio_file="../secret.wav",
            mood="happy",
        )
        entries = tmp_path / "my-dataset" / "entries"
        entries.mkdir(parents=True)
        for entry in (first, second):
            (entries / f"{entry['id']}.json").write_text(json.dumps(entry), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        code = _block_with('load("my-dataset")', "python", DATASETS_README)
        with pytest.raises(DatasetError) as caught:
            exec(code, {"__name__": "__readme__"})  # noqa: S102
        shown = _block_after('load("my-dataset")', "text", DATASETS_README)
        assert f"DatasetError: {caught.value}\n" == shown

    def test_split_example_runs_on_the_fixture_it_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(ROOT)
        namespace: dict[str, object] = {"__name__": "__readme__"}
        exec(_block_with("loader.split(", "python", DATASETS_README), namespace)  # noqa: S102
        dataset = namespace["dataset"]
        assert isinstance(dataset, Dataset)
        assert len(dataset.entries) == 3
        splits = [namespace[name] for name in ("train", "val", "test")]
        assert sum(len(split) for split in splits if isinstance(split, list)) == 3

    def test_fixture_sizes_are_as_described(self) -> None:
        fixtures = ROOT / "tests" / "fixtures" / "datasets"
        text = DATASETS_README.read_text(encoding="utf-8")
        for name in ("sample", "training_synthetic"):
            count = len(list((fixtures / name / "entries").glob("*.json")))
            assert re.search(rf"datasets/{name}`: {count} ", text), (name, count)

    def test_rule_table_matches_the_loader(self) -> None:
        """Rules and severities in the README table equal those in datasets.py."""
        docstring = datasets.__doc__ or ""
        rules = re.findall(r"^\s+(D\d+)\s.*?(ERROR|WARNING)", docstring, re.M | re.S)
        in_code = {rule: severity.lower() for rule, severity in rules}
        text = DATASETS_README.read_text(encoding="utf-8")
        in_readme = dict(re.findall(r"^\| (D\d+) \| .* \| (error|warning) \|$", text, re.M))
        assert in_readme == in_code
        assert len(in_code) == 12


_LINK = re.compile(r"\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def _anchors(path: Path) -> set[str]:
    """The fragment ids GitHub gives the Markdown headings of *path*."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            continue
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", match.group(1))  # links -> text
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def test_anchor_slugs_follow_github() -> None:
    """Spot-check the slug rules against headings GitHub renders."""
    spec = _anchors(ROOT / "spec.md")
    assert {"82-prohibited-use-cases", "7-user-prosody-profiles"} <= spec
    assert "what-is-real-and-what-is-heuristic" in _anchors(README)


@pytest.mark.parametrize(
    "doc",
    ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "datasets/README.md", "CLAUDE.md"],
)
def test_relative_links_exist(doc: str) -> None:
    """Every relative link points at a file, and every #fragment at a heading."""
    path = ROOT / doc
    missing = []
    for target in _LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+:", target):
            continue
        file_part, _, fragment = target.partition("#")
        linked = (path.parent / file_part) if file_part else path
        if not linked.exists() or (
            fragment and linked.suffix == ".md" and fragment not in _anchors(linked)
        ):
            missing.append(target)
    assert not missing, f"{doc} links to missing files or headings: {missing}"


def test_py_typed_marker_ships_with_the_package() -> None:
    """PEP 561: type checkers use the package's annotations."""
    assert (ROOT / "src" / "prosody_protocol" / "py.typed").exists()
