"""Execute the code examples in the user-facing Markdown docs.

Every fenced block in README.md, docs/**/*.md and examples/README.md is
checked so the documentation cannot drift from the code:

- ``python`` blocks of one file run in order in a shared namespace, in a
  scratch working directory where ``examples/`` points at the repository's
  examples directory;
- ``xml`` blocks whose root element is ``<iml>`` or ``<utterance>`` must
  pass :class:`~prosody_protocol.IMLValidator`; fragments rooted at
  ``<prosody>``, ``<emphasis>``, ``<pause>`` or ``<segment>`` are checked
  inside an ``<utterance>``;
- ``bash``/``shell`` lines, and ``console`` lines after a ``$ `` prompt,
  that start with ``prosody-protocol`` run through
  :func:`prosody_protocol.cli.main` and must exit 0.

An HTML comment on the line directly above a fence changes this:

- ``<!-- docs-test: skip -->`` -- not run (e.g. it calls an external service);
- ``<!-- docs-test: invalid -->`` -- an IML example that must NOT validate.

Blocks that need an optional extra are skipped when it is not installed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest

from prosody_protocol import IMLValidator

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = sorted(
    [ROOT / "README.md", ROOT / "examples" / "README.md", *(ROOT / "docs").rglob("*.md")]
)
DOC_FILES = [p for p in DOC_FILES if p.exists()]

# Modules whose absence means "extra not installed" rather than a doc bug.
# The modules each optional extra provides. A failure that names an extra is
# a skip only when that extra really is missing here; otherwise it is a bug.
EXTRA_MODULES = {
    "audio": ("numpy", "parselmouth"),
    "whisper": ("whisper",),
    "ml": ("sklearn", "yaml", "joblib"),
    "api": ("fastapi", "uvicorn", "multipart"),
}
OPTIONAL_MODULES = {m for mods in EXTRA_MODULES.values() for m in mods} | {"httpx"}
_EXTRA_IN_TEXT = re.compile(r"prosody-protocol\[(\w+)\]")


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def _missing_extra_named_in(text: str) -> str | None:
    """The first extra named in *text* whose modules are not installed."""
    for extra in _EXTRA_IN_TEXT.findall(text):
        if any(_missing(m) for m in EXTRA_MODULES.get(extra, ())):
            return extra
    return None

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^`\n]*)$")
_DIRECTIVE = re.compile(r"<!--\s*docs-test:\s*(?P<what>[\w-]+)\s*-->")


@dataclass(frozen=True)
class Block:
    path: Path
    line: int  # 1-based line of the opening fence
    lang: str
    code: str
    directive: str | None

    def where(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}"


def extract_blocks(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, indent = m.group("fence"), m.group("indent")
        lang = m.group("info").strip().split(" ")[0].lower() if m.group("info").strip() else ""
        directive = None
        if i > 0:
            d = _DIRECTIVE.search(lines[i - 1])
            if d:
                directive = d.group("what")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith(fence[0] * len(fence)):
            line = lines[j]
            body.append(line[len(indent):] if line.startswith(indent) else line)
            j += 1
        blocks.append(Block(path, i + 1, lang, "\n".join(body), directive))
        i = j + 1
    return blocks


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(ROOT)) for p in paths]


@pytest.fixture()
def docs_cwd(tmp_path: Path) -> object:
    """Run examples in a scratch directory that can see ``examples/``."""
    examples = ROOT / "examples"
    if examples.exists():
        (tmp_path / "examples").symlink_to(examples, target_is_directory=True)
    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old)


def _skip_if_optional_missing(exc: BaseException) -> None:
    if isinstance(exc, ImportError):
        name = (getattr(exc, "name", None) or "").split(".")[0]
        if (name in OPTIONAL_MODULES and _missing(name)) or _missing_extra_named_in(str(exc)):
            pytest.skip(f"optional dependency missing: {exc}")


@pytest.mark.parametrize("path", DOC_FILES, ids=_ids(DOC_FILES))
def test_python_examples_run(path: Path, docs_cwd: Path) -> None:
    blocks = [b for b in extract_blocks(path) if b.lang in ("python", "py")]
    runnable = [b for b in blocks if b.directive != "skip"]
    if not runnable:
        pytest.skip("no runnable python examples")
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for block in runnable:
        try:
            code = compile(block.code, block.where(), "exec")
            exec(code, namespace)  # noqa: S102 - executing our own documentation
        except BaseException as exc:  # noqa: BLE001
            _skip_if_optional_missing(exc)
            raise AssertionError(
                f"example at {block.where()} failed: {type(exc).__name__}: {exc}"
            ) from exc


def _iml_root(code: str) -> str | None:
    stripped = re.sub(r"<\?xml[^>]*\?>", "", code).lstrip()
    m = re.match(r"<([A-Za-z_][\w.-]*)", stripped)
    return m.group(1) if m else None


DOCUMENT_ROOTS = ("iml", "utterance")
FRAGMENT_ROOTS = ("prosody", "emphasis", "pause", "segment")

IML_BLOCKS = [
    b
    for p in DOC_FILES
    for b in extract_blocks(p)
    if b.lang == "xml"
    and b.directive != "skip"
    and _iml_root(b.code) in DOCUMENT_ROOTS + FRAGMENT_ROOTS
]


@pytest.mark.parametrize("block", IML_BLOCKS, ids=[b.where() for b in IML_BLOCKS])
def test_iml_examples_validate(block: Block) -> None:
    code = block.code
    if _iml_root(code) in FRAGMENT_ROOTS:
        code = f"<utterance>{code}</utterance>"
    result = IMLValidator().validate(code)
    if block.directive == "invalid":
        assert not result.valid, f"{block.where()} is marked invalid but validates"
        return
    errors = [f"{i.rule}: {i.message}" for i in result.issues if i.severity == "error"]
    assert result.valid, f"IML example at {block.where()} is invalid: {errors}"


def _cli_commands(block: Block) -> list[list[str]]:
    commands: list[list[str]] = []
    pending = ""
    for raw in block.code.splitlines():
        line = raw.strip()
        if line.startswith("$ "):
            line = line[2:]
        elif block.lang == "console" and not pending:
            continue  # console blocks mix commands ("$ ...") with their output
        if pending:
            line = pending + " " + line
            pending = ""
        if line.endswith("\\"):
            pending = line[:-1].strip()
            continue
        if not line.startswith("prosody-protocol"):
            continue
        line = line.split(" #", 1)[0]
        argv = shlex.split(line)[1:]
        if any(tok in ("|", ">", "<", "&&") for tok in argv):
            argv = argv[: min(argv.index(t) for t in ("|", ">", "<", "&&") if t in argv)]
        commands.append(argv)
    return commands


CLI_BLOCKS = [
    b
    for p in DOC_FILES
    for b in extract_blocks(p)
    if b.lang in ("bash", "sh", "shell", "console") and b.directive != "skip" and _cli_commands(b)
]


@pytest.mark.parametrize("block", CLI_BLOCKS, ids=[b.where() for b in CLI_BLOCKS])
def test_cli_examples_run(block: Block, docs_cwd: Path) -> None:
    from prosody_protocol.cli import main

    for argv in _cli_commands(block):
        if argv and argv[0] == "serve":
            continue
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:  # argparse --help / errors
                code = exc.code if isinstance(exc.code, int) else 1
        if code != 0 and _missing_extra_named_in(err.getvalue()):
            pytest.skip(f"optional extra missing for {argv}: {err.getvalue().strip()}")
        assert code == 0, (
            f"`prosody-protocol {' '.join(argv)}` at {block.where()} exited {code}: "
            f"{err.getvalue().strip()}"
        )
