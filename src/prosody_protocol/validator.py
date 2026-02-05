"""IML validator -- checks IML documents against the spec rules.

Implements validation rules V1-V16 as defined in the execution guide:

  V1  Document is well-formed XML                              ERROR
  V2  At least one <utterance> exists                          ERROR
  V3  confidence present when emotion is set                   ERROR
  V4  confidence is float between 0.0 and 1.0                  ERROR
  V5  <pause> has duration attribute                           ERROR
  V6  <pause> duration is a positive integer                   ERROR
  V7  <pause> has no child content                             ERROR
  V8  <emphasis> has level attribute                           ERROR
  V9  <emphasis> level is one of: strong, moderate, reduced    WARNING
  V10 <segment> is direct child of <utterance>                 ERROR
  V11 <segment> not nested in another <segment>                ERROR
  V12 Nesting depth of prosody/emphasis does not exceed 2      WARNING
  V13 pitch value matches valid format                         WARNING
  V14 volume value matches valid format                        WARNING
  V15 emotion is from core vocabulary                          INFO
  V16 No unknown elements present                              INFO

Spec reference: Sections 5-6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lxml import etree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORE_EMOTIONS = frozenset({
    "neutral",
    "sincere",
    "sarcastic",
    "frustrated",
    "joyful",
    "uncertain",
    "angry",
    "sad",
    "fearful",
    "surprised",
    "disgusted",
    "calm",
    "empathetic",
})

_VALID_EMPHASIS_LEVELS = frozenset({"strong", "moderate", "reduced"})

_KNOWN_ELEMENTS = frozenset({"iml", "utterance", "prosody", "pause", "emphasis", "segment"})

# Patterns per spec Section 3.2
_PITCH_RE = re.compile(r"^[+\-]\d+(\.\d+)?(%|st)$|^\d+(\.\d+)?Hz$")
_VOLUME_RE = re.compile(r"^[+\-]\d+(\.\d+)?dB$")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    severity: Literal["error", "warning", "info"]
    rule: str
    message: str
    line: int | None = None
    column: int | None = None


@dataclass
class ValidationResult:
    """Outcome of validating an IML document."""

    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_ns(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _loc(el: etree._Element) -> dict[str, int | None]:
    """Return line/column from an element (if available from the parser)."""
    return {"line": el.sourceline, "column": None}


class _Walker:
    """Stateful tree walker that accumulates validation issues."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def _add(
        self,
        severity: Literal["error", "warning", "info"],
        rule: str,
        message: str,
        el: etree._Element | None = None,
    ) -> None:
        line: int | None = None
        if el is not None:
            line = el.sourceline
        self.issues.append(ValidationIssue(severity=severity, rule=rule, message=message, line=line))

    # -- element visitors ---------------------------------------------------

    def check_utterance(self, el: etree._Element) -> None:
        emotion = el.get("emotion")
        confidence_raw = el.get("confidence")

        # V3: confidence required when emotion is present
        if emotion is not None and confidence_raw is None:
            self._add("error", "V3", f'<utterance> has emotion="{emotion}" but no confidence attribute', el)

        # V4: confidence is float in [0.0, 1.0]
        if confidence_raw is not None:
            try:
                conf = float(confidence_raw)
                if conf < 0.0 or conf > 1.0:
                    self._add("error", "V4", f"confidence={confidence_raw} is outside the valid range [0.0, 1.0]", el)
            except ValueError:
                self._add("error", "V4", f'confidence="{confidence_raw}" is not a valid float', el)

        # V15: emotion from core vocabulary
        if emotion is not None and emotion not in _CORE_EMOTIONS:
            self._add("info", "V15", f'emotion="{emotion}" is not in the core vocabulary', el)

        # Walk children
        self._walk_children(el, parent_tag="utterance", depth=0, in_segment=False)

    def check_prosody(self, el: etree._Element, depth: int, in_segment: bool) -> None:
        # V13: pitch format
        pitch = el.get("pitch")
        if pitch is not None and not _PITCH_RE.match(pitch):
            self._add("warning", "V13", f'pitch="{pitch}" does not match a valid format (+N%, +Nst, NHz)', el)

        # V14: volume format
        volume = el.get("volume")
        if volume is not None and not _VOLUME_RE.match(volume):
            self._add("warning", "V14", f'volume="{volume}" does not match a valid format (+NdB, -NdB)', el)

        # V12: nesting depth
        if depth > 2:
            self._add("warning", "V12", f"<prosody> nesting depth {depth} exceeds recommended max of 2", el)

        self._walk_children(el, parent_tag="prosody", depth=depth, in_segment=in_segment)

    def check_emphasis(self, el: etree._Element, depth: int, in_segment: bool) -> None:
        level = el.get("level")

        # V8: level attribute required
        if level is None:
            self._add("error", "V8", "<emphasis> is missing required attribute 'level'", el)
        elif level not in _VALID_EMPHASIS_LEVELS:
            # V9: known level value
            self._add("warning", "V9", f'<emphasis> level="{level}" is not one of: strong, moderate, reduced', el)

        # V12: nesting depth
        if depth > 2:
            self._add("warning", "V12", f"<emphasis> nesting depth {depth} exceeds recommended max of 2", el)

        self._walk_children(el, parent_tag="emphasis", depth=depth, in_segment=in_segment)

    def check_pause(self, el: etree._Element) -> None:
        duration_raw = el.get("duration")

        # V5: duration attribute required
        if duration_raw is None:
            self._add("error", "V5", "<pause> is missing required attribute 'duration'", el)
        else:
            # V6: positive integer
            try:
                val = int(duration_raw)
                if val <= 0:
                    self._add("error", "V6", f"<pause> duration={duration_raw} must be a positive integer", el)
            except ValueError:
                self._add("error", "V6", f'<pause> duration="{duration_raw}" is not a valid integer', el)

        # V7: no child content
        if el.text and el.text.strip():
            self._add("error", "V7", "<pause> must be a self-closing empty element but has text content", el)
        if len(el) > 0:
            self._add("error", "V7", "<pause> must be a self-closing empty element but has child elements", el)

    def check_segment(self, el: etree._Element, parent_tag: str, in_segment: bool) -> None:
        # V10: must be direct child of <utterance>
        if parent_tag != "utterance":
            self._add("error", "V10", f"<segment> must be a direct child of <utterance>, found inside <{parent_tag}>", el)

        # V11: not nested in another segment
        if in_segment:
            self._add("error", "V11", "<segment> must not be nested inside another <segment>", el)

        self._walk_children(el, parent_tag="segment", depth=0, in_segment=True)

    def _walk_children(
        self,
        el: etree._Element,
        parent_tag: str,
        depth: int,
        in_segment: bool,
    ) -> None:
        for child in el:
            if isinstance(child, etree._Comment):
                continue
            tag = _strip_ns(child.tag)  # type: ignore[arg-type]

            # V16: unknown elements
            if tag not in _KNOWN_ELEMENTS:
                self._add("info", "V16", f"Unknown element <{tag}>", child)
                continue

            if tag == "utterance":
                self.check_utterance(child)
            elif tag == "prosody":
                self.check_prosody(child, depth=depth + 1, in_segment=in_segment)
            elif tag == "emphasis":
                self.check_emphasis(child, depth=depth + 1, in_segment=in_segment)
            elif tag == "pause":
                self.check_pause(child)
            elif tag == "segment":
                self.check_segment(child, parent_tag=parent_tag, in_segment=in_segment)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLValidator:
    """Validate IML documents against the specification."""

    def validate(self, iml_string: str) -> ValidationResult:
        """Validate an IML XML string.

        Returns a :class:`ValidationResult` with ``valid=True`` when no
        errors are found (warnings and info issues are allowed).
        """
        result = ValidationResult()

        # V1: well-formed XML
        try:
            root = etree.fromstring(iml_string.encode("utf-8"))  # noqa: S320
        except etree.XMLSyntaxError as exc:
            result.valid = False
            result.issues.append(
                ValidationIssue(
                    severity="error",
                    rule="V1",
                    message=f"Malformed XML: {exc}",
                    line=getattr(exc, "lineno", None),
                )
            )
            return result

        walker = _Walker()
        root_tag = _strip_ns(root.tag)

        if root_tag == "iml":
            # Check for at least one utterance
            utterance_children = [
                c for c in root
                if not isinstance(c, etree._Comment) and _strip_ns(c.tag) == "utterance"  # type: ignore[arg-type]
            ]
            if not utterance_children:
                walker._add("error", "V2", "<iml> contains no <utterance> elements", root)
            for child in root:
                if isinstance(child, etree._Comment):
                    continue
                tag = _strip_ns(child.tag)  # type: ignore[arg-type]
                if tag == "utterance":
                    walker.check_utterance(child)
                elif tag not in _KNOWN_ELEMENTS:
                    walker._add("info", "V16", f"Unknown element <{tag}>", child)
        elif root_tag == "utterance":
            walker.check_utterance(root)
        else:
            walker._add("error", "V2", f"Expected root element <iml> or <utterance>, got <{root_tag}>", root)

        result.issues = walker.issues
        result.valid = not any(i.severity == "error" for i in result.issues)
        return result

    def validate_file(self, path: str | Path) -> ValidationResult:
        """Validate an IML XML file from disk."""
        p = Path(path)
        return self.validate(p.read_text(encoding="utf-8"))
