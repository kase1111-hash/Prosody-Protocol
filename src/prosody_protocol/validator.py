"""IML validator -- checks IML documents against the spec rules.

Implements validation rules V1-V32. Violations of a spec MUST (including
6.3's "valid attribute types") are errors and make a document invalid;
SHOULD violations are warnings; info issues are notes that never affect
validity.

  V1  Document is well-formed XML                              ERROR
  V2  Root is <iml> or <utterance>; at least one <utterance>   ERROR
  V3  confidence present when emotion is set                   ERROR
  V4  confidence is a float between 0.0 and 1.0                ERROR
  V5  <pause> has duration attribute                           ERROR
  V6  <pause> duration is a positive integer (<= 2147483647)   ERROR
  V7  <pause> has no elements or non-whitespace text           ERROR
  V8  <emphasis> has level attribute                           ERROR
  V9  <emphasis> level is one of: strong, moderate, reduced    ERROR
  V10 <segment> is direct child of <utterance>                 ERROR
  V11 <segment> not nested in another <segment>                ERROR
  V12 Nesting depth of prosody/emphasis does not exceed 2      WARNING
  V13 pitch value matches valid format                         ERROR
  V14 volume value matches valid format                        ERROR
  V15 emotion is from core vocabulary                          INFO
  V16 No unknown elements present                              INFO
  V17 consent is "explicit", "implicit" or "none"              ERROR
  V18 processing is "local", "remote" or "hybrid"              ERROR
  V19 <iml> contains only <utterance> elements                 ERROR
  V20 <utterance>/<iml> not nested inside another element      ERROR
  V21 <emphasis> not a direct child of <emphasis>              ERROR
  V22 rate is fast, slow, medium or a percentage               ERROR
  V23 pitch_contour value is from the spec vocabulary          ERROR
  V24 quality value is from the spec vocabulary                ERROR
  V25 tempo value is from the spec vocabulary                  ERROR
  V26 rhythm value is from the spec vocabulary                 ERROR
  V27 extended attribute values have their Section 4 type      ERROR
  V28 version is a Semantic Version                            ERROR
  V29 language is a BCP 47 tag                                 ERROR
  V30 document is encoded in UTF-8                             ERROR
  V31 document has no DOCTYPE declaration                      ERROR
  V32 attributes are IML-defined, x- prefixed or in a foreign  WARNING
      namespace (not unprefixed-unknown, not in the IML namespace)

Unknown elements are transparent (spec 6.2): the rules apply to their
content as if their tags were absent. The exception is <pause>, which
must not contain any element (V7). Comments and processing instructions
are ignored.

Spec reference: Sections 2-6, 8, 9.2.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from lxml import etree

from .exceptions import IMLValidationError
from .parser import (
    _IML_ELEMENTS,
    _SECURE_PARSER,
    IML_NAMESPACE,
    MAX_INTEGER,
    _declared_encoding,
    _display_name,
    _has_doctype,
    _iml_name,
    _is_blank,
    _is_utf8_name,
    _iter_content,
    _iter_iml_content,
    _parse_float,
    _parse_positive_int,
    _snippet,
    _syntax_error_location,
    _utf8_error,
)

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
_VALID_PITCH_CONTOURS = frozenset(
    {"rise", "fall", "rise-fall", "fall-rise", "fall-sharp", "rise-sharp", "flat"}
)
_VALID_QUALITIES = frozenset({"modal", "breathy", "tense", "creaky", "whispery", "harsh"})
_VALID_NAMED_RATES = frozenset({"fast", "slow", "medium"})
_VALID_TEMPOS = frozenset({"rushed", "steady", "drawn-out"})
_VALID_RHYTHMS = frozenset({"staccato", "legato", "syncopated"})

_VALID_CONSENT_VALUES = frozenset({"explicit", "implicit", "none"})
_VALID_PROCESSING_VALUES = frozenset({"local", "remote", "hybrid"})

# Patterns per spec Sections 2.4, 2.6, 3.2 and 4 (ASCII digits only).
_PITCH_RE = re.compile(r"[+-][0-9]+(?:\.[0-9]+)?(?:%|st)|[0-9]+(?:\.[0-9]+)?Hz")
_VOLUME_RE = re.compile(r"[+-][0-9]+(?:\.[0-9]+)?dB")
_RATE_PERCENT_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?%")
_F0_RANGE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?")
_F0_CONTOUR_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:,[0-9]+(?:\.[0-9]+)?)*")
_SEMVER_RE = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_BCP47_RE = re.compile(r"[A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*")

# Attributes each IML element defines; anything else must be an x- or
# namespaced extension attribute (spec 9.2).
_EXTENDED_ATTRIBUTES = (
    "f0_mean", "f0_range", "f0_contour", "intensity_mean", "intensity_range",
    "speech_rate", "duration_ms", "jitter", "shimmer", "hnr",
)
_KNOWN_ATTRIBUTES: dict[str, frozenset[str]] = {
    "iml": frozenset({"version", "language", "consent", "processing"}),
    "utterance": frozenset({"emotion", "confidence", "speaker_id"}),
    "prosody": frozenset(
        {"pitch", "pitch_contour", "volume", "rate", "quality", *_EXTENDED_ATTRIBUTES}
    ),
    "pause": frozenset({"duration"}),
    "emphasis": frozenset({"level"}),
    "segment": frozenset({"tempo", "rhythm"}),
}

# Attributes in the IML namespace are not IML attributes (spec 2.3).
_IML_ATTRIBUTE_PREFIX = f"{{{IML_NAMESPACE}}}"


def _is_number(raw: str) -> bool:
    return _parse_float(raw) is not None


def _is_non_negative_number(raw: str) -> bool:
    value = _parse_float(raw)
    return value is not None and value >= 0.0


# Extended attribute checks (Section 4): predicate and expected-value text.
_EXTENDED_CHECKS: dict[str, tuple[Callable[[str], bool], str]] = {
    "f0_mean": (_is_non_negative_number, "a non-negative number of Hz"),
    "f0_range": (lambda raw: bool(_F0_RANGE_RE.fullmatch(raw)), '"low-high" in Hz, e.g. "120-240"'),
    "f0_contour": (
        lambda raw: bool(_F0_CONTOUR_RE.fullmatch(raw)),
        'comma-separated Hz values, e.g. "150,165,180"',
    ),
    "intensity_mean": (_is_number, "a number of dB"),
    "intensity_range": (_is_non_negative_number, "a non-negative number of dB"),
    "speech_rate": (_is_non_negative_number, "a non-negative number of syllables/second"),
    "duration_ms": (
        lambda raw: _parse_positive_int(raw) is not None,
        f"a positive integer of ms, at most {MAX_INTEGER}",
    ),
    "jitter": (_is_non_negative_number, "a non-negative percentage"),
    "shimmer": (_is_non_negative_number, "a non-negative percentage"),
    "hnr": (_is_number, "a number of dB"),
}


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

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues with severity ``"error"``; any one makes the document invalid."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Issues with severity ``"warning"`` (SHOULD-level; the document stays valid)."""
        return [i for i in self.issues if i.severity == "warning"]

    def raise_for_errors(self) -> None:
        """Raise :class:`~prosody_protocol.exceptions.IMLValidationError` if invalid.

        The exception's ``issues`` are the error issues; its message lists them.
        Does nothing for a valid result.
        """
        errors = self.errors
        if self.valid and not errors:
            return
        shown = "; ".join(
            f"{i.rule}: {i.message}" + (f" (line {i.line})" if i.line is not None else "")
            for i in errors[:5]
        )
        if len(errors) > 5:
            shown += f"; and {len(errors) - 5} more"
        raise IMLValidationError(
            f"IML document is invalid: {shown or 'no error details'}", issues=errors
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _line(el: etree._Element) -> int | None:
    """Return the line of *el* (lxml-stubs mistypes ``sourceline``)."""
    return cast("int | None", el.sourceline)


def _attribute_display_name(el: etree._Element, name: str) -> str:
    """Return the attribute *name* (a Clark name) with a prefix bound on *el*."""
    uri, local = name[1:].split("}", 1)
    prefix = next((p for p, u in el.nsmap.items() if p and u == uri), None)
    return f"{prefix}:{local}" if prefix else name


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
        line = _line(el) if el is not None else None
        self.issues.append(
            ValidationIssue(severity=severity, rule=rule, message=message, line=line)
        )

    def _check_enum(
        self, el: etree._Element, attr: str, allowed: frozenset[str], rule: str
    ) -> None:
        value = el.get(attr)
        if value is not None and value not in allowed:
            expected = ", ".join(sorted(allowed))
            self._add("error", rule, f'{attr}="{value}" is not one of: {expected}', el)

    # -- document-level checks ----------------------------------------------

    def check_iml(self, root: etree._Element) -> None:
        # V17: consent attribute value
        consent = root.get("consent")
        if consent is not None and consent not in _VALID_CONSENT_VALUES:
            self._add(
                "error", "V17",
                f'consent="{consent}" is not a recognized value (expected: explicit, '
                "implicit, none)",
                root,
            )

        # V18: processing attribute value
        processing = root.get("processing")
        if processing is not None and processing not in _VALID_PROCESSING_VALUES:
            self._add(
                "error", "V18",
                f'processing="{processing}" is not a recognized value (expected: local, '
                "remote, hybrid)",
                root,
            )

        # V28: version is a Semantic Version
        version = root.get("version")
        if version is not None and not _SEMVER_RE.fullmatch(version):
            self._add(
                "error", "V28",
                f'version="{version}" is not a Semantic Version (e.g. "0.1.0")', root,
            )

        # V29: language is a BCP 47 tag
        language = root.get("language")
        if language is not None and not _BCP47_RE.fullmatch(language):
            self._add(
                "error", "V29",
                f'language="{language}" is not a BCP 47 language tag (e.g. "en-US")', root,
            )

        has_utterance = False
        for item in _iter_iml_content(root):
            if isinstance(item, str):
                # V19: no text outside utterances
                if not _is_blank(item):
                    self._add(
                        "error", "V19",
                        f"<iml> contains text outside any <utterance>: {_snippet(item)!r}",
                        root,
                    )
                continue
            tag = _iml_name(item)
            if tag == "utterance":
                has_utterance = True
                self.check_utterance(item)
            elif tag == "iml":
                # V20: <iml> is only the root
                self._add("error", "V20", "<iml> must not be nested inside another element", item)
            else:
                # V19: only utterances directly in <iml>
                self._add(
                    "error", "V19",
                    f"<{tag}> must be inside an <utterance>, not directly inside <iml>", item,
                )

        # V2: at least one utterance
        if not has_utterance:
            self._add("error", "V2", "<iml> contains no <utterance> elements", root)

    def check_unknown(self, root: etree._Element) -> None:
        """Report unknown elements (V16) and unknown attributes (V32) anywhere."""
        for el in root.iter():
            tag = _iml_name(el)
            if tag is None:
                continue
            if tag not in _IML_ELEMENTS:
                self._add("info", "V16", f"Unknown element <{_display_name(el)}> is ignored", el)
                continue
            known = _KNOWN_ATTRIBUTES[tag]
            for attr in el.attrib:
                name = str(attr)
                if name.startswith(_IML_ATTRIBUTE_PREFIX):
                    self._add(
                        "warning", "V32",
                        f"Attribute '{_attribute_display_name(el, name)}' on <{tag}> is "
                        "ignored; IML attributes are never namespace-qualified (spec 2.3)",
                        el,
                    )
                    continue
                if name in known or name.startswith(("x-", "{")):
                    continue
                self._add(
                    "warning", "V32",
                    f"Unknown attribute '{name}' on <{tag}> is ignored; custom attributes "
                    "use the x- prefix (spec 9.2)",
                    el,
                )

    # -- element visitors ---------------------------------------------------

    def check_utterance(self, el: etree._Element) -> None:
        emotion = el.get("emotion")
        confidence_raw = el.get("confidence")

        # V3: confidence required when emotion is present
        if emotion is not None and confidence_raw is None:
            self._add(
                "error", "V3",
                f'<utterance> has emotion="{emotion}" but no confidence attribute',
                el,
            )

        # V4: confidence is float in [0.0, 1.0]
        if confidence_raw is not None:
            conf = _parse_float(confidence_raw)
            if conf is None:
                self._add("error", "V4", f'confidence="{confidence_raw}" is not a valid float', el)
            elif conf < 0.0 or conf > 1.0:
                self._add(
                    "error", "V4",
                    f"confidence={confidence_raw} is outside the valid range [0.0, 1.0]",
                    el,
                )

        # V15: emotion from core vocabulary
        if emotion is not None and emotion not in _CORE_EMOTIONS:
            self._add("info", "V15", f'emotion="{emotion}" is not in the core vocabulary', el)

        # Walk children
        self._walk_children(el, parent_tag="utterance", depth=0, in_segment=False)

    def check_prosody(self, el: etree._Element, depth: int, in_segment: bool) -> None:
        # V13: pitch format
        pitch = el.get("pitch")
        if pitch is not None and not _PITCH_RE.fullmatch(pitch):
            self._add(
                "error", "V13",
                f'pitch="{pitch}" does not match a valid format (+N%, +Nst, NHz)',
                el,
            )

        # V14: volume format
        volume = el.get("volume")
        if volume is not None and not _VOLUME_RE.fullmatch(volume):
            self._add(
                "error", "V14",
                f'volume="{volume}" does not match a valid format (+NdB, -NdB)',
                el,
            )

        # V22: rate is a named value or a percentage
        rate = el.get("rate")
        if (
            rate is not None
            and rate not in _VALID_NAMED_RATES
            and not _RATE_PERCENT_RE.fullmatch(rate)
        ):
            self._add(
                "error", "V22",
                f'rate="{rate}" is not fast, slow, medium or a percentage (N%)',
                el,
            )

        # V23, V24: pitch_contour and quality vocabularies
        self._check_enum(el, "pitch_contour", _VALID_PITCH_CONTOURS, "V23")
        self._check_enum(el, "quality", _VALID_QUALITIES, "V24")

        # V27: extended attributes (Section 4)
        for attr, (is_valid, expected) in _EXTENDED_CHECKS.items():
            raw = el.get(attr)
            if raw is not None and not is_valid(raw):
                self._add(
                    "error", "V27",
                    f'{attr}="{_snippet(raw)}" is not valid (expected {expected})', el,
                )

        # V12: nesting depth
        if depth > 2:
            self._add(
                "warning", "V12",
                f"<prosody> nesting depth {depth} exceeds recommended max of 2",
                el,
            )

        self._walk_children(el, parent_tag="prosody", depth=depth, in_segment=in_segment)

    def check_emphasis(
        self, el: etree._Element, parent_tag: str, depth: int, in_segment: bool
    ) -> None:
        level = el.get("level")

        # V8: level attribute required
        if level is None:
            self._add("error", "V8", "<emphasis> is missing required attribute 'level'", el)
        elif level not in _VALID_EMPHASIS_LEVELS:
            # V9: known level value
            self._add(
                "error", "V9",
                f'<emphasis> level="{level}" is not one of: strong, moderate, reduced',
                el,
            )

        # V21: no emphasis directly inside emphasis
        if parent_tag == "emphasis":
            self._add(
                "error", "V21",
                "<emphasis> must not be a direct child of another <emphasis>",
                el,
            )

        # V12: nesting depth
        if depth > 2:
            self._add(
                "warning", "V12",
                f"<emphasis> nesting depth {depth} exceeds recommended max of 2",
                el,
            )

        self._walk_children(el, parent_tag="emphasis", depth=depth, in_segment=in_segment)

    def check_pause(self, el: etree._Element) -> None:
        duration_raw = el.get("duration")

        # V5: duration attribute required
        if duration_raw is None:
            self._add("error", "V5", "<pause> is missing required attribute 'duration'", el)
        elif _parse_positive_int(duration_raw) is None:
            # V6: positive integer
            self._add(
                "error", "V6",
                f'<pause> duration="{_snippet(duration_raw)}" must be a positive integer '
                f"of ms, at most {MAX_INTEGER}",
                el,
            )

        # V7: no content. Text inside unknown elements counts as text, and
        # unlike elsewhere an unknown element is itself content (spec 3.3).
        if any(not _is_blank(text) for text in _iter_content(el) if isinstance(text, str)):
            self._add(
                "error", "V7",
                "<pause> must be a self-closing empty element but has text content",
                el,
            )
        if any(_iml_name(child) is not None for child in el):
            self._add(
                "error", "V7",
                "<pause> must be a self-closing empty element but has child elements",
                el,
            )

    def check_segment(self, el: etree._Element, parent_tag: str, in_segment: bool) -> None:
        # V10: must be direct child of <utterance>
        if parent_tag != "utterance":
            self._add(
                "error", "V10",
                f"<segment> must be a direct child of <utterance>, found inside <{parent_tag}>",
                el,
            )

        # V11: not nested in another segment
        if in_segment:
            self._add("error", "V11", "<segment> must not be nested inside another <segment>", el)

        # V25, V26: tempo and rhythm vocabularies
        self._check_enum(el, "tempo", _VALID_TEMPOS, "V25")
        self._check_enum(el, "rhythm", _VALID_RHYTHMS, "V26")

        self._walk_children(el, parent_tag="segment", depth=0, in_segment=True)

    def _walk_children(
        self,
        el: etree._Element,
        parent_tag: str,
        depth: int,
        in_segment: bool,
    ) -> None:
        # _iter_content skips comments and processing instructions and looks
        # through unknown elements, so parent_tag is the nearest IML ancestor.
        for child in _iter_content(el, text=False):
            if isinstance(child, str):
                continue
            tag = _iml_name(child)

            if tag in ("utterance", "iml"):
                # V20: <utterance> and <iml> only at the top level
                self._add(
                    "error", "V20",
                    f"<{tag}> must not be nested inside <{parent_tag}>",
                    child,
                )
                if tag == "utterance":
                    self.check_utterance(child)
            elif tag == "prosody":
                self.check_prosody(child, depth=depth + 1, in_segment=in_segment)
            elif tag == "emphasis":
                self.check_emphasis(
                    child, parent_tag=parent_tag, depth=depth + 1, in_segment=in_segment
                )
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

        # V30: the text must be encodable as UTF-8 (no lone surrogates).
        try:
            data = iml_string.encode("utf-8")
        except UnicodeEncodeError as exc:
            return self._fail("V30", f"Document cannot be encoded as UTF-8: {exc.reason}")

        # V1: well-formed XML
        try:
            root = etree.fromstring(data, parser=_SECURE_PARSER)  # noqa: S320
        except etree.XMLSyntaxError as exc:
            line, column = _syntax_error_location(exc)
            return self._fail("V1", f"Malformed XML: {exc}", line, column)

        walker = _Walker()

        # V30: a declared encoding must be UTF-8 (spec M1).
        encoding = _declared_encoding(iml_string)
        if encoding is not None and not _is_utf8_name(encoding):
            walker._add(
                "error", "V30",
                f'Document declares encoding="{encoding}"; IML must be UTF-8',
            )

        # V31: no DOCTYPE (spec 2.5)
        if _has_doctype(root):
            walker._add("error", "V31", "Document must not contain a DOCTYPE declaration")

        root_tag = _iml_name(root)
        try:
            if root_tag == "iml":
                walker.check_iml(root)
            elif root_tag == "utterance":
                walker.check_utterance(root)
            else:
                walker._add(
                    "error", "V2",
                    f"Expected root element <iml> or <utterance>, got <{_display_name(root)}>",
                    root,
                )
            walker.check_unknown(root)
        except RecursionError:
            walker._add("error", "V1", "Document is nested too deeply to validate")

        result.issues = walker.issues
        result.valid = not any(i.severity == "error" for i in result.issues)
        return result

    def validate_file(self, path: str | Path) -> ValidationResult:
        """Validate an IML XML file from disk.

        A file that is not valid UTF-8 gets a V30 error rather than an
        exception; :class:`OSError` is raised when it cannot be read.
        """
        data = Path(path).read_bytes()
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            return self._fail("V30", *_utf8_error(data, exc))
        return self.validate(text)

    @staticmethod
    def _fail(
        rule: str, message: str, line: int | None = None, column: int | None = None
    ) -> ValidationResult:
        issue = ValidationIssue(
            severity="error", rule=rule, message=message, line=line, column=column
        )
        return ValidationResult(valid=False, issues=[issue])
