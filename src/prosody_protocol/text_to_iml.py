"""TextToIML -- predict prosody for plain text (no audio input).

Supports a rule-based baseline and a pluggable ML model backend.
Spec reference: Section 3 (output conforms to IML tag set).
"""

from __future__ import annotations


class TextToIML:
    """Predict prosodic markup for plain text."""

    def __init__(
        self,
        model: str = "rule-based",
        default_confidence: float = 0.6,
    ) -> None:
        self.model = model
        self.default_confidence = default_confidence

    def predict(self, text: str, context: str | None = None) -> str:
        """Predict prosody and return an IML XML string."""
        raise NotImplementedError
