"""Use a trained SER baseline as the SDK's emotion classifier.

:class:`TrainedEmotionClassifier` implements the
:class:`~prosody_protocol.emotion_classifier.EmotionClassifier` protocol, so
a model trained with ``training/scripts/train.py`` can label the utterances
that :class:`~prosody_protocol.AudioToIML` produces::

    from prosody_protocol import AudioToIML
    from training.inference import TrainedEmotionClassifier

    # A directory written by: export.py --checkpoint ... --format json
    classifier = TrainedEmotionClassifier("exports/ser")
    converter = AudioToIML(emotion_classifier=classifier)
    print(converter.convert("clip.wav", transcript="I can't believe it"))

The classifier summarises the utterance's word features with the same
function data preparation uses (:func:`training.features.feature_vector`),
so the model sees the kind of input it was trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from prosody_protocol import SpanFeatures

from .features import SER_FEATURES, feature_vector, has_voiced_speech
from .portable import PortableModel


class TrainedEmotionClassifier:
    """Emotion classifier backed by a trained SER model.

    Parameters
    ----------
    model:
        A JSON export: the directory ``export.py --format json`` wrote, its
        ``model.json`` file, or a loaded
        :class:`~training.portable.PortableModel`. Loading a JSON export
        never runs code from the file. For a pickled checkpoint, use
        :meth:`from_checkpoint`.

    The confidence :meth:`classify` returns is the model's probability for
    the label. It is not calibrated: a model trained on little data can be
    confidently wrong, especially on recordings unlike its training data
    (another microphone, another speaker). Utterances without voiced
    speech get ``("neutral", 0.0)``, which
    :class:`~prosody_protocol.assembler.IMLAssembler`'s confidence
    threshold turns into no emotion at all.
    """

    def __init__(self, model: str | Path | PortableModel) -> None:
        if not isinstance(model, PortableModel):
            model = PortableModel.load(model)
        unknown = [n for n in model.feature_names if n not in SER_FEATURES]
        if unknown:
            raise ValueError(
                f"The model was not trained on SER features: {unknown} are not among "
                f"{list(SER_FEATURES)}"
            )
        self.model = model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> TrainedEmotionClassifier:
        """Build a classifier from a training checkpoint (``model.joblib``).

        .. warning::
           The checkpoint is a pickle, and loading it runs code from the
           file. Only use checkpoints you trust; share JSON exports instead.

        Needs scikit-learn and joblib (the ``ml`` extra).
        """
        from .models import BaseModel

        return cls(PortableModel(BaseModel.load(path).portable_params()))

    @property
    def labels(self) -> tuple[str, ...]:
        """The emotion labels the model can return."""
        return self.model.classes

    def probabilities(self, features: Sequence[SpanFeatures]) -> dict[str, float]:
        """The model's probability for each label, given an utterance's span features.

        All zero when the spans contain no voiced speech.
        """
        if not has_voiced_speech(features):
            return {label: 0.0 for label in self.labels}
        vector = feature_vector(features, self.model.feature_names)
        proba = self.model.predict_proba(vector)[0]
        return {label: float(p) for label, p in zip(self.labels, proba, strict=True)}

    def classify(self, features: list[SpanFeatures]) -> tuple[str, float]:
        """Classify the emotion of an utterance from its span features.

        Returns ``(label, confidence)`` with the confidence rounded to two
        decimals, or ``("neutral", 0.0)`` when the spans contain no voiced
        speech.
        """
        probabilities = self.probabilities(features)
        label = max(probabilities, key=lambda k: probabilities[k])
        if probabilities[label] == 0.0:
            return ("neutral", 0.0)
        return (label, round(probabilities[label], 2))
