"""Tests for Phase 13: Documentation & Adoption.

Verifies the adoption checklist from the execution guide:
- pip install prosody-protocol works (import test)
- from prosody_protocol import IMLParser, IMLValidator works
- README quick-start code runs without modification
- At least 3 integration examples are tested end-to-end
- API server is deployable via Docker (Dockerfile exists and is valid)
- Benchmark results published in README or docs
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Adoption Checklist Tests
# ---------------------------------------------------------------------------


class TestAdoptionChecklist:
    """Verify the 6-point adoption checklist."""

    def test_sdk_import_works(self):
        """AC1: 'from prosody_protocol import IMLParser, IMLValidator' works."""
        from prosody_protocol import IMLParser, IMLValidator

        assert IMLParser is not None
        assert IMLValidator is not None

    def test_all_public_classes_importable(self):
        """All classes listed in __all__ should be importable."""
        from prosody_protocol import (
            AudioToIML,
            Benchmark,
            BenchmarkReport,
            Dataset,
            DatasetEntry,
            DatasetLoader,
            EmotionClassifier,
            Emphasis,
            IMLAssembler,
            IMLDocument,
            IMLParser,
            IMLToAudio,
            IMLToSSML,
            IMLValidator,
            Pause,
            PauseInterval,
            ProfileApplier,
            ProfileLoader,
            Prosody,
            ProsodyAnalyzer,
            ProsodyMapping,
            ProsodyProfile,
            RuleBasedEmotionClassifier,
            Segment,
            SpanFeatures,
            TextToIML,
            Utterance,
            ValidationIssue,
            ValidationResult,
            WordAlignment,
        )

        # Just verify they're all real classes/objects
        assert IMLParser is not None
        assert Benchmark is not None
        assert DatasetLoader is not None

    def test_readme_parse_example(self):
        """AC3: README quick-start code (parse) runs without modification."""
        from prosody_protocol import IMLParser

        parser = IMLParser()
        doc = parser.parse('''
<utterance emotion="sarcastic" confidence="0.87">
  Oh, that's
  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">
    GREAT
  </prosody>.
</utterance>
''')
        assert doc.utterances[0].emotion == "sarcastic"
        plain = parser.to_plain_text(doc)
        assert "GREAT" in plain

    def test_readme_validate_example(self):
        """AC3: README quick-start code (validate) runs without modification."""
        from prosody_protocol import IMLValidator

        validator = IMLValidator()
        result = validator.validate(
            '<utterance emotion="happy">'
            '<prosody pitch="+20%">This is great!</prosody>'
            '</utterance>'
        )
        # emotion without confidence should fail validation
        assert not result.valid

    def test_readme_text_to_iml_example(self):
        """AC3: README quick-start code (TextToIML) runs without modification."""
        from prosody_protocol import TextToIML

        predictor = TextToIML()
        iml = predictor.predict("I can't believe this happened.", context="frustrated")
        assert "<utterance" in iml
        assert "</utterance>" in iml

    def test_integration_whisper_example(self):
        """AC4: Whisper integration example runs (SDK parts)."""
        from prosody_protocol import AudioToIML, IMLParser, IMLValidator

        # Verify the converter can be instantiated
        converter = AudioToIML()
        parser = IMLParser()
        validator = IMLValidator()

        # Verify the pipeline works with TextToIML as a stand-in
        from prosody_protocol import TextToIML
        predictor = TextToIML()
        iml = predictor.predict("Hello world")
        doc = parser.parse(iml)
        result = validator.validate(iml)

        assert len(doc.utterances) > 0
        assert result.valid

    def test_integration_claude_example(self):
        """AC4: Claude integration example runs (SDK parts, no API call)."""
        from prosody_protocol import IMLParser

        parser = IMLParser()

        iml_input = '''<utterance emotion="sarcastic" confidence="0.89">
          Oh that's <prosody pitch_contour="fall-rise">wonderful</prosody>.
        </utterance>'''

        doc = parser.parse(iml_input)
        utt = doc.utterances[0]

        assert utt.emotion == "sarcastic"
        assert utt.confidence == pytest.approx(0.89)

        plain = parser.to_plain_text(doc)
        assert "wonderful" in plain

    def test_integration_elevenlabs_example(self):
        """AC4: ElevenLabs integration example runs (SDK parts, no API call)."""
        from prosody_protocol import IMLToSSML

        converter = IMLToSSML()
        ssml = converter.convert('''
<utterance emotion="empathetic" confidence="0.8">
  I understand <prosody pitch="-5%" quality="breathy">how you feel</prosody>.
</utterance>
''')

        assert ssml.startswith("<speak")
        assert "how you feel" in ssml

    def test_dockerfile_exists_and_valid(self):
        """AC5: API server is deployable via Docker."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"

        content = dockerfile.read_text()
        assert "FROM python:" in content
        assert "uvicorn" in content
        assert "EXPOSE" in content
        assert "api.app:app" in content

    def test_benchmark_docs_exist(self):
        """AC6: Benchmark results published in README or docs."""
        # Check that benchmarks are documented in quickstart or API docs
        quickstart = (PROJECT_ROOT / "docs" / "quickstart.md").read_text()
        api_doc = (PROJECT_ROOT / "docs" / "API.md").read_text()

        assert "Benchmark" in quickstart or "benchmark" in quickstart
        assert "BenchmarkReport" in api_doc


# ---------------------------------------------------------------------------
# Documentation Completeness Tests
# ---------------------------------------------------------------------------


class TestDocumentation:
    """Verify all required documentation files exist."""

    def test_quickstart_exists(self):
        assert (PROJECT_ROOT / "docs" / "quickstart.md").exists()

    def test_api_reference_exists(self):
        assert (PROJECT_ROOT / "docs" / "API.md").exists()

    def test_contributing_guide_exists(self):
        assert (PROJECT_ROOT / "CONTRIBUTING.md").exists()

    def test_changelog_exists(self):
        assert (PROJECT_ROOT / "CHANGELOG.md").exists()

    def test_integration_whisper_guide_exists(self):
        assert (PROJECT_ROOT / "docs" / "integrations" / "whisper.md").exists()

    def test_integration_claude_guide_exists(self):
        assert (PROJECT_ROOT / "docs" / "integrations" / "claude.md").exists()

    def test_integration_elevenlabs_guide_exists(self):
        assert (PROJECT_ROOT / "docs" / "integrations" / "elevenlabs.md").exists()

    def test_integration_coqui_guide_exists(self):
        assert (PROJECT_ROOT / "docs" / "integrations" / "coqui_tts.md").exists()


# ---------------------------------------------------------------------------
# Package Publishing Tests
# ---------------------------------------------------------------------------


class TestPackagePublishing:
    """Verify package publishing infrastructure."""

    def test_py_typed_marker_exists(self):
        """PEP 561: py.typed marker for typed package."""
        assert (PROJECT_ROOT / "src" / "prosody_protocol" / "py.typed").exists()

    def test_publish_workflow_exists(self):
        assert (PROJECT_ROOT / ".github" / "workflows" / "publish.yml").exists()

    def test_ci_workflow_exists(self):
        assert (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_pyproject_toml_has_package_name(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert 'name = "prosody-protocol"' in content

    def test_version_file_exists(self):
        from prosody_protocol import __version__
        assert __version__ is not None
        assert len(__version__) > 0


# ---------------------------------------------------------------------------
# Quickstart Code Smoke Tests
# ---------------------------------------------------------------------------


class TestQuickstartSmoke:
    """Verify quickstart code examples actually work end-to-end."""

    def test_parse_and_validate_pipeline(self):
        """Full parse -> validate -> plain text pipeline."""
        from prosody_protocol import IMLParser, IMLValidator

        iml = '''<utterance emotion="frustrated" confidence="0.92">
          I've been waiting for <emphasis level="strong">hours</emphasis>!
        </utterance>'''

        parser = IMLParser()
        doc = parser.parse(iml)
        assert doc.utterances[0].emotion == "frustrated"
        assert doc.utterances[0].confidence == pytest.approx(0.92)

        validator = IMLValidator()
        result = validator.validate(iml)
        assert result.valid

        text = parser.to_plain_text(doc)
        assert "hours" in text

    def test_text_to_iml_to_ssml_pipeline(self):
        """Full text -> IML -> SSML pipeline."""
        from prosody_protocol import TextToIML, IMLToSSML, IMLValidator

        predictor = TextToIML()
        iml = predictor.predict("This is amazing!")

        validator = IMLValidator()
        result = validator.validate(iml)
        assert result.valid

        converter = IMLToSSML()
        ssml = converter.convert(iml)
        assert ssml.startswith("<speak")

    def test_iml_to_audio_pipeline(self):
        """IML -> Audio synthesis pipeline."""
        from prosody_protocol import IMLToAudio

        synthesizer = IMLToAudio()
        wav_bytes = synthesizer.synthesize(
            '<utterance emotion="calm" confidence="0.9">'
            'Please <prosody rate="slow">listen carefully</prosody>.'
            '</utterance>'
        )

        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0
        # WAV files start with RIFF header
        assert wav_bytes[:4] == b"RIFF"
