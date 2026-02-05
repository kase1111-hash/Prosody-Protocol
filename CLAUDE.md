# CLAUDE.md

Project guidance for Claude Code when working on the Prosody Protocol repository.

## Project Overview

Prosody Protocol defines **Intent Markup Language (IML)** -- an XML-based markup format that preserves prosodic information (pitch, volume, tempo, voice quality, rhythm) alongside transcribed text. The goal is to prevent the loss of emotional intent during speech-to-text conversion so that downstream AI systems can distinguish sarcasm from sincerity, urgency from calm, etc.

**Current status:** Specification phase (v0.1.0-alpha). No runnable code yet -- the repository contains the spec, dataset schema definitions, and planned API/SDK designs.

## Repository Structure

```
Prosody-Protocol/
├── README.md          # Project overview, use cases, code examples, roadmap
├── spec.md            # Formal IML specification (tags, attributes, processing model)
├── CLAUDE.md          # This file
└── (planned)
    ├── datasets/      # Training datasets (Mavis corpus, emotional speech, etc.)
    ├── docs/          # API documentation
    ├── src/           # Python SDK source (prosody_protocol package)
    ├── tests/         # Test suite
    ├── CONTRIBUTING.md
    └── LICENSE
```

## Key Concepts

- **IML (Intent Markup Language):** The XML markup format. See `spec.md` for the full specification.
- **Core tags:** `<utterance>`, `<prosody>`, `<pause>`, `<emphasis>` (stable)
- **Experimental tags:** `<segment>` (may change)
- **Extended attributes:** `f0_mean`, `f0_range`, `intensity_mean`, `jitter`, `shimmer`, etc. (for research use)
- **Prosody profiles:** JSON documents that map atypical prosodic patterns to intended meanings (accessibility feature)

## Specification Rules

When editing IML examples or the spec:

- `<utterance>` must include `confidence` whenever `emotion` is present
- `<pause>` is always self-closing: `<pause duration="800"/>`
- `<segment>` can only be a direct child of `<utterance>`, not nested inside `<prosody>` or `<emphasis>`
- Nesting depth of `<prosody>` inside `<emphasis>` (or vice versa) should not exceed 2 levels
- Confidence values are floats between 0.0 and 1.0
- Pitch is expressed as relative percentage (`+15%`), semitones (`+3st`), or absolute Hz (`185Hz`)
- Volume is expressed in relative dB (`+6dB`, `-3dB`)
- Pause duration is in whole milliseconds

## Planned Python SDK

The Python package will be called `prosody_protocol` and expose:

- `AudioToIML` -- convert audio files to IML-annotated transcripts
- `IMLToAudio` -- synthesize speech from IML markup
- `TextToIML` -- predict prosody for plain text
- `IMLValidator` -- validate IML markup structure and semantics
- `IMLParser` -- parse IML into structured data
- `ProsodyAnalyzer` -- analyze audio prosody given a transcript
- `IMLToSSML` -- convert IML to SSML for TTS engines

## Planned REST API

```
POST /v1/convert/audio-to-iml    # Audio file -> IML
POST /v1/synthesize               # IML -> Audio
POST /v1/validate                  # Validate IML document
```

## Build and Test

No build system or tests exist yet. When they are added:

- The Python SDK will use `pip install -e .` for local development
- Tests will likely use `pytest`
- IML validation tests should cover well-formedness, attribute types, nesting rules, and confidence requirements

## Style Guidelines

- IML examples in documentation should be valid XML
- Use realistic prosodic values in examples (not extreme or nonsensical values)
- Always include `confidence` with `emotion` in example `<utterance>` tags
- Prefer the core emotion vocabulary defined in `spec.md` Section 3.1
- Keep the specification language precise: use MUST, SHOULD, MAY, MUST NOT per RFC 2119

## Related Projects

- **Mavis** -- vocal typing instrument that generates IML training data
- **intent-engine** -- prosody-aware AI system that consumes IML
- **Agent-OS** -- constitutional governance layer that uses intent verification

## License

- Specification: CC-BY-4.0
- Code: MIT
- Datasets: individual licenses per dataset
