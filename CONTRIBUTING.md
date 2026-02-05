# Contributing to Prosody Protocol

Thank you for your interest in contributing to the Prosody Protocol project.

## Getting Started

1. Fork and clone the repository.
2. Install in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite to verify your setup:

   ```bash
   pytest
   ```

## Development Workflow

1. Create a branch from `main` for your work.
2. Make your changes, following the style guidelines below.
3. Add or update tests for any new or changed behavior.
4. Run the full check suite before submitting:

   ```bash
   ruff check src/ tests/
   mypy src/
   pytest --cov=prosody_protocol
   ```

5. Open a pull request with a clear description of the change.

## Code Style

- Python 3.10+ with type annotations on all public APIs.
- Formatted and linted with [Ruff](https://docs.astral.sh/ruff/) (line length 100).
- Type-checked with [mypy](https://mypy-lang.org/) in strict mode.
- Docstrings on all public classes and methods (Google style).

## IML Examples

When adding or editing IML examples anywhere in the project:

- Examples must be valid XML.
- Use realistic prosodic values (not extreme or nonsensical).
- Always include `confidence` when `emotion` is present on `<utterance>`.
- Prefer the core emotion vocabulary from `spec.md` Section 3.1.
- Use RFC 2119 language (MUST, SHOULD, MAY) in spec text.

## Tests

- Tests use [pytest](https://docs.pytest.org/).
- Place IML test fixtures in `tests/fixtures/valid/` or `tests/fixtures/invalid/`.
- Name fixtures descriptively: `valid_sarcasm.xml`, `invalid_missing_confidence.xml`.
- Aim for high coverage on parser and validator code.

## Datasets

If contributing dataset entries:

- Each entry must conform to `schemas/dataset-entry.schema.json` (when available).
- Audio must be WAV format, 16kHz mono recommended.
- IML annotations must pass `IMLValidator`.
- Include consent confirmation in the entry metadata.

## Reporting Issues

Open an issue on GitHub with:

- A clear title and description.
- Steps to reproduce (if applicable).
- Expected vs. actual behavior.
- IML snippets or audio samples if relevant.

## License

By contributing, you agree that your contributions will be licensed under the MIT License (code) or CC-BY-4.0 (specification and documentation).
