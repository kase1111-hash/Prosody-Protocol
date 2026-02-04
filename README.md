# Prosody Protocol

**A protocol for preserving human intent across the speech-to-text boundary.**

![Status](https://img.shields.io/badge/status-specification-blue)
![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange)
![License](https://img.shields.io/badge/license-CC--BY--4.0-green)

---

## The Problem

Current voice AI systems destroy prosodic information during speech-to-text conversion:

```
Human speaks: "Oh, that's GREAT." [sarcastic intonation]
      ↓
   STT converts to: "oh that's great"
      ↓
   LLM interprets as: Genuine enthusiasm
      ↓
   Response: "I'm glad you're happy!"
      ↓
Human: "It didn't understand I was being sarcastic."
```

**60-80% of emotional meaning** is lost because STT strips away:
- Pitch contours (rising/falling)
- Volume dynamics (loud/soft)
- Tempo variations (fast/slow)
- Voice quality (breathy/tense)
- Rhythm and timing

This creates AI that can't distinguish:
- Sarcasm from sincerity
- Urgency from calm
- Confidence from uncertainty
- Joy from frustration

---

## The Solution: Intent Markup Language (IML)

IML is a **structured markup format** that preserves prosodic information as machine-readable metadata alongside transcribed text.

### Basic Example

**Human speech:**
> "Oh, that's GREAT." [sarcastic tone - high pitch followed by sharp fall]

**IML transcription:**
```xml
<utterance emotion="sarcastic" confidence="0.87">
  Oh, that's 
  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">
    GREAT
  </prosody>.
</utterance>
```

**LLM can now:**
- Detect sarcasm from `emotion="sarcastic"` tag
- Understand emphasis from `pitch` and `volume` attributes
- Infer frustration from `pitch_contour="fall-sharp"`

---

## Specification

### Core Tags

#### `<utterance>`
Container for a complete spoken phrase.

**Attributes:**
- `emotion` - Detected emotion (sarcastic, sincere, frustrated, joyful, uncertain, etc.)
- `confidence` - Detection confidence (0.0-1.0)
- `speaker_id` - Optional speaker identifier

**Example:**
```xml
<utterance emotion="frustrated" confidence="0.92" speaker_id="user_001">
  I've been waiting for hours!
</utterance>
```

#### `<prosody>`
Marks text with specific prosodic features.

**Attributes:**
- `pitch` - Relative pitch (+/-N% or +/-N semitones)
- `pitch_contour` - Pitch movement (rise, fall, rise-fall, fall-rise, flat)
- `volume` - Relative volume (+/-N dB)
- `rate` - Speech rate (fast, slow, or percentage: 150%)
- `quality` - Voice quality (breathy, tense, creaky, modal)

**Example:**
```xml
I <prosody pitch="-10%" volume="-3dB" quality="breathy">really</prosody> don't care.
```

#### `<pause>`
Explicit timing gap (significant for interpretation).

**Attributes:**
- `duration` - Length in milliseconds

**Example:**
```xml
Well<pause duration="800"/> I suppose that could work.
```
*(Long pause indicates uncertainty or reluctance)*

#### `<emphasis>`
Marked stress or emphasis.

**Attributes:**
- `level` - strong, moderate, reduced

**Example:**
```xml
I <emphasis level="strong">said</emphasis> I'm fine.
```
*(Strong emphasis on "said" indicates frustration)*

#### `<segment>`
Groups words/phonemes with shared prosodic characteristics.

**Attributes:**
- `tempo` - Overall speed (rushed, steady, drawn-out)
- `rhythm` - Rhythmic pattern (staccato, legato, syncopated)

**Example:**
```xml
<segment tempo="rushed" rhythm="staccato">
  I don't have time for this right now okay?
</segment>
```

### Extended Attributes

For research and detailed analysis:

- `f0_mean` - Mean fundamental frequency (Hz)
- `f0_range` - Pitch range (Hz)
- `intensity_mean` - Mean intensity (dB)
- `intensity_range` - Dynamic range (dB)
- `speech_rate` - Syllables per second
- `jitter` - Voice quality measure
- `shimmer` - Voice quality measure

**Example:**
```xml
<prosody f0_mean="185" f0_range="120-240" intensity_mean="68" speech_rate="4.2">
  This is getting ridiculous!
</prosody>
```

---

## Use Cases

### 1. Emotion-Aware Voice Assistants

**Traditional STT:**
```
Input: [frustrated tone] "The app crashed again."
Output: "the app crashed again"
LLM: "I'm sorry to hear that. Have you tried restarting?"
```

**With IML:**
```xml
Input: <utterance emotion="frustrated" confidence="0.89">
  The app <emphasis>crashed</emphasis> 
  <prosody pitch="+8%" volume="+5dB">again</prosody>.
</utterance>

LLM detects: High frustration, repeated problem
Response: "I can tell this is really frustrating - this is the third crash. 
           Let me escalate this to engineering immediately."
```

### 2. Constitutional AI Governance

Agent needs to verify user intent before dangerous actions:

```xml
<!-- Likely genuine request -->
<utterance emotion="calm" confidence="0.94">
  <prosody pitch="flat" rate="100%">
    Delete all temporary files.
  </prosody>
</utterance>

<!-- Likely NOT genuine (sarcastic or frustrated venting) -->
<utterance emotion="sarcastic" confidence="0.87">
  <prosody pitch_contour="fall-rise" volume="+8dB">
    Yeah, just delete EVERYTHING, that'll help.
  </prosody>
</utterance>
```

Agent can require calm, deliberate prosody for destructive actions.

### 3. Accessibility & Communication

People with atypical prosody (autism, stroke, neurological conditions) can:

**Define their prosodic language:**
```json
{
  "user_profile": "user_789",
  "prosody_mappings": {
    "monotone_with_fast_rate": "excitement",
    "flat_pitch_with_pauses": "thinking_carefully",
    "volume_spike": "emphasis_not_anger"
  }
}
```

AI learns: "When THIS user exhibits X prosody, they mean Y emotion."

### 4. Training Data for Models

IML enables:
- Fine-tuning LLMs to understand emotional context
- Training TTS systems to sound more human
- Building prosody-detection models
- Cross-lingual emotion recognition

---

## The Dataset

This repository hosts **training datasets** for prosody ↔ intent mapping.

### Dataset Structure

```
datasets/
├── mavis-corpus/           # Generated from Mavis vocal typing game
│   ├── text/               # Original typed text with markup
│   ├── audio/              # Synthesized audio files
│   ├── annotations/        # IML markup files
│   └── metadata.json       # Dataset statistics
│
├── emotional-speech/       # Labeled emotional speech samples
│   ├── sarcasm/
│   ├── sincerity/
│   ├── frustration/
│   ├── joy/
│   └── uncertainty/
│
├── conversational/         # Natural dialogue with prosody
│   ├── audio/
│   ├── transcripts/
│   └── iml/
│
└── accessibility/          # Atypical prosody samples
    ├── autism-spectrum/
    ├── stroke-recovery/
    └── voice-disorders/
```

### Mavis Corpus

**Source:** Generated by users playing [Mavis](https://github.com/yourusername/mavis)

Every Mavis performance creates:
- **Typed text** with prosody markup (input)
- **Audio file** with synthesized speech (output)
- **IML annotation** mapping text → prosody features

**Example entry:**
```json
{
  "id": "mavis_001234",
  "timestamp": "2026-02-04T10:30:00Z",
  "typed_text": "the SUN... is falling _down_",
  "iml": "<utterance>the <prosody pitch='+20%' volume='+8dB'>SUN</prosody><pause duration='300'/> is falling <prosody volume='-3dB' quality='breathy'>down</prosody>.</utterance>",
  "audio_file": "mavis_001234.wav",
  "user_consent": true,
  "difficulty_level": "intermediate"
}
```

**Current size:** 0 samples (dataset launches with Mavis v1.0)  
**Target size:** 100,000+ samples within first year

### Dataset Access

**License:** CC-BY-4.0 (attribution required)

```bash
# Download the dataset
git clone https://github.com/yourusername/prosody-protocol.git
cd prosody-protocol/datasets

# Or use the API
curl https://api.prosody-protocol.org/v1/datasets/mavis-corpus
```

**Citation:**
```bibtex
@dataset{prosody_protocol_2026,
  title={Prosody Protocol: Intent Markup Language and Training Datasets},
  author={True North Construction LLC},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yourusername/prosody-protocol}
}
```

---

## Converter Tools

### Audio → IML

Convert audio files to IML-annotated transcripts.

```python
from prosody_protocol import AudioToIML

converter = AudioToIML(model="whisper-large-v3-prosody")

# Process audio file
iml = converter.convert("speech.wav")
print(iml)
```

**Output:**
```xml
<utterance emotion="frustrated" confidence="0.87">
  I <emphasis level="strong">told</emphasis> you 
  <prosody pitch="+12%" volume="+6dB">yesterday</prosody>!
</utterance>
```

### IML → Audio

Synthesize speech from IML markup.

```python
from prosody_protocol import IMLToAudio

synthesizer = IMLToAudio(voice="en_US-female-medium")

# Generate audio from IML
audio = synthesizer.synthesize("""
<utterance emotion="calm">
  Please <prosody rate="slow">listen carefully</prosody>.
</utterance>
""")

audio.save("output.wav")
```

### Text → IML (Prediction)

Predict prosody for plain text (useful for TTS).

```python
from prosody_protocol import TextToIML

predictor = TextToIML(model="prosody-bert-large")

# Predict likely prosody
iml = predictor.predict("I can't believe this happened.", context="user_frustrated")
print(iml)
```

**Output:**
```xml
<utterance emotion="frustrated" confidence="0.76">
  I <emphasis level="strong">can't</emphasis> believe 
  <prosody pitch="+8%" volume="+4dB">this</prosody> happened.
</utterance>
```

### Validation

Ensure IML markup is well-formed and semantically valid.

```python
from prosody_protocol import IMLValidator

validator = IMLValidator()

iml = """<utterance emotion="happy">
  <prosody pitch="+20%">This is great!</prosody>
</utterance>"""

result = validator.validate(iml)
if result.valid:
    print("✓ Valid IML")
else:
    print(f"✗ Errors: {result.errors}")
```

---

## Integration Examples

### With Whisper (STT)

```python
import whisper
from prosody_protocol import ProsodyAnalyzer

# Load models
stt_model = whisper.load_model("large-v3")
prosody_analyzer = ProsodyAnalyzer()

# Transcribe with prosody
result = stt_model.transcribe("audio.wav")
text = result["text"]

# Analyze prosody
iml = prosody_analyzer.analyze("audio.wav", text)
print(iml)
```

### With Claude (LLM)

```python
import anthropic
from prosody_protocol import IMLParser

client = anthropic.Anthropic()

# User speaks with sarcasm
iml_input = """<utterance emotion="sarcastic" confidence="0.89">
  Oh that's <prosody pitch_contour="fall-rise">wonderful</prosody>.
</utterance>"""

# Send to Claude with prosody context
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    system="""You receive text with IML prosody markup. 
    The markup tells you HOW words were spoken, revealing intent.
    Respond appropriately to the emotional intent, not just the literal words.""",
    messages=[{
        "role": "user",
        "content": iml_input
    }]
)

print(message.content)
# Output: "I can tell you're being sarcastic. What's actually wrong?"
```

### With ElevenLabs (TTS)

```python
from elevenlabs import generate
from prosody_protocol import IMLToSSML

# Convert IML to SSML for TTS
converter = IMLToSSML()
ssml = converter.convert("""
<utterance emotion="empathetic">
  I understand <prosody pitch="-5%" quality="breathy">how you feel</prosody>.
</utterance>
""")

# Generate speech with prosody
audio = generate(
    text=ssml,
    voice="Bella",
    model="eleven_multilingual_v2"
)
```

---

## API Reference

### REST API (Coming Soon)

```bash
# Convert audio to IML
POST /v1/convert/audio-to-iml
Content-Type: multipart/form-data

{
  "audio": <file>,
  "language": "en-US"
}

# Synthesize IML to audio
POST /v1/synthesize
Content-Type: application/json

{
  "iml": "<utterance>...</utterance>",
  "voice": "en_US-female-medium"
}

# Validate IML
POST /v1/validate
Content-Type: application/json

{
  "iml": "<utterance>...</utterance>"
}
```

### Python SDK

```bash
pip install prosody-protocol
```

```python
from prosody_protocol import (
    AudioToIML,
    IMLToAudio,
    TextToIML,
    IMLValidator,
    IMLParser,
    ProsodyAnalyzer
)
```

See [API Documentation](./docs/API.md) for complete reference.

---

## Specification Status

### Current Version: 0.1.0-alpha

**Stable tags:** `<utterance>`, `<prosody>`, `<pause>`, `<emphasis>`

**Experimental tags:** `<segment>`, extended f0/intensity attributes

**In discussion:**
- Multi-speaker conversations
- Backchannel markers (mm-hmm, uh-huh)
- Laughter and non-verbal vocalizations
- Cross-lingual prosody patterns

**Submit proposals:** [GitHub Issues](https://github.com/yourusername/prosody-protocol/issues)

### Versioning

- **Breaking changes:** Major version (1.0.0 → 2.0.0)
- **New features:** Minor version (1.0.0 → 1.1.0)
- **Bug fixes:** Patch version (1.0.0 → 1.0.1)

---

## Roadmap

### Phase 1: Specification (Current)
- [x] Define core IML tags
- [x] Document basic attributes
- [ ] Gather feedback from researchers
- [ ] Publish v1.0 stable spec

### Phase 2: Dataset
- [ ] Launch Mavis corpus (100K+ samples)
- [ ] Partner with universities for labeled data
- [ ] Add emotional speech datasets
- [ ] Accessibility corpus (atypical prosody)

### Phase 3: Tools
- [ ] Python SDK (converters, validators)
- [ ] REST API
- [ ] Model training pipelines
- [ ] Evaluation metrics

### Phase 4: Adoption
- [ ] Integration guides for major platforms
- [ ] Academic papers demonstrating effectiveness
- [ ] Industry partnerships (voice AI companies)
- [ ] Standardization proposal (W3C or similar)

---

## Contributing

We welcome contributions in:

### Specification Design
- Tag proposals
- Attribute refinements
- Use case validation
- Cross-lingual considerations

### Dataset Curation
- Annotated audio samples
- Prosody labeling guidelines
- Quality assurance protocols
- Privacy-preserving collection methods

### Tooling
- Converter implementations
- Validation tools
- Analysis pipelines
- Integration examples

### Research
- Prosody detection algorithms
- Intent classification models
- Cross-cultural prosody patterns
- Accessibility applications

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## Related Projects

Part of the **Natural Language Ecosystem:**

- **[mavis](https://github.com/yourusername/mavis)** - Vocal typing instrument (generates training data)
- **[intent-engine](https://github.com/yourusername/intent-engine)** - Prosody-aware AI system (consumes IML)
- **[Agent-OS](https://github.com/yourusername/agent-os)** - Constitutional governance (uses intent verification)

---

## Research & Papers

### Citing This Work

If you use the Prosody Protocol or IML specification in research:

```bibtex
@misc{prosody_protocol_2026,
  title={Intent Markup Language: Preserving Prosodic Information for AI Systems},
  author={True North Construction LLC},
  year={2026},
  url={https://github.com/yourusername/prosody-protocol},
  note={Version 0.1.0-alpha}
}
```

### Related Research

- **Prosody and Emotion:** Scherer, K. R. (2003). Vocal communication of emotion
- **Speech Synthesis:** Taylor, P. (2009). Text-to-Speech Synthesis
- **Computational Paralinguistics:** Schuller, B., & Batliner, A. (2013)
- **Intent Recognition:** Intentions in Communication (MIT Press)

---

## FAQ

**Q: Why not just use SSML?**  
A: SSML is designed for TTS output (text → speech). IML is designed for STT + intent understanding (speech → text + prosody → intent). Different use cases, different requirements.

**Q: How accurate is emotion detection from prosody?**  
A: Current state-of-the-art: 70-85% for basic emotions. IML includes confidence scores. Combining prosody with context and linguistic cues improves accuracy to 85-95%.

**Q: Does this work across languages/cultures?**  
A: Prosodic patterns vary significantly across languages. IML is language-agnostic (tags are universal), but interpretation requires language-specific models. We're building multi-lingual datasets.

**Q: Privacy concerns with emotional detection?**  
A: Valid concern. IML is a *tool* - applications must implement appropriate consent and privacy controls. We recommend: explicit opt-in, local processing where possible, emotional data treated as sensitive PII.

**Q: Can this detect lying or deception?**  
A: No. Prosody indicates emotional state, not truthfulness. Don't use for deception detection.

---

## License

### Specification
Creative Commons Attribution 4.0 International (CC-BY-4.0)

You are free to:
- Share: Copy and redistribute
- Adapt: Remix and build upon

Under these terms:
- Attribution: Must credit "Prosody Protocol / True North Construction LLC"
- No restrictions: Cannot apply legal terms that restrict others

### Datasets
Individual datasets may have specific licenses. Check each dataset's README.

### Code
MIT License (see [LICENSE](./LICENSE))

---

## Contact

**Maintainer:** Kase Branham
**Email:** kase1111@gmail.com
**Discussions:** [GitHub Discussions](https://github.com/kase1111-hash/prosody-protocol/discussions)  
**Issues:** [GitHub Issues](https://github.com/kase1111-hash/prosody-protocol/issues)

---

## Acknowledgments

Built on research from:
- Speech prosody community
- Computational paralinguistics field
- Voice AI/assistive technology researchers
- Constitutional AI framework (Anthropic)

Special thanks to Mavis users who contribute training data.

---

**The future of AI communication isn't just understanding *what* you said - it's understanding *how you meant it*.**

🎯 **Preserve. Interpret. Respond.**
