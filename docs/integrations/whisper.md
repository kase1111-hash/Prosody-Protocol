# Whisper Integration

Use OpenAI Whisper for speech-to-text and Prosody Protocol for prosodic annotation.

## Setup

```bash
pip install prosody-protocol[audio]
```

The `audio` extras install Whisper, librosa, and Praat for feature extraction.

## Basic Usage

```python
from prosody_protocol import AudioToIML

converter = AudioToIML(stt_model="base")
iml = converter.convert("recording.wav")
print(iml)
```

The `AudioToIML` converter handles the full pipeline:

1. Whisper transcribes with word-level timestamps
2. `ProsodyAnalyzer` extracts acoustic features per word
3. `RuleBasedEmotionClassifier` determines the utterance emotion
4. `IMLAssembler` builds the final IML document

## Custom Whisper Model

```python
converter = AudioToIML(stt_model="large-v3")
doc = converter.convert_to_doc("interview.wav")

for utt in doc.utterances:
    print(f"Emotion: {utt.emotion} (confidence: {utt.confidence})")
```

## Advanced: Manual Pipeline

For more control, use the components individually:

```python
from prosody_protocol import ProsodyAnalyzer, WordAlignment, IMLAssembler

# 1. Run Whisper separately
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav", word_timestamps=True)

# 2. Build alignments from Whisper output
alignments = []
for segment in result["segments"]:
    for word_info in segment.get("words", []):
        alignments.append(WordAlignment(
            word=word_info["word"].strip(),
            start_ms=int(word_info["start"] * 1000),
            end_ms=int(word_info["end"] * 1000),
        ))

# 3. Extract prosodic features
analyzer = ProsodyAnalyzer()
features = analyzer.analyze("audio.wav", alignments)

# 4. Build IML
assembler = IMLAssembler()
for feat in features:
    assembler.add_word(feat.text)
    if feat.f0_mean and feat.f0_mean > 200:
        assembler.add_prosody(pitch="+10%")

doc = assembler.build()
```

## Batch Processing

```python
from pathlib import Path
from prosody_protocol import AudioToIML

converter = AudioToIML()
audio_dir = Path("interviews/")

for wav_file in audio_dir.glob("*.wav"):
    iml = converter.convert(wav_file)
    output = wav_file.with_suffix(".iml")
    output.write_text(iml)
    print(f"Processed: {wav_file.name}")
```
