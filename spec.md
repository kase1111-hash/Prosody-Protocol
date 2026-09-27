# Intent Markup Language (IML) Specification

**Version:** 0.1.0-alpha
**Status:** Draft
**License:** CC-BY-4.0
**Maintainer:** Kase Branham / True North Construction LLC

---

## 1. Introduction

### 1.1 Purpose

The Intent Markup Language (IML) is a structured markup format designed to preserve prosodic information as machine-readable metadata alongside transcribed text. It bridges the gap between speech-to-text (STT) output and downstream language model interpretation by encoding **how** something was said, not just **what** was said.

### 1.2 Problem Statement

Current STT systems discard 60-80% of emotional meaning by stripping away:

- Pitch contours (rising/falling intonation)
- Volume dynamics (loud/soft)
- Tempo variations (fast/slow speech rate)
- Voice quality (breathy/tense/creaky)
- Rhythm and timing (pauses, stress patterns)

This causes AI systems to misinterpret sarcasm as sincerity, urgency as calm, confidence as uncertainty, and joy as frustration.

### 1.3 Scope

This specification defines:

- The XML-based tag set for encoding prosodic features
- Attribute schemas and valid value ranges
- Nesting rules and document structure
- Semantic interpretation guidelines
- Conformance requirements for producers and consumers

### 1.4 Design Principles

1. **Human-readable:** IML documents should be understandable by non-specialists.
2. **Machine-parseable:** Standard XML parsers can process IML.
3. **Language-agnostic:** Tags and attributes work across natural languages; interpretation requires language-specific models.
4. **Incrementally adoptable:** Consumers can ignore tags they don't understand and still process the text content.
5. **Confidence-aware:** All classifications include confidence scores to express uncertainty.

### 1.5 Relationship to SSML

SSML (Speech Synthesis Markup Language) is designed for text-to-speech **output** (controlling how synthesized speech sounds). IML is designed for speech-to-text **input** (describing how natural speech was produced). While there is overlap in vocabulary, the use cases and semantics differ:

| Aspect | SSML | IML |
|--------|------|-----|
| Direction | Text -> Speech | Speech -> Text + Prosody |
| Purpose | Control synthesis | Describe observation |
| Values | Prescriptive targets | Measured/inferred values |
| Confidence | N/A | Required on classifications |

---

## 2. Document Structure

### 2.1 Content Model

The root element of an IML document is either a single `<utterance>` or an `<iml>` element that wraps one or more utterances (Section 2.4). Each utterance contains mixed content: plain text interspersed with inline prosodic markup elements.

```xml
<utterance emotion="frustrated" confidence="0.92">
  I <emphasis level="strong">told</emphasis> you
  <prosody pitch="+12%" volume="+6dB">yesterday</prosody>!
</utterance>
```

### 2.2 Encoding

IML documents MUST be encoded in UTF-8. The XML declaration is optional but recommended:

```xml
<?xml version="1.0" encoding="UTF-8"?>
```

If the XML declaration includes an encoding declaration, it MUST name `UTF-8`. A UTF-8 byte order mark MAY precede the document. Consumers MUST NOT decode an IML document with any encoding other than UTF-8; a document that is not valid UTF-8 is not a valid IML document.

### 2.3 Namespacing

IML uses no namespace by default. When embedded in other XML formats, the namespace `http://prosody-protocol.org/iml/0.1` SHOULD be used:

```xml
<iml:utterance xmlns:iml="http://prosody-protocol.org/iml/0.1" emotion="calm" confidence="0.94">
  Hello world.
</iml:utterance>
```

An element in the IML namespace is equivalent to the un-namespaced element with the same local name. A document SHOULD use one form for all of its IML elements. Elements in any other namespace are unknown elements (Section 6.2), even when their local name matches an IML tag. IML attributes are never namespace-qualified, including on namespaced elements: `iml:emotion` is not the `emotion` attribute, and consumers ignore it like any other unknown attribute. Producers SHOULD NOT put attributes in the IML namespace.

The published XML Schema (`schemas/iml-1.0.xsd`) describes un-namespaced IML. Namespaced IML can be validated with a schema whose `targetNamespace` is the IML namespace and which includes `iml-1.0.xsd`.

### 2.4 Multi-Utterance Documents

Multiple utterances can be wrapped in a root `<iml>` element:

```xml
<iml version="0.1.0" language="en-US">
  <utterance emotion="calm" confidence="0.91" speaker_id="user_001">
    Hey, how's it going?
  </utterance>
  <utterance emotion="frustrated" confidence="0.88" speaker_id="user_002">
    Not great, honestly.
  </utterance>
</iml>
```

**Content model:** `<utterance>` elements only. Whitespace, comments and processing instructions MAY appear between utterances; other text and other IML elements MUST NOT appear directly inside `<iml>`.

**`<iml>` Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `version` | SemVer | RECOMMENDED | IML specification version (e.g., `"0.1.0"`, `"0.1.0-alpha"`) |
| `language` | BCP-47 | OPTIONAL | Primary language of the content (e.g., `"en-US"`) |
| `consent` | Enum | OPTIONAL | Consent recorded for the prosodic and emotional annotations (Section 8.3) |
| `processing` | Enum | OPTIONAL | Where the audio was analyzed (Section 8.3) |

**Consent values:**

- `explicit` - The speaker explicitly opted in to prosodic and emotional analysis
- `implicit` - Consent was inferred from context rather than given as an opt-in. This does not meet the explicit-consent requirement of Section 8.1.
- `none` - No consent was recorded

**Processing values:**

- `local` - Analysis ran on the user's device
- `remote` - Analysis ran on a remote service
- `hybrid` - Analysis ran partly on the device and partly remotely

### 2.5 XML Features

1. IML documents MUST NOT contain a document type declaration (`<!DOCTYPE ...>`). IML defines no DTD, and DTD-declared entities enable XML external entity (XXE) and entity-expansion attacks. Consumers MAY reject a document that contains one, and MUST NOT resolve external entities or fetch external resources while parsing IML.
2. Comments and processing instructions MAY appear anywhere XML allows them. They carry no IML content: consumers MUST NOT treat their contents as text.
3. CDATA sections are ordinary text.

### 2.6 Attribute Value Syntax

Attribute values MUST conform to the types given in this specification:

- **Integer:** one or more ASCII digits (`0`-`9`), optionally preceded by `+`. Where a positive integer is required, the value MUST be greater than zero and MUST NOT exceed 2147483647 (2^31 - 1; as milliseconds, about 24.8 days).
- **Float:** ASCII digits with an optional sign, decimal point and exponent (`0.87`, `-3.5`, `1e-1`). The value MUST be finite as an IEEE 754 double: `NaN`, infinities and values too large to represent (such as `1e400`) are not valid, and neither are digit-group separators (`1_000`) or non-ASCII digits.
- **Enumerations** and patterned values (Sections 2.4, 3.2, 4) are case-sensitive and MUST NOT contain surrounding whitespace. XML whitespace around an Integer or Float is ignored.

---

## 3. Core Tags

### 3.1 `<utterance>`

Container for a complete spoken phrase or sentence.

**Content model:** Mixed content (text, `<prosody>`, `<pause>`, `<emphasis>`, `<segment>`). An `<utterance>` MUST NOT contain another `<utterance>` or an `<iml>`.

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `emotion` | Enum/String | OPTIONAL | Detected emotional state |
| `confidence` | Float (0.0-1.0) | REQUIRED when `emotion` is present | Classification confidence |
| `speaker_id` | String | OPTIONAL | Unique speaker identifier |

**Emotion values (core set):**

- `neutral` - No marked emotional state
- `sincere` - Genuine, straightforward
- `sarcastic` - Saying the opposite of what is meant
- `frustrated` - Annoyed, exasperated
- `joyful` - Happy, delighted
- `uncertain` - Hesitant, unsure
- `angry` - Hostile, aggressive
- `sad` - Unhappy, sorrowful
- `fearful` - Scared, anxious
- `surprised` - Startled, astonished
- `disgusted` - Repulsed, contemptuous
- `calm` - Relaxed, composed
- `empathetic` - Showing understanding of others' feelings

Producers MAY use values outside this set. Consumers SHOULD handle unknown emotion values gracefully by treating them as `neutral`.

**Example:**

```xml
<utterance emotion="frustrated" confidence="0.92" speaker_id="user_001">
  I've been waiting for hours!
</utterance>
```

### 3.2 `<prosody>`

Marks a span of text with specific prosodic features. This is the primary tag for encoding pitch, volume, rate, and voice quality.

**Content model:** Mixed content (text, `<prosody>`, `<emphasis>`, `<pause>`)

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `pitch` | Relative | OPTIONAL | Pitch offset from speaker baseline |
| `pitch_contour` | Enum | OPTIONAL | Direction of pitch movement |
| `volume` | Relative | OPTIONAL | Volume offset from speaker baseline |
| `rate` | Enum/Percentage | OPTIONAL | Speech rate |
| `quality` | Enum | OPTIONAL | Voice quality descriptor |

**Pitch values:**

- Relative percentage: `+15%`, `-10%`
- Semitone offset: `+3st`, `-2st`
- Absolute Hz (extended): `185Hz`

Relative values MUST carry an explicit sign; absolute values MUST NOT. Each is a decimal number (e.g., `+2.5st`) immediately followed by its unit.

**Pitch contour values:**

- `rise` - Upward pitch movement
- `fall` - Downward pitch movement
- `rise-fall` - Up then down (emphasis/certainty)
- `fall-rise` - Down then up (sarcasm/uncertainty/continuation)
- `fall-sharp` - Rapid downward movement (finality/frustration)
- `rise-sharp` - Rapid upward movement (surprise/alarm)
- `flat` - No significant pitch movement

**Volume values:**

- Relative dB with an explicit sign: `+6dB`, `-3dB`

**Rate values:**

- Named: `fast`, `slow`, `medium`
- Percentage of baseline, without a sign: `150%`, `80%`

**Quality values:**

- `modal` - Normal phonation
- `breathy` - Excess airflow (intimacy, sadness, exhaustion)
- `tense` - Constricted voice (stress, anger)
- `creaky` - Vocal fry (disinterest, fatigue, casual register)
- `whispery` - Near-whisper (secrecy, emphasis)
- `harsh` - Rough voice quality (anger, shouting)

**Example:**

```xml
I <prosody pitch="-10%" volume="-3dB" quality="breathy">really</prosody> don't care.
```

### 3.3 `<pause>`

An explicit timing gap that is significant for interpretation. Not every silence needs to be marked -- only pauses that carry meaning (hesitation, emphasis, turn-taking).

**Content model:** Empty. A `<pause>` MUST NOT contain elements, including unknown elements (Section 6.2), or text other than whitespace; comments and processing instructions are ignored (Section 2.5). Producers SHOULD write it as a self-closing tag: `<pause duration="800"/>`.

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `duration` | Positive integer (ms) | REQUIRED | Duration in whole milliseconds |

**Semantic guidelines:**

| Duration Range | Typical Interpretation |
|----------------|----------------------|
| < 200ms | Natural speech rhythm, not semantically significant |
| 200-500ms | Mild hesitation, breath pause, list boundary |
| 500-1000ms | Noticeable hesitation, uncertainty, topic shift |
| 1000-2000ms | Strong uncertainty, reluctance, emotional processing |
| > 2000ms | Possible communication breakdown, deep thought |

**Example:**

```xml
Well<pause duration="800"/> I suppose that could work.
```

### 3.4 `<emphasis>`

Marks words or phrases with notable stress or emphasis.

**Content model:** Mixed content (text, `<prosody>`, `<pause>`). An `<emphasis>` MUST NOT directly contain another `<emphasis>`.

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `level` | Enum | REQUIRED | Degree of emphasis |

**Level values:**

- `strong` - Heavy stress (contrastive focus, frustration, insistence)
- `moderate` - Standard emphasis (new information, clarification)
- `reduced` - De-emphasized (given information, parenthetical asides)

**Example:**

```xml
I <emphasis level="strong">said</emphasis> I'm fine.
```

### 3.5 `<segment>`

Groups a stretch of speech sharing overall prosodic characteristics. Useful for characterizing entire phrases or clauses rather than individual words.

**Status:** Stable (promoted from experimental in v0.1.0-alpha)

**Content model:** Mixed content (text, `<prosody>`, `<pause>`, `<emphasis>`)

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `tempo` | Enum | OPTIONAL | Overall speed pattern |
| `rhythm` | Enum | OPTIONAL | Rhythmic pattern |

**Tempo values:**

- `rushed` - Faster than baseline, compressed timing
- `steady` - Even, measured pace
- `drawn-out` - Slower than baseline, elongated syllables

**Rhythm values:**

- `staccato` - Clipped, choppy delivery (each word distinct)
- `legato` - Smooth, flowing delivery (words connected)
- `syncopated` - Irregular, unpredictable timing

**Example:**

```xml
<segment tempo="rushed" rhythm="staccato">
  I don't have time for this right now okay?
</segment>
```

---

## 4. Extended Attributes

These attributes provide detailed acoustic measurements for research and advanced analysis. They appear on `<prosody>`. Producers are NOT required to include them; consumers MUST NOT depend on their presence. When present, their values MUST have the types below (Section 2.6); frequencies, ranges, rates and perturbation percentages MUST NOT be negative.

### 4.1 Fundamental Frequency (F0)

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| `f0_mean` | Float, >= 0 | Hz | Mean fundamental frequency over the span |
| `f0_range` | String | Hz | Pitch range as `"low-high"` (e.g., `"120-240"`) |
| `f0_contour` | String | Hz sequence | Sampled F0 values, comma-separated without spaces (e.g., `"150,165,180,170,140"`) |

### 4.2 Intensity

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| `intensity_mean` | Float | dB | Mean intensity over the span |
| `intensity_range` | Float, >= 0 | dB | Dynamic range (max - min) |

### 4.3 Temporal

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| `speech_rate` | Float, >= 0 | syllables/sec | Articulation rate |
| `duration_ms` | Positive integer | ms | Total duration of the span |

### 4.4 Voice Quality Measures

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| `jitter` | Float, >= 0 | % | Cycle-to-cycle frequency perturbation |
| `shimmer` | Float, >= 0 | % | Cycle-to-cycle amplitude perturbation |
| `hnr` | Float | dB | Harmonics-to-noise ratio |

**Example:**

```xml
<prosody f0_mean="185" f0_range="120-240" intensity_mean="68" speech_rate="4.2" jitter="1.2" shimmer="3.8">
  This is getting ridiculous!
</prosody>
```

---

## 5. Nesting Rules

### 5.1 Valid Nesting Hierarchy

```
<iml>
  └── <utterance>
        ├── text
        ├── <prosody>
        │     ├── text
        │     ├── <prosody>
        │     ├── <emphasis>
        │     └── <pause/>
        ├── <emphasis>
        │     ├── text
        │     ├── <prosody>
        │     └── <pause/>
        ├── <pause/>
        └── <segment>
              ├── text
              ├── <prosody>
              ├── <emphasis>
              └── <pause/>
```

Every element MUST contain only the content its content model allows (Appendix A). The constraints below restate the consequences.

### 5.2 Constraints

1. `<utterance>` MUST be the top-level content element (or wrapped in `<iml>`). `<utterance>` and `<iml>` MUST NOT appear inside `<utterance>` or any inline element, and `<iml>` MUST contain only `<utterance>` elements (Section 2.4).
2. `<prosody>` MAY nest inside `<emphasis>` and vice versa, and `<prosody>` MAY nest inside `<prosody>`, but the combined nesting depth of `<prosody>` and `<emphasis>` SHOULD NOT exceed 2 levels. Depth is counted from the enclosing `<utterance>` or `<segment>`.
3. `<pause>` MUST NOT have content other than whitespace (Section 3.3) and SHOULD be written as a self-closing tag.
4. `<segment>` MUST be a direct child of `<utterance>` (not nested inside `<prosody>` or `<emphasis>`).
5. `<segment>` elements MUST NOT nest within other `<segment>` elements.
6. `<emphasis>` MUST NOT be a direct child of another `<emphasis>`.

Unknown elements (Section 6.2) do not count as parents: constraints 1, 2 and 4-6 apply as if their start and end tags were absent. Constraint 3 is the exception: an unknown element inside `<pause>` is content, which `<pause>` MUST NOT have.

---

## 6. Processing Model

### 6.1 Producer Requirements

A **producer** is any system that generates IML (e.g., a prosody-aware STT pipeline).

1. Producers MUST generate well-formed XML.
2. Producers MUST include `confidence` when `emotion` is specified on `<utterance>`.
3. Producers SHOULD omit attributes when values are at speaker baseline (e.g., don't emit `pitch="+0%"`).
4. Producers SHOULD use the core emotion vocabulary (Section 3.1) where applicable.
5. Producers MUST express `<pause>` durations in whole milliseconds.

### 6.2 Consumer Requirements

A **consumer** is any system that reads and interprets IML (e.g., an LLM, a TTS engine, an analytics tool).

1. Consumers MUST be able to extract plain text by stripping all tags (graceful degradation).
2. Consumers MUST ignore unknown tags and attributes without error.
3. Consumers SHOULD treat `confidence` values below 0.5 as low-confidence annotations.
4. Consumers SHOULD use speaker-specific baselines when `speaker_id` is present.
5. Consumers MUST NOT assume the presence of extended attributes (Section 4).

An unknown element is one that this specification does not define, including any element in a namespace other than the IML namespace (Section 2.3). Ignoring an unknown element means processing its content as if its start and end tags were absent: text inside it remains part of the enclosing utterance, and IML elements inside it are processed in place. Text inside an unknown element that is not within any `<utterance>` (for example, a metadata element directly inside `<iml>`) is not IML content and is ignored; `<utterance>` elements inside such an element are processed as if they were children of `<iml>`.

### 6.3 Validation

A valid IML document:

1. Is well-formed XML, encoded in UTF-8, with no document type declaration (Sections 2.2, 2.5).
2. Contains at least one `<utterance>` element.
3. Has valid attribute types (Section 2.6; e.g., `confidence` is a float between 0.0 and 1.0), including the enumerations and value formats of Sections 2.4, 3 and 4.
4. Follows the content models and nesting rules defined in Section 5.
5. Does not contain `emotion` without an accompanying `confidence` value.
6. Has every REQUIRED attribute (`duration` on `<pause>`, `level` on `<emphasis>`).

Unknown elements and attributes do not make a document invalid, except that a `<pause>` MUST NOT contain any element (Section 3.3). Validators SHOULD report violations of MUST-level requirements as errors, which make a document invalid, and SHOULD-level recommendations (such as the nesting depth in Section 5.2) as warnings, which do not.

*Non-normative:* the reference validator (`prosody_protocol.IMLValidator`) implements these rules, and `schemas/iml-1.0.xsd` expresses the subset that XML Schema 1.0 can.

---

## 7. User Prosody Profiles

For accessibility and personalization, IML supports external prosody profile documents that map atypical prosodic patterns to intended meanings.

### 7.1 Profile Format (JSON)

```json
{
  "profile_version": "0.1.0",
  "user_id": "user_789",
  "description": "Autism spectrum - monotone speech with rate-based expression",
  "prosody_mappings": [
    {
      "pattern": {
        "pitch_contour": "flat",
        "rate": "fast"
      },
      "interpretation": {
        "emotion": "excitement",
        "confidence_boost": 0.15
      }
    },
    {
      "pattern": {
        "pitch_contour": "flat",
        "pause_frequency": "high"
      },
      "interpretation": {
        "emotion": "thinking_carefully",
        "confidence_boost": 0.10
      }
    },
    {
      "pattern": {
        "volume": "spike"
      },
      "interpretation": {
        "emotion": "emphasis_not_anger",
        "confidence_boost": 0.20
      }
    }
  ]
}
```

### 7.2 Profile Application

When a prosody profile is active, consumers SHOULD:

1. Apply profile mappings before default emotion classification.
2. Add the `confidence_boost` to the baseline confidence score (capped at 1.0).
3. Indicate profile usage in any downstream reporting.

---

## 8. Security and Privacy Considerations

### 8.1 Emotional Data as PII

Prosodic and emotional annotations constitute sensitive personal information. Systems handling IML data MUST:

1. Obtain explicit user consent before collecting or storing emotional annotations.
2. Treat emotion data with the same protections as other PII.
3. Provide mechanisms for users to review and delete their emotional data.
4. Support local/on-device processing where feasible.

### 8.2 Prohibited Use Cases

IML MUST NOT be used for:

1. **Deception detection** - Prosody does not reliably indicate truthfulness.
2. **Covert emotional surveillance** - Users must be informed when prosody is being analyzed.
3. **Discriminatory profiling** - Emotional classifications must not be used in hiring, lending, or law enforcement decisions.

### 8.3 Consent Model

The `consent` and `processing` attributes on `<iml>` (Section 2.4) record how the annotations were obtained, so that downstream systems can apply the requirements of Section 8.1:

```xml
<iml version="0.1.0" consent="explicit" processing="local">
  <!-- User has explicitly opted in; processing occurs on-device -->
  <utterance emotion="calm" confidence="0.91">
    Sure, go ahead and record this.
  </utterance>
</iml>
```

Consumers that re-serialize an IML document MUST preserve `consent` and `processing`.

---

## 9. Versioning and Extensibility

### 9.1 Version Scheme

IML follows Semantic Versioning:

- **Major** (1.0.0 -> 2.0.0): Breaking changes to tag semantics or structure
- **Minor** (1.0.0 -> 1.1.0): New tags, attributes, or enum values (backwards-compatible)
- **Patch** (1.0.0 -> 1.0.1): Clarifications, typo fixes, non-normative changes

### 9.2 Extension Mechanism

Custom attributes MAY be added using the `x-` prefix, or as namespace-qualified attributes:

```xml
<prosody x-formant-shift="+200Hz" x-nasality="0.7">
  Something custom here.
</prosody>
```

Consumers MUST ignore `x-` prefixed attributes they do not understand. Producers SHOULD NOT emit other unprefixed attributes that this specification does not define; consumers ignore them like any unknown attribute (Section 6.2).

### 9.3 Stability Levels

| Level | Tags | Commitment |
|-------|------|------------|
| **Stable** | `<utterance>`, `<prosody>`, `<pause>`, `<emphasis>`, `<segment>` | Will not change in backwards-incompatible ways within a major version |
| **Experimental** | Extended attributes | May change or be removed in minor versions |
| **Proposed** | Backchannel markers, laughter notation | Under discussion, not yet part of the spec |

---

## 10. Future Work

The following features are under discussion for future versions:

- **Multi-speaker turn-taking:** Explicit turn boundaries and overlap markers.
- **Backchannel markers:** Tags for `mm-hmm`, `uh-huh`, and other listener signals.
- **Non-verbal vocalizations:** Laughter, sighs, coughs, and other paralinguistic sounds.
- **Cross-lingual prosody:** Language-specific attribute values and interpretation rules.
- **Streaming IML:** Incremental prosody annotation for real-time processing.
- **Prosody diffs:** Representing changes in prosody over the course of a conversation.

---

## Appendix A: Complete Tag Reference

This table is normative for content models (Section 5.1).

| Tag | Status | Content | Self-closing | Parent |
|-----|--------|---------|-------------|--------|
| `<iml>` | Stable | `<utterance>` elements | No | Root |
| `<utterance>` | Stable | Mixed (text + `<prosody>`, `<pause>`, `<emphasis>`, `<segment>`) | No | `<iml>` or root |
| `<prosody>` | Stable | Mixed (text + `<prosody>`, `<emphasis>`, `<pause>`) | No | `<utterance>`, `<prosody>`, `<emphasis>`, `<segment>` |
| `<pause>` | Stable | Empty | Yes | `<utterance>`, `<prosody>`, `<emphasis>`, `<segment>` |
| `<emphasis>` | Stable | Mixed (text + `<prosody>`, `<pause>`) | No | `<utterance>`, `<prosody>`, `<segment>` |
| `<segment>` | Stable | Mixed (text + `<prosody>`, `<emphasis>`, `<pause>`) | No | `<utterance>` |

## Appendix B: Attribute Quick Reference

| Attribute | Tags | Type | Example Values |
|-----------|------|------|----------------|
| `version` | `<iml>` | SemVer | `0.1.0` |
| `language` | `<iml>` | BCP-47 | `en-US` |
| `consent` | `<iml>` | Enum | `explicit`, `implicit`, `none` |
| `processing` | `<iml>` | Enum | `local`, `remote`, `hybrid` |
| `emotion` | `<utterance>` | String | `sarcastic`, `frustrated`, `calm` |
| `confidence` | `<utterance>` | Float 0.0-1.0 | `0.87` |
| `speaker_id` | `<utterance>` | String | `user_001` |
| `pitch` | `<prosody>` | Relative | `+15%`, `-2st`, `185Hz` |
| `pitch_contour` | `<prosody>` | Enum | `rise`, `fall`, `fall-rise` |
| `volume` | `<prosody>` | Relative dB | `+6dB`, `-3dB` |
| `rate` | `<prosody>` | Enum/Pct | `fast`, `150%` |
| `quality` | `<prosody>` | Enum | `breathy`, `tense`, `creaky` |
| `duration` | `<pause>` | Positive integer ms | `800` |
| `level` | `<emphasis>` | Enum | `strong`, `moderate`, `reduced` |
| `tempo` | `<segment>` | Enum | `rushed`, `steady`, `drawn-out` |
| `rhythm` | `<segment>` | Enum | `staccato`, `legato`, `syncopated` |

## Appendix C: Example Documents

### C.1 Simple Sarcasm Detection

```xml
<utterance emotion="sarcastic" confidence="0.87">
  Oh, that's
  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">
    GREAT
  </prosody>.
</utterance>
```

### C.2 Multi-Speaker Dialogue

```xml
<iml version="0.1.0" language="en-US">
  <utterance emotion="neutral" confidence="0.95" speaker_id="agent">
    How can I help you today?
  </utterance>
  <utterance emotion="frustrated" confidence="0.89" speaker_id="caller">
    I've been on hold for
    <prosody pitch="+8%" volume="+5dB">thirty minutes</prosody>
    and my
    <emphasis level="strong">account is still locked</emphasis>.
  </utterance>
  <utterance emotion="empathetic" confidence="0.82" speaker_id="agent">
    I'm
    <prosody quality="breathy" pitch="-5%">so sorry</prosody>
    about that. Let me fix this right away.
  </utterance>
</iml>
```

### C.3 Accessibility Profile Application

```xml
<!-- Speaker has autism spectrum prosody profile active -->
<iml version="0.1.0" language="en-US">
  <utterance emotion="excitement" confidence="0.78" speaker_id="user_789">
    <segment tempo="rushed" rhythm="legato">
      I just got the new keyboard and it works perfectly with the setup.
    </segment>
  </utterance>
</iml>
```

### C.4 Research-Grade Annotation

```xml
<utterance emotion="frustrated" confidence="0.92">
  <prosody f0_mean="220" f0_range="180-310" intensity_mean="72"
           intensity_range="15" speech_rate="5.1" jitter="2.1" shimmer="4.5">
    I <emphasis level="strong">can't believe</emphasis>
    <pause duration="350"/>
    this happened
    <prosody pitch="+18%" volume="+8dB" pitch_contour="rise-fall">again</prosody>!
  </prosody>
</utterance>
```

## Appendix D: RFC-2119 Conformance Summary

This specification uses the key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, REQUIRED, and OPTIONAL as defined in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119). The following table catalogues every normative requirement.

### D.1 MUST / REQUIRED / MUST NOT

| ID | Requirement | Section |
|----|-------------|---------|
| M1 | IML documents MUST be encoded in UTF-8. | 2.2 |
| M2 | `<utterance>` MUST be the top-level content element (or wrapped in `<iml>`). | 5.2 |
| M3 | `<pause>` MUST NOT contain elements or text other than whitespace. | 3.3, 5.2 |
| M4 | `<segment>` MUST be a direct child of `<utterance>`. | 5.2 |
| M5 | `<segment>` elements MUST NOT nest within other `<segment>` elements. | 5.2 |
| M6 | Producers MUST generate well-formed XML. | 6.1 |
| M7 | Producers MUST include `confidence` when `emotion` is specified on `<utterance>`. | 6.1 |
| M8 | Producers MUST express `<pause>` durations in whole milliseconds. | 6.1 |
| M9 | Consumers MUST be able to extract plain text by stripping all tags. | 6.2 |
| M10 | Consumers MUST ignore unknown tags and attributes without error. | 6.2 |
| M11 | Consumers MUST NOT assume the presence of extended attributes. | 6.2 |
| M12 | Systems handling IML data MUST obtain explicit user consent before collecting emotional annotations. | 8.1 |
| M13 | Systems handling IML data MUST treat emotion data with the same protections as other PII. | 8.1 |
| M14 | Systems handling IML data MUST provide mechanisms for users to review and delete their emotional data. | 8.1 |
| M15 | IML MUST NOT be used for deception detection. | 8.2 |
| M16 | IML MUST NOT be used for covert emotional surveillance. | 8.2 |
| M17 | IML MUST NOT be used for discriminatory profiling. | 8.2 |
| M18 | Consumers MUST ignore `x-` prefixed attributes they do not understand. | 9.2 |
| M19 | An encoding declaration, if present, MUST name UTF-8; consumers MUST NOT decode IML with another encoding. | 2.2 |
| M20 | IML documents MUST NOT contain a document type declaration. | 2.5 |
| M21 | Consumers MUST NOT resolve external entities or fetch external resources while parsing IML. | 2.5 |
| M22 | Consumers MUST NOT treat the contents of comments or processing instructions as text. | 2.5 |
| M23 | Attribute values MUST conform to their types and enumerations (Integers at most 2147483647, Floats finite). | 2.6, 3, 4 |
| M24 | `<iml>` MUST contain only `<utterance>` elements (no text or other IML elements). | 2.4, 5.2 |
| M25 | `<utterance>` and `<iml>` MUST NOT appear inside `<utterance>` or any inline element. | 3.1, 5.2 |
| M26 | `<emphasis>` MUST NOT be a direct child of another `<emphasis>`. | 3.4, 5.2 |
| M27 | Every element MUST contain only the content its content model allows. | 5.1, App. A |
| M28 | Consumers that re-serialize an IML document MUST preserve `consent` and `processing`. | 8.3 |

### D.2 SHOULD / SHOULD NOT

| ID | Requirement | Section |
|----|-------------|---------|
| S1 | The IML namespace `http://prosody-protocol.org/iml/0.1` SHOULD be used when embedding IML in other XML formats. | 2.3 |
| S2 | The combined nesting depth of `<prosody>` and `<emphasis>` SHOULD NOT exceed 2 levels. | 5.2 |
| S3 | Producers SHOULD omit attributes when values are at speaker baseline. | 6.1 |
| S4 | Producers SHOULD use the core emotion vocabulary where applicable. | 6.1 |
| S5 | Consumers SHOULD treat `confidence` values below 0.5 as low-confidence annotations. | 6.2 |
| S6 | Consumers SHOULD use speaker-specific baselines when `speaker_id` is present. | 6.2 |
| S7 | Consumers SHOULD handle unknown emotion values gracefully by treating them as `neutral`. | 3.1 |
| S8 | When a prosody profile is active, consumers SHOULD apply profile mappings before default emotion classification. | 7.2 |
| S9 | When a prosody profile is active, consumers SHOULD add the `confidence_boost` to the baseline confidence score (capped at 1.0). | 7.2 |
| S10 | When a prosody profile is active, consumers SHOULD indicate profile usage in any downstream reporting. | 7.2 |
| S11 | Systems handling IML data SHOULD support local/on-device processing where feasible. | 8.1 |
| S12 | A document SHOULD use one form (namespaced or un-namespaced) for all of its IML elements. | 2.3 |
| S13 | Producers SHOULD NOT emit unprefixed attributes that this specification does not define. | 9.2 |
| S14 | Validators SHOULD report MUST violations as errors and SHOULD violations as warnings. | 6.3 |
| S15 | Producers SHOULD write `<pause>` as a self-closing tag. | 3.3, 5.2 |
| S16 | Producers SHOULD NOT put attributes in the IML namespace. | 2.3 |

### D.3 MAY / OPTIONAL

| ID | Requirement | Section |
|----|-------------|---------|
| O1 | The XML declaration is OPTIONAL but recommended. | 2.2 |
| O2 | The `version` attribute on `<iml>` is RECOMMENDED. | 2.4 |
| O3 | The `language` attribute on `<iml>` is OPTIONAL. | 2.4 |
| O4 | `emotion` on `<utterance>` is OPTIONAL. | 3.1 |
| O5 | `speaker_id` on `<utterance>` is OPTIONAL. | 3.1 |
| O6 | Producers MAY use emotion values outside the core set. | 3.1 |
| O7 | All `<prosody>` attributes are OPTIONAL. | 3.2 |
| O8 | `<segment>` attributes (`tempo`, `rhythm`) are OPTIONAL. | 3.5 |
| O9 | Extended attributes (Section 4) are OPTIONAL; producers are NOT required to include them. | 4 |
| O10 | Custom attributes MAY be added using the `x-` prefix or as namespace-qualified attributes. | 9.2 |
| O11 | `consent` and `processing` on `<iml>` are OPTIONAL. | 2.4 |
| O12 | A UTF-8 byte order mark MAY precede the document. | 2.2 |
| O13 | Comments and processing instructions MAY appear anywhere XML allows them. | 2.5 |
| O14 | Consumers MAY reject a document that contains a document type declaration. | 2.5 |
