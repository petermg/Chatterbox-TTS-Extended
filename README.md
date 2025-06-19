# 🚀 Chatterbox-TTS-Extended — All Features & Technical Explanations

Chatterbox-TTS-Extended is a *power-user TTS pipeline* for advanced single and batch speech synthesis, voice conversion, and artifact-free audio generation. It is based on [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox), but adds:

- **Multi-file input & batch output**
- **Custom candidate generation & validation**
- **Rich audio post-processing**
- **Whisper/faster-whisper validation**
- **Voice conversion (VC) tab**
- **Full-featured persistent UI with parallelism and artifact reduction**

---

## 📋 Table of Contents

- [Feature Summary Table](#feature-summary-table)
- [Text Input & File Handling](#text-input--file-handling)
- [Reference Audio](#reference-audio)
- [Voice/Emotion/Synthesis Controls](#voiceemotionsynthesis-controls)
- [Batching, Chunking & Grouping](#batching-chunking--grouping)
- [Text Preprocessing](#text-preprocessing)
- [Audio Post-Processing](#audio-post-processing)
- [Export & Output Options](#export--output-options)
- [Generation Logic & Quality Control](#generation-logic--quality-control)
- [Whisper Sync & Validation](#whisper-sync--validation)
- [Parallel Processing & Performance](#parallel-processing--performance)
- [Persistent Settings & UI](#persistent-settings--ui)
- [🎙️ Voice Conversion (VC) Tab](#️-voice-conversion-vc-tab)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Installation](#-installation)
- [Feedback & Contributions](#-feedback--contributions)

---

## Feature Summary Table

| Feature                                   | UI Exposed?   | Script Logic |
| ----------------------------------------- | ------------- | ------------ |
| Text input (box + multi-file upload)      | ✔             | Yes          |
| Reference audio (conditioning)            | ✔             | Yes          |
| Separate/merge file output                | ✔             | Yes          |
| Emotion, CFG, temperature, seed           | ✔             | Yes          |
| Batch/smart-append/split (sentences)      | ✔             | Yes          |
| Sound word remove/replace                 | ✔             | Yes          |
| Inline reference number removal           | ✔             | Yes          |
| Dot-letter ("J.R.R.") correction          | ✔             | Yes          |
| Lowercase & whitespace normalization      | ✔             | Yes          |
| Auto-Editor post-processing               | ✔             | Yes          |
| FFmpeg normalization (EBU/peak)           | ✔             | Yes          |
| WAV/MP3/FLAC export                       | ✔             | Yes          |
| Candidates per chunk, retries, fallback   | ✔             | Yes          |
| Parallelism (workers)                     | ✔             | Yes          |
| Whisper/faster-whisper backend            | ✔             | Yes          |
| Persistent settings (JSON/CSV per output) | ✔             | Yes          |
| Settings load/save in UI                  | ✔             | Yes          |
| Pause Tag Support                         | ✔             | Yes          |
| Audio preview & download                  | ✔             | Yes          |
| Help/Instructions                         | ✔ (Accordion) | Yes          |
| Voice Conversion (VC tab)                 | ✔             | Yes          |

---

## Text Input & File Handling

- **Text box:** For direct text entry (single or multi-line).
- **Multi-file upload:** Drag-and-drop any number of `.txt` files.
  - Choose to merge them into one audio or process each as a separate output file.
  - Outputs are named for sorting and reproducibility.
- **Reference audio input:** Upload or record a sample to condition the generated voice.
- **Settings file support:** Load or save all UI settings as JSON for easy workflow repeatability.

---

## Reference Audio

- **Voice Prompt (Conditioning):**
  - Upload or record an audio reference.
  - The TTS engine mimics the style, timbre, or emotion from the provided sample.
  - Handles missing/invalid reference audio gracefully.

---

## Voice/Emotion/Synthesis Controls

- **Emotion exaggeration:** Slider (0 = flat/neutral, 1 = normal, 2 = exaggerated emotion).
- **CFG Weight/Pace:** Controls strictness and speech pacing. High = literal, monotone. Low = expressive, dynamic.
- **Temperature:** Controls voice randomness/variety.
- **Random seed:** 0 = new random each run. Any number = repeatable generations.

---

## Batching, Chunking & Grouping

- **Sentence batching:** Groups sentences up to 300 characters per chunk (adjustable in code).
- **Smart-append short sentences:** When batching is off, merges very short sentences for smooth prosody.
- **Recursive long sentence splitting:** Automatically splits long sentences at `; : - ,` or by character count.
- **Parallel chunk processing:** Multiple chunks are generated at once for speed (user control).

---

## Text Preprocessing

- **Lowercase conversion:** Makes all text lowercase (optional).
- **Whitespace normalization:** Strips extra spaces/newlines.
- **Dot-letter fix:** Converts `"J.R.R."` to `"J R R"` to improve initialisms and names.
- **Inline reference number removal:** Automatically removes numbers after sentence-ending punctuation (e.g., `.188` or `.”3`).
- **Sound word removal/replacement:** Configurable box for unwanted noises or phrases, e.g. `um`, `ahh`, or custom mappings like `zzz=>sigh`.
  - Handles standalone words, possessives, quoted patterns, and dash/punctuation-only removals.
- **Pause Tag:** `[pause:x.xs]` tags are parsed to add custom, mandatory silence lengths at precisions of down to 0.1s .  Ex: `Hello[pause:0.5s]world`
---

## Audio Post-Processing

- **Auto-Editor integration:**
  - Trims silences/stutters/artifacts after generation.
  - **Threshold** and **margin** are adjustable in UI.
  - **Option to keep original WAV** before cleanup.
- **FFmpeg normalization:**
  - **EBU R128:** Target loudness, true peak, dynamic range.
  - **Peak:** Quick normalization to prevent clipping.
  - All normalization parameters are user-adjustable.

---

## Export & Output Options

- **Multiple audio formats:** WAV (uncompressed), MP3 (320k), FLAC (lossless). Any/all selectable in UI.
- **Output file naming:** Each output includes base name, timestamp, generation, and seed for tracking.
- **Batch export:** If “separate files” is checked, each text file gets its own processed output.

---

## Generation Logic & Quality Control

- **Number of generations:** Generate multiple different outputs at once (“takes”).
- **Candidates per chunk:** For each chunk, generate multiple variants.
- **Max attempts per candidate:** If validation fails, retries up to N times for best result.
- **Whisper validation:** Uses speech-to-text to check each candidate and picks the closest transcript match (can bypass for speed).
- **Fallback strategies:** If all candidates fail, use the longest transcript or highest similarity score.

---

## Whisper Sync & Validation

- **Model choice:** Select between OpenAI Whisper and faster-whisper (SYSTRAN). Both have multiple model sizes (VRAM vs. speed tradeoff).
- **Whisper backend and size exposed in UI:** Shows VRAM estimates and auto-disables if not needed.
- **Per-chunk Whisper validation:** Each audio chunk is transcribed and compared to its intended text.
- **Fallbacks:** If all candidates fail, configurable selection of longest transcript or highest score.
- **Bypass option:** Skip Whisper entirely (faster, but riskier for artifacts).

---

## Parallel Processing & Performance

- **Full parallelism:** User-configurable worker count (default 4).
- **Worker control:** Set to 1 for low-memory or debugging, higher for speed.
- **VRAM management:** Cleans up GPU memory after Whisper use to avoid leaks.

---

## Persistent Settings & UI

- **JSON settings:** UI choices are saved/restored automatically, with option to import/export.
- **Per-output settings:** Every output audio file also gets a `.settings.json` and `.settings.csv` with all relevant parameters (for reproducibility and workflow management).
- **Complete Gradio UI:** All options available as toggles, sliders, dropdowns, checkboxes, and file pickers.
- **Audio preview/download:** Listen to or download any generated output from the UI.
- **Help/Instructions:** Accordion panel with detailed explanations of every feature and control.

---

## 🎙️ Voice Conversion (VC) Tab

Convert any voice to sound like another!\
**The Voice Conversion tab lets you:**

- Upload or record the **input audio** (the voice to convert).
- Upload or record the **target/reference voice** (the voice to match).
- Click **Run Voice Conversion** — get a new audio file with the same words but the target voice!

**Technical highlights:**

- Handles long audio by splitting into overlapping chunks, recombining with crossfades for seamless transitions.
- Output matches model’s sample rate and fidelity.
- Automatic chunking and processing—no manual intervention needed.
- Option to disable watermarking.

---

## Tips & Troubleshooting

- **Out of VRAM or slow?**
  - Lower parallel workers
  - Use a smaller/faster Whisper model
  - Reduce number of candidates
- **Artifacts/Errors?**
  - Increase candidates/retries
  - Adjust auto-editor threshold/margin
  - Refine sound word replacements
- **Choppy audio?**
  - Increase auto-editor margin
  - Lower threshold
- **Reproducibility**
  - Use a fixed random seed

---

## 📝 Installation

Requires Python 3.10.x and [FFMPEG](https://ffmpeg.org/download.html).

Clone the repo:

```bash
git clone https://github.com/petermg/Chatterbox-TTS-Extended
```

Install requirements:

```bash
pip install --force-reinstall -r requirements.txt
# If needed, try requirements.base.with.versions.txt or requirements_frozen.txt
```

Run:

```bash
python Chatter.py
```

If FFMPEG isn’t in your PATH, put the executable in the same directory as your script.

---

## 📣 Feedback & Contributions

Open an issue or pull request for suggestions, bug reports, or improvements!

