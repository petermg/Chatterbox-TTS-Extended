# 🚀 Chatterbox-TTS-Extended — All Features & Technical Explanations

Chatterbox-TTS-Extended is a *power-user TTS pipeline* for advanced single and batch speech synthesis, voice conversion, and artifact-reduced audio generation. It is based on [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox), but adds:

- **Multi-file input & batch output**
- **Custom candidate generation & validation**
- **Rich audio post-processing**
- **Whisper/faster-whisper validation**
- **Voice conversion (VC) tab**
- **Full-featured persistent UI with parallelism and artifact reduction**
- **Optional audio denoising with pyrnnoise - removes most artifacts**

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
- [Known Bugs](#known-bugs)

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
| **pyrnnoise denoising (RNNoise)**         | ✔             | Yes          |
| FFmpeg normalization (EBU/peak)           | ✔             | Yes          |
| WAV/MP3/FLAC export                       | ✔             | Yes          |
| Candidates per chunk, retries, fallback   | ✔             | Yes          |
| Parallelism (workers)                     | ✔             | Yes          |
| Whisper/faster-whisper backend            | ✔             | Yes          |
| Persistent settings (JSON/CSV per output) | ✔             | Yes          |
| Settings load/save in UI                  | ✔             | Yes          |
| Audio preview & download                  | ✔             | Yes          |
| Help/Instructions                         | ✔ (Accordion) | Yes          |
| Voice Conversion (VC tab)                 | ✔             | Yes          |

---

## Text Input & File Handling

- **Text box:** For direct text entry (single or multi-line).
- **Multi-file upload:** Drag-and-drop any number of `.txt` files.
  - Choose to merge them into one audio or process **each as a separate output** (toggle in the UI).
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

- **Sentence batching:** Groups sentences up to ~300 characters per chunk (adjustable in code).
- **Smart-append short sentences:** When batching is off, merges very short sentences for smoother prosody.
- **Recursive long sentence splitting:** Automatically splits long sentences at `; : - ,` or by character count.
- **Parallel chunk processing:** Multiple chunks are generated at once for speed (user control).

---

## Text Preprocessing

- **Lowercase conversion:** Makes all text lowercase (optional).
- **Whitespace normalization:** Strips extra spaces/newlines.
- **Dot-letter fix:** Converts `"J.R.R."` to `"J R R"` to improve initialisms and names.
- **Inline reference number removal:** Automatically removes numbers after sentence-ending punctuation (e.g., `.188` or `.”3`).
- **Sound word removal/replacement:** Configurable list for unwanted noises or phrases, e.g., `um`, `ahh`, or mappings like `zzz=>sigh`.
  - Handles standalone words, possessives, quoted patterns, and dash/punctuation-only removals.

---

## Audio Post-Processing

- **pyrnnoise Denoising (RNNoise):**
  - Optional toggle for almost 100% artifact removal.
  - Runs **before** Auto-Editor and normalization.
  - Uses the `denoise` CLI if available; otherwise falls back to the Python API.
  - Temporary conversion to **48 kHz mono s16** for best compatibility; output is restored to original sample rate.
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
- **Batch export:** If “separate files” is checked, each uploaded text file gets its own processed output.
- **Disable watermarking:** Optional toggle to disable watermarking during generation.

---

## Generation Logic & Quality Control

- **Number of generations:** Produce multiple different outputs at once (“takes”).  
- **Candidates per chunk:** For each chunk, generate multiple variants.  
- **Max attempts per candidate:** If validation fails, retry up to N times.  
- **Deterministic seeding:** A per-chunk/per-candidate/per-attempt seed is derived from the base seed for reproducibility.  
- **Fallback strategies:** If all candidates fail validation, use the longest transcript or highest similarity score.

---

## Whisper Sync & Validation

- **Backends:** Choose **OpenAI Whisper** or **faster-whisper** (SYSTRAN), with multiple model sizes (VRAM vs. speed tradeoff).
- **Per-chunk validation:** Each audio chunk is transcribed and compared to its intended text.
- **Bypass option:** Skip Whisper entirely (faster, but may allow more TTS errors).
- **Use-longest-transcript-on-fail:** Optional fallback if no candidate passes validation.

---

## Parallel Processing & Performance

- **Full parallelism:** User-configurable worker count (default 4).
- **Worker control:** Set to 1 for low-memory or debugging, higher for speed.
- **VRAM management:** Clears Whisper model and GPU cache after validation to avoid leaks.

---

## Persistent Settings & UI

- **JSON settings:** UI choices are saved/restored automatically; import/export supported.
- **Per-output settings artifacts:** Each output also writes a `.settings.json` and `.settings.csv` capturing all parameters and output filenames.
- **Complete Gradio UI:** All options available as toggles, sliders, dropdowns, checkboxes, and file pickers.
- **Audio preview & download:** Listen to or download any generated output from the UI.
- **Help/Instructions:** Accordion with detailed guidance for each setting.

---

## 🎙️ Voice Conversion (VC) Tab

Convert any voice to sound like another!

**The Voice Conversion tab lets you:**
- Upload or record the **input audio** (the voice to convert).
- Upload or record the **target/reference voice** (the voice to match).
- Adjust pitch (optional)
- Click **Run Voice Conversion** — get a new audio file with the same words but the target voice!

**Technical highlights:**
- Handles long audio by splitting into overlapping chunks and recombining with crossfades.
- Output matches the model’s sample rate and fidelity.
- Automatic chunking and processing—no manual intervention needed.
- **Pitch shift** control.
- Option to **disable watermarking**.

---

## Tips & Troubleshooting

- **Background noise in output?**
  - Enable **pyrnnoise denoising** in the UI to clean up artifacts.
  - Denoising runs before Auto-Editor and normalization for best results.
- **Out of VRAM or slow?**
  - Lower parallel workers, pick a smaller/faster Whisper model, reduce candidates.
- **Artifacts/Errors?**
  - Increase candidates/retries, adjust Auto-Editor threshold/margin, refine sound word replacements.
- **Choppy audio?**
  - Increase Auto-Editor margin; lower threshold.
- **Reproducibility**
  - Set a fixed random seed.

---

## 📝 Installation

Requires **Python 3.10.x** and **[FFmpeg](https://ffmpeg.org/download.html)** (on PATH).

Clone the repo:
```bash
git clone https://github.com/petermg/Chatterbox-TTS-Extended
```

Install requirements:
```bash
pip install --force-reinstall -r requirements.txt
# If needed, try requirements.base.with.versions.txt or requirements_frozen.txt
```

### Updating dependencies

`requirements.txt` pins exact versions for reproducible installs. When upgrading a
package, update its version in `requirements.txt` and reinstall:

```bash
pip install --upgrade <package>
pip install --force-reinstall -r requirements.txt
```
After adjusting versions, run the test suite to ensure everything works as expected.

Run:
```bash
# Use your repo's main file. For example:
python Chatter.py
# or, if your file is named like this branch:
python zChatter.py
```

If [FFmpeg](https://ffmpeg.org/download.html) isn’t in your PATH, place the executable alongside the script or add it to PATH.

---

## 📣 Feedback & Contributions

Open an issue or pull request for suggestions, bug reports, or improvements!

---

## Known Bugs:

It seems if you use fasterwhisper for validation, sometimes it just silently crashes. Apparently this has to do with using the fasterwhisper model. It's not actually the python code. So if you are experiencing this, switch back to the original WhisperSync model. 
UPDATE: with the latest update this bug may have been resolved.

---


