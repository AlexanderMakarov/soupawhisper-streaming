# SoupaWhisper Streaming

**SoupaWhisper Streaming** is a local voice dictation tool for **Linux** and **macOS**, aimed at low-resource machines and low latency. Speak into your microphone; your words are transcribed with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and either pasted to the clipboard and typed into the active input field, or streamed in as you talk.

Target workflow (streaming mode) is:
- press a hotkey and speak into your microphone until need to enter custom term or complex characters sequence (even most sophisticated speech recognition systems can't handle URLs, email addresses, names, etc.),
- type custom term/characters sequence on the keyboard,
- continue speaking,
- press a hotkey when you finished.

It increases your words-per-minute speed in 3-4 times, from ~40 WPM to ~150 WPM. Even professionals can't type faster than 75 WPM ([ref](https://www.medrxiv.org/content/10.1101/2025.05.11.25327386v1.full)).

- **Streaming:** Start/stop with the hotkey; speech is split by silence (VAD) and transcribed in chunks so text appears as you speak.
- **Push-to-talk (non-streaming):** Hold a hotkey to record, release to transcribe the full recording and insert the text. May not behave correctly in terminals or other environments that handle keyboard input in special ways.

Runs entirely on your machine — no cloud or API keys. Optional NVIDIA GPU support for faster transcription.

[![Watch the demo](https://img.youtube.com/vi/fRiqNzupudI/0.jpg)](https://youtu.be/fRiqNzupudI)

---

## Requirements

- **Python 3.10+**
- **Poetry** or **uv**
- **[PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)** — on Linux, install PortAudio via your distro (see table below). On macOS, wheels usually suffice; if `pip` fails to build PyAudio, install PortAudio with Homebrew: `brew install portaudio`.

### Linux

- **X11** (for `xclip`, `xdotool`, notifications on typical setups)

### macOS

- Clipboard and typing use **pbcopy** / **AppleScript** (no X11).
- **Global hotkeys** need **Accessibility** permission for the app that runs SoupaWhisper (e.g. Terminal or iTerm). Function keys (F10, F12, …) are matched correctly on macOS (virtual key codes vs. `Key.f10` style symbols).
- **launchd** user **LaunchAgent** is optional (`./install.sh` or `make service-reinstall`); background services **cannot** receive global hotkeys on macOS — use a **foreground** Terminal session for dictation. Use **`--test-keys`** to confirm the configured hotkey is seen (`[MATCH]` when you press it).

### System dependencies by distro (Linux)

| Distro | Install command |
|--------|-----------------|
| Ubuntu / Pop!_OS / Debian | `sudo apt install xclip xdotool libnotify-bin portaudio19-dev` |
| Fedora | `sudo dnf install xclip xdotool libnotify portaudio-devel` |
| Arch Linux | `sudo pacman -S xclip xdotool libnotify portaudio` |
| openSUSE | `sudo zypper install xclip xdotool libnotify portaudio-devel` |

---

## Installation

```bash
git clone https://github.com/AlexanderMakarov/soupawhisper-streaming.git
cd soupawhisper-streaming
chmod +x install.sh
./install.sh
```

The installer supports **Linux** and **macOS**. On Linux it uses your package manager for system libraries; on macOS it skips that step. It installs Python dependencies (Poetry or uv), creates the config file, and can install a user service — **systemd** on Linux or **launchd** (LaunchAgent) on macOS.

### Manual setup

Install system packages from the table above, then:

```bash
poetry install
# or
uv sync
```

Copy and customize the config:

```bash
mkdir -p ~/.config/soupawhisper
cp config.example.ini ~/.config/soupawhisper/config.ini
```

### Language (English, Spanish, Russian, auto-detect)

Whisper uses [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) language codes (`en`, `es`, `ru`, …). **Any language other than English-only dictation requires a multilingual model** — names **without** the `.en` suffix (for example `base`, not `base.en`). Models ending in `.en` transcribe English only.

**English (optimized):** default config often uses `model = base.en` and `language = en`.

**Spanish (or another single language):** set the code and a multilingual model, for example:

```ini
[whisper]
model = base
language = es
```

Transcription is emitted in that language’s script (e.g. Cyrillic for Russian). The focused app must accept that text (system keyboard/layout or IME as needed).

**Several languages in one session** (mixed speech or switching): use auto-detect and a multilingual model:

```ini
[whisper]
model = base
language = auto
```

**English and Russian only** (auto-detect, but never Hindi, German, etc.): with `language = auto`, Whisper can mis-guess the language on very short or noisy audio (for example while filling subscription fields). Restrict detection to a small set:

```ini
[whisper]
model = base
language = auto
language_allowlist = en, ru
```

SoupaWhisper runs language detection once per utterance, keeps only the languages you list, and transcribes using the highest-scoring one among them—so output stays in Latin or Cyrillic from those two choices instead of drifting into another language.

**Optional: enforce auto-detect using current keyboard layout (best for `en, ru`)**: if you have `language = auto` + `language_allowlist = en, ru`, you can also tell SoupaWhisper to pick the language based on your **current OS keyboard layout/input source** (Latin vs Cyrillic). This language is captured **once when you start dictation** (hotkey press) and used for the whole dictation session (no re-checks mid-stream).

```ini
[behavior]
enforce_language_from_layout = true
layout_to_language = com.apple.keylayout.US:en, com.apple.keylayout.Russian:ru, us:en, ru:ru
```

#### How `layout_to_language` works

- **Format**: comma-separated `<layout_id>:<iso639-1>` pairs (no spaces inside the id).
- **Allowed values**: any two-letter ISO 639-1 codes Whisper supports (for example `en`, `ru`, `es`, `de`).
- **Interaction with `language_allowlist`**: if `language_allowlist` is set, the enforced language must be included (otherwise the allowlist wins and normal detection is used).

#### How to find the right `layout_id`

- **macOS (recommended)**
  - The app reads the current input source id from `com.apple.HIToolbox` (selected input sources).
    Quick way to see the current `InputSourceID`:

    ```bash
    defaults export com.apple.HIToolbox - | python3 -c 'import sys, plistlib; d=plistlib.loads(sys.stdin.buffer.read()); sel=d.get("AppleSelectedInputSources") or []; print((sel[0] or {}).get("InputSourceID",""))'
    ```
  - Example outputs:
    - `com.apple.keylayout.US`
    - `com.apple.keylayout.Russian`
    - `com.apple.keylayout.ABC`

  Start with a minimal mapping:

  ```ini
  [behavior]
  enforce_language_from_layout = true
  layout_to_language = com.apple.keylayout.US:en, com.apple.keylayout.ABC:en, com.apple.keylayout.Russian:ru
  ```

  If you want to see which languages macOS associates with each input source (what SoupaWhisper uses as a fallback when no explicit mapping matches), run:

  ```bash
  defaults export com.apple.HIToolbox - | python3 -c '
import sys, plistlib
d = plistlib.loads(sys.stdin.buffer.read())
items = (d.get("AppleSelectedInputSources") or []) + (d.get("AppleInputSourceHistory") or [])
for it in items:
    if isinstance(it, dict) and "InputSourceID" in it:
        print(it.get("InputSourceID"), it.get("InputSourceLanguages"))
'
  ```

- **Linux/X11**
  - To reliably detect the **active** layout group, install one of:
    - `xkb-switch`
    - `xkblayout-state`
  - Your `layout_id` values are typically short XKB tokens like `us`, `ru`, `de`.

  Example mapping:

  ```ini
  [behavior]
  enforce_language_from_layout = true
  layout_to_language = us:en, ru:ru
  ```

  If you don’t have `xkb-switch` / `xkblayout-state`, this feature can’t reliably determine the active layout and will do nothing.

**Does this add a “second heavy” model pass?** Whisper’s CTranslate2 backend does not let you restrict the language classifier to a subset of ISO codes; it always produces scores for every language token. We reuse that same vector and only **filter** it to your allowlist—there is no lighter instruction path inside the model. The expensive step is the **text decoder** (autoregressive), and that still runs **once** per chunk. The extra work for `en, ru` is one **encoder + language head** pass via `detect_language`, which is small next to decoding. If you only need **one** language, set `language = ru` (or put a single code in `language_allowlist`) so detection is skipped entirely.

**Spanish or other languages** forced to one language: same pattern as Russian — `language = es` (etc.) and a non-`*.en` `model`.

### GPU support (optional)

For NVIDIA GPU acceleration, install cuDNN 9 and set in `~/.config/soupawhisper/config.ini`:

```ini
device = cuda
compute_type = float16
```

(See [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) for package instructions for your distro.)

---

## Usage

**Recommended:** use the Makefile (auto-picks Poetry or uv):

```bash
make run           # default mode (config-driven)
make run-stream    # streaming
make run-no-stream # non-streaming
make run-file F=path/to/audio.wav   # transcribe a file
```

Or run directly:

```bash
poetry run python dictate.py
# or
uv run python dictate.py
```

Add `--streaming` or `--no-streaming` to override config. Use `--verbose` to list audio devices.

### Behavior

- **Hotkey** (default **F12**): start/stop recording (or hold for push-to-talk in non-streaming).
- **Non-streaming:** release hotkey → full recording is transcribed, then text is copied and typed into the focused field.
- **Streaming:** press hotkey again to stop; text is emitted in chunks as silence is detected.
- **Ctrl+C** quits when run in the foreground.

### Custom terms / glossary

Whisper sometimes mishears domain-specific words (`Claude` → `cloud`) or rare phrases (`ML repository` → `a male repository`). You can give it a glossary of custom terms — names, brands, acronyms, technical vocabulary — that it should prefer. The list is passed to faster-whisper as both **`initial_prompt`** (in-context priming) and **`hotwords`** (decoder logit bias), so it works in **streaming and non-streaming modes**.

```ini
[behavior]
custom_terms = Claude, Kubernetes, GraphQL, ML repository
```

- Comma-separated; multi-word phrases are fine. Newlines also work as separators.
- Case is preserved (Whisper is case-sensitive for many terms).
- Leave empty / unset to disable.
- This is a **hint**, not a guarantee. Strongly mismatched audio can still produce other words.
- Subject to Whisper's ~224-token prompt cap; very long glossaries are silently truncated by the decoder.

### Streaming: reject specific noise phrases

Whisper can sometimes output short “filler” phrases from noise (coughs, mic bumps, etc.). In **streaming mode only**, you can configure a list of phrases to suppress.

- A chunk is skipped only if it **exactly equals** one configured phrase, ignoring punctuation (so `thank you`, `thank you.` and `THANK YOU!!!` are treated the same).
- If the chunk contains **anything else** (multiple phrases, extra words), it is **not** rejected.
- If `reject_phrases` is empty/unset, the feature is **disabled** and all chunks are emitted as-is.
- When a chunk is skipped, SoupaWhisper logs an info line: `[reject] Skipping chunk (matched reject phrase): ...`

Config example:

```ini
[behavior]
reject_phrases = thank you, thanks, okay, ok, um, hmm
```

### Modes

| Mode | Flag / config | Best for |
|------|----------------|----------|
| **Non-streaming** | `--no-streaming` or `default_streaming = false` | Short, precise dictation; full-sentence accuracy. |
| **Streaming** | `--streaming` or `default_streaming = true` | Longer dictation; text appears incrementally after short pauses. |

---

## Run as a background service

You can run SoupaWhisper as a **user service** so it starts at login: **systemd** on Linux, **launchd** on macOS. The same `make service-*` targets work on both; they call `systemctl` or `launchctl` as appropriate.

If you didn’t enable the service during install:

```bash
./install.sh   # choose 'y' when prompted (systemd on Linux, LaunchAgent on macOS)
```

Or reinstall only the service (Python deps already installed):

```bash
make service-reinstall
```

**Service commands** (Linux → `systemctl`; macOS → LaunchAgent at `~/Library/LaunchAgents/com.soupawhisper.dictate.plist`, logs at `~/Library/Logs/soupawhisper.log`):

```bash
make service-start
make service-stop
make service-restart
make service-status
make service-logs     # Linux: journalctl; macOS: tail -f ~/Library/Logs/soupawhisper.log
```

**Linux (manual systemd):**

```bash
systemctl --user start soupawhisper
systemctl --user stop soupawhisper
journalctl --user -u soupawhisper -f
```

---

## Configuration

Edit `~/.config/soupawhisper/config.ini`. Options are documented in the file.

### Audio input device

Recording uses PyAudio. To choose the input device:

1. Run with `--verbose` to print available input devices and indices.
2. In `config.ini`, under `[streaming]`, set `audio_input_device` to a device index (e.g. `0`) or a partial name (e.g. `"pulse"`, `"HDA Intel"`). Leave unset to use the system default.

---

## Troubleshooting

**No audio / wrong device**

- Run with `--verbose` and check the listed input devices.
- Set `audio_input_device` in config to the correct index or name.
- Ensure the microphone works in system settings and isn’t muted.

**Keyboard / permissions**

- **Linux:** Add your user to the `input` group so the app can read the keyboard:
  ```bash
  sudo usermod -aG input $USER
  # Log out and back in.
  ```
- **macOS – hotkey (e.g. F10) does nothing:**
  1. **Run in the foreground:** Start the app from Terminal (`uv run python dictate.py` or `make run`) so it can receive key events. The launchd service runs in the background and on macOS typically cannot receive global hotkeys.
  2. **Grant Accessibility:** Open **System Settings → Privacy & Security → Accessibility** and add **Terminal** (or iTerm / the app you use to run the command). Restart Terminal after adding.
  3. **Function keys:** If you use F10/F12 etc., ensure **System Settings → Keyboard → "Use F1, F2, etc. keys as standard function keys"** is enabled, or hold **Fn** when pressing F10 so the key is sent as F10 and not as a special key (e.g. mute).
  4. **Verify key is seen:** Run `uv run python dictate.py --test-keys` (or `poetry run python dictate.py --test-keys`). Press your hotkey (e.g. F10); you should see a line with `[MATCH]`. Press Ctrl+C to exit. If no key events appear, the app is not receiving keyboard input (permissions or run context).
  5. If you see a pynput warning about "This process is not trusted" or "Input event monitoring will not be possible", add your terminal app (Terminal, iTerm, etc.) to **System Settings → Privacy & Security → Accessibility**, then restart the app.

**Hotkey debugging (`--test-keys`)**

To check whether your configured hotkey is detected at all, run:

```bash
uv run python dictate.py --test-keys
# or: poetry run python dictate.py --test-keys
```

The app will print every key press and mark the configured hotkey with `[MATCH]`. Press keys (e.g. F10) and confirm you see the match; press Ctrl+C to exit. If keys never appear, the process is not receiving keyboard input (on macOS: add your terminal to Accessibility and run in the foreground).

**Bad transcription quality**

To verify what is being captured (e.g. wrong device, silence, or bad quality), enable persistent recordings in config and listen to the files:

- In `~/.config/soupawhisper/config.ini`, under `[behavior]`, set:
  ```ini
  save_recordings = true
  ```
- Run a short dictation. Recordings are written under `/tmp`:
  - Non-streaming: `recording_YYYYMMDD_HHMMSS.wav`
  - Streaming: `stream_chunk_YYYYMMDD_HHMMSS.wav` (one file per speech chunk)
- Play a file to hear how it sounds. Often issues are in the recording, not the transcription model.
- Set `save_recordings = false` when you are done debugging.

**cuDNN / GPU errors**

If you see errors about `libcudnn_ops.so.9`, install cuDNN 9 for your CUDA version or set `device = cpu` in config.

---

## Model sizes

Sizes are approximate. **English-only** models (`*.en`) are a bit better for English-only dictation. **Multilingual** models (same size tier, name **without** `.en`: `tiny`, `base`, `small`, `medium`, `large-v3`) are required for Russian, other non-English languages, or `language = auto`.

| Model      | Scope           | Size   | Speed   | Accuracy |
| ---------- | --------------- | ------ | ------- | -------- |
| tiny.en    | English only    | ~75MB  | Fastest | Basic    |
| base.en    | English only    | ~150MB | Fast    | Good     |
| small.en   | English only    | ~500MB | Medium  | Better   |
| medium.en  | English only    | ~1.5GB | Slower  | Great    |
| tiny       | Multilingual    | ~75MB  | Fastest | Basic    |
| base       | Multilingual    | ~150MB | Fast    | Good     |
| small      | Multilingual    | ~500MB | Medium  | Better   |
| medium     | Multilingual    | ~1.5GB | Slower  | Great    |
| large-v3   | Multilingual    | ~3GB   | Slowest | Best     |

For **English-only** streaming dictation, `base.en` or `small.en` is usually the best tradeoff. For **Russian or multilingual** use, prefer `base` / `small` / `medium` / `large-v3` (no `.en`) and set `language` as in [Language](#language-english-spanish-russian-auto-detect) above. Larger models are much slower; improving quality often starts with a better microphone.

---

## Testing

```bash
make test
```

Or:

```bash
poetry run pytest dictate_tests.py
# or
uv run pytest dictate_tests.py
```

---

# TODO/Roadmap

- [x] PyAudio for all recordings (no `arecord` dependency).
- [x] Streaming: fixes for voice duplication, race conditions, and skipped segments; corrected transcriber duration reporting.
- [x] Multiple languages support.
- [x] Support list of custom terms or pronunciation features (like accents or speech patterns).
- [ ] Option to reuse previous transcription as context (e.g. `initial_prompt`).
- [ ] Context from a first word (e.g. “Python” → prompt about Python without "Python" in the output).
- [ ] Expose more `model.transcribe()` options in config.
