# SoupaWhisper

**SoupaWhisper** is a local voice dictation tool for Linux designed for low resources machines and low latency. Speak into your microphone; your words are transcribed with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and either pasted to the clipboard and typed into the active input field, or streamed in as you talk.

Target workflow (streaming mode) is:
- press a hotkey and speak into your microphone until need to enter custom term or complex characters sequence,
- wait a moment until text appears in the active input field,
- type custom term/characters sequence,
- continue speaking,
- press a hotkey again to stop transcribing and get recent words in the active input field.

It increases your words-per-minute speed in 3-4 times, from ~40 WPM to ~150 WPM. Even professionals can't type faster than 75 WPM ([ref](https://www.medrxiv.org/content/10.1101/2025.05.11.25327386v1.full)).

- **Push-to-talk (non-streaming):** Hold a hotkey to record, release to transcribe the full recording and insert the text.
- **Streaming:** Start/stop with the hotkey; speech is split by silence (VAD) and transcribed in chunks so text appears as you speak.

Runs entirely on your machine — no cloud or API keys. Optional NVIDIA GPU support for faster transcription.

> **Note:** Streaming push-to-talk works in most apps but may not behave correctly in terminals or other environments that handle keyboard input in special ways.

---

## Requirements

- **Python 3.10+**
- **Poetry** or **uv**
- **Linux with X11** (for `xclip`, `xdotool`, notifications)
- **[PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)** (system packages below)

### System dependencies by distro

| Distro | Install command |
|--------|-----------------|
| Ubuntu / Pop!_OS / Debian | `sudo apt install xclip xdotool libnotify-bin portaudio19-dev` |
| Fedora | `sudo dnf install xclip xdotool libnotify portaudio-devel` |
| Arch Linux | `sudo pacman -S xclip xdotool libnotify portaudio` |
| openSUSE | `sudo zypper install xclip xdotool libnotify portaudio-devel` |

---

## Installation

```bash
git clone https://github.com/ksred/soupawhisper.git
cd soupawhisper
chmod +x install.sh
./install.sh
```

The installer will detect your package manager, install system and Python dependencies (Poetry or uv), create the config file, and optionally install a systemd user service.

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

### Modes

| Mode | Flag / config | Best for |
|------|----------------|----------|
| **Non-streaming** | `--no-streaming` or `default_streaming = false` | Short, precise dictation; full-sentence accuracy. |
| **Streaming** | `--streaming` or `default_streaming = true` | Longer dictation; text appears incrementally after short pauses. |

---

## Run as a systemd service

If you didn’t enable it during install:

```bash
./install.sh   # choose 'y' for systemd
```

Or reinstall only the service (deps already installed):

```bash
make service-reinstall
```

**Service commands:**

```bash
make service-start    # or systemctl --user start soupawhisper
make service-stop     # or systemctl --user stop soupawhisper
make service-restart  # or systemctl --user restart soupawhisper
make service-status   # or systemctl --user status soupawhisper
make service-logs     # or journalctl --user -u soupawhisper -f
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

```bash
sudo usermod -aG input $USER
# then log out and back in
```

**cuDNN / GPU errors**

If you see errors about `libcudnn_ops.so.9`, install cuDNN 9 for your CUDA version or set `device = cpu` in config.

---

## Model sizes

| Model     | Size   | Speed   | Accuracy |
| --------- | ------ | ------- | -------- |
| tiny.en   | ~75MB  | Fastest | Basic    |
| base.en   | ~150MB | Fast    | Good     |
| small.en  | ~500MB | Medium  | Better   |
| medium.en | ~1.5GB | Slower  | Great    |
| large-v3  | ~3GB   | Slowest | Best     |

For dictation, `base.en` or `small.en` is usually the best tradeoff.

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
- [ ] Support list of custom terms or pronunciation features (like accents or speech patterns).
- [ ] Option to reuse previous transcription as context (e.g. `initial_prompt`).
- [ ] Context from a first word (e.g. “Python” → prompt about Python without "Python" in the output).
- [ ] Multiple languages support.
- [ ] Expose more `model.transcribe()` options in config.
