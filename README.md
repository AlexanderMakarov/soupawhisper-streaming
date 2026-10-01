# SoupaWhisper Streaming

Built-in speech-to-text (macOS dictation / F6, Windows voice typing) and typical local tools (Handy and similar) are push-to-talk: you speak, stop, and get one block of text. That breaks down the moment you need a URL, patient ID, case cite, API path, or proper name the model will mangle — you cancel dictation, type by hand, then start again.

**SoupaWhisper Streaming** is for that gap. It is local voice dictation for **Linux** and **macOS** where you **speak and type in the same breath**: dictate the boilerplate, type the precise bit yourself, keep talking. Technical workers (IT, medical, legal, and anyone living in forms, tickets, and docs) get the speed of speech without giving up control over the tokens that must be exact.

On-device [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no cloud, no API keys. Optional NVIDIA GPU. A tray / menu-bar icon shows when the mic is live and which language the session is using.

### How you use it (streaming)

1. Press the hotkey and speak until you need a URL, email, ID, cite, or other awkward sequence.
2. Type that bit on the keyboard (text already inserted keeps flowing around it).
3. Keep speaking.
4. Press the hotkey again when you are done.

Also available: **push-to-talk** (hold to record, release to insert the full utterance). Streaming is usually smoother in terminals and apps that mishandle held keys.

Typical typing is ~40 WPM for most people and ~75 WPM for professionals ([ref](https://www.medrxiv.org/content/10.1101/2025.05.11.25327386v1.full)); this flow often reaches ~150 WPM on the prose while you still type the irreplaceable tokens by hand.

[![Watch the demo](https://img.youtube.com/vi/fRiqNzupudI/0.jpg)](https://youtu.be/fRiqNzupudI)

### Why you should try it

- **Speak + type in parallel** — the main advantage over OS dictation and one-shot tools: insert speech in chunks while the keyboard stays yours.
- **Always-visible status** — tray / menu-bar icon for idle, loading, recording, transcribing, error, plus an `EN` / `RU` / `AUTO` language chip; Start/Stop and Quit from the menu.
- **Two languages that follow the keyboard** — with `language=auto`, allowlists and `enforce_language_from_layout` lock Whisper to the layout at hotkey press for the whole session.
- **Domain vocabulary** — `custom_terms` bias the decoder while it runs (`hotwords` + `initial_prompt`), not a post-hoc find-and-replace.
- **Background-friendly** — user service (systemd / launchd), one hotkey, config file; built for ordinary laptops.

---

## How it differs from other local dictation tools

| Capability | SoupaWhisper | Typical push-to-talk / OS dictation |
|---|---|---|
| Text insertion | Chunk by chunk **while you speak** — type a URL or name mid-dictation and keep talking | One paste after you stop (even if a preview was shown live) |
| Status UI | Tray / menu-bar: mic state, language chip, Start/Stop menu | Often none, or only a brief notification |
| Two-language dictation | Allowlist + active keyboard layout at hotkey press | One global language, or auto-detect that drifts on short chunks |
| Custom terms | Bias during transcription | Usually replace after the fact |
| Control | Tray + config + CLI + user service | Full GUI, or no tray |

Not yet: a model manager, or backends other than faster-whisper.

---

## Install

Needs **Python 3.10+**, **Poetry** or **uv**, and PortAudio (Linux packages below; on macOS `brew install portaudio` only if PyAudio fails to build). Linux dictation uses **X11** tools (`xclip`, `xdotool`, notifications).

```bash
git clone https://github.com/AlexanderMakarov/soupawhisper-streaming.git
cd soupawhisper-streaming
chmod +x install.sh
./install.sh
```

`install.sh` installs system libraries (Linux), Python deps, copies `config.example.ini` → `~/.config/soupawhisper/config.ini`, and can enable a user service (**systemd** on Linux, **launchd** LaunchAgent on macOS).

**Linux packages** (if you skip the installer or need to reinstall):

| Distro | Packages |
|--------|----------|
| Ubuntu / Pop!_OS / Debian / Mint | `sudo apt install xclip xdotool libnotify-bin portaudio19-dev gir1.2-ayatanaappindicator3-0.1 python3-gi python3-gi-cairo` |
| Fedora | `sudo dnf install xclip xdotool libnotify portaudio-devel libappindicator-gtk3` |
| Arch | `sudo pacman -S xclip xdotool libnotify portaudio libayatana-appindicator` |
| openSUSE | `sudo zypper install xclip xdotool libnotify portaudio-devel` |

**Tray on Linux:** pystray needs a StatusNotifier/AppIndicator host **and** `gi` for the **same** Python that runs SoupaWhisper. Prefer distro Python + `--system-site-packages` (Option A). Without a tray host or `gi`, set `tray_icon = false` or the process exits when the tray is required.

```bash
# Option A (Mint/Ubuntu 24.04): OS Python 3.12 + apt gi
sudo apt install python3-gi python3-gi-cairo gir1.2-ayatanaappindicator3-0.1 python3.12-dev portaudio19-dev
uv venv --python /usr/bin/python3.12 --system-site-packages && uv sync
# Expect: .venv/bin/python -c "import pystray; print(pystray.Icon.__module__)" → pystray._appindicator
```

Option B (uv/pyenv Python without matching `gi`): build PyGObject into that venv, or use a toolchain that ships GObject packages for that interpreter — there is no pure-PyPI tray path. On GNOME, enable an AppIndicator extension.

**macOS:** clipboard uses pbcopy and typing uses Quartz key events, so **Accessibility** is the only permission needed — grant it to whatever runs SoupaWhisper (the terminal, or the interpreter the LaunchAgent starts). The LaunchAgent receives global hotkeys once **Accessibility** is granted to the interpreter it launches (`.venv/bin/python`), which macOS lists simply as "Python"; until then the log warns `This process is not trusted`. Check with `uv run python dictate.py --test-keys` (`[MATCH]` on your hotkey).

**Manual deps only** (no installer): install the packages above, then `poetry install` or `uv sync`, and `cp config.example.ini ~/.config/soupawhisper/config.ini`.

**GPU (optional):** install cuDNN 9, then `device = cuda` and `compute_type = float16` in config.

---

## Run

```bash
make run            # config default
make run-stream     # streaming
make run-no-stream  # push-to-talk
make run-file F=path/to/audio.wav
make transcribe F=path/to/video.mov   # needs ffmpeg → writes .srt next to the file (EXT=txt for plain text)
make test
```

`make` picks Poetry or uv. Equivalents: `uv run python dictate.py` / `poetry run python dictate.py`, with `--streaming` / `--no-streaming` / `--verbose` as needed. Hotkey default is **F12**. Ctrl+C quits a foreground process.

### User service

```bash
make service-reinstall   # or choose 'y' during ./install.sh
make service-start | service-stop | service-restart | service-status | service-logs
```

Same targets map to `systemctl --user` (Linux) or the LaunchAgent / `~/Library/Logs/soupawhisper.log` (macOS). Reminder: on macOS, global hotkeys generally need a foreground run, not the agent.

---

## Configuration

Edit `~/.config/soupawhisper/config.ini` (comments in the file are authoritative). Highlights:

**Language** — ISO 639-1 codes. Non-English or `auto` needs a **multilingual** model (name **without** `.en`, e.g. `base`). English-only: `base.en` + `language = en`. Mixed / bilingual:

```ini
[whisper]
model = base
language = auto
language_allowlist = en, ru

[behavior]
enforce_language_from_layout = true
layout_to_language = us:en, ru:ru, com.apple.keylayout.US:en, com.apple.keylayout.Russian:ru
```

Layout language is captured **once at hotkey press** for the session. On Linux, active layout uses `xkb-switch` / `xkblayout-state` if present, else X11 `XkbGetState` + `setxkbmap`. Map `us` → `en`. Allowlist detection still runs Whisper’s normal language scores and **filters** them — not a second full decode; one language in config/allowlist skips detect entirely.

**Custom terms** — `custom_terms = Claude, Kubernetes, GraphQL` (comma or newlines; hint to the decoder, not a guarantee).

**Streaming noise rejects** — `reject_phrases = thank you, thanks, okay` skips a chunk only when the whole chunk equals a phrase (punctuation ignored).

**Tray** — `tray_icon = true`, `tray_show_language = true` (on-icon chip; idle stays `AUTO` with `language=auto` until a session starts). Fail-hard if tray cannot start; set `tray_icon = false` for headless.

**Audio device** — `--verbose` lists devices; set `audio_input_device` under `[streaming]` to an index or partial name.

### Modes (behavior)

| Mode | Flag / config | What happens |
|------|----------------|--------------|
| **Streaming** | `--streaming` / `default_streaming = true` | Press hotkey to start, press again to stop. Silence (VAD) splits speech into chunks; each chunk is transcribed and inserted as you go — this is the speak+type flow. |
| **Non-streaming (push-to-talk)** | `--no-streaming` / `default_streaming = false` | Hold hotkey to record, release to transcribe the whole buffer once, then clipboard + type. Better for short precise utterances; held keys can misbehave in some terminals/apps. |

Streaming uses recent text for better recognition. Under `[streaming]`, `context_words` controls its size (default `50`; `0` disables) and `context_reset_seconds` sets the silence reset interval (default `5.0`).

Default hotkey is **F12** (configurable). Ctrl+C quits a foreground process.

---

## Meeting recording

Records a call — Google Meet, Zoom, Teams, anything — as **two separate tracks**, then transcribes both and merges them into one speaker-labelled Markdown transcript. Because each speaker has their own file, "who said what" comes from the filename; no diarization model is involved.

- **Your voice** — the default source (mic).
- **Everyone else** — the default sink's *monitor*, i.e. whatever your speakers play. This works below the app layer, so the meeting app neither cooperates nor notices.

Capturing the other side means recording what your speakers play, which is an OS-level facility rather than something the meeting app offers. That is currently wired to PipeWire's `pw-record`, so meeting mode is Linux-only for now; the equivalent on macOS needs a virtual audio device and on Windows WASAPI loopback. Dictation itself runs on all three.

> **Use headphones.** On open speakers your mic also picks up the other participants, so their words land on **both** tracks and are repeated under `Me:`. Headphones make the two tracks cleanly separate.

Meeting mode and dictation are **mutually exclusive**: while a meeting records, the dictation hotkey does nothing. That is deliberate — a stray F12 mid-call would type transcribed text into whatever window has focus.

### Using it

Start and stop from the tray menu (**Start meeting recording**), or set `hotkey = shift+f12` under `[meeting]`. The tray icon switches to a two-people glyph for the whole call, and the menu shows how much scratch audio is on disk with a **Delete recorded audio files** action.

Nothing is transcribed while recording — two `pw-record` processes write WAV straight to disk at near-zero CPU, so nothing competes with the video call. Transcription runs once, after you stop.

### Where files go

| What | Where | Lifetime |
|---|---|---|
| Merged transcript (`.md`) | `transcript_dir` (default `~/Documents/meetings`) **and** the session dir | Kept |
| Per-track `.srt`, optional `.words.json`, `transcribe.log` | `$TMPDIR/soupawhisper-streaming/meetings/<timestamp>/` | Kept until the 30-day `/tmp` sweep |
| Track audio (`.wav`) | same session dir | Deleted once the transcript is written |

Audio is deleted **only after** the transcript lands. If transcription fails, the WAVs are kept and the notification says where — they are the only copy of the meeting.

### Speed

Meeting transcription uses its **own** model instance, so dictation is usable the moment recording stops rather than queueing behind it. `cpu_threads` (default: half the logical cores) caps it.

Measured on an i5-10300H (4 physical cores + HT), 57s of silence-heavy audio:

| `cpu_threads` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| time | 32.2s | 19.1s | **12.9s** | 37.1s |

Past half the logical cores it oversubscribes and gets slower, so that default is both the fastest setting and the one that leaves CPU for dictation.

Progress goes to `transcribe.log` in the session directory — one line per speech run with its offset, duration and detected language, plus an ETA every 10% — and to the tray as `Transcribing meeting… mic 45%`.

### Quality

**What works.** Each track is cut into speech runs and the language is detected per run, so a call that switches languages decodes correctly and timestamps land on real speech. Set `language_allowlist` (e.g. `en, ru`) to keep detection between plausible languages; short or noisy runs otherwise score highest on things like `la`. Silence is never sent to Whisper, and low-confidence output is dropped — together these stop the invented text Whisper produces when it has nothing to hear (`Thanks for watching!`, subtitle-translator credits).

**What does not.** Strongly accented speech is sometimes detected as the wrong language and comes back as phonetic nonsense; on one 73-minute interview this hit 9% of runs. A larger model detects better if you can afford the decode time. Long monologues split every `run_max_seconds`, so a language change inside one is only caught at that boundary.

**Tuning.** Set `keep_audio = true`, record a sample, then re-run the same WAVs with different values and compare — `transcribe.log` shows what each run was detected as.

| Setting | Default | Raise it to… | Lower it to… |
|---|---|---|---|
| `run_min_silence_ms` | `700` | keep a sentence's pauses in one run | let the language switch more often |
| `run_pad_ms` | `200` | stop clipped first/last words | keep neighbouring noise out |
| `run_vad_threshold` | `0.5` | admit only clear speech | catch quiet or distant talk |
| `run_min_speech_ms` | `0` | drop coughs and clicks | keep one-word answers |
| `run_max_seconds` | `30` | transcribe faster | let a long monologue change language |
| `min_avg_logprob` | `-1.5` | cut more invented text | keep more short real words |
| `me_max_block_seconds` | `60` | keep a long answer whole | get finer timestamps |
| `them_max_block_seconds` | `20` | keep a long answer whole | separate speakers' turns better |

`[behavior] custom_terms` and `reject_phrases` apply here too. A glossary fixes domain words — without it Whisper wrote `sync 8 group` and `cup flow, air flow`, with it `sync.WaitGroup` and `Kubeflow, Airflow` — but a listed term can also be forced onto audio that never contained it, so keep the list to words that actually recur.

### Measuring conversation timing

Every transcript opens with YAML frontmatter summarising the conversation:

```yaml
---
me_them_speaking_ratio: 5.36        # your speaking time / theirs
me_speaking_s: 111.4
them_speaking_s: 20.8
me_wpm_avg: 147.6           # words per minute, averaged over your turns
me_wpm_stdev: 13.0
them_wpm_avg: 173.6
them_wpm_stdev: 3.6
me_think_time_avg_s: 1.8    # your pause before answering them
me_think_time_stdev_s: 0.7
them_think_time_avg_s: 3.2
them_think_time_stdev_s:    # empty when there was too little to measure
---
```

Rates and pauses are measured per turn, not per subtitle cue, so a half-second cue holding one word cannot skew them. Think time counts only pauses where the speaker actually changed — a gap between two of your own turns is you drawing breath, not deciding — and overlaps are excluded rather than averaged in as negative waits. Spread is the population standard deviation, in the same unit as the average.

For finer analysis, `word_timestamps = true` additionally writes `mic.words.json` and `them.words.json` with per-word start and end times. Both tracks share one clock, so these support response latency, pause length, speaking rate, talk-time ratio and interruption counts. It costs extra transcription time and leaves the `.srt` and `.md` output unchanged.


### Requirements

`pw-record` (from `pipewire-bin` / `pipewire-utils`). `parec` is deliberately not used: on PipeWire's PulseAudio shim it was measured dropping the first ~2 s of every mic capture. The participants' track uses `stream.capture.sink=true`, so it follows the default sink — plugging in headphones mid-meeting does not leave it recording a dead device.

---

## Troubleshooting

**No audio / wrong device**

If no usable audio is captured, the app shows a “No audio detected” notification once per dictation session by default.

- Run with `--verbose` and check the listed input devices.
- Set `audio_input_device` in config to the correct index or name.
- Ensure the microphone works in system settings and is not muted.

**Bad transcription / suspect the mic, not the model**

Enable persistent capture, dictate briefly, then listen:

```ini
[behavior]
save_recordings = true
```

Files land under `/tmp`: `recording_YYYYMMDD_HHMMSS.wav` (non-streaming) or `stream_chunk_YYYYMMDD_HHMMSS.wav` (one per streaming chunk). Disable `save_recordings` when finished debugging.

**Linux: hotkey does nothing**

Add your user to the `input` group, then log out and back in:

```bash
sudo usermod -aG input $USER
```

**macOS: hotkey (e.g. F10/F12) does nothing**

1. **System Settings → Privacy & Security → Accessibility** — enable the entry for whatever runs SoupaWhisper: the LaunchAgent's interpreter (listed as "Python"), or Terminal/iTerm for a foreground run. Then restart it — the grant is only read at process start.
2. Note that macOS keys this permission to the interpreter's path, so rebuilding `.venv` or upgrading Homebrew Python revokes it and you must grant the new binary.
3. Prefer **Use F1, F2, etc. as standard function keys**, or hold **Fn** so F-keys are not media keys.
4. Verify with `--test-keys` (below). A “process is not trusted” / monitoring warning means Accessibility is still missing.

**Hotkey debugging (`--test-keys`)**

```bash
uv run python dictate.py --test-keys
# or: poetry run python dictate.py --test-keys
```

Every key press is printed; the configured hotkey is marked `[MATCH]`. Ctrl+C exits. If nothing appears, the process is not receiving keyboard input (permissions or background launch on macOS).

**Tray icon missing (Linux)**

Confirm AppIndicator backend (`pystray._appindicator`, not `_xorg`), a panel StatusNotifier / AppIndicator host, and `gi` for this Python (see Install Option A). Or set `tray_icon = false`.

**cuDNN / GPU errors**

If you see errors about `libcudnn_ops.so.9`, install cuDNN 9 for your CUDA version or set `device = cpu` in config.

---

## Model sizes

English-only `*.en` models are slightly better for English-only work. Multilingual names omit `.en`.

| Model | Scope | Size | Speed | Accuracy |
| ---------- | --------------- | ------ | ------- | -------- |
| tiny.en / tiny | EN-only / multi | ~75MB | Fastest | Basic |
| base.en / base | EN-only / multi | ~150MB | Fast | Good |
| small.en / small | EN-only / multi | ~500MB | Medium | Better |
| medium.en / medium | EN-only / multi | ~1.5GB | Slower | Great |
| large-v3 | Multilingual | ~3GB | Slowest | Best |

For English streaming, `base.en` or `small.en` is usually enough. For Russian / bilingual / `auto`, use `base`+ without `.en`. Larger models are much slower; mic quality often matters more.

---

# TODO/Roadmap

- [x] PyAudio for all recordings (no `arecord` dependency).
- [x] Streaming: fixes for voice duplication, race conditions, and skipped segments; corrected transcriber duration reporting.
- [x] Multiple languages support.
- [x] Support list of custom terms or pronunciation features (like accents or speech patterns).
- [x] Status icon in the menu bar / system tray, so it is always visible when the microphone is live.
- [ ] Option to reuse previous transcription as context (e.g. `initial_prompt`).
- [ ] Context from a first word (e.g. “Python” → prompt about Python without "Python" in the output).
- [ ] Expose more `model.transcribe()` options in config.
