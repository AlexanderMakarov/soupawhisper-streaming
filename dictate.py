#!/usr/bin/env python3

import argparse
import configparser
import ctypes
import ctypes.util
import subprocess
import tempfile
import threading
import signal
import sys
import os
import platform
import queue
import time
import wave
import logging
import math
import plistlib
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Tuple
from faster_whisper.transcribe import Segment
from faster_whisper.vad import VadOptions

import numpy as np
import pyaudio
import webrtcvad

import meeting as meeting_mod

from faster_whisper import WhisperModel


def _import_keyboard():
    """Import pynput.keyboard only when needed.

    On Linux pynput opens an X11 display at import time, so a top-level import
    breaks headless ``--file`` / ``make transcribe`` when DISPLAY is unset.
    """
    from pynput import keyboard

    return keyboard

__version__ = "0.1.0"

# Logger will be configured in main()
logger = logging.getLogger(__name__)

# Streaming chunks are already speech from WebRTC VAD; Silero uses looser settings than
# the library default (min_silence_duration_ms=2000 is for long files and can wipe short clips).
_STREAMING_VAD_PARAMETERS = {
    "threshold": 0.33,
    "min_speech_duration_ms": 0,
    "min_silence_duration_ms": 120,
    "speech_pad_ms": 200,
}


_DEFAULT_STREAMING_CONTEXT_WORDS = 50
_DEFAULT_STREAMING_CONTEXT_RESET_S = 5.0


@dataclass(frozen=True)
class _StreamingChunk:
    """Audio plus speech bounds on the capture timeline, excluding trailing silence."""

    audio: np.ndarray
    speech_start: float
    speech_end: float


def resolve_transcription_language(
    model,
    audio_float32: np.ndarray,
    language: Optional[str],
    language_allowlist: Optional[list[str]],
) -> Optional[str]:
    """
    Pick the language argument for WhisperModel.transcribe().

    - Fixed language (en, ru, …): returned as-is.
    - Auto (language is None) with no allowlist: None → Whisper does full multilingual detection.
    - Auto with allowlist of **one** code: use that language without calling detect_language
      (same as setting ``language`` explicitly; no extra encoder pass).
    - Auto with allowlist of **two or more**: run ``detect_language`` once, then pick the
      allowlisted language with the highest probability. CTranslate2 does not expose a way to
      “decode only from these languages”; the model always scores all language tokens. We only
      **filter** those scores to your list. Cost is one encoder forward + language head (the
      heavy work is the autoregressive decoder, which still runs once per ``transcribe``).
    """
    if language is not None:
        return language
    if not language_allowlist:
        return None
    if len(language_allowlist) == 1:
        return language_allowlist[0]
    try:
        _, _, all_probs = model.detect_language(
            audio_float32,
            vad_filter=False,
            language_detection_segments=1,
        )
    except Exception as e:
        logger.warning(
            "language_allowlist: detect_language failed (%s); using %s",
            e,
            language_allowlist[0],
        )
        return language_allowlist[0]
    allowed = frozenset(language_allowlist)
    filtered = [(lang, p) for lang, p in all_probs if lang in allowed]
    if filtered:
        chosen = max(filtered, key=lambda x: x[1])[0]
        logger.debug("language_allowlist: chose %s (candidates %s)", chosen, allowed)
        return chosen
    logger.debug(
        "language_allowlist: no scores for %s; defaulting to %s",
        allowed,
        language_allowlist[0],
    )
    return language_allowlist[0]


def _streaming_segments_to_text(segments: Iterable[Segment]) -> str:
    parts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        parts.append(text)
    return " ".join(parts).strip()

def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp: HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

_REJECT_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)

def _normalize_reject_phrase(text: str) -> str:
    """
    Normalize text for reject-phrase matching.

    Rule: reject phrase matches the entire chunk ignoring punctuation.
    Examples:
      "Thank you" == "thank you." == "thank you!!!"
    """
    s = " ".join(text.lower().split())
    if not s:
        return ""
    return s.translate(_REJECT_PUNCT_TRANSLATION).strip()


def _parse_custom_terms(raw: str) -> list[str]:
    """
    Parse a comma/newline-separated custom-terms string into a deduplicated list.

    Order is preserved; case is preserved (Whisper's tokenizer is case-sensitive
    for some terms like brand names). Empty / whitespace-only entries are dropped.
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _build_custom_terms_kwargs(terms: list[str]) -> dict:
    """
    Build the faster-whisper kwargs that bias transcription toward custom terms.

    Returns an empty dict when there are no terms (so the feature is fully off).
    Otherwise returns both `initial_prompt` (in-context priming) and `hotwords`
    (per-token logit boost). Using both is intentional: in streaming mode
    `condition_on_previous_text=True` causes the initial prompt to be displaced
    by accumulated transcription over time, while hotwords stays active per chunk.
    """
    if not terms:
        return {}
    return {
        "initial_prompt": "Glossary: " + ", ".join(terms) + ".",
        "hotwords": " ".join(terms),
    }


# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"
DEFAULT_HOTKEY = "f12"
IS_MACOS = platform.system() == "Darwin"


def _run_cmd(cmd: list[str], timeout_s: float = 0.8) -> Optional[str]:
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None
    if res.returncode != 0:
        return None
    out = (res.stdout or "").strip()
    return out or None


def detect_current_keyboard_layout() -> Optional[str]:
    """
    Best-effort OS keyboard layout / input source identifier.

    Returns:
      - macOS: InputSourceID (e.g. "com.apple.keylayout.US", "com.apple.keylayout.Russian")
      - Linux/X11: xkb layout string from `setxkbmap -query` (e.g. "us", "ru", "us,ru")
    """
    if IS_MACOS:
        # Prefer HIToolbox current-layout id (exists on many systems; see README for example).
        data = _macos_hitoolbox_plist()
        if isinstance(data, dict):
            current_id = data.get("AppleCurrentKeyboardLayoutInputSourceID")
            if isinstance(current_id, str) and current_id.strip():
                return current_id.strip()

            # Fallback: selected input sources may not carry InputSourceID (often only name/id),
            # but in some setups it does.
            selected = data.get("AppleSelectedInputSources")
            if isinstance(selected, list) and selected:
                first = selected[0]
                if isinstance(first, dict):
                    src_id = first.get("InputSourceID")
                    if isinstance(src_id, str) and src_id.strip():
                        return src_id.strip()

        # Fallbacks (may be missing on some systems).
        for cmd in (
            ["defaults", "read", "-g", "AppleCurrentKeyboardLayoutInputSourceID"],
            [
                "defaults",
                "read",
                str(Path.home() / "Library" / "Preferences" / "com.apple.HIToolbox.plist"),
                "AppleCurrentKeyboardLayoutInputSourceID",
            ],
        ):
            out = _run_cmd(cmd)
            if out:
                return out
        return None

    # Linux (X11): prefer setxkbmap if present.
    if subprocess.run(["which", "setxkbmap"], capture_output=True).returncode == 0:
        out = _run_cmd(["setxkbmap", "-query"], timeout_s=0.8)
        if not out:
            return None
        for line in out.splitlines():
            if line.strip().startswith("layout:"):
                return line.split("layout:", 1)[1].strip() or None
    return None


def _macos_hitoolbox_plist() -> Optional[dict]:
    """
    Parse com.apple.HIToolbox preferences as a plist.

    We use `defaults export` so we get a structured plist (XML), not an ambiguous human string.
    """
    out = _run_cmd(["defaults", "export", "com.apple.HIToolbox", "-"], timeout_s=1.2)
    if not out:
        return None
    try:
        return plistlib.loads(out.encode("utf-8"))
    except Exception:
        return None


def _macos_input_source_languages_for_id(input_source_id: str) -> Optional[list[str]]:
    """
    Given a macOS InputSourceID, return its InputSourceLanguages (ISO codes) if available.
    """
    data = _macos_hitoolbox_plist()
    if not data:
        return None
    # Keys observed in the wild: AppleSelectedInputSources, AppleInputSourceHistory.
    candidates = []
    for k in ("AppleSelectedInputSources", "AppleInputSourceHistory"):
        v = data.get(k)
        if isinstance(v, list):
            candidates.extend([x for x in v if isinstance(x, dict)])
    for entry in candidates:
        if entry.get("InputSourceID") != input_source_id:
            continue
        langs = entry.get("InputSourceLanguages")
        if isinstance(langs, list) and all(isinstance(x, str) for x in langs):
            return langs
    return None


def _linux_xkb_layouts_from_setxkbmap() -> Optional[list[str]]:
    """Ask setxkbmap for the configured layout list."""
    if subprocess.run(["which", "setxkbmap"], capture_output=True).returncode != 0:
        return None
    out = _run_cmd(["setxkbmap", "-query"], timeout_s=0.8)
    return parse_setxkbmap_layouts(out)


def parse_setxkbmap_layouts(query_text: Optional[str]) -> Optional[list[str]]:
    """Parse `setxkbmap -query` output into layout tokens (e.g. us,ru,am -> ['us','ru','am'])."""
    if not query_text:
        return None
    for line in query_text.splitlines():
        if line.strip().startswith("layout:"):
            raw = line.split("layout:", 1)[1].strip()
            layouts = [p.strip() for p in raw.split(",") if p.strip()]
            return layouts or None
    return None


class _XkbStateRec(ctypes.Structure):
    _fields_ = [
        ("group", ctypes.c_ubyte),
        ("locked_group", ctypes.c_ubyte),
        ("base_group", ctypes.c_ushort),
        ("latched_group", ctypes.c_ushort),
        ("mods", ctypes.c_ubyte),
        ("base_mods", ctypes.c_ubyte),
        ("latched_mods", ctypes.c_ubyte),
        ("locked_mods", ctypes.c_ubyte),
        ("compat_state", ctypes.c_ubyte),
        ("grab_mods", ctypes.c_ubyte),
        ("compat_grab_mods", ctypes.c_ubyte),
        ("lookup_mods", ctypes.c_ubyte),
        ("compat_lookup_mods", ctypes.c_ubyte),
        ("ptr_buttons", ctypes.c_ushort),
    ]


# Opening an X11 display costs ~3.4ms. That runs inside the pynput callback on every
# trigger-key press, and it is also a race: the modifier can be released before the
# query lands. Keep one connection open instead -- the query then takes microseconds.
_x11_lock = threading.Lock()
_x11_conn: Optional[tuple] = None


def _x11_connection() -> Optional[tuple]:
    global _x11_conn
    if _x11_conn is not None:
        return _x11_conn
    lib_name = ctypes.util.find_library("X11")
    if not lib_name:
        return None
    x11 = ctypes.CDLL(lib_name)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XkbGetState.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(_XkbStateRec)]
    x11.XkbGetState.restype = ctypes.c_int
    dpy = x11.XOpenDisplay(None)
    if not dpy:
        return None
    _x11_conn = (x11, dpy)
    return _x11_conn


def _xkb_state() -> Optional["_XkbStateRec"]:
    """Current XKB state via libX11 XkbGetState (no extra packages), or None."""
    global _x11_conn
    try:
        with _x11_lock:
            conn = _x11_connection()
            if conn is None:
                return None
            x11, dpy = conn
            state = _XkbStateRec()
            # XkbUseCoreKbd; Status Success == 0
            if x11.XkbGetState(dpy, 0x0100, ctypes.byref(state)) != 0:
                return None
            return state
    except Exception as e:
        logger.debug("XkbGetState failed: %s", e)
        _x11_conn = None  # force a reconnect next time (e.g. X restarted)
        return None


def _linux_xkb_group_index() -> Optional[int]:
    """Current XKB group index, used to resolve the active keyboard layout."""
    state = _xkb_state()
    return None if state is None else int(state.group)


def _which_ok(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def _linux_active_xkb_layout() -> Optional[str]:
    """
    Try to detect the *active* XKB layout (current group), not just the configured list.

    Returns a single layout token like "us" or "ru" when possible.
    Order: xkb-switch, xkblayout-state, then setxkbmap list + XkbGetState.
    """
    if _which_ok("xkb-switch"):
        out = _run_cmd(["xkb-switch", "-p"], timeout_s=0.5) or _run_cmd(["xkb-switch"], timeout_s=0.5)
        if out:
            return out.strip()
    if _which_ok("xkblayout-state"):
        out = _run_cmd(["xkblayout-state", "print", "%s"], timeout_s=0.5)
        if out:
            return out.strip()
    layouts = _linux_xkb_layouts_from_setxkbmap()
    if not layouts:
        return None
    if len(layouts) == 1:
        return layouts[0]
    idx = _linux_xkb_group_index()
    if idx is None or idx < 0 or idx >= len(layouts):
        return None
    return layouts[idx]


def detect_current_keyboard_language(layout_to_language: dict[str, str]) -> Optional[str]:
    """
    Robustly detect current "language expectation" from the OS keyboard input source.

    Priority:
    1) Explicit config mapping (layout_to_language) for the detected layout/source id
    2) macOS: read InputSourceLanguages for the active InputSourceID (HIToolbox)
    3) Linux/X11: read active XKB layout (xkb-switch, xkblayout-state, or X11 XkbGetState); return if mapped or ISO 639-1
    """
    def looks_like_iso_639_1(s: str) -> bool:
        return len(s) == 2 and s.isalpha()

    layout_id = detect_current_keyboard_layout()
    if layout_id:
        mapped = layout_to_language.get(layout_id)
        if mapped:
            return mapped

    if IS_MACOS:
        if not layout_id:
            return None
        langs = _macos_input_source_languages_for_id(layout_id)
        if not langs:
            return None
        lang = langs[0].strip().lower()
        return lang if looks_like_iso_639_1(lang) else None

    active = _linux_active_xkb_layout()
    if not active:
        return None
    mapped = layout_to_language.get(active)
    if mapped:
        return mapped
    active = active.strip().lower()
    return active if looks_like_iso_639_1(active) else None


def language_from_layout(layout_id: Optional[str], layout_to_language: dict[str, str]) -> Optional[str]:
    """
    Backward-compatible wrapper (kept for external callers / tests).
    Prefer `detect_current_keyboard_language()` for actual runtime detection.
    """
    if not layout_id:
        return None
    return layout_to_language.get(layout_id)


def load_config():
    config = configparser.ConfigParser()

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    language_raw = config.get("whisper", "language", fallback="en").strip().lower()
    language = None if language_raw in {"auto", "none"} else (language_raw or "en")

    allow_raw = config.get("whisper", "language_allowlist", fallback="").strip()
    language_allowlist: Optional[list[str]] = None
    if allow_raw:
        language_allowlist = [
            code.strip().lower()
            for code in allow_raw.replace(",", " ").split()
            if code.strip()
        ]
        if not language_allowlist:
            language_allowlist = None

    context_words = config.getint(
        "streaming", "context_words", fallback=_DEFAULT_STREAMING_CONTEXT_WORDS
    )
    context_reset_seconds = config.getfloat(
        "streaming", "context_reset_seconds", fallback=_DEFAULT_STREAMING_CONTEXT_RESET_S
    )
    if context_words < 0:
        raise ValueError("[streaming] context_words must be non-negative (0 disables text context)")
    if not math.isfinite(context_reset_seconds) or context_reset_seconds < 0:
        raise ValueError("[streaming] context_reset_seconds must be finite and non-negative")

    return {
        # Whisper
        "model": config.get("whisper", "model", fallback="base.en"),
        "device": config.get("whisper", "device", fallback="cpu"),
        "compute_type": config.get("whisper", "compute_type", fallback="int8"),
        "language": language,
        "language_allowlist": language_allowlist,
        # Input device
        "audio_input_device": config.get("input", "audio_input_device", fallback=None),
        # Hotkey
        "key": config.get("hotkey", "key", fallback=DEFAULT_HOTKEY),
        # Behavior
        "default_streaming": config.getboolean("behavior", "default_streaming", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
        "clipboard": config.getboolean("behavior", "clipboard", fallback=True),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "auto_sentence": config.getboolean("behavior", "auto_sentence", fallback=True),
        "typing_delay": config.getfloat("behavior", "typing_delay", fallback=0.01),
        "save_recordings": config.getboolean("behavior", "save_recordings", fallback=False),
        # Desktop toast when the microphone delivers silence (once per session). Log always records it.
        "notify_no_audio": config.getboolean("behavior", "notify_no_audio", fallback=True),
        # Streaming-only: comma-separated phrases to suppress if a chunk equals one of them
        # (punctuation ignored). If empty, feature is disabled.
        "reject_phrases": config.get("behavior", "reject_phrases", fallback="").strip(),
        # Custom terms / glossary biased into transcription via faster-whisper's
        # initial_prompt + hotwords. Comma-separated; empty disables the feature.
        "custom_terms": config.get("behavior", "custom_terms", fallback="").strip(),
        # Language enforcement (optional): use current OS keyboard layout as the "field expectation"
        "enforce_language_from_layout": config.getboolean("behavior", "enforce_language_from_layout", fallback=False),
        # Comma-separated list: "<layout_id>:<lang>,<layout_id>:<lang>"
        "layout_to_language": config.get("behavior", "layout_to_language", fallback="").strip(),
        # Menu bar / system tray (default on; if true and tray cannot start, process exits)
        "tray_icon": config.getboolean("behavior", "tray_icon", fallback=True),
        "tray_show_language": config.getboolean("behavior", "tray_show_language", fallback=True),
        # Streaming
        "streaming_context_words": context_words,
        "streaming_context_reset_seconds": context_reset_seconds,
        "min_speech_length_seconds": config.getfloat("streaming", "min_speech_length_seconds", fallback=1.0),
        "vad_silence_threshold_seconds": config.getfloat("streaming", "vad_silence_threshold_seconds", fallback=1.0),
        "vad_sample_rate": config.getint("streaming", "vad_sample_rate", fallback=16000),
        "vad_chunk_size_ms": config.getfloat("streaming", "vad_chunk_size_ms", fallback=20.0),
        "vad_min_speech_chunks": config.getint("streaming", "vad_min_speech_chunks", fallback=10),
        "vad_threshold": config.getfloat("streaming", "vad_threshold", fallback=0.5),
        # Meeting recording mode. Transcripts are kept; the WAVs they came from are
        # scratch and live under meeting.TMP_ROOT until transcription succeeds.
        "meeting_transcript_dir": os.path.expanduser(
            config.get("meeting", "transcript_dir", fallback="~/Documents/meetings")
        ),
        "meeting_max_duration_min": config.getint("meeting", "max_duration_min", fallback=120),
        # 0 disables the silence watchdog. Generous by default: a false stop loses
        # the rest of the meeting, over-recording only costs disk.
        "meeting_silence_stop_min": config.getint("meeting", "silence_stop_min", fallback=10),
        # Empty = tray only. Meeting mode disables the dictation hotkey while it runs,
        # so transcribed text can never be typed into the call.
        "meeting_hotkey": config.get("meeting", "hotkey", fallback="").strip().lower(),
        # Keep the WAVs after a successful transcript. Off by default (~230 MB/hour),
        # but the only way to re-run a bad transcript with different settings.
        "meeting_keep_audio": config.getboolean("meeting", "keep_audio", fallback=False),
        # Threads for the meeting model. Half the logical cores by default: measured
        # on an i5-10300H (4 physical + HT), 4 threads was BOTH the fastest setting
        # (2.5x over 1) and 8 was slower than 1 through oversubscription -- so this
        # leaves CPU for dictation at no speed cost.
        "meeting_cpu_threads": config.getint(
            "meeting", "cpu_threads", fallback=max(1, (os.cpu_count() or 2) // 2)
        ),
        # Speech-run segmentation. Fallbacks come from meeting.py so the defaults are
        # defined once; meeting.RunSettings.from_config() turns these back into the
        # VadOptions the transcription pass uses.
        "meeting_run_min_silence_ms": config.getint(
            "meeting", "run_min_silence_ms", fallback=meeting_mod.RUN_MIN_SILENCE_MS
        ),
        "meeting_run_pad_ms": config.getint(
            "meeting", "run_pad_ms", fallback=meeting_mod.RUN_PAD_MS
        ),
        "meeting_run_vad_threshold": config.getfloat(
            "meeting", "run_vad_threshold", fallback=meeting_mod.RUN_VAD_THRESHOLD
        ),
        "meeting_run_min_speech_ms": config.getint(
            "meeting", "run_min_speech_ms", fallback=meeting_mod.RUN_MIN_SPEECH_MS
        ),
        "meeting_run_max_seconds": config.getfloat(
            "meeting", "run_max_seconds", fallback=meeting_mod.LANGUAGE_WINDOW_S
        ),
        "meeting_min_avg_logprob": config.getfloat(
            "meeting", "min_avg_logprob", fallback=meeting_mod.MIN_AVG_LOGPROB
        ),
        "meeting_word_timestamps": config.getboolean(
            "meeting", "word_timestamps", fallback=meeting_mod.WORD_TIMESTAMPS
        ),
        # How long a single speaker block may run before it is split with a fresh
        # timestamp. Your own track defaults to 60s: measured against a professional
        # transcript of the same 73-minute interview, that reproduces its shape
        # (86 blocks, 110-word longest vs 88 and 121); the previous 600s gave
        # 415-word walls.
        "meeting_me_max_block_seconds": config.getfloat(
            "meeting", "me_max_block_seconds", fallback=meeting_mod.ME_MAX_BLOCK_S
        ),
        "meeting_them_max_block_seconds": config.getfloat(
            "meeting", "them_max_block_seconds", fallback=meeting_mod.THEM_MAX_BLOCK_S
        ),
    }


def scratch_wav_path(filename: str, root: Path = None) -> Path:
    """Path for a debug/mirror WAV under the shared scratch root.

    Everything this app writes lives under one directory so the tray can report its
    size and wipe it in one action, and so we never scatter files among other apps'
    entries in /tmp.
    """
    base = Path(root if root is not None else meeting_mod.TMP_ROOT) / "recordings"
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


# How often the meeting watchdog re-checks duration, silence and recorder health.
MEETING_WATCHDOG_POLL_S = 30.0

MODIFIER_NAMES = frozenset({"shift", "ctrl", "alt", "cmd"})


# X11 modifier mask bits (X.h): ShiftMask, ControlMask, Mod1Mask (Alt), Mod4Mask (Super).
X11_MODIFIER_MASKS = (("shift", 1), ("ctrl", 4), ("alt", 8), ("cmd", 64))


def active_modifiers_x11() -> Optional[set[str]]:
    """Modifiers physically held right now, read from X11, or None if unavailable.

    Tracking press/release events cannot be trusted here: when layout switching is
    bound to Shift, X11 reports Shift's RELEASE as an ISO group-switch keysym (65032)
    rather than Shift, so a tracked set never clears and every later hotkey test is
    wrong. XkbGetState reports the true current mask and is self-healing.
    """
    state = _xkb_state()
    if state is None:
        return None
    return {name for name, mask in X11_MODIFIER_MASKS if state.mods & mask}


def modifier_name(key) -> Optional[str]:
    """Normalise a pynput modifier key to "shift"/"ctrl"/"alt"/"cmd", else None.

    pynput reports sided variants (shift_l, ctrl_r, alt_gr); a hotkey spec should not
    have to care which physical key was used.
    """
    name = getattr(key, "name", None)
    if not isinstance(name, str):
        return None
    base = name.split("_")[0]
    return base if base in MODIFIER_NAMES else None


def parse_hotkey_spec(spec: str) -> tuple[frozenset[str], str]:
    """Split "shift+f12" into ({"shift"}, "f12").

    Dictation's hotkey is a single key, but meeting mode wants a deliberate combo
    so it cannot be triggered by accident mid-call.
    """
    parts = [p.strip().lower() for p in (spec or "").split("+") if p.strip()]
    modifiers = frozenset(p for p in parts if p in MODIFIER_NAMES)
    key = next((p for p in parts if p not in MODIFIER_NAMES), "")
    return modifiers, key


def get_hotkey(key_name: str):
    """Map key name to pynput key."""
    keyboard = _import_keyboard()
    key_name = key_name.lower()
    if hasattr(keyboard.Key, key_name):
        return getattr(keyboard.Key, key_name)
    elif len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)
    else:
        logger.warning(f"Unknown key: {key_name}, defaulting to {DEFAULT_HOTKEY}")
        return get_hotkey(DEFAULT_HOTKEY)


def keys_match(
    event_key,
    hotkey,
    hotkey_value=None,
    hotkey_vk=None,
) -> bool:
    """Return True if the event key matches the hotkey (cross-platform, e.g. macOS KeyCode vs Key).

    Pass hotkey_value and hotkey_vk (e.g. from _hotkey_value, _hotkey_vk) to avoid
    repeated getattr on every keypress.
    """
    if event_key == hotkey:
        return True
    if hotkey_value is not None and event_key == hotkey_value:
        return True
    if hotkey_vk is not None:
        event_vk = getattr(event_key, "vk", None)
        if event_vk is not None and event_vk == hotkey_vk:
            return True
        return False
    # Fallback when precomputed values not passed (e.g. tests)
    if hasattr(hotkey, "value") and hotkey.value is not None and event_key == hotkey.value:
        return True
    event_vk = getattr(event_key, "vk", None)
    hotkey_ref = getattr(hotkey, "value", None) or hotkey
    hotkey_vk_val = getattr(hotkey_ref, "vk", None)
    return (
        event_vk is not None
        and hotkey_vk_val is not None
        and event_vk == hotkey_vk_val
    )


class Typer:
    """Types and removes characters in any input field via xdotool (Linux) or Quartz (macOS)."""

    def __init__(self, delay_ms: int = 10, start_delay_ms: int = 250):
        # 0 means "no pacing": type the whole string in one burst.
        self.delay_ms = max(0, int(delay_ms))
        self.start_delay_ms = int(start_delay_ms)
        self._controller = None
        self._keyboard = None
        if IS_MACOS:
            # Quartz key events need only Accessibility, which the hotkey listener already
            # holds. Driving System Events over AppleScript would additionally require
            # Automation access, and blocks on its consent dialog until the call times out.
            self._keyboard = _import_keyboard()
            self._controller = self._keyboard.Controller()
            self.enabled = True
        else:
            self.enabled = subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0
            if not self.enabled:
                logger.warning("[typer] xdotool not found, typing disabled")

    def type_rewrite(self, text: str, previous_length: int = 0):
        """
        Type text into the focused field.

        Args:
            text: The text to type.
            previous_length: The number of characters to delete before typing the new text.
        """
        if not self.enabled or not text:
            return
        if self.start_delay_ms > 0:
            time.sleep(self.start_delay_ms / 1000.0)
        if IS_MACOS:
            self._type_macos(text, previous_length)
        else:
            self._type_linux(text, previous_length)

    def _type_macos(self, text: str, previous_length: int = 0):
        # Quartz carries the characters as a Unicode string, so the active keyboard layout
        # does not rewrite them — Latin and Cyrillic both arrive as spoken.
        for _ in range(previous_length):
            self._controller.tap(self._keyboard.Key.backspace)
        if self.delay_ms <= 0:
            self._controller.type(text)
            return
        for char in text:
            self._controller.type(char)
            time.sleep(self.delay_ms / 1000.0)

    def _type_linux(self, text: str, previous_length: int = 0):
        if previous_length > 0:
            subprocess.run(
                ["xdotool", "key", "BackSpace", "--clearmodifiers", "--repeat", str(previous_length)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        subprocess.run(
            ["xdotool", "type", "--delay", str(self.delay_ms), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )


class Dictation:
    def __init__(self, config: dict, *, interactive: bool = True):
        self.config = config
        self._interactive = interactive
        self.recording = False
        self.transcribing = False
        # Meeting recording is a separate, mutually exclusive mode: while it runs the
        # dictation hotkey is inert, so nothing can be typed into the call.
        self.meeting_active = False
        self.meeting: Optional[meeting_mod.MeetingRecorder] = None
        self._meeting_tmp_root = meeting_mod.TMP_ROOT
        self._meeting_spawn = subprocess.Popen
        self._meeting_watchdog: Optional[threading.Thread] = None
        self._meeting_stop_event = threading.Event()
        meeting_mods, meeting_key_name = parse_hotkey_spec(config.get("meeting_hotkey", ""))
        self._meeting_modifiers = meeting_mods
        self._held_modifiers: set[str] = set()
        self._modifier_source = active_modifiers_x11
        self._meeting_model = None
        self.meeting_transcribing = False
        self.meeting_progress: Optional[str] = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self._tray_error = None
        self._tray = None
        self.running = True
        self.typer: Optional[Typer] = None
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream = None
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_data: list[np.ndarray] = []
        # True once the "no audio" notification was shown for the current session.
        self._audio_problem_notified = False
        self.sample_rate = 16000
        self.frames_per_buffer = 4096  # ~0.25 seconds of audio
        self._layout_to_language_map = self._parse_layout_to_language_map(
            self.config.get("layout_to_language", "")
        )
        self._session_enforced_language: Optional[str] = None
        self._hotkey_action_lock = threading.Lock()
        self._last_hotkey_event_monotonic: Optional[float] = None
        self._hotkey_action_in_progress_since: Optional[float] = None
        self._hotkey_action_in_progress_name: Optional[str] = None
        self._keyboard_listener = None
        self._listener_restart_backoff_s = 1.0

        # Hotkeys / pynput need a display on Linux; skip for headless --file mode.
        if interactive:
            self.hotkey = get_hotkey(config["key"])
            # Precompute for keys_match (avoids getattr on every keypress)
            self._hotkey_value = getattr(self.hotkey, "value", None)
            hv = self._hotkey_value
            self._hotkey_vk = getattr(hv, "vk", None) if hv is not None else getattr(self.hotkey, "vk", None)
            self._meeting_key = get_hotkey(meeting_key_name) if meeting_key_name else None
        else:
            self.hotkey = None
            self._hotkey_value = None
            self._hotkey_vk = None
            self._meeting_key = None

        custom_terms = _parse_custom_terms(self.config.get("custom_terms") or "")
        # One glossary for dictation and meetings alike; meeting.finish_session()
        # reads this attribute.
        self.custom_terms_kwargs = _build_custom_terms_kwargs(custom_terms)
        # reject_phrases likewise applies to every mode.
        self._reject_phrase_set = self._build_reject_phrase_set()
        if custom_terms:
            logger.info(f"Custom terms glossary active ({len(custom_terms)} term(s)): {custom_terms}")

        if interactive and self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100,  # Delay to avoid modifiers from hotkey
            )

        # Load model in background.
        logger.debug(f"Loading Whisper model ({config['model']})...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def get_hotkey_name(self) -> str:
        if self.hotkey is None:
            return self.config.get("key", DEFAULT_HOTKEY)
        return getattr(self.hotkey, 'name', None) or getattr(self.hotkey, 'char', DEFAULT_HOTKEY)

    def _load_model(self):
        try:
            self.model = WhisperModel(self.config["model"], device=self.config["device"], compute_type=self.config["compute_type"])
            self.model_loaded.set()
            logger.info(f"Model {self.config['model']} ({self.config['device']}, {self.config['compute_type']}) loaded.")
            self._finish_model_loading()
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            logger.error(f"Failed to load model: {e}", exc_info=True)
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                logger.info("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def _finish_model_loading(self):
        if not self._interactive:
            logger.info("Model ready (file transcription mode).")
            return
        logger.info(f"Hold [{self.get_hotkey_name()}] to start dictation, release to transcribe. Press Ctrl+C to quit.")

    def _get_input_device_index(self) -> Optional[int]:
        """Get the input device index from config, or None to use default."""
        audio_input_device = self.config.get("audio_input_device")
        if not audio_input_device or not isinstance(audio_input_device, str):
            return None
        audio_input_device = audio_input_device.strip()
        if not audio_input_device:
            return None
        try:
            device_index = int(audio_input_device)
            return device_index
        except ValueError:
            if self.audio_interface is None:
                self.audio_interface = pyaudio.PyAudio()
            for i in range(self.audio_interface.get_device_count()):
                try:
                    info = self.audio_interface.get_device_info_by_index(i)
                    max_inputs = int(info.get('maxInputChannels', 0))
                    device_name = str(info.get('name', ''))
                    if max_inputs > 0 and audio_input_device.lower() in device_name.lower():
                        logger.info(f"[record] Found audio device matching '{audio_input_device}': {i} - {device_name}")
                        return i
                except Exception:
                    pass
            logger.warning(f"[record] Audio device '{audio_input_device}' not found, using default")
            return None

    def _get_available_input_devices(self) -> list[tuple[int, str, int]]:
        """
        Get list of available audio input devices.

        Returns:
            List of tuples (device_index, device_name, max_input_channels)
        """
        if self.audio_interface is None:
            self.audio_interface = pyaudio.PyAudio()
        devices = []
        for i in range(self.audio_interface.get_device_count()):
            try:
                info = self.audio_interface.get_device_info_by_index(i)
                max_inputs = int(info.get('maxInputChannels', 0))
                if max_inputs > 0:
                    device_name = str(info.get('name', ''))
                    devices.append((i, device_name, max_inputs))
            except Exception:
                pass
        return devices

    def _report_audio_problem(self, issue_description: str = "No audio detected"):
        """Report that the microphone delivered no usable audio.

        Always logged at WARNING. By default a desktop toast is shown at most once
        per dictation session (``notify_no_audio``); streaming pauses would otherwise
        stack banners. Set ``notify_no_audio = false`` to suppress the toast.

        Args:
            issue_description: Description of the audio input issue (e.g., "Audio input contains only zeros" or "Audio input has too low amplitude")
        """
        logger.warning(f"[record] {issue_description}")
        if not self.config.get("notify_no_audio", True):
            return
        if self._audio_problem_notified:
            return
        self._audio_problem_notified = True
        logger.debug("[record] Building list of devices...")
        devices = self._get_available_input_devices()
        devices_list = [f"  {i}: {name}" for i, name, _ in devices]
        devices_text = "\n".join(devices_list[:8])
        if len(devices_list) > 8:
            devices_text += f"\n  ... and {len(devices_list) - 8} more"
        config_path = CONFIG_PATH
        message = f"{issue_description}\n\nSet 'audio_input_device' in {config_path}\nSupported (by pyaudio, not on this machine!) devices:\n{devices_text}"
        self.notify("No audio detected - check device", message, logging.WARNING, 5000)

    def _check_valid_audio_input(self, audio_buffer: np.ndarray) -> bool:
        """
        Check if audio segment contains only zeros or is effectively silent.
        If detected, logs error and shows device help.
        This indicates an audio input problem, not just absence of speech.

        Args:
            audio_buffer: Audio array (int16 format).

        Returns:
            True if segment is all zeros or has very low amplitude (audio input problem)
        """
        if audio_buffer.dtype == np.int16:
            # Check if all zeros
            if np.all(audio_buffer == 0):
                self._report_audio_problem("Audio input contains only zeros")
                return True
            # Check max amplitude - if very small, likely no audio input
            # For int16, normal speech would have max amplitude > 100
            max_amplitude = np.max(np.abs(audio_buffer))
            if max_amplitude < 100:
                self._report_audio_problem("Audio input is effectively silent (too low amplitude)")
                return True
        return False

    def _start_pyaudio_stream(self, frames_per_buffer: int) -> Optional[pyaudio.Stream]:
        """
        Start a pyaudio stream for recording.

        Args:
            frames_per_buffer: Number of frames per buffer

        Returns:
            pyaudio.Stream instance or None if failed
        """
        if self.audio_interface is None:
            self.audio_interface = pyaudio.PyAudio()
        input_device_index = self._get_input_device_index()
        if input_device_index is not None:
            device_info = self.audio_interface.get_device_info_by_index(input_device_index)
            logger.info(f"[record] Using audio input device {input_device_index}: {device_info['name']}")
        else:
            default_device = self.audio_interface.get_default_input_device_info()
            logger.info(f"[record] Using default audio input device {default_device['index']}: {default_device['name']}")
            input_device_index = int(default_device['index'])
        try:
            audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=frames_per_buffer
            )
            return audio_stream
        except OSError as e:
            logger.error(f"[record] Failed to open audio stream: {e}", exc_info=True)
            devices_text = "\n".join([f"  {i}: {name} (inputs: {max_inputs})" for i, name, max_inputs in self._get_available_input_devices()])
            self.notify("Error", f"Failed to open audio device: {str(e)[:50]}...\nSupported (by pyaudio, not on this machine!) devices:\n{devices_text}", logging.ERROR, 10000)
            return None

    def notify(self, title, message, level=logging.INFO, timeout=2000, icon=None):
        """Send a desktop notification."""
        icon_map = {
            logging.DEBUG: "dialog-information",
            logging.INFO: "dialog-information",
            logging.WARNING: "dialog-warning",
            logging.ERROR: "dialog-error",
            logging.CRITICAL: "dialog-error",
        }
        if icon is None:
            icon = icon_map.get(level, "dialog-information")
        log_method_map = {
            logging.DEBUG: logger.debug,
            logging.INFO: logger.info,
            logging.WARNING: logger.warning,
            logging.ERROR: logger.error,
            logging.CRITICAL: logger.critical,
        }
        log_method = log_method_map.get(level, logger.info)
        log_method("Showing notification: %s, %s, %s, %s", title, message, icon, timeout)
        if not self.config["notifications"]:
            return
        if IS_MACOS:
            escaped_msg = message.replace("\\", "\\\\").replace('"', '\\"')
            escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
            script = f'display notification "{escaped_msg}" with title "{escaped_title}" subtitle "SoupaWhisper"'
            subprocess.run(["osascript", "-e", script], capture_output=True)
        else:
            subprocess.run(
                [
                    "notify-send",
                    "-a", "SoupaWhisper",
                    "-i", icon,
                    "-t", str(timeout),
                    "-h", "string:x-canonical-private-synchronous:soupawhisper",
                    title,
                    message
                ],
                capture_output=True
            )

    def _segments_to_text(self, segments: Iterable[Segment], auto_sentence: bool) -> str:
        """Format text as a sentence: capitalize first letter and add period at end if needed."""
        text = " ".join(segment.text.strip() for segment in segments)
        if not text or not self.config.get("auto_sentence", False):
            return text
        text = text.strip()
        if not text:
            return text
        if auto_sentence and len(text) > 0:
            text = text[0].upper() + text[1:]
        if auto_sentence and not text.endswith(('.', '!', '?', ':', ';')):
            text = text + '.'
        return text

    @staticmethod
    def _parse_layout_to_language_map(raw: str) -> dict[str, str]:
        """
        Parse `behavior.layout_to_language` config into a dict.
        Format: "<layout_id>:<lang>, <layout_id>:<lang>".
        """
        out: dict[str, str] = {}
        if not raw:
            return out
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            layout_id, lang = part.split(":", 1)
            layout_id = layout_id.strip()
            lang = lang.strip().lower()
            if layout_id and lang:
                out[layout_id] = lang
        return out

    def _maybe_enforced_language_for_field(self) -> Optional[str]:
        """
        Return the language captured at session start when layout enforcement is on.

        Captured once in start_recording (hotkey press) and cleared when the session ends,
        so idle tray shows AUTO again and the next hotkey re-reads the current layout.
        """
        if not self.config.get("enforce_language_from_layout", False):
            return None
        return self._session_enforced_language

    def _begin_session_language(self) -> None:
        """Capture layout language for this dictation session (hotkey / Start)."""
        self._capture_session_enforced_language()

    def _end_session_language(self) -> None:
        """Clear session language so idle UI returns to AUTO / configured language."""
        self._clear_session_enforced_language()

    def _capture_session_enforced_language(self) -> None:
        """
        Capture current field-expectation language once at the start of a dictation session.
        Does not re-check during a running session (determinism + fewer layout queries).
        """
        self._session_enforced_language = None
        if not self.config.get("enforce_language_from_layout", False):
            return
        if self.config.get("language") is not None:
            # Fixed whisper language: don't override.
            return
        layout_id = detect_current_keyboard_layout()
        mapped = self._layout_to_language_map.get(layout_id) if layout_id else None
        lang = detect_current_keyboard_language(self._layout_to_language_map)
        logger.info(
            "[lang/layout] detected layout_id=%s mapped=%s captured=%s",
            layout_id,
            mapped,
            lang,
        )
        if lang:
            self._session_enforced_language = lang
            logger.debug("[lang/layout] captured=%s", lang)

    def _clear_session_enforced_language(self) -> None:
        self._session_enforced_language = None

    def _stop_tray(self) -> None:
        tray = getattr(self, "_tray", None)
        if tray is None:
            return
        try:
            tray.stop()
        except Exception as e:
            logger.debug("tray.stop() failed: %s", e)

    def _start_tray_or_exit(self):
        """Prepare the tray icon or exit non-zero when tray_icon is required."""
        from tray_status import TrayStartError, TrayStatus

        try:
            tray = TrayStatus(self)
            tray.prepare()
            return tray
        except TrayStartError as e:
            logger.error("Tray icon required but failed to start: %s", e)
            raise SystemExit(1) from e
        except Exception as e:
            logger.error("Tray icon required but failed to start: %s", e, exc_info=True)
            raise SystemExit(1) from e

    def _transcribe_audio_array(self, audio_array: np.ndarray) -> Tuple[str, float]:
        """
        Check if model is loaded and transcribe an audio numpy array.

        Args:
            audio_array: numpy array of audio data (int16 format, 16kHz, mono)

        Returns:
            Tuple of (transcribed_text, audio_duration)

        Raises:
            RuntimeError: If model failed to load or is not available
            Exception: If transcription fails
        """
        if self.model_error:
            self.notify("Error", "Model failed to load", logging.ERROR, 5000)
            return "", 0.0
        if self.model is None:
            logger.error("Cannot transcribe: model not loaded")
            return "", 0.0
        # Convert int16 to float32 and normalize to [-1.0, 1.0]
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        enforced = self._maybe_enforced_language_for_field()
        configured_lang = self.config.get("language")
        allowlist = self.config.get("language_allowlist")
        if enforced and configured_lang is None:
            # Only override in auto mode. If allowlist exists, enforce only if it is allowed.
            if not allowlist or enforced in allowlist:
                lang = enforced
            else:
                lang = resolve_transcription_language(self.model, audio_array, configured_lang, allowlist)
        else:
            lang = resolve_transcription_language(self.model, audio_array, configured_lang, allowlist)
        segments, info = self.model.transcribe(
            audio_array,
            vad_filter=True,
            language=lang,
            **self.custom_terms_kwargs,
        )
        # Convert segments to text.
        text = self._segments_to_text(segments, self.config["auto_sentence"])
        return text, info.duration

    def _build_reject_phrase_set(self) -> frozenset[str]:
        raw = (self.config.get("reject_phrases") or "").strip()
        if not raw:
            return frozenset()
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        normalized = {_normalize_reject_phrase(p) for p in parts}
        normalized.discard("")
        return frozenset(normalized)

    def should_reject_text(self, text: str) -> bool:
        """
        Reject only if the *whole text* equals one configured phrase (punctuation
        ignored). Empty reject_phrases disables the feature. Shared by dictation
        chunks and meeting cues alike.
        """
        if not self._reject_phrase_set:
            return False
        normalized = _normalize_reject_phrase(text)
        if not normalized:
            return False
        return normalized in self._reject_phrase_set

    def meeting_model(self):
        """Whisper model used for meeting transcription, built on first use.

        Deliberately separate from the dictation model: one CTranslate2 model
        serialises concurrent calls, so sharing it would make dictation wait behind a
        long meeting transcription. This one is also capped to fewer threads.
        """
        if self._meeting_model is None:
            threads = self.config["meeting_cpu_threads"]
            logger.info("[meeting] Loading model %s (%d threads)", self.config["model"], threads)
            self._meeting_model = WhisperModel(
                self.config["model"],
                device=self.config["device"],
                compute_type=self.config["compute_type"],
                cpu_threads=threads,
            )
        return self._meeting_model

    def start_meeting(self) -> None:
        """Enter meeting mode: capture both tracks and suspend dictation.

        Nothing is transcribed while recording -- two pw-record subprocesses write
        WAV straight to disk, so the video call keeps the CPU.
        """
        if self.meeting_active:
            logger.info("[meeting] Already recording")
            return
        self.meeting = meeting_mod.MeetingRecorder(
            transcript_dir=Path(self.config["meeting_transcript_dir"]),
            tmp_root=self._meeting_tmp_root,
            max_duration_s=self.config["meeting_max_duration_min"] * 60,
            silence_stop_s=self.config["meeting_silence_stop_min"] * 60,
            spawn=self._meeting_spawn,
        )
        try:
            self.meeting.start()
        except OSError as e:
            self.meeting = None
            self._tray_error = f"Meeting recording failed: {e}"
            self.notify("Meeting", f"Could not start recording: {e}", logging.ERROR, 5000)
            return
        self.meeting_active = True
        logger.info("[meeting] Recording to %s", self.meeting.session_dir)
        self.notify("Meeting", "Recording started (dictation paused)", logging.INFO, 2500)
        self._meeting_stop_event.clear()
        self._meeting_watchdog = threading.Thread(
            target=self._meeting_watchdog_loop, daemon=True
        )
        self._meeting_watchdog.start()

    def stop_meeting(self, transcribe: bool = True) -> None:
        """Leave meeting mode, then transcribe off the hotkey thread."""
        if not self.meeting_active or self.meeting is None:
            return
        session_dir = self.meeting.session_dir
        self._meeting_stop_event.set()
        self.meeting.stop_recording()
        self.meeting_active = False
        logger.info("[meeting] Recording stopped")
        if not transcribe:
            return
        threading.Thread(
            target=self._transcribe_meeting, args=(session_dir,), daemon=True
        ).start()

    def _meeting_watchdog_loop(self) -> None:
        """Stop the meeting on its own for a dead recorder, the cap, or silence.

        A dead pw-record is nearly always a full disk and is otherwise silent -- the
        file simply stops growing -- so it must surface rather than leave us
        "recording" into nothing.
        """
        while not self._meeting_stop_event.wait(MEETING_WATCHDOG_POLL_S):
            if not self.meeting_active or self.meeting is None:
                return
            try:
                failure = self.meeting.track_failure()
                if failure:
                    logger.error("[meeting] %s", failure)
                    self._tray_error = failure
                    self.notify("Meeting", failure, logging.ERROR, 8000)
                    self.stop_meeting()
                    return
                reason = self.meeting.auto_stop_reason()
                if reason:
                    logger.info("[meeting] Auto-stopping (%s)", reason)
                    message = {
                        "max_duration": "Recording hit the time limit; transcribing.",
                        "silence": "No audio for a while; transcribing.",
                    }[reason]
                    self.notify("Meeting", message, logging.INFO, 5000)
                    self.stop_meeting()
                    return
            except Exception as e:
                logger.error("[meeting] Watchdog error: %s", e, exc_info=True)

    def toggle_meeting(self) -> None:
        self.stop_meeting() if self.meeting_active else self.start_meeting()

    def _transcribe_meeting(self, session_dir) -> None:
        self.meeting_transcribing = True
        try:
            out = meeting_mod.finish_session(
                self, session_dir, Path(self.config["meeting_transcript_dir"])
            )
            self.notify("Meeting", f"Transcript: {out.name}", logging.INFO, 5000)
        except Exception as e:
            # The WAVs are deliberately left in place so the meeting is recoverable.
            logger.error("[meeting] Transcription failed: %s", e, exc_info=True)
            self._tray_error = f"Meeting transcription failed: {e}"
            self.notify(
                "Meeting",
                f"Transcription failed; audio kept in {session_dir}",
                logging.ERROR,
                8000,
            )
        finally:
            self.transcribing = False

    def _handle_modifier(self, key, pressed: bool) -> bool:
        """Track held modifiers. Returns True if the key was a modifier."""
        mod = modifier_name(key)
        if mod is None:
            return False
        self._held_modifiers.add(mod) if pressed else self._held_modifiers.discard(mod)
        return True

    def active_modifiers(self) -> set[str]:
        """Modifiers held now. Live from X11 where possible, tracked state otherwise."""
        live = self._modifier_source()
        return live if live is not None else set(self._held_modifiers)

    def _meeting_hotkey_pressed(self, key) -> bool:
        """True when the configured meeting combo is exactly satisfied.

        Checked before the dictation gate so the combo can also STOP a meeting --
        the gate that silences dictation must not silence its own off switch.

        The key is matched first so the (cheap) common case short-circuits before we
        ask X11 for the modifier state, which happens once per trigger-key press.
        """
        if self._meeting_key is None:
            return False
        if not keys_match(key, self._meeting_key):
            return False
        return self.active_modifiers() == set(self._meeting_modifiers)

    def _dictation_suspended(self) -> bool:
        """True while meeting mode owns the app and dictation must not fire."""
        if self.meeting_active:
            logger.debug("[hotkey] Ignored: meeting recording in progress")
            return True
        return False

    def on_press(self, key):
        try:
            if self._handle_modifier(key, pressed=True):
                return
            if self._meeting_hotkey_pressed(key):
                self._schedule_hotkey_action(self.toggle_meeting, "toggle_meeting")
                return
            if self._dictation_suspended():
                return
            if keys_match(key, self.hotkey, self._hotkey_value, self._hotkey_vk):
                self._last_hotkey_event_monotonic = time.monotonic()
                self._schedule_hotkey_action(self.start_recording, "start_recording")
        except Exception as e:
            # Never let an exception kill the pynput listener thread.
            logger.error("[hotkey] on_press failed: %s", e, exc_info=True)

    def on_release(self, key):
        try:
            if self._handle_modifier(key, pressed=False):
                return
            if self._meeting_hotkey_pressed(key):
                return  # the press already toggled; do not also stop dictation
            if self._dictation_suspended():
                return
            if keys_match(key, self.hotkey, self._hotkey_value, self._hotkey_vk):
                self._last_hotkey_event_monotonic = time.monotonic()
                self._schedule_hotkey_action(self.stop_recording, "stop_recording")
        except Exception as e:
            logger.error("[hotkey] on_release failed: %s", e, exc_info=True)

    def _schedule_hotkey_action(self, fn, name: str) -> None:
        """
        Run hotkey actions (start/stop) off the pynput callback thread.

        This prevents long operations (thread joins, IO) from blocking key processing and
        making the app appear to "stop reacting" after quick toggles.
        """

        def runner():
            if not self._hotkey_action_lock.acquire(blocking=False):
                held_s = None
                if self._hotkey_action_in_progress_since is not None:
                    held_s = time.monotonic() - self._hotkey_action_in_progress_since
                logger.warning(
                    "[hotkey] action %s skipped (in progress=%s held_s=%s)",
                    name,
                    self._hotkey_action_in_progress_name or "unknown",
                    f"{held_s:.2f}" if held_s is not None else "unknown",
                )
                # If we get stuck holding the hotkey action lock, we become permanently unresponsive.
                # In that case, prefer a clean restart under launchd KeepAlive.
                if held_s is not None and held_s > 10.0:
                    logger.critical(
                        "[hotkey] action lock stuck for %.2fs (in_progress=%s). Exiting for launchd restart.",
                        held_s,
                        self._hotkey_action_in_progress_name,
                    )
                    try:
                        if self.config.get("notifications"):
                            self.notify(
                                "SoupaWhisper stalled",
                                "Hotkey actions got stuck; restarting the service to recover.",
                                logging.ERROR,
                                5000,
                                "dialog-error",
                            )
                    finally:
                        os._exit(1)
                return
            try:
                self._hotkey_action_in_progress_since = time.monotonic()
                self._hotkey_action_in_progress_name = name
                fn()
            except Exception as e:
                logger.error("[hotkey] action %s failed: %s", name, e, exc_info=True)
            finally:
                self._hotkey_action_in_progress_since = None
                self._hotkey_action_in_progress_name = None
                self._hotkey_action_lock.release()

        threading.Thread(target=runner, daemon=True, name=f"hotkey_{name}").start()

    def stop(self):
        logger.info("\nExiting...")
        self.running = False
        # Stop the keyboard listener to release X11 grabs
        if hasattr(self, '_keyboard_listener') and self._keyboard_listener:
            self._keyboard_listener.stop()
        self._stop_tray()

    def _create_hotkey_listener(self):
        keyboard = _import_keyboard()
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        listener.daemon = True
        return listener

    def _start_hotkey_listener(self):
        listener = self._create_hotkey_listener()
        listener.start()
        self._keyboard_listener = listener
        return listener

    def restart_hotkey_listener(self) -> None:
        """Stop and start the pynput listener (menu / supervisor recovery)."""
        old = getattr(self, "_keyboard_listener", None)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        time.sleep(self._listener_restart_backoff_s)
        self._start_hotkey_listener()
        self._listener_restart_backoff_s = 1.0
        logger.info("[hotkey] Listener restarted")

    def _run_supervisor_loop(self) -> None:
        try:
            self._start_hotkey_listener()
        except Exception as e:
            exe = sys.executable
            logger.error("[hotkey] Failed to start keyboard listener: %s", e, exc_info=True)
            if self.config.get("notifications"):
                self.notify(
                    "Hotkey listener failed",
                    "SoupaWhisper could not listen for global hotkeys.\n\n"
                    "On macOS, this usually means missing Accessibility / Input Monitoring permission "
                    f"for the running executable:\n{exe}",
                    logging.ERROR,
                    10000,
                    "dialog-error",
                )
            raise

        last_heartbeat = 0.0
        while self.running:
            time.sleep(0.2)
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                last_heartbeat = now
                if self._last_hotkey_event_monotonic is not None:
                    _ = now - self._last_hotkey_event_monotonic

            listener = self._keyboard_listener
            if listener is not None and getattr(listener, "running", True) is False:
                try:
                    self.restart_hotkey_listener()
                except Exception as e:
                    self._listener_restart_backoff_s = min(30.0, self._listener_restart_backoff_s * 2.0)
                    logger.error("[hotkey] Listener restart failed: %s", e, exc_info=True)
                    if self.config.get("notifications"):
                        self.notify(
                            "Hotkey listener stopped",
                            "SoupaWhisper stopped receiving hotkeys and could not restart.\n\n"
                            f"Executable: {sys.executable}\n\n"
                            "If this keeps happening on macOS, re-check Accessibility / Input Monitoring permissions.",
                            logging.ERROR,
                            10000,
                            "dialog-error",
                        )

        try:
            if self._keyboard_listener:
                self._keyboard_listener.stop()
        except Exception:
            pass

    def _startup_notification(self) -> None:
        if self.config.get("notifications"):
            self.notify(
                "SoupaWhisper running",
                f"Hold {self.get_hotkey_name().upper()} to record, release to transcribe.",
                logging.INFO,
                1500,
                "dialog-information",
            )

    def run(self):
        self._startup_notification()
        if self.config.get("tray_icon", True):
            tray = self._start_tray_or_exit()
            self._tray = tray
            threading.Thread(
                target=self._run_supervisor_loop,
                daemon=True,
                name="hotkey_supervisor",
            ).start()
            tray.run_blocking()
            return
        self._run_supervisor_loop()

    def _audio_recording_worker(self):
        """Audio recording worker thread that collects audio data into one array."""
        chunk_size = 4096  # ~0.25 seconds of audio
        try:
            while self.recording:
                if not self.audio_stream:
                    break
                data = self.audio_stream.read(chunk_size, exception_on_overflow=False)
                if not data:
                    continue
                samples = np.frombuffer(data, dtype=np.int16)
                self.audio_data.append(samples)
        except Exception as e:
            logger.error(f"[record] Error in audio recording worker: {e}", exc_info=True)

    def start_recording(self):
        if self.recording:
            return
        self.model_loaded.wait()
        if self.model_error or self.model is None:
            logger.error("Recording is not ready yet.")
            return

        # Capture enforced language once for this session (hotkey press).
        self._begin_session_language()

        self.recording = True
        self.audio_data = []
        self._audio_problem_notified = False

        self.audio_stream = self._start_pyaudio_stream(self.frames_per_buffer)
        if self.audio_stream is None:
            self.recording = False
            self._end_session_language()
            return

        self.audio_thread = threading.Thread(
            target=self._audio_recording_worker,
            daemon=True,
            name="audio_recording_worker"
        )
        self.audio_thread.start()
        self.notify("Recording...", f"Release {self.get_hotkey_name().upper()} when done", logging.INFO, 2000, "emblem-synchronizing")

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self.transcribing = True

        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)

        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None

        self.notify("Transcribing...", "Processing your speech", logging.INFO, 1500)

        try:
            if not self.audio_data:
                self._report_audio_problem("No audio data recorded")
                return

            audio_array = np.concatenate(self.audio_data)
            # Check if audio contains only zeros or is effectively silent (audio input problem)
            if self._check_valid_audio_input(audio_array):
                return
            if self.config.get("save_recordings", False):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = str(scratch_wav_path(f"recording_{timestamp}.wav"))
                try:
                    with wave.open(file_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_array.tobytes())
                    logger.info(f"[record] Saved recording to {file_path}")
                except Exception as e:
                    logger.error(f"[record] Failed to save recording: {e}", exc_info=True)
            trans_start = time.monotonic()
            text, duration = self._transcribe_audio_array(audio_array)
            trans_duration = time.monotonic() - trans_start
            if text:
                logger.info(f"Transcribed {duration:.2f}s in {trans_duration:.2f}s: {text}")
                if self.config["clipboard"]:
                    clip_cmd = ["pbcopy"] if IS_MACOS else ["xclip", "-selection", "clipboard"]
                    process = subprocess.Popen(clip_cmd, stdin=subprocess.PIPE)
                    process.communicate(input=text.encode())
                    logger.info(f"Pasted to clipboard: {text}")
                if self.config["auto_type"] and self.typer:
                    self.typer.type_rewrite(text, 0)
                if self.config["notifications"]:
                    self.notify(
                        f"Transcribed {duration:.2f}s speech:",
                        text[:100] + ("..." if len(text) > 100 else ""),
                        logging.INFO,
                        3000,
                        "emblem-ok-symbolic",
                    )
            else:
                self.notify("No speech detected", "Check your microphone or try speaking louder", logging.WARNING, 2000, "audio-input-microphone")
        except Exception as e:
            logger.error(f"Error transcribing: {e}", exc_info=True)
            self.notify("Error", str(e)[:50], logging.ERROR, 3000)
            self._tray_error = str(e)
        finally:
            self.audio_data = []
            self.transcribing = False
            self._end_session_language()

    def transcribe_file(self, wav_file_path: str) -> str:
        """
        Transcribe a WAV file using non-streaming transcription.
        Expects WAV files (16-bit, 16kHz, mono).

        Args:
            wav_file_path: Path to the WAV file to transcribe

        Returns:
            The transcribed text
        """
        logger.info(f"[file] Transcribing WAV file: {wav_file_path}")
        try:
            with wave.open(wav_file_path, "rb") as wf:
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            text, duration = self._transcribe_audio_array(audio_array)
            if text:
                logger.info(f"[file] Transcribed {duration:.2f}s: {text}")
            else:
                logger.info("[file] No speech detected")
            return text
        except Exception as e:
            logger.error(f"[file] Error transcribing: {e}", exc_info=True)
            raise

    def transcribe_file_to_output(self, wav_file_path: str, output_path: str) -> None:
        """
        Transcribe a WAV file and write the result to output_path.
        Writes SRT subtitles (with segment timestamps) if output_path ends with ".srt",
        plain text otherwise. Expects WAV files (16-bit, 16kHz, mono).

        Args:
            wav_file_path: Path to the WAV file to transcribe
            output_path: Path to write the transcript to
        """
        logger.info(f"[file] Transcribing WAV file: {wav_file_path} -> {output_path}")
        with wave.open(wav_file_path, "rb") as wf:
            audio_data = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.model_error or self.model is None:
            raise RuntimeError("Model not loaded")
        lang = resolve_transcription_language(
            self.model, audio_array, self.config.get("language"), self.config.get("language_allowlist")
        )
        # No VAD filter: for pre-recorded files nothing should be dropped,
        # and subtitle timestamps must stay aligned with the original audio.
        segments, info = self.model.transcribe(
            audio_array,
            vad_filter=False,
            language=lang,
            **self.custom_terms_kwargs,
        )
        with open(output_path, "w", encoding="utf-8") as out:
            if output_path.lower().endswith(".srt"):
                index = 0
                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue
                    index += 1
                    out.write(
                        f"{index}\n"
                        f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}\n"
                        f"{text}\n\n"
                    )
            else:
                out.write(self._segments_to_text(segments, self.config["auto_sentence"]) + "\n")
        logger.info(f"[file] Transcribed {info.duration:.2f}s to {output_path}")


class StreamingDictation(Dictation):
    """Streaming dictation mode.
    """

    def __init__(self, config: dict):
        # Initialize base class (sets up config, hotkey, model loading, etc.)
        super().__init__(config)
        self.min_speech_length_seconds = config["min_speech_length_seconds"]
        self.vad_silence_threshold_seconds = config["vad_silence_threshold_seconds"]
        self.vad_sample_rate = config["vad_sample_rate"]
        self.vad_chunk_size_ms = config["vad_chunk_size_ms"]
        self.vad_min_speech_chunks = config["vad_min_speech_chunks"]
        # Validate vad_chunk_size_ms - webrtcvad only supports 10ms, 20ms, or 30ms
        if self.vad_chunk_size_ms not in [10, 20, 30]:
            raise ValueError(f"vad_chunk_size_ms must be 10, 20, or 30 (got {self.vad_chunk_size_ms}). webrtcvad only supports these frame sizes.")
        self.vad_threshold = config["vad_threshold"]
        self.vad_frame_size_ms = self.vad_chunk_size_ms
        self.vad_frame_size = int(self.vad_sample_rate * self.vad_frame_size_ms / 1000)

        # Workers, queues, etc.
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.transcription_queue: queue.Queue[_StreamingChunk | None] = queue.Queue()
        self.typing_queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        self.file_saving_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self.audio_stream = None
        self.audio_thread: Optional[threading.Thread] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.typing_thread: Optional[threading.Thread] = None
        self.file_saving_thread: Optional[threading.Thread] = None
        self.vad = webrtcvad.Vad(int(self.vad_threshold))
        self.typer: Optional[Typer] = None
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100,
            )

        # State variables.
        self.in_speech = False
        self.file_mode = False
        self.stopping = False
        self.recording_start_time: float = 0.0
        self.speech_segment_chunks: list[np.ndarray] = []
        self.speech_start_time: Optional[float] = None
        self.speech_silence_duration: float = 0.0
        self.speech_end_time: float = 0.0
        self.accumulated_text = ""
        self._last_hotkey_event_monotonic: Optional[float] = None

    def _finish_model_loading(self):
        if not self._interactive:
            logger.info("Model ready (file transcription mode).")
            return
        logger.info(f"Press [{self.get_hotkey_name().upper()}] to start transcribing, press one more time to stop. Press Ctrl+C to quit.")

    def _create_hotkey_listener(self):
        keyboard = _import_keyboard()
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        listener.daemon = True
        return listener

    def on_release(self, key):
        """Streaming toggles on press only; releases just keep modifier state fresh."""
        try:
            self._handle_modifier(key, pressed=False)
        except Exception as e:
            logger.error("[hotkey] on_release failed: %s", e, exc_info=True)

    def _startup_notification(self) -> None:
        if self.config.get("notifications"):
            self.notify(
                "SoupaWhisper running",
                f"Press {self.get_hotkey_name().upper()} to start/stop streaming transcription.",
                logging.INFO,
                1500,
                "dialog-information",
            )

    def on_press(self, key):
        try:
            if self._handle_modifier(key, pressed=True):
                return
            if self._meeting_hotkey_pressed(key):
                self._schedule_hotkey_action(self.toggle_meeting, "toggle_meeting")
                return
            if self._dictation_suspended():
                return
            if keys_match(key, self.hotkey, self._hotkey_value, self._hotkey_vk):
                self._last_hotkey_event_monotonic = time.monotonic()
                if not self.recording:
                    self._schedule_hotkey_action(self.start_recording, "start_recording")
                else:
                    self._schedule_hotkey_action(self.stop_recording, "stop_recording")
        except Exception as e:
            logger.error("[hotkey] on_press failed: %s", e, exc_info=True)

    def start_recording(self):
        if self.recording:
            logger.error("[record] Recording is already started")
            return
        if self.stopping:
            self.notify("Error", "Previous recording is still shutting down. Please wait a moment.", logging.ERROR, 3000)
            return
        self.model_loaded.wait()
        if self.model_error or self.model is None:
            self.notify("Error", "Model is not loaded yet", logging.ERROR, 3000)
            return
        # Capture enforced language once for this streaming session (hotkey press).
        self._begin_session_language()
        # Reset transcription state.
        self.recording = True
        self._audio_problem_notified = False
        self.accumulated_text = ""
        self.in_speech = False
        self.speech_segment_chunks = []
        self.speech_start_time = None
        # Clear queues to remove any leftover sentinels from previous recording
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except queue.Empty:
                break
        while not self.typing_queue.empty():
            try:
                self.typing_queue.get_nowait()
            except queue.Empty:
                break
        frames_per_buffer = int(self.vad_sample_rate * self.vad_chunk_size_ms / 1000.0)
        self.audio_stream = self._start_pyaudio_stream(frames_per_buffer)
        if self.audio_stream is None:
            self.recording = False
            self._end_session_language()
            return
        self.audio_thread = threading.Thread(
            target=self._continuous_audio_stream_worker,
            args=(frames_per_buffer,),
            daemon=True,
            name="audio_stream_worker"
        )
        self.audio_thread.start()
        self.recording_start_time = time.monotonic()
        self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.transcription_thread.start()
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100,
            )
            self.typing_thread = threading.Thread(target=self._typing_worker, daemon=True)
            self.typing_thread.start()
        if self.config.get("save_recordings", False):
            self.file_saving_thread = threading.Thread(target=self._file_saving_worker, daemon=True)
            self.file_saving_thread.start()
        # Notify about recording start.
        self.notify("Recording...", f"Press {self.get_hotkey_name().upper()} when done", logging.INFO, 1500, "audio-input-microphone")

    def _continuous_audio_stream_worker(self, frames_per_buffer: int):
        captured_samples = 0
        try:
            while self.recording:
                if not self.audio_stream:
                    break
                data = self.audio_stream.read(frames_per_buffer, exception_on_overflow=False)
                if not data:
                    continue
                samples = np.frombuffer(data, dtype=np.int16)
                self._process_audio_chunk(
                    samples, captured_samples / float(self.vad_sample_rate),
                    len(samples) / float(self.vad_sample_rate),
                )
                captured_samples += len(samples)
        except Exception as e:
            logger.error(f"[record] Error in audio stream worker: {e}", exc_info=True)
        finally:
            self._finalize_segment()

    def _enqueue_speech_segment(self, segment: np.ndarray):
        start = self.speech_start_time if self.speech_start_time is not None else 0.0
        self.transcription_queue.put(_StreamingChunk(
            segment, start, max(start, self.speech_end_time),
        ))

    def _reset_speech_mode(self):
        self.in_speech = False
        self.speech_segment_chunks = []
        self.speech_start_time = None
        self.speech_silence_duration = 0.0
        self.speech_end_time = 0.0

    def _process_audio_chunk(self, frame: np.ndarray, frame_start_time: float, frame_duration: float):
        """
        Process a single audio frame using VAD to determine segment boundaries.
        """
        if len(frame) != self.vad_frame_size:
            logger.error(f"[vad] Invalid frame size: expected {self.vad_frame_size} samples, got {len(frame)}")
        # Use VAD to detect if this frame contains speech.
        has_speech = False
        try:
            has_speech = self.vad.is_speech(frame.tobytes(), self.vad_sample_rate)
        except Exception as e:
            logger.error(f"[vad] VAD processing failed: {e}", exc_info=True)
            self.notify("Error", f"VAD processing failed: {e}", logging.ERROR, 3000)
            has_speech = False
        # Handle frame data.
        if has_speech:
            # Include the first accepted speech frame, before in_speech becomes true.
            self.speech_end_time = frame_start_time + frame_duration
            if not self.in_speech:
                segment_length = len(self.speech_segment_chunks)
                # Check it is the first speech chunk.
                if segment_length == 0:
                    self.speech_start_time = frame_start_time
                # Check if we have enough consecutive speech chunks to officially start a segment.
                if segment_length >= self.vad_min_speech_chunks:
                    self.in_speech = True
                    # Log "speech detected" as soon as we're sure speech is active.
                    if segment_length == self.vad_min_speech_chunks and self.speech_start_time is not None:
                        logger.info(f"[SAD] {frame_start_time:.3f} speech detected (started {frame_start_time - self.speech_start_time:.3f} seconds ago)")
            # Add the frame to the current speech segment exactly once.
            self.speech_segment_chunks.append(frame)
        else:  # This frame does not contain speech.
            if self.in_speech:
                self.speech_silence_duration += frame_duration
                if self.speech_end_time == 0.0:
                    self.speech_end_time = frame_start_time
                # Check duration of a silence is not long enough to finalize the segment.
                if self.speech_silence_duration < self.vad_silence_threshold_seconds:
                    self.speech_segment_chunks.append(frame)
                    return
                # Otherwise finalize the segment.
                # Check state variables.
                if self.speech_start_time is None:
                    self.notify("Error", "Speech start time is not set", logging.ERROR, 2000)
                    return
                # Concatenate all chunks (speech + non-speech at the end) into a single segment.
                segment: np.ndarray = np.concatenate(self.speech_segment_chunks)
                segment_duration = len(segment) / float(self.vad_sample_rate)
                logger.info(f"[SAD] {frame_start_time:.3f} speech finished ({self.speech_end_time:.3f}s ago), handling chunk of {segment_duration:.3f} seconds audio")
                # Send complete segment to transcription queue
                self._enqueue_speech_segment(segment)
                if not self.file_mode and self.config.get("save_recordings", False):
                    self.file_saving_queue.put((segment, self.vad_sample_rate))
                self._reset_speech_mode()
            elif len(self.speech_segment_chunks) > 0:
                # Reset speech mode.
                self._reset_speech_mode()

    def _finalize_segment(self):
        """Force finalize remaining speech segment."""
        if self.in_speech and self.speech_segment_chunks:
            # Finish segment forcefully.
            segment = np.concatenate(self.speech_segment_chunks)
            segment_audio_duration = len(segment) / float(self.vad_sample_rate)
            current_in_segment_time = self.speech_end_time
            # Calculate current in segment time.
            if current_in_segment_time <= 0.0:
                if self.speech_start_time is not None:
                    current_in_segment_time = self.speech_start_time + segment_audio_duration
                else:
                    current_in_segment_time = segment_audio_duration
            # Log finalization.
            logger.info(f"[SAD] {current_in_segment_time:.3f} speech finished (forced), handling chunk of {segment_audio_duration:.3f} seconds audio")
            # Send complete segment to transcription queue.
            self._enqueue_speech_segment(segment)
            # Save segment to file if needed.
            if not self.file_mode and self.config.get("save_recordings", False):
                self.file_saving_queue.put((segment, self.vad_sample_rate))
            self._reset_speech_mode()

    def _transcription_worker(self):
        """Transcription worker thread - processes chunks in order."""
        chunk_idx = 0
        context_words = self.config.get("streaming_context_words", _DEFAULT_STREAMING_CONTEXT_WORDS)
        context_reset_seconds = self.config.get(
            "streaming_context_reset_seconds", _DEFAULT_STREAMING_CONTEXT_RESET_S
        )
        # Worker-local history starts fresh for each live/file session.
        recent_text = ""
        previous_language = None
        previous_speech_end = None
        # FYI: continue during "stopping" state to process all remaining items before exiting.
        while self.recording or self.stopping or not self.transcription_queue.empty():
            try:
                chunk: _StreamingChunk | None = self.transcription_queue.get(timeout=0.1)
                if chunk is None:
                    # If we're stopping and there might be more segments, continue processing
                    # This handles the race condition where None is added before the final segment
                    if self.stopping and not self.transcription_queue.empty():
                        continue
                    break
                if (
                    previous_speech_end is not None
                    and chunk.speech_start - previous_speech_end >= context_reset_seconds
                ):
                    recent_text = ""
                    previous_language = None
                    logger.debug("[transcriber] Cleared text context after speech pause")
                previous_speech_end = chunk.speech_end
                segment = chunk.audio
                if self.model is None:
                    recent_text = ""
                    logger.error("[transcriber] Model not loaded")
                    continue
                # Check if segment is all zeros or effectively silent (audio input problem)
                if self._check_valid_audio_input(segment):
                    recent_text = ""
                    continue
                # Convert int16 to float32 and normalize to [-1.0, 1.0]
                if segment.dtype == np.int16:
                    segment = segment.astype(np.float32) / 32768.0
                # Note: transcribe() returns an iterator; iteration runs the decoder.
                trans_start = time.monotonic()
                enforced = self._maybe_enforced_language_for_field()
                configured_lang = self.config.get("language")
                allowlist = self.config.get("language_allowlist")
                if enforced and configured_lang is None:
                    if not allowlist or enforced in allowlist:
                        lang = enforced
                    else:
                        lang = resolve_transcription_language(self.model, segment, configured_lang, allowlist)
                else:
                    lang = resolve_transcription_language(self.model, segment, configured_lang, allowlist)
                # With unrestricted auto detection, resolve the new chunk's language
                # before supplying history so a switch cannot inherit the old prompt.
                if recent_text and lang is None:
                    try:
                        lang, _, _ = self.model.detect_language(
                            segment, vad_filter=True,
                            vad_parameters=VadOptions(**_STREAMING_VAD_PARAMETERS),
                        )
                    except Exception:
                        recent_text = ""
                        logger.warning("[transcriber] Could not resolve context language", exc_info=True)
                if lang != previous_language:
                    recent_text = ""
                prompt_kwargs = dict(self.custom_terms_kwargs)
                if recent_text:
                    glossary = prompt_kwargs.get("initial_prompt", "")
                    prompt_kwargs["initial_prompt"] = (
                        glossary + "\n" + recent_text if glossary else recent_text
                    )
                segments, info = self.model.transcribe(
                    segment,
                    # Second VAD pass with options tuned for short clips (see _STREAMING_VAD_PARAMETERS).
                    vad_filter=True,
                    vad_parameters=_STREAMING_VAD_PARAMETERS,
                    # FYI: don't set temperature=0.0, it will cause errors like
                    # "Log probability threshold is not met with temperature 0.0 (-1.359611 < -1.000000)"
                    # temperature=0.0,
                    language=lang,
                    condition_on_previous_text=True,
                    without_timestamps=True,
                    **prompt_kwargs,
                )
                previous_language = lang or getattr(info, "language", None)
                text = _streaming_segments_to_text(segments)
                trans_duration = time.monotonic() - trans_start
                if not text:
                    recent_text = ""
                    logger.info(f"[transcriber] Empty transcription (transcribed in {trans_duration:.2f}s)")
                    continue
                if self.should_reject_text(text):
                    logger.info("[reject] Skipping chunk (matched reject phrase): %r", text)
                    continue
                else:
                    logger.info(f"[transcriber] Transcribed {len(segment) / float(self.vad_sample_rate):.2f}s in {trans_duration:.2f}s: {text}")
                # Only accepted output becomes prompt context, never glossary text.
                recent_text = (
                    " ".join((recent_text + " " + text).split()[-context_words:])
                    if context_words else ""
                )
                # Add space before chunk if it's not the first one
                if chunk_idx > 0:
                    text = " " + text
                self.accumulated_text += text
                chunk_idx += 1
                if not self.file_mode and self.typing_queue:
                    self.typing_queue.put((text, chunk_idx))
            except queue.Empty:
                continue
            except Exception as e:
                recent_text = ""
                previous_language = None
                logger.error(f"[transcriber] {e}", exc_info=True)

    def _typing_worker(self):
        """Typing worker thread - types text in order of chunks."""
        # FYI: continue during "stopping" state to process all remaining items before exiting.
        while self.recording or self.stopping or not self.typing_queue.empty():
            if not self.typer:
                self.notify("Error", "Typer is not initialized", logging.ERROR, 3000)
                return
            try:
                typing_task = self.typing_queue.get(timeout=0.1)
                # Check for signal to stop typing.
                if typing_task is None:
                    break
                # Get typing task and log it.
                text_to_type, chunk_idx = typing_task
                logger.info(f"[typer] Chunk {chunk_idx} typing: {text_to_type}")
                # Run typing.
                try:
                    self.typer.type_rewrite(text_to_type, 0)
                except Exception as e:
                    logger.error(f"[typer] Typing failed: {e}", exc_info=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[typer] {e}", exc_info=True)

    def _file_saving_worker(self):
        """File saving worker thread - saves audio segments to files asynchronously."""
        # FYI: continue during "stopping" state to process all remaining items before exiting.
        while self.recording or self.stopping or not self.file_saving_queue.empty():
            try:
                task = self.file_saving_queue.get(timeout=0.1)
                if task is None:
                    break
                segment, sample_rate = task
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = str(scratch_wav_path(f"stream_chunk_{timestamp}.wav"))
                try:
                    with wave.open(file_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(segment.tobytes())
                except Exception as e:
                    logger.error(f"[record] Failed to save streaming chunk: {e}", exc_info=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[file_saver] {e}", exc_info=True)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.stopping = True
        # Notify about stopping.
        self.notify("Transcribing stopped", "Processing remaining chunks", logging.INFO, 1500, "emblem-synchronizing")
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None
        if self.audio_interface:
            try:
                self.audio_interface.terminate()
            except Exception:
                pass
            self.audio_interface = None
        # Signal transcription worker to stop.
        self.transcription_queue.put(None)
        if self.config.get("save_recordings", False):
            self.file_saving_queue.put(None)
        # Wait for transcription worker to finish processing all segments
        # (including the last one that might be in progress).
        if self.transcription_thread:
            self.transcription_thread.join(timeout=10.0)
        # Now that transcription is done, signal typing worker to stop.
        if not self.file_mode and self.config["auto_type"]:
            self.typing_queue.put(None)
        # Join remaining worker threads.
        if self.typing_thread:
            self.typing_thread.join(timeout=1.0)
        if self.file_saving_thread:
            self.file_saving_thread.join(timeout=1.0)
        self.stopping = False
        # Finalize any remaining chunks
        final_text = self.accumulated_text.strip()
        if final_text:
            if self.config["clipboard"]:
                clip_cmd = ["pbcopy"] if IS_MACOS else ["xclip", "-selection", "clipboard"]
                process = subprocess.Popen(clip_cmd, stdin=subprocess.PIPE)
                process.communicate(input=final_text.encode())
                logger.info(f"[idle] Pasted to clipboard: {final_text}")
            logger.info(f"[idle] Final text: {final_text}")
            self.notify("Got:", final_text[:100] + ("..." if len(final_text) > 100 else ""), logging.INFO, 3000)
        else:
            logger.info("[idle] No speech detected")
            self.notify("No speech detected", "Try speaking louder or check audio device", logging.WARNING, 2000)
        self._end_session_language()

    def transcribe_file(self, wav_file_path: str) -> str:
        """
        Transcribe a WAV file using the streaming transcription pipeline.
        Expects WAV files created by Dictation (16-bit, 16kHz, mono).

        Args:
            wav_file_path: Path to the WAV file to transcribe

        Returns:
            The transcribed text
        """
        logger.info(f"[file] Transcribing WAV file: {wav_file_path}")
        # Set file mode to skip typing. No other preparations needed.
        self.file_mode = True
        # Read WAV file (expects 16-bit, 16kHz, mono).
        try:
            with wave.open(wav_file_path, "rb") as wf:
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
                audio_buffer = np.frombuffer(audio_data, dtype=np.int16)
        except Exception as e:
            raise RuntimeError(f"Failed to read WAV file: {e}")
        # Process audio in chunks similar to streaming mode.
        frames_per_buffer = int(self.vad_sample_rate * self.vad_chunk_size_ms / 1000.0)
        chunk_duration = frames_per_buffer / float(self.vad_sample_rate)
        # Start transcription worker thread (no typing worker for file mode).
        self.recording = True
        transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        transcription_thread.start()
        try:
            # Process audio samples in chunks.
            for i in range(0, len(audio_buffer), frames_per_buffer):
                if not self.recording:
                    break
                chunk = audio_buffer[i:i + frames_per_buffer]
                if len(chunk) < frames_per_buffer:
                    chunk = np.pad(chunk, (0, frames_per_buffer - len(chunk)), mode='constant')
                self._process_audio_chunk(chunk, i / float(self.vad_sample_rate), chunk_duration)
            # Finalize any remaining speech segment.
            self._finalize_segment()
            # Signal transcription worker to stop.
            self.recording = False
            self.transcription_queue.put(None)
            # Wait for transcription to complete.
            transcription_thread.join(timeout=30.0)
            if transcription_thread.is_alive():
                logger.warning("[file] Transcription thread did not finish in time")
            final_text = self.accumulated_text.strip()
            logger.info(f"[file] Transcription complete: {final_text}")
            return final_text
        except Exception as e:
            logger.error(f"[file] Error during transcription: {e}", exc_info=True)
            raise
        finally:
            self.file_mode = False

    def stop(self):
        logger.info("\nExiting...")
        self.running = False
        if self.recording:
            self.stop_recording()
        # Stop the keyboard listener to release X11 grabs
        if hasattr(self, '_keyboard_listener') and self._keyboard_listener:
            self._keyboard_listener.stop()
        self._stop_tray()


def check_dependencies(config: dict):
    """Check that required system commands are available."""
    if IS_MACOS:
        # macOS types through Quartz and copies with pbcopy; neither needs installing.
        return
    missing = []
    required_cmds = []
    if config["clipboard"]:
        required_cmds.append("xclip")
    if config["auto_type"]:
        required_cmds.append("xdotool")
    for cmd in required_cmds:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            missing.append((cmd, cmd))
    if missing:
        logger.error("Missing dependencies:")
        for cmd, pkg in missing:
            logger.error(f"  {cmd} - install with something like: sudo apt install {pkg}")
        sys.exit(1)


def get_model_cache_path():
    """Get the path where faster-whisper models are cached."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(cache_home, "huggingface", "hub")


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)d [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load and print configuration
    config = load_config()
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Prepare arguments parser.
    cache_path = get_model_cache_path()
    description = f"""SoupaWhisper - voice dictation tool.

Works in both streaming and non-streaming modes.
- Non-streaming mode: push-to-talk, text is available only at the end of recording, good quality transcription.
- Streaming mode: press to toggle transcribing, text is appearing incrementally as you speak (by small chunks), quality is lower.

Version: {__version__}
Config file: {CONFIG_PATH}
Model cache: {cache_path}

Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3
"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming transcription mode (default: from config)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming transcription mode (default: from config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output to troubleshoot"
    )
    parser.add_argument(
        "--file",
        type=str,
        metavar="WAV_FILE",
        help="Transcribe provided WAV file and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="OUTPUT_FILE",
        help="With --file: write transcript to this path (.srt for subtitles with timestamps, anything else for plain text)"
    )
    args = parser.parse_args()

    # Apply arguments.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle file transcription mode (headless: no X11 / pynput / clipboard tools).
    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"WAV file not found: {args.file}")
        dictation = Dictation(config, interactive=False)
        dictation.model_loaded.wait()
        if dictation.model_error or dictation.model is None:
            logger.error("Cannot transcribe file: model failed to load")
            sys.exit(1)
        try:
            if args.output:
                dictation.transcribe_file_to_output(args.file, args.output)
            else:
                dictation.transcribe_file(args.file)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}", exc_info=True)
            sys.exit(1)

    check_dependencies(config)

    # Determine streaming mode
    use_streaming = config["default_streaming"]
    if args.streaming:
        use_streaming = True
    elif args.no_streaming:
        use_streaming = False

    # Create dictation object.
    if use_streaming:
        dictation = StreamingDictation(config)
    else:
        dictation = Dictation(config)

    # Handle Ctrl+C and SIGTERM gracefully
    def handle_signal(sig, frame):
        dictation.stop()
        # Use sys.exit() instead of os._exit() to allow cleanup
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)  # For systemd stop

    # Start dictation.
    dictation.run()


if __name__ == "__main__":
    main()
