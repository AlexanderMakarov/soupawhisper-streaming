"""Meeting recording mode for SoupaWhisper.

Captures two tracks with pw-record (your mic + the default sink's monitor),
then transcribes each to SRT and merges them into one speaker-labelled
Markdown transcript.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import webrtcvad
from faster_whisper.vad import VadOptions, get_speech_timestamps

logger = logging.getLogger(__name__)

# Every WAV this app writes -- meeting tracks and save_recordings debug mirrors --
# lives under one root so the tray can size it and wipe it in one place, and so we
# never scatter files among other apps' /tmp entries.
TMP_ROOT = Path(tempfile.gettempdir()) / "soupawhisper-streaming"

# "00:01:02,345 --> 00:01:04,120" as written by Dictation.transcribe_file_to_output().
_SRT_TIMING_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)

# One voice on the mic track: merge freely across normal breathing pauses. The cap
# is what actually splits blocks (gap splitting proved inert in practice), so it sets
# how readable the transcript is. Measured against a professional transcript of the
# same 73-minute interview: 600s gave 415-word walls, while 60s gives 86 blocks with
# a 110-word longest -- almost exactly the reference's 88 blocks and 121 words.
ME_GAP_S = 30.0
ME_MAX_BLOCK_S = 60.0
# The monitor track may carry several people, but silence length does NOT mark a
# speaker change -- turn transitions are often faster than intra-sentence pauses.
# So the gap is only a weak hint and the duration cap does the real splitting,
# giving a fresh timestamp roughly every 20s of continuous participant speech.
THEM_GAP_S = 2.0
THEM_MAX_BLOCK_S = 20.0

# WebRTC VAD aggressiveness for the auto-stop watchdog. Must be >= 2: at 0 and 1 a
# real mic's noise floor (measured: +/-4 LSB hiss) scores 0.14 speech, so silence
# would never be recognised and the watchdog would never fire.
SILENCE_VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 20
SAMPLE_RATE = 16000

# Your ask: a forgotten recording must not run forever (it costs ~230 MB/hour).
MAX_DURATION_S = 2 * 3600
# Both tracks must be silent this long before we call the meeting over. Generous on
# purpose: a false stop loses the rest of the meeting, while over-recording only
# costs disk. Real meetings do go quiet for a minute while people read.
SILENCE_STOP_S = 600.0

# Whisper decodes in fixed 30s windows and detects language from one such window,
# so a speech run longer than this is split: otherwise a long monologue would be
# locked to whatever language its first 30s scored.
LANGUAGE_WINDOW_S = 30.0
# Context is only a fallback for uncertain scores, never a language lock.
LANGUAGE_CONTEXT_MAX_GAP_S = 2.0
LANGUAGE_MIN_PROBABILITY = 0.35
LANGUAGE_MIN_SHARE = 0.80
# Silero VAD settings for cutting a track into speech runs. faster-whisper's own
# default silence gap is 2000ms, long enough to swallow a whole turn change -- and
# with it a language switch -- into one run. 700ms still keeps a sentence's internal
# pauses together while separating one utterance from the next.
RUN_MIN_SILENCE_MS = 700
# Padding restored around each run so the decoder hears the attack of the first word
# and the tail of the last; VAD boundaries sit tight against the speech itself.
RUN_PAD_MS = 200
# Silero's own defaults, restated here so config fallbacks have one source of truth.
# The threshold is the speech probability a 32ms frame must clear; the minimum
# duration drops blips (0 = keep everything the VAD called speech).
RUN_VAD_THRESHOLD = 0.5
RUN_MIN_SPEECH_MS = 0
# Per-word timings, for measuring pauses and response latency. Off by default: it
# adds a cross-attention alignment pass per run.
WORD_TIMESTAMPS = False
# Backstop for noise the VAD still accepts as speech, which makes Whisper invent
# text. Its avg_logprob separates the two -- measured -2.2 on invented credits vs
# -0.26 on a clear sentence. The floor sits at -1.5, not at Whisper's -1.0 default:
# a real one-word answer ("Всё.") measured -1.09, and short utterances are naturally
# less confident. Silero now rejects silence before it ever reaches the model, so
# this no longer has to carry the hallucination guard on its own.
MIN_AVG_LOGPROB = -1.5


@dataclass
class WavUsage:
    """How much disk our scratch WAVs currently occupy."""

    total_bytes: int
    count: int

    @property
    def human(self) -> str:
        mb = self.total_bytes / 1e6
        return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def wav_usage(root: Path = TMP_ROOT) -> WavUsage:
    """Total size and count of scratch WAVs under root (missing root = zero)."""
    total = count = 0
    for wav in Path(root).rglob("*.wav"):
        try:
            total += wav.stat().st_size
            count += 1
        except OSError:
            continue  # vanished mid-scan; it costs us nothing either way
    return WavUsage(total, count)


def wav_tail(path: Path, seconds: float) -> np.ndarray:
    """Last `seconds` of a mono int16 WAV, for polling a file pw-record is still writing."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        want = min(frames, int(wf.getframerate() * seconds))
        wf.setpos(frames - want)
        data = wf.readframes(want)
    return np.frombuffer(data, dtype=np.int16)


def pw_record_cmd(path: Path, capture_sink: bool) -> list[str]:
    """Build a pw-record command writing Whisper-ready audio (16kHz mono s16).

    pw-record is used rather than parec because the PulseAudio shim was measured
    dropping the first ~2s of every mic capture on this stack. With capture_sink,
    the stream property makes PipeWire link us to the DEFAULT sink's monitor and
    re-link on default-sink changes -- so plugging in headphones mid-meeting does
    not silently leave us recording a dead device.
    """
    cmd = ["pw-record", "--rate=16000", "--channels=1", "--format=s16"]
    if capture_sink:
        cmd += ["-P", "stream.capture.sink=true"]
    return cmd + [str(path)]


def speech_ratio(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    aggressiveness: int = SILENCE_VAD_AGGRESSIVENESS,
    frame_ms: int = VAD_FRAME_MS,
) -> float:
    """Fraction of VAD frames in `samples` (int16 mono) that contain speech."""
    frame_len = sample_rate * frame_ms // 1000
    frame_count = len(samples) // frame_len
    if frame_count == 0:
        return 0.0
    vad = webrtcvad.Vad(aggressiveness)
    speech = sum(
        vad.is_speech(samples[i * frame_len : (i + 1) * frame_len].tobytes(), sample_rate)
        for i in range(frame_count)
    )
    return speech / frame_count


def free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding path (0 if it cannot be determined)."""
    import shutil

    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def purge_wavs(root: Path = TMP_ROOT, keep: Path | None = None) -> int:
    """Delete scratch WAVs under root, returning bytes freed.

    `keep` is the session directory currently being recorded; its files are left
    alone so a tray-triggered cleanup can never pull the floor out from under a
    meeting in progress.
    """
    keep = Path(keep).resolve() if keep else None
    freed = 0
    for wav in Path(root).rglob("*.wav"):
        if keep and keep in wav.resolve().parents:
            continue
        try:
            size = wav.stat().st_size
            wav.unlink()
            freed += size
        except OSError as e:
            logger.warning("Could not delete %s: %s", wav, e)
    return freed


@dataclass
class Word:
    """One word with its own span, from Whisper's cross-attention alignment."""

    start: float
    end: float
    text: str
    probability: float


@dataclass
class Cue:
    """One subtitle cue: a timestamped span of transcribed speech."""

    start: float
    end: float
    text: str
    words: list["Word"] = field(default_factory=list)


@dataclass
class Block:
    """Consecutive speech from one speaker, rendered as one transcript entry."""

    speaker: str
    start: float
    end: float
    text: str


def merge_tracks(
    me: list[Cue],
    them: list[Cue],
    me_gap_s: float = ME_GAP_S,
    them_gap_s: float = THEM_GAP_S,
    me_max_block_s: float = ME_MAX_BLOCK_S,
    them_max_block_s: float = THEM_MAX_BLOCK_S,
) -> list[Block]:
    """Interleave both tracks into one chronological, speaker-labelled transcript.

    Consecutive cues from the same speaker join into one block while the silence
    between them stays under that speaker's gap and the block stays under that
    speaker's length cap. "Them" may hold several people and cannot be split by
    silence alone, so its length cap is short and carries the splitting; "Me" is
    one voice and merges freely.
    """
    tagged = [("Me", c) for c in me] + [("Them", c) for c in them]
    tagged.sort(key=lambda pair: pair[1].start)
    gaps = {"Me": me_gap_s, "Them": them_gap_s}
    caps = {"Me": me_max_block_s, "Them": them_max_block_s}

    blocks: list[Block] = []
    for speaker, c in tagged:
        current = blocks[-1] if blocks else None
        if (
            current
            and current.speaker == speaker
            and c.start - current.end <= gaps[speaker]
            and c.end - current.start <= caps[speaker]
        ):
            current.text = f"{current.text} {c.text}".strip()
            current.end = max(current.end, c.end)
            continue
        blocks.append(Block(speaker, c.start, c.end, c.text))
    return blocks


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def conversation_metrics(blocks: list[Block]) -> dict:
    """Turn-taking statistics for the transcript frontmatter.

    Measured on blocks rather than cues: a block is one speaker's turn as a reader
    sees it, so words-per-minute is not skewed by a half-second cue holding one word.

    Think time is the pause before someone starts, counted only where the speaker
    actually changed -- a gap between two of your own blocks is you drawing breath,
    not you deciding. Overlaps (interrupting, or a mic catching the far side) give a
    negative gap and are excluded rather than averaged in, which would understate
    how long people really waited.

    Spread is the population standard deviation -- these blocks are the whole
    meeting, not a sample of one -- in the same unit as the average, so
    "142 wpm, sd 31" reads directly.
    """
    stats: dict = {}
    for label, speaker in (("me", "Me"), ("them", "Them")):
        mine = [b for b in blocks if b.speaker == speaker]
        stats[f"{label}_speaking_s"] = round(float(sum(b.end - b.start for b in mine)), 1)
        rates = [
            len(b.text.split()) / ((b.end - b.start) / 60.0)
            for b in mine
            if b.end > b.start and b.text.split()
        ]
        stats[f"{label}_wpm_avg"] = round(statistics.fmean(rates), 1) if rates else None
        stats[f"{label}_wpm_stdev"] = (
            round(statistics.pstdev(rates), 1) if len(rates) > 1 else None
        )

    gaps: dict[str, list[float]] = {"Me": [], "Them": []}
    for previous, current in zip(blocks, blocks[1:]):
        if current.speaker == previous.speaker:
            continue
        gap = current.start - previous.end
        if gap >= 0:
            gaps[current.speaker].append(gap)
    for label, speaker in (("me", "Me"), ("them", "Them")):
        waits = gaps[speaker]
        stats[f"{label}_think_time_avg_s"] = (
            round(statistics.fmean(waits), 2) if waits else None
        )
        stats[f"{label}_think_time_stdev_s"] = (
            round(statistics.pstdev(waits), 2) if len(waits) > 1 else None
        )

    them = stats["them_speaking_s"]
    stats["me_them_speaking_ratio"] = round(stats["me_speaking_s"] / them, 2) if them else None
    return stats


_FRONTMATTER_ORDER = (
    "me_them_speaking_ratio", "me_speaking_s", "them_speaking_s",
    "me_wpm_avg", "me_wpm_stdev", "them_wpm_avg", "them_wpm_stdev",
    "me_think_time_avg_s", "me_think_time_stdev_s",
    "them_think_time_avg_s", "them_think_time_stdev_s",
)


def render_markdown(
    blocks: list[Block], header: str = "", metrics: dict | None = None
) -> str:
    """Render merged blocks as a readable speaker-labelled transcript.

    With metrics, a YAML frontmatter block leads -- readable as-is and parsed by
    note tools. A metric that could not be computed is written as an empty value
    rather than omitted, so the key set stays stable across meetings.
    """
    parts = []
    if metrics is not None:
        lines = ["---"]
        lines += [
            f"{key}:" if metrics.get(key) is None else f"{key}: {metrics[key]}"
            for key in _FRONTMATTER_ORDER
        ]
        lines.append("---")
        parts.append("\n".join(lines))
    if header:
        parts.append(header.rstrip())
    parts += [f"**[{_clock(b.start)}] {b.speaker}:** {b.text}" for b in blocks]
    return "\n\n".join(parts) + "\n"


def _srt_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def parse_srt(text: str) -> list[Cue]:
    """Parse SRT into cues, ignoring indices and blank padding."""
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        timing_at = next(
            (i for i, line in enumerate(lines) if _SRT_TIMING_RE.match(line)), None
        )
        if timing_at is None:
            continue
        groups = _SRT_TIMING_RE.match(lines[timing_at]).groups()
        body = " ".join(lines[timing_at + 1 :]).strip()
        if not body:
            continue
        cues.append(Cue(_srt_seconds(*groups[:4]), _srt_seconds(*groups[4:]), body))
    return cues


@dataclass(frozen=True)
class RunSettings:
    """Tunables for cutting a track into speech runs, from [meeting] in config.ini.

    Defaults are the module constants above, so code and config never disagree.
    """

    min_silence_ms: int = RUN_MIN_SILENCE_MS
    pad_ms: int = RUN_PAD_MS
    vad_threshold: float = RUN_VAD_THRESHOLD
    min_speech_ms: int = RUN_MIN_SPEECH_MS
    max_run_s: float = LANGUAGE_WINDOW_S
    min_avg_logprob: float = MIN_AVG_LOGPROB
    word_timestamps: bool = WORD_TIMESTAMPS

    _KEYS = {
        "min_silence_ms": "meeting_run_min_silence_ms",
        "pad_ms": "meeting_run_pad_ms",
        "vad_threshold": "meeting_run_vad_threshold",
        "min_speech_ms": "meeting_run_min_speech_ms",
        "max_run_s": "meeting_run_max_seconds",
        "min_avg_logprob": "meeting_min_avg_logprob",
        "word_timestamps": "meeting_word_timestamps",
    }

    @classmethod
    def from_config(cls, config: dict | None) -> "RunSettings":
        """Build from a Dictation config dict; absent keys keep their default."""
        config = config or {}
        given = {
            field: config[key] for field, key in cls._KEYS.items() if key in config
        }
        return cls(**given)


def speech_runs(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    settings: "RunSettings | None" = None,
) -> list[tuple[int, int]]:
    """Sample spans of actual speech, as (start, end) pairs.

    Everything between the spans is silence and is never decoded as text. Whisper
    invents training-data text ("Редактор субтитров ...", "Thanks for watching!")
    when asked to transcribe nothing. This silero VAD is a stricter guard than the
    webrtcvad ratio it replaced: webrtcvad scored broadband noise 0.9+, silero
    rejects it outright.
    """
    settings = settings or RunSettings()
    options = VadOptions(
        min_silence_duration_ms=settings.min_silence_ms,
        speech_pad_ms=settings.pad_ms,
        threshold=settings.vad_threshold,
        min_speech_duration_ms=settings.min_speech_ms,
    )
    return [(run["start"], run["end"]) for run in get_speech_timestamps(audio, options)]


def _split_long_runs(runs, limit: int) -> list[tuple[int, int]]:
    """Cut any run longer than `limit` samples, so each piece gets its own language."""
    pieces: list[tuple[int, int]] = []
    for start, end in runs:
        while end - start > limit:
            pieces.append((start, start + limit))
            start += limit
        if end > start:
            pieces.append((start, end))
    return pieces


def transcribe_runs(
    model,
    audio: np.ndarray,
    allowlist: list[str] | None = None,
    sample_rate: int = SAMPLE_RATE,
    settings: "RunSettings | None" = None,
    transcribe_kwargs: dict | None = None,
    progress=None,
    find_runs=None,
) -> list[Cue]:
    """Decode each speech run independently, resolving uncertain language scores
    with up to 30 seconds of nearby same-track audio.

    Confident run-local evidence wins even when neighboring speech uses another
    language. Context stops at long pauses and is used only for language detection:
    decoding, cue offsets, and word timestamps retain the original run boundaries.
    No language state is shared between runs or tracks.

    `allowlist` filters language candidates. `progress(done, total)` runs once per
    speech run; `transcribe_kwargs` forwards the shared glossary to the decoder.
    `find_runs` is injectable so tests can pin the segmentation.
    """
    settings = settings or RunSettings()
    detect = find_runs or speech_runs
    runs = _split_long_runs(
        detect(audio, sample_rate, settings), int(settings.max_run_s * sample_rate)
    )
    cues: list[Cue] = []
    dropped = 0
    for index, (start, end) in enumerate(runs, 1):
        chunk = audio[start:end]
        offset, limit = start / sample_rate, end / sample_rate
        context = _language_context(audio, runs, index - 1, sample_rate)
        lang = _detect_window_language(model, chunk, allowlist, context=context)
        logger.info(
            "run %d/%d at %.1fs (%.1fs) -> %s", index, len(runs), offset, limit - offset, lang
        )
        extra = dict(transcribe_kwargs or {})
        if settings.word_timestamps:
            extra["word_timestamps"] = True
        segments, _ = model.transcribe(
            chunk, vad_filter=False, language=lang, **extra
        )
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            confidence = getattr(seg, "avg_logprob", None)
            if confidence is not None and confidence < settings.min_avg_logprob:
                dropped += 1
                logger.debug("Dropped (avg_logprob %.2f): %s", confidence, text[:60])
                continue
            # Whisper can place a segment past the audio it was given; the run's own
            # end is the truth, since that is where the VAD heard speech stop.
            cue_start = offset + seg.start
            words = [
                Word(offset + w.start, offset + w.end, w.word.strip(), w.probability)
                for w in (getattr(seg, "words", None) or [])
            ] if settings.word_timestamps else []
            cues.append(
                Cue(cue_start, max(cue_start, min(offset + seg.end, limit)), text, words)
            )
        if progress:
            progress(index, len(runs))
    if dropped:
        logger.info("Dropped %d low-confidence segment(s) as likely hallucination", dropped)
    return cues


def _language_context(audio, runs, index: int, sample_rate: int) -> np.ndarray:
    """A centered, bounded audio view within the current cluster of speech runs."""
    start, end = runs[index]
    window = int(LANGUAGE_WINDOW_S * sample_rate)
    if end - start >= window:
        return audio[start:end]
    center = (start + end) // 2
    left, right = max(0, center - window // 2), center + window // 2
    gap = int(LANGUAGE_CONTEXT_MAX_GAP_S * sample_rate)
    first = last = index
    while first > 0:
        _, prev_end = runs[first - 1]
        if prev_end <= left or runs[first][0] - prev_end > gap:
            break
        first -= 1
    while last + 1 < len(runs):
        next_start, _ = runs[last + 1]
        if next_start >= right or next_start - runs[last][1] > gap:
            break
        last += 1
    return audio[max(left, runs[first][0]):min(right, runs[last][1])]


def _language_evidence(model, chunk: np.ndarray, allowlist: list[str] | None):
    """Return the selected language and whether its acoustic evidence is strong.

    Absolute probability prevents a tiny allowlisted winner from seeming certain
    after filtering out other languages. Relative share rejects close contests.
    These conservative heuristics are not calibrated recognition probabilities.
    """
    try:
        detected, probability, all_probs = model.detect_language(
            chunk, vad_filter=False, language_detection_segments=1
        )
    except Exception as e:
        logger.warning("Language detection failed (%s)", e)
        return None, False
    candidates = [
        (lang, p) for lang, p in all_probs if not allowlist or lang in allowlist
    ]
    if allowlist:
        if not candidates:
            return allowlist[0], False
        detected, probability = max(candidates, key=lambda pair: pair[1])
    mass = sum(p for _, p in candidates)
    confident = (
        probability >= LANGUAGE_MIN_PROBABILITY
        and mass > 0
        and probability / mass >= LANGUAGE_MIN_SHARE
    )
    return detected, confident


def _detect_window_language(
    model,
    chunk: np.ndarray,
    allowlist: list[str] | None,
    context: np.ndarray | None = None,
):
    language, confident = _language_evidence(model, chunk, allowlist)
    if confident or context is None or len(context) <= len(chunk):
        return language
    contextual_language, contextual_confident = _language_evidence(
        model, context, allowlist
    )
    if contextual_confident:
        logger.info(
            "Language context resolved uncertain run: %s -> %s",
            language, contextual_language,
        )
        return contextual_language
    # Weak context must not force a switch. None delegates to decoder detection
    # when the run-local detector failed too.
    return language


def _write_srt(cues: list[Cue], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for i, c in enumerate(cues, 1):
            out.write(
                f"{i}\n{_srt_timestamp(c.start)} --> {_srt_timestamp(c.end)}\n{c.text}\n\n"
            )


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _write_words_json(cues: list[Cue], path: Path) -> None:
    """Per-word timings beside the SRT, for measuring pauses and response latency.

    Kept out of the SRT and the Markdown on purpose: both are for reading, and a cue
    per word would make them useless for that.
    """
    payload = [
        {
            "start": round(c.start, 3),
            "end": round(c.end, 3),
            "text": c.text,
            "words": [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "text": w.text,
                    "probability": round(w.probability, 3),
                }
                for w in c.words
            ],
        }
        for c in cues
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_log_handler(session_dir: Path) -> logging.Handler:
    """Per-meeting log file, so a transcription run can be inspected on its own.

    The service journal interleaves every meeting with dictation activity; this
    keeps one run's detail in one place next to its audio.
    """
    handler = logging.FileHandler(session_dir / "transcribe.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    handler.setLevel(logging.DEBUG)
    return handler


def _publish_progress(dictation, track: str | None, done: int, total: int) -> None:
    """Expose transcription progress for the tray; cleared with track=None."""
    try:
        dictation.meeting_progress = (
            None if track is None else f"{track} {100 * done // total}%"
        )
    except Exception as e:
        logger.debug("Could not publish progress: %s", e)


def finish_session(dictation, session_dir: Path, transcript_dir: Path) -> Path:
    """Transcribe a finished session and write the merged Markdown transcript.

    Language is detected per speech run, with surrounding same-track audio used
    to resolve uncertain scores. Each run is decoded at its original offset.

    The transcript is written to transcript_dir (kept) and beside the audio in the
    session directory. The WAVs are deleted only after the transcript is safely on
    disk: if transcription fails, that audio is the only copy of the meeting.
    """
    session_dir, transcript_dir = Path(session_dir), Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    handler = _session_log_handler(session_dir)
    logger.addHandler(handler)
    # The module logger inherits root's level (WARNING by default), which would drop
    # our INFO lines before they ever reach the file handler. Lower it for the run.
    previous_level = logger.level
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    started = time.monotonic()
    try:
        model = dictation.meeting_model()
        allowlist = (dictation.config or {}).get("language_allowlist")
        settings = RunSettings.from_config(dictation.config)
        # One glossary for every mode, built by Dictation from [behavior] custom_terms.
        glossary = getattr(dictation, "custom_terms_kwargs", None) or {}
        # Same for reject_phrases: the noise it filters is Whisper's, not dictation's.
        reject = getattr(dictation, "should_reject_text", None)

        tracks: dict[str, list[Cue]] = {}
        for name in ("mic", "them"):
            wav = session_dir / f"{name}.wav"
            if not wav.exists():
                logger.warning("%s.wav missing; skipping that track", name)
                tracks[name] = []
                continue
            audio = _read_wav(wav)
            logger.info("Transcribing %s.wav (%.1fs)", name, len(audio) / SAMPLE_RATE)
            t0 = time.monotonic()

            # A meeting has one run per utterance -- hundreds of them -- so the tray
            # updates on every run while the log only marks each 10% crossed.
            def report(done, total, _name=name, _t0=t0, _last=[-1]):
                decile = 10 * done // total
                if decile > _last[0]:
                    _last[0] = decile
                    elapsed = time.monotonic() - _t0
                    eta = (elapsed / done) * (total - done)
                    logger.info(
                        "%s.wav run %d/%d (%.0f%%), %.0fs elapsed, ~%.0fs left",
                        _name, done, total, 100 * done / total, elapsed, eta,
                    )
                _publish_progress(dictation, _name, done, total)

            tracks[name] = transcribe_runs(
                model, audio, allowlist=allowlist, settings=settings,
                transcribe_kwargs=glossary, progress=report,
            )
            if reject:
                kept = [c for c in tracks[name] if not reject(c.text)]
                if len(kept) != len(tracks[name]):
                    logger.info(
                        "Dropped %d cue(s) matching reject_phrases",
                        len(tracks[name]) - len(kept),
                    )
                tracks[name] = kept
            logger.info(
                "%s.wav -> %d cues in %.1fs", name, len(tracks[name]), time.monotonic() - t0
            )
            _write_srt(tracks[name], session_dir / f"{name}.srt")
            if settings.word_timestamps:
                _write_words_json(tracks[name], session_dir / f"{name}.words.json")

        config = dictation.config or {}
        blocks = merge_tracks(
            tracks["mic"], tracks["them"],
            me_max_block_s=config.get("meeting_me_max_block_seconds", ME_MAX_BLOCK_S),
            them_max_block_s=config.get("meeting_them_max_block_seconds", THEM_MAX_BLOCK_S),
        )
        text = render_markdown(
            blocks, f"# Meeting {session_dir.name}", conversation_metrics(blocks)
        )
        out_path = transcript_dir / f"{session_dir.name}.md"
        out_path.write_text(text, encoding="utf-8")
        # A copy beside the audio and the SRTs, so one meeting's artifacts stay together.
        (session_dir / f"{session_dir.name}.md").write_text(text, encoding="utf-8")

        if (dictation.config or {}).get("meeting_keep_audio"):
            logger.info("Keeping audio (meeting_keep_audio is on)")
        else:
            freed = purge_wavs(session_dir)
            logger.info("Deleted audio, freed %.0f MB", freed / 1e6)
        logger.info(
            "Transcript written to %s in %.1fs", out_path, time.monotonic() - started
        )
        return out_path
    finally:
        _publish_progress(dictation, None, 0, 0)
        logger.setLevel(previous_level)
        logger.removeHandler(handler)
        handler.close()


class MeetingRecorder:
    """Owns one meeting: two pw-record subprocesses and the session directory.

    Recording is deliberately cheap -- pw-record writes WAV straight to disk and we
    load no model -- so nothing competes with the video call. Transcription happens
    only after stop().
    """

    def __init__(
        self,
        transcript_dir: Path,
        tmp_root: Path = TMP_ROOT,
        max_duration_s: float = MAX_DURATION_S,
        silence_stop_s: float = SILENCE_STOP_S,
        spawn=subprocess.Popen,
        clock=time.monotonic,
    ):
        self.transcript_dir = Path(transcript_dir)
        self.tmp_root = Path(tmp_root)
        self.max_duration_s = max_duration_s
        self.silence_stop_s = silence_stop_s
        self._spawn = spawn
        self._clock = clock
        self.procs: list = []
        self._track_paths: list[Path] = []
        self.session_dir: Path | None = None
        self.started_at: float | None = None
        self.started_wall: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.procs)

    @property
    def mic_wav(self) -> Path:
        return self.session_dir / "mic.wav"

    @property
    def them_wav(self) -> Path:
        return self.session_dir / "them.wav"

    def start(self) -> None:
        if self.active:
            raise RuntimeError("A meeting is already being recorded")
        self.started_wall = time.strftime("%Y-%m-%d_%H%M%S")
        self.session_dir = self.tmp_root / "meetings" / self.started_wall
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = self._clock()
        try:
            self.procs = [
                self._spawn(pw_record_cmd(self.mic_wav, capture_sink=False)),
                self._spawn(pw_record_cmd(self.them_wav, capture_sink=True)),
            ]
            self._track_paths = [self.mic_wav, self.them_wav]
        except Exception:
            self.stop_recording()
            raise

    def elapsed_s(self) -> float:
        return 0.0 if self.started_at is None else self._clock() - self.started_at

    def auto_stop_reason(self) -> str | None:
        """Why this meeting should stop on its own, or None to keep recording."""
        if not self.active:
            return None
        if self.max_duration_s and self.elapsed_s() >= self.max_duration_s:
            return "max_duration"
        if self.silence_stop_s and self.elapsed_s() >= self.silence_stop_s:
            if self._both_tracks_silent():
                return "silence"
        return None

    def track_failure(self) -> str | None:
        """Message naming a track whose recorder exited early, else None.

        A pw-record that dies mid-meeting is nearly always the disk filling up, and
        it is silent: the file just stops growing. Surfacing it lets the tray go to
        the error state instead of us "recording" a meeting into nothing.
        """
        for path, proc in zip(self._track_paths, self.procs):
            if proc.poll() is not None:
                free = free_bytes(path.parent)
                return (
                    f"Recorder for {path.name} stopped unexpectedly "
                    f"({free / 1e6:.0f} MB free on disk)"
                )
        return None

    def _both_tracks_silent(self) -> bool:
        """True only when NEITHER track has speech in the trailing window.

        Checking the monitor alone would stop the recording while you are the one
        talking, so both must be quiet.
        """
        for track in (self.mic_wav, self.them_wav):
            try:
                tail = wav_tail(track, self.silence_stop_s)
            except (OSError, wave.Error) as e:
                logger.debug("Silence check skipped for %s: %s", track, e)
                return False
            if len(tail) < SAMPLE_RATE * self.silence_stop_s:
                return False  # not enough audio yet to judge
            if speech_ratio(tail) > 0.0:
                return False
        return True

    def stop_recording(self) -> None:
        """Stop both tracks with SIGTERM so pw-record patches the WAV headers."""
        for proc in self.procs:
            try:
                proc.terminate()
            except Exception as e:
                logger.warning("Could not terminate recorder: %s", e)
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except Exception as e:
                logger.warning("Recorder did not exit cleanly: %s", e)
        self.procs = []
