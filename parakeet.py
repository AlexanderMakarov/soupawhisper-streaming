"""NVIDIA Parakeet TDT (via onnx-asr), shaped like faster-whisper's WhisperModel.

Only the part of the WhisperModel API this app uses is provided: ``transcribe()``
returning ``(segments, info)``, where segments carry text, start/end, avg_logprob
and optional words. There is deliberately no ``detect_language()``: Parakeet has no
language detector, so callers check ``hasattr(model, "detect_language")`` and take
the language from the keyboard layout instead.

Decoding is onnx-asr's greedy TDT loop, re-implemented here so that each call can
change token scores before the pick, without touching shared model state:
- a script mask keeps output in the layout language's alphabet (Parakeet otherwise
  writes accented English in Cyrillic and vice versa);
- a glossary bonus nudges tokens that start or continue a custom term.
Both were measured in spikes/2026-11-13-parakeet (REPORT.md).

The greedy loop uses onnx-asr private methods (``_encode``, ``_decode``,
``_create_state``), so pyproject pins onnx-asr to a minor version.
"""
from __future__ import annotations

import logging
import os
import re
import string
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"
QUANTIZATION = "int8"
SAMPLE_RATE = 16000
# Silence added at both ends before encoding. Without it Parakeet returned nothing
# for some short utterances that start or end right at the chunk edge.
EDGE_PAD_S = 0.3
# Glossary bonus added to a token's score (log space). 2.0 fixed "agent session" ->
# "agentic session" in the spike without inserting terms where none were spoken.
GLOSSARY_BONUS = 2.0
# A glossary term may only be *started* by a first piece at least this long
# (" Kube", not " m"), so short common pieces don't pull terms into the text.
GLOSSARY_MIN_START_PIECE = 4

# Parakeet TDT 0.6B v3 languages and the alphabet each is written in.
SCRIPT_OF_LANGUAGE = {
    **dict.fromkeys(
        ("en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sk", "sl", "hr", "hu",
         "ro", "da", "sv", "fi", "et", "lv", "lt", "mt"),
        "LATIN",
    ),
    **dict.fromkeys(("ru", "uk", "bg"), "CYRILLIC"),
    "el": "GREEK",
}

# Hesitation sounds Parakeet transcribes and Whisper drops. Compared after lowercasing
# and stripping punctuation; only whole words are removed.
FILLER_WORDS = frozenset({
    "uh", "uhh", "um", "umm", "uhm", "er", "erm", "ah", "eh", "hmm", "hm", "mm", "mhm",
    "mm-hmm", "uh-huh",
    "э", "ээ", "эм", "мм", "хм",
})

_PUNCT = string.punctuation + "…«»“”„"


# How strongly the layout language keeps output in its own alphabet, per language.
# None forbids other alphabets; a number is subtracted from their scores (log space),
# so they win only when the audio clearly calls for them; 0 disables the mask.
# "*" covers languages not listed. Measured in the spike (REPORT.md, "Forced language"):
# - English layout: only a ban stopped accented English coming out in Cyrillic
#   ("Файл ищу ин гитхаб фор" -> "File issue in GitHub for"); penalty 3 did not.
# - Russian layout: a ban transliterated English terms in Russian speech
#   ("Инглиш вордс"); penalty 3 kept them in Latin.
DEFAULT_SCRIPT_PENALTY: dict[str, Optional[float]] = {"*": None, "ru": 3.0, "uk": 3.0, "bg": 3.0}


def parse_script_penalty(raw: str) -> dict[str, Optional[float]]:
    """Parse "<lang>:<ban|number>, ..." ("*" = other languages). Raises ValueError."""
    out: dict[str, Optional[float]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        lang, sep, value = (part.strip().lower() for part in entry.partition(":"))
        if not sep or not lang or not value:
            raise ValueError(f"entry {entry!r} must be <language>:<ban|number>")
        if value == "ban":
            out[lang] = None
            continue
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"entry {entry!r}: {value!r} is not 'ban' or a number") from None
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"entry {entry!r}: penalty must be 'ban' or a number >= 0")
        out[lang] = number
    return out


def _token_scripts(piece: str) -> frozenset[str]:
    return frozenset(
        unicodedata.name(ch, "?").split()[0] for ch in piece if ch.isalpha()
    )


def _log_softmax(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    peak = finite.max() if finite.size else 0.0
    shifted = x - peak
    return shifted - np.log(np.exp(shifted).sum())


def _bare(word: str) -> str:
    return word.strip(_PUNCT).lower()


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    words: list[Word] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionInfo:
    language: Optional[str]
    language_probability: float
    duration: float


class _Glossary:
    """Token sequences for custom terms, used to boost them during decoding."""

    def __init__(self, terms: list[str], piece_to_id: dict[str, int], pieces: dict[int, str]):
        self.sequences: list[list[int]] = []
        for term in terms:
            for variant in {term, term.lower(), term[:1].upper() + term[1:]}:
                seq = _encode_text(variant, piece_to_id)
                if seq:
                    self.sequences.append(seq)
        self.starters = frozenset(
            seq[0] for seq in self.sequences
            if len(pieces[seq[0]].strip()) >= GLOSSARY_MIN_START_PIECE
        )

    def boosted(self, previous: list[int]) -> set[int]:
        """Tokens that continue a term already under way, plus distinctive term starts."""
        out = set(self.starters)
        for seq in self.sequences:
            for k in range(min(len(seq) - 1, len(previous)), 0, -1):
                if previous[-k:] == seq[:k]:
                    out.add(seq[k])
                    break
        return out


def _encode_text(text: str, piece_to_id: dict[str, int]) -> list[int]:
    """Greedy longest-match split into vocabulary pieces (a leading space starts a word)."""
    longest = max(map(len, piece_to_id))
    ids: list[int] = []
    for word in text.split():
        rest = " " + word
        while rest:
            for n in range(min(len(rest), longest), 0, -1):
                if rest[:n] in piece_to_id:
                    ids.append(piece_to_id[rest[:n]])
                    rest = rest[n:]
                    break
            else:
                rest = rest[1:]  # character the vocabulary lacks
    return ids


def resolve_model_name(model_id: str) -> tuple[str, Optional[str]]:
    """Map a config model value to onnx-asr's (model, path) arguments."""
    if os.path.isdir(model_id):
        return "nemo-conformer-tdt", model_id
    return model_id, None


def download(model_id: str = DEFAULT_MODEL) -> None:
    """Fetch the model files into the Hugging Face cache without loading them."""
    from onnx_asr.loader import create_asr_resolver

    name, path = resolve_model_name(model_id)
    if path is None:
        create_asr_resolver(name).resolve_model(quantization=QUANTIZATION)


class ParakeetModel:
    """A Parakeet TDT model with the WhisperModel calls this app makes."""

    def __init__(self, model_id: str = DEFAULT_MODEL, cpu_threads: int = 0,
                 script_penalty: Optional[dict[str, Optional[float]]] = None):
        import onnx_asr
        import onnxruntime as ort
        from onnx_asr.models.nemo import NemoConformerTdt

        options = ort.SessionOptions()
        if cpu_threads:
            # Measured on 4 cores / 8 threads: 4 threads beat both 2 and 8.
            options.intra_op_num_threads = cpu_threads
            options.inter_op_num_threads = 1
        name, path = resolve_model_name(model_id)
        adapter = onnx_asr.load_model(name, path, quantization=QUANTIZATION, sess_options=options)
        asr = adapter.asr
        if not isinstance(asr, NemoConformerTdt):
            raise ValueError(f"{model_id!r} is not a Parakeet TDT model")
        self._attach(asr, model_id, script_penalty)

    def _attach(self, asr, model_id: str,
                script_penalty: Optional[dict[str, Optional[float]]] = None) -> None:
        """Index the vocabulary of a loaded onnx-asr TDT model (tests pass a fake)."""
        self.model_id = model_id
        self._script_penalty = dict(DEFAULT_SCRIPT_PENALTY if script_penalty is None else script_penalty)
        self._asr = asr
        self._blank = asr._blank_idx
        self._frame_s = asr.window_step * asr._subsampling_factor
        size = asr._vocab_size
        self._pieces: dict[int, str] = {i: asr._vocab[i] for i in range(size)}
        tags = np.array([p.startswith("<") and p.endswith(">") for p in self._pieces.values()])
        # Tags such as <|en|> never belong in text; <blk> must stay selectable.
        self._tags = tags & (np.arange(size) != self._blank)
        self._scripts = [frozenset() if tags[i] else _token_scripts(p) for i, p in self._pieces.items()]
        self._piece_to_id = {p: i for i, p in self._pieces.items() if not tags[i]}
        self._bias_cache: dict[Optional[str], np.ndarray] = {}
        self._glossary_cache: dict[str, _Glossary] = {}
        self._cache_lock = threading.Lock()

    # --- per-call decoding knobs -------------------------------------------------

    def _bias(self, language: Optional[str]) -> np.ndarray:
        """Score offsets for the language: tags banned, other alphabets penalised."""
        with self._cache_lock:
            if language not in self._bias_cache:
                bias = np.zeros(len(self._pieces), np.float32)
                script = SCRIPT_OF_LANGUAGE.get(language or "")
                penalties = self._script_penalty
                penalty = penalties.get(language or "", penalties.get("*"))
                if script and penalty != 0:
                    foreign = np.array([bool(s) and s != {script} for s in self._scripts])
                    bias[foreign] = -np.inf if penalty is None else -penalty
                bias[self._tags] = -np.inf
                self._bias_cache[language] = bias
            return self._bias_cache[language]

    def _glossary(self, hotwords: Optional[str]) -> Optional[_Glossary]:
        if not hotwords:
            return None
        with self._cache_lock:
            if hotwords not in self._glossary_cache:
                self._glossary_cache[hotwords] = _Glossary(
                    hotwords.split(), self._piece_to_id, self._pieces
                )
            return self._glossary_cache[hotwords]

    # --- decoding -----------------------------------------------------------------

    def _greedy(self, encodings: np.ndarray, length: int, bias: np.ndarray,
                glossary: Optional[_Glossary]) -> tuple[list[int], list[int], list[float]]:
        """onnx-asr's TDT greedy loop, with score edits before each pick."""
        asr = self._asr
        state = asr._create_state()
        tokens: list[int] = []
        frames: list[int] = []
        logprobs: list[float] = []
        t = emitted = 0
        while t < length:
            logits, step, next_state = asr._decode(tokens, state, encodings[t])
            logits = logits + bias
            if glossary is not None:
                boost = list(glossary.boosted(tokens))
                if boost:
                    logits[boost] += GLOSSARY_BONUS
            token = int(logits.argmax())
            if token != self._blank:
                state = next_state
                tokens.append(token)
                frames.append(t)
                logprobs.append(float(_log_softmax(logits)[token]))
                emitted += 1
            if step > 0:
                t += step
                emitted = 0
            elif token == self._blank or emitted == asr._max_tokens_per_step:
                t += 1
                emitted = 0
        return tokens, frames, logprobs

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        vad_filter: bool = False,
        vad_parameters=None,
        hotwords: Optional[str] = None,
        word_timestamps: bool = False,
        **_whisper_only,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Transcribe 16 kHz mono audio. Whisper-only options (prompts, temperature,
        condition_on_previous_text, ...) are accepted and ignored."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / SAMPLE_RATE
        info = TranscriptionInfo(language, 1.0, duration)
        if vad_filter and not self._has_speech(audio, vad_parameters):
            # Parakeet types "Yeah." / "Thank you." on breaths and noise.
            return [], info
        pad = np.zeros(int(EDGE_PAD_S * SAMPLE_RATE), np.float32)
        padded = np.concatenate([pad, audio, pad])[None, :]
        lengths = np.array([padded.shape[1]], np.int64)
        encoder_out, encoder_lens = self._asr._encode(*self._asr._preprocessor(padded, lengths))
        tokens, frames, logprobs = self._greedy(
            encoder_out[0], int(encoder_lens[0]), self._bias(language), self._glossary(hotwords)
        )
        words = self._words(tokens, frames, logprobs, duration)
        return _segments(words), info

    @staticmethod
    def _has_speech(audio: np.ndarray, vad_parameters) -> bool:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        if isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        return bool(get_speech_timestamps(audio, vad_parameters or VadOptions()))

    def _words(self, tokens, frames, logprobs, duration) -> list[tuple[Word, list[float]]]:
        """Group pieces into words (a leading space starts one), on the unpadded timeline."""
        groups: list[list] = []  # [text, start, end, logprobs]
        for i, (token, frame, lp) in enumerate(zip(tokens, frames, logprobs)):
            piece = self._pieces[token]
            start = frame * self._frame_s - EDGE_PAD_S
            end = (frames[i + 1] if i + 1 < len(frames) else frame + 1) * self._frame_s - EDGE_PAD_S
            if piece.startswith(" ") or not groups:
                groups.append([piece.strip(), start, end, [lp]])
            else:
                groups[-1][0] += piece
                groups[-1][2] = end
                groups[-1][3].append(lp)
        clamp = lambda s: min(max(s, 0.0), duration)  # noqa: E731
        return [
            (Word(clamp(s), clamp(e), text, float(np.exp(np.mean(lps)))), lps)
            for text, s, e, lps in groups if text
        ]


def drop_fillers(words: list[tuple[Word, list[float]]]) -> list[tuple[Word, list[float]]]:
    """Remove hesitation words, keeping sentence ends and sentence-initial capitals."""
    kept: list[tuple[Word, list[float]]] = []
    capitalize_next = False
    for word, lps in words:
        if _bare(word.word) in FILLER_WORDS:
            ending = re.search(r"[.!?]+$", word.word)
            if ending and kept and not re.search(r"[.!?]$", kept[-1][0].word):
                prev, prev_lps = kept[-1]
                kept[-1] = (Word(prev.start, prev.end, prev.word + ending.group(0), prev.probability), prev_lps)
            capitalize_next = capitalize_next or word.word[:1].isupper() or bool(ending)
            continue
        if capitalize_next and word.word[:1].islower():
            word = Word(word.start, word.end, word.word[:1].upper() + word.word[1:], word.probability)
        capitalize_next = False
        kept.append((word, lps))
    return kept


def _segments(words: list[tuple[Word, list[float]]]) -> list[Segment]:
    """One segment per sentence, ended by . ! or ?"""
    words = drop_fillers(words)
    segments: list[Segment] = []
    current: list[tuple[Word, list[float]]] = []
    for item in words:
        current.append(item)
        if re.search(r"[.!?]$", item[0].word):
            segments.append(_segment(current))
            current = []
    if current:
        segments.append(_segment(current))
    return segments


def _segment(items: list[tuple[Word, list[float]]]) -> Segment:
    words = [w for w, _ in items]
    logprobs = [lp for _, lps in items for lp in lps]
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.word for w in words),
        avg_logprob=float(np.mean(logprobs)),
        words=words,
    )
