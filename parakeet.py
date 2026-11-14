"""NVIDIA Parakeet TDT (via onnx-asr), shaped like faster-whisper's WhisperModel.

Only the part of the WhisperModel API this app uses is provided: ``transcribe()``
returning ``(segments, info)``, where segments carry text, start/end, avg_logprob
and optional words.

Parakeet takes no language: it hears the language and writes pieces from one shared
vocabulary of all its alphabets. Its typical mistake is the wrong alphabet (accented
English written in Cyrillic), so a "language" here selects an alphabet mask. With
``language=None`` the model detects the alphabet itself: it encodes the audio once,
decodes it once per alphabet of ``language_allowlist`` and keeps the decode with the
best confidence and coverage. There is deliberately no ``detect_language()``;
callers check ``hasattr(model, "detect_language")`` and pass ``language=None``.

Decoding is onnx-asr's greedy TDT loop, re-implemented here so that each call can
change token scores before the pick, without touching shared model state:
- a script mask keeps output in the layout language's alphabet (Parakeet otherwise
  writes accented English in Cyrillic and vice versa);
- a glossary bonus nudges the next piece of a custom term once the term has begun.
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
# Loudness each piece is brought to before encoding (RMS of float audio). Parakeet
# returns nothing for quiet or distant speech that Whisper still hears: in a phone
# video, "здоровое и умное" at RMS 0.018 came out empty and appeared once raised.
# 0.05 recovered the most words there; 0.1 started splitting sentences.
TARGET_RMS = 0.05
# Upper limit on that gain, so a piece of near silence is not raised into noise.
MAX_GAIN = 10.0
# Glossary bonus added to a token's score (log space). 2.0 fixed "agent session" ->
# "agentic session" in the spike without inserting terms where none were spoken.
GLOSSARY_BONUS = 2.0
# Longest audio encoded in one pass. The encoder's attention grows with the square
# of the length (a 1-hour file would need tens of GB), so longer audio is split at
# pauses into speech pieces of at most this length, as Whisper splits into 30 s windows.
MAX_PASS_S = 30.0
# Alphabet detection score = mean token log probability + COVERAGE_WEIGHT *
# log(share of the audio covered by decoded tokens). The coverage term stops a short
# confident fragment from beating a full transcript in the other alphabet: on a 21 s
# Russian run, the Latin decode "5-7-10. Technical challenge." averaged -0.12 and
# the full Russian text -0.13. Not yet validated on more mixed recordings.
COVERAGE_WEIGHT = 1.0

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

# Hesitation sounds Parakeet transcribes and Whisper drops. Only whole words are
# removed, compared after lowercasing and stripping punctuation. A word is removed
# only if it is a filler in the language being decoded: "er" and "um" are fillers in
# English but real words in German ("er" = he, "um" = at) and Portuguese ("um" = a).
# FILLERS_ANY_LANGUAGE holds sounds that are not a word in any Parakeet language;
# it is all that applies when the language is unknown.
FILLERS_ANY_LANGUAGE = frozenset({
    "uh", "uhh", "umm", "uhm", "hmm", "hm", "mm", "mmm", "mhm", "mm-hmm", "uh-huh",
})
_CYRILLIC_FILLERS = frozenset({"э", "ээ", "эээ", "эм", "мм", "ммм", "хм"})
FILLERS_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "en": frozenset({"um", "er", "erm"}),
    "ru": _CYRILLIC_FILLERS,
    "uk": _CYRILLIC_FILLERS,
    "bg": _CYRILLIC_FILLERS,
}


def fillers_for(language: Optional[str]) -> frozenset[str]:
    return FILLERS_ANY_LANGUAGE | FILLERS_BY_LANGUAGE.get(language or "", frozenset())

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
    """Token sequences of custom terms, used to boost the rest of a term once begun.

    Terms are not boosted from their first piece: the vocabulary splits them into
    short, common pieces (Claude -> " C", "la", "ude"), so boosting those would pull
    terms into ordinary words. A multi-word term ("ML repository") is one sequence.
    """

    def __init__(self, terms: list[str], piece_to_id: dict[str, int]):
        self.sequences: list[list[int]] = []
        for term in terms:
            for variant in {term, term.lower(), term[:1].upper() + term[1:]}:
                seq = _encode_text(variant, piece_to_id)
                if len(seq) > 1 and seq not in self.sequences:
                    self.sequences.append(seq)

    def boosted(self, previous: list[int]) -> set[int]:
        """Next pieces of every term whose beginning ends the text so far."""
        out: set[int] = set()
        for seq in self.sequences:
            for k in range(min(len(seq) - 1, len(previous)), 0, -1):
                if previous[-k:] == seq[:k]:
                    out.add(seq[k])
                    break
        return out


def _encode_text(text: str, piece_to_id: dict[str, int]) -> list[int]:
    """Greedy longest-match split into vocabulary pieces (a leading space starts a word).

    Returns [] if any character has no piece: a sequence with letters skipped would
    start mid-term and boost unrelated words.
    """
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
                return []
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
                 script_penalty: Optional[dict[str, Optional[float]]] = None,
                 glossary: Optional[list[str]] = None,
                 language_allowlist: Optional[list[str]] = None):
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
        self._attach(asr, model_id, script_penalty, glossary, language_allowlist)

    def _attach(self, asr, model_id: str,
                script_penalty: Optional[dict[str, Optional[float]]] = None,
                glossary: Optional[list[str]] = None,
                language_allowlist: Optional[list[str]] = None) -> None:
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
        self._cache_lock = threading.Lock()
        # From [behavior] custom_terms. Whisper's joined `hotwords` string can't be used:
        # it loses the boundaries of multi-word terms.
        self._glossary = _Glossary(glossary, self._piece_to_id) if glossary else None
        self._candidates = self._detection_candidates(language_allowlist or [])

    def _detection_candidates(self, allowlist: list[str]) -> list[tuple[str, Optional[str]]]:
        """(language to decode with, language to report) per distinct alphabet mask.

        Languages sharing an alphabet and penalty decode identically, so they are
        decoded once and reported as unknown (None): "en, de" can't be told apart.
        """
        groups: dict[tuple, list[str]] = {}
        for lang in allowlist:
            penalty = self._script_penalty.get(lang, self._script_penalty.get("*"))
            groups.setdefault((SCRIPT_OF_LANGUAGE.get(lang), penalty), []).append(lang)
        return [(langs[0], langs[0] if len(langs) == 1 else None) for langs in groups.values()]

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

    # --- decoding -----------------------------------------------------------------

    def _greedy(self, encodings: np.ndarray, length: int, bias: np.ndarray,
                glossary: Optional[_Glossary]) -> tuple[list[int], list[int], list[int], list[float]]:
        """onnx-asr's TDT greedy loop, with score edits before each pick.

        Returns tokens, their first frame, their duration in frames (from the TDT
        duration head; at least 1) and log probabilities.
        """
        asr = self._asr
        state = asr._create_state()
        tokens: list[int] = []
        frames: list[int] = []
        durations: list[int] = []
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
                durations.append(max(step, 1))
                logprobs.append(float(_log_softmax(logits)[token]))
                emitted += 1
            if step > 0:
                t += step
                emitted = 0
            elif token == self._blank or emitted == asr._max_tokens_per_step:
                t += 1
                emitted = 0
        return tokens, frames, durations, logprobs

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        vad_filter: bool = False,
        vad_parameters=None,
        word_timestamps: bool = False,
        **_whisper_only,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Transcribe 16 kHz mono audio of any length. Whisper-only options (prompts,
        hotwords, temperature, condition_on_previous_text, ...) are accepted and
        ignored; the glossary comes from the constructor."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / SAMPLE_RATE
        info = TranscriptionInfo(language, 1.0, duration)
        if vad_filter and not self._has_speech(audio, vad_parameters):
            # Parakeet types "Yeah." / "Thank you." on breaths and noise.
            return [], info
        if duration <= MAX_PASS_S:
            pieces = [(0, len(audio))]
        else:
            pieces = self._speech_pieces(audio)
        segments: list[Segment] = []
        detected: list[Optional[str]] = []
        for start, end in pieces:
            offset = start / SAMPLE_RATE
            piece_words, piece_language = self._decode_piece(audio[start:end], language)
            detected.append(piece_language)
            # Fillers depend on the language: "er" is English hesitation, German "he".
            words = [
                (Word(w.start + offset, w.end + offset, w.word, w.probability), lps)
                for w, lps in drop_fillers(piece_words, piece_language)
            ]
            # Segments end at piece boundaries too: a pause between pieces ends a
            # sentence even without punctuation, and the next piece may switch language.
            segments.extend(_segments(words))
        if language is None and detected:
            info = TranscriptionInfo(max(detected, key=detected.count), 1.0, duration)
        return segments, info

    def _decode_piece(self, audio: np.ndarray, language: Optional[str]):
        """Encode at most MAX_PASS_S of audio once; decode it in `language`, or in
        each candidate alphabet when `language` is None. Returns (words, language)."""
        pad = np.zeros(int(EDGE_PAD_S * SAMPLE_RATE), np.float32)
        padded = np.concatenate([pad, normalize_loudness(audio), pad])[None, :]
        lengths = np.array([padded.shape[1]], np.int64)
        encoder_out, encoder_lens = self._asr._encode(*self._asr._preprocessor(padded, lengths))
        encodings, length = encoder_out[0], int(encoder_lens[0])
        duration = len(audio) / SAMPLE_RATE
        if language is not None or len(self._candidates) < 2:
            decode_as, report = (
                (language, language) if language is not None
                else self._candidates[0] if self._candidates else (None, None)
            )
            decoded = self._greedy(encodings, length, self._bias(decode_as), self._glossary)
            return self._words(*decoded, duration=duration), report
        best = None
        scores = {}
        for decode_as, report in self._candidates:
            decoded = self._greedy(encodings, length, self._bias(decode_as), self._glossary)
            score = detection_score(decoded[2], decoded[3], duration / self._frame_s)
            scores[decode_as] = round(score, 2)
            if best is None or score > best[0]:
                best = (score, decoded, report)
        logger.debug("Alphabet detection scores %s -> %s", scores, best[2])
        return self._words(*best[1], duration=duration), best[2]

    @staticmethod
    def _speech_pieces(audio: np.ndarray) -> list[tuple[int, int]]:
        """Split long audio at pauses into speech pieces no longer than MAX_PASS_S."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(
            min_silence_duration_ms=500, speech_pad_ms=200,
            # Silero cuts at the longest pause it can find before this length.
            max_speech_duration_s=MAX_PASS_S - 1,
        )
        return [(run["start"], run["end"]) for run in get_speech_timestamps(audio, options)]

    @staticmethod
    def _has_speech(audio: np.ndarray, vad_parameters) -> bool:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        if isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        return bool(get_speech_timestamps(audio, vad_parameters or VadOptions()))

    def _words(self, tokens, frames, durations, logprobs, duration) -> list[tuple[Word, list[float]]]:
        """Group pieces into words (a leading space starts one), on the unpadded timeline.

        A word ends where its last piece ends, by that piece's predicted duration, so
        pauses between words stay visible in word and cue times.
        """
        groups: list[list] = []  # [text, start, end, logprobs]
        for token, frame, length, lp in zip(tokens, frames, durations, logprobs):
            piece = self._pieces[token]
            start = frame * self._frame_s - EDGE_PAD_S
            end = (frame + length) * self._frame_s - EDGE_PAD_S
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


def _scaled(audio: np.ndarray) -> np.ndarray:
    """Scale audio to TARGET_RMS, by at most MAX_GAIN either way, clipped to [-1, 1]."""
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms <= 0.0:
        return audio
    gain = min(max(TARGET_RMS / rms, 1.0 / MAX_GAIN), MAX_GAIN)
    return np.clip(audio * gain, -1.0, 1.0)


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Bring each speech run in `audio` to TARGET_RMS on its own.

    One gain for the whole piece is not enough: a phone video mixes a close, loud
    speaker with quiet ones, and the loud one sets the gain. Pauses between runs keep
    their level. Without detected runs, the whole piece is scaled."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = np.asarray(audio, dtype=np.float32)
    runs = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))
    if not runs:
        return _scaled(audio).astype(np.float32)
    out = audio.copy()
    for run in runs:
        out[run["start"]:run["end"]] = _scaled(audio[run["start"]:run["end"]])
    return out


def detection_score(durations: list[int], logprobs: list[float], frames: float) -> float:
    """Confidence of a decode, penalised by how little of the audio it covers."""
    if not logprobs:
        return float("-inf")
    coverage = min(1.0, sum(durations) / max(frames, 1.0))
    return float(np.mean(logprobs)) + COVERAGE_WEIGHT * float(np.log(coverage))


def drop_fillers(words: list[tuple[Word, list[float]]],
                 language: Optional[str] = None) -> list[tuple[Word, list[float]]]:
    """Remove hesitation words of `language`, keeping sentence ends and capitals."""
    fillers = fillers_for(language)
    kept: list[tuple[Word, list[float]]] = []
    capitalize_next = False
    for word, lps in words:
        if _bare(word.word) in fillers:
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
