"""Unit tests for parakeet.py with a tiny fake onnx-asr model (no download, no ONNX)."""
import threading

import numpy as np
import pytest
from onnx_asr.asr import _AsrWithTransducerDecoding

import parakeet

BLANK = "<blk>"
# Pieces as in the real vocabulary: a leading space starts a word, and terms split
# into short pieces (Kubernetes -> " Ku", "ber", "net", "es").
VOCAB = [" Hello", " world", ".", " Привет", " uh", " Ku", "ber", "net", "es", "<|en|>",
         " ok", " er", " um", " э", " Er", " sagt", " fünf", " Мир", " rep", "osi", "tor", "y",
         " M", "L", BLANK]
IDS = {p: i for i, p in enumerate(VOCAB)}
FRAME_S = 0.08


class FakeAsr:
    """Emits one scripted piece per frame unless score edits change the winner.

    `script[t]` is the favourite at frame t (score 5.0), `runner_up[t]` scores 4.0.
    `steps[t]` is the TDT duration head's answer (default 1 = advance one frame).
    """

    window_step = 0.01
    _subsampling_factor = 8
    _max_tokens_per_step = 10
    use_low_precision = False

    def __init__(self, script, runner_up=None, steps=None):
        self._vocab = dict(enumerate(VOCAB))
        self._vocab_size = len(VOCAB)
        self._blank_idx = IDS[BLANK]
        self.script = script
        self.runner_up = runner_up or {}
        self.steps = steps or {}

    def _create_state(self):
        return 0

    def _decode(self, prev_tokens, prev_state, frame):
        t = int(frame[0])
        logits = np.full(self._vocab_size, -10.0, np.float32)
        logits[IDS[self.script[t]]] = 5.0
        if t in self.runner_up:
            logits[IDS[self.runner_up[t]]] = 4.0
        return logits, self.steps.get(t, 1), prev_state + 1

    def _preprocessor(self, audio, lengths):
        return audio, lengths

    def _encode(self, audio, lengths):
        frames = np.arange(len(self.script), dtype=np.float32)[:, None]
        return frames[None], np.array([len(self.script)])


def model(script, runner_up=None, penalty=None, steps=None, glossary=None, allowlist=None):
    """A model whose penalty applies to every language (None = ban)."""
    m = object.__new__(parakeet.ParakeetModel)
    m._attach(FakeAsr(script, runner_up, steps), "fake", {"*": penalty}, glossary, allowlist)
    return m


def audio(seconds=1.0):
    return np.zeros(int(16000 * seconds), np.float32)


def text(segments):
    return " ".join(s.text for s in segments)


class TestTranscribe:
    def test_words_join_into_sentence_segments(self):
        m = model([" Hello", " world", ".", " ok", BLANK])

        segments, info = m.transcribe(audio())

        assert [s.text for s in segments] == ["Hello world.", "ok"]
        assert info.duration == pytest.approx(1.0)

    def test_int16_audio_is_accepted(self):
        segments, _ = model([" ok"]).transcribe(np.zeros(1600, np.int16))

        assert text(segments) == "ok"

    def test_tags_never_reach_the_text(self):
        segments, _ = model(["<|en|>", " ok"], runner_up={0: BLANK}).transcribe(audio(0.1))

        assert text(segments) == "ok"

    def test_whisper_only_options_are_ignored(self):
        segments, _ = model([" ok"]).transcribe(
            audio(0.1), initial_prompt="x", hotwords="y", condition_on_previous_text=True,
            without_timestamps=True, temperature=0.2,
        )

        assert text(segments) == "ok"

    def test_has_no_language_detector(self):
        # Callers rely on this to pass language=None and let the model pick.
        assert not hasattr(model([" ok"]), "detect_language")


class TestTimestamps:
    """Times are on the caller's timeline: the 0.3 s edge padding is removed."""

    def test_word_start_excludes_the_padding(self):
        segments, _ = model([BLANK] * 10 + [" Hello"]).transcribe(audio())

        assert segments[0].words[0].start == pytest.approx(10 * FRAME_S - parakeet.EDGE_PAD_S)

    def test_word_ends_after_its_own_duration_not_at_the_next_word(self):
        # " Hello" at frame 5 lasts 2 frames; then a pause; " world" at frame 30.
        script = [BLANK] * 5 + [" Hello", BLANK] + [BLANK] * 23 + [" world"]
        segments, _ = model(script, steps={5: 2}).transcribe(audio(3.0))

        hello, world = segments[0].words
        assert hello.end == pytest.approx(7 * FRAME_S - parakeet.EDGE_PAD_S)
        assert world.start - hello.end == pytest.approx(23 * FRAME_S)

    def test_multi_piece_word_spans_its_pieces(self):
        segments, _ = model([BLANK] * 5 + [" Ku", "ber", "net", "es"]).transcribe(audio())

        word = segments[0].words[0]
        assert word.word == "Kubernetes"
        assert word.end - word.start == pytest.approx(4 * FRAME_S)

    def test_times_are_clamped_to_the_audio(self):
        # Pieces in the leading/trailing padding would otherwise get negative or
        # past-the-end times.
        segments, _ = model([" Hello"] + [BLANK] * 20 + [" world"]).transcribe(audio(1.0))

        hello, world = segments[0].words
        assert hello.start == 0.0
        assert world.end == pytest.approx(1.0)
        assert 0.0 <= hello.probability <= 1.0


class TestDecodingMatchesOnnxAsr:
    """Our greedy loop must pick exactly what onnx-asr's does when no edits apply."""

    class RandomAsr(FakeAsr):
        def __init__(self, rng, frames):
            super().__init__([BLANK] * frames)
            self._max_tokens_per_step = 3
            self.rng = rng

        def _decode(self, prev_tokens, prev_state, frame):
            logits = self.rng.normal(size=self._vocab_size).astype(np.float32)
            logits[self._blank_idx] += 1.0
            step = int(self.rng.integers(0, 3))  # includes 0: several pieces per frame
            return logits, step, prev_state + 1

    @pytest.mark.parametrize("seed", range(20))
    def test_same_tokens_frames_and_logprobs(self, seed):
        frames = 40
        ours = object.__new__(parakeet.ParakeetModel)
        ours._attach(self.RandomAsr(np.random.default_rng(seed), frames), "fake", {"*": 0})
        encodings = np.zeros((frames, 1), np.float32)
        zero_bias = np.zeros(len(VOCAB), np.float32)

        tokens, at, _, lps = ours._greedy(encodings, frames, zero_bias, None)
        reference = self.RandomAsr(np.random.default_rng(seed), frames)
        ref_tokens, ref_at, ref_lps = next(_AsrWithTransducerDecoding._decoding(
            reference, encodings[None], np.array([frames]), need_logprobs=True))

        assert tokens == list(ref_tokens)
        assert at == list(ref_at)
        assert lps == pytest.approx(list(ref_lps), abs=1e-5)


class TestLongAudio:
    def test_long_audio_is_decoded_in_speech_pieces(self, monkeypatch):
        m = model([" ok"])
        calls = []
        monkeypatch.setattr(parakeet.ParakeetModel, "_speech_pieces",
                            staticmethod(lambda a: [(0, 16000), (20 * 16000, 21 * 16000)]))
        original = m._decode_piece

        def spy(piece, language):
            calls.append(len(piece))
            return original(piece, language)

        m._decode_piece = spy

        segments, info = m.transcribe(audio(parakeet.MAX_PASS_S + 30))

        assert calls == [16000, 16000]
        assert [round(w.start, 2) for s in segments for w in s.words] == [0.0, 20.0]
        assert info.duration == pytest.approx(parakeet.MAX_PASS_S + 30)

    def test_a_sentence_never_spans_two_pieces(self, monkeypatch):
        # No final punctuation in the first piece: the pause still ends its segment.
        m = model([" Hello", " world"])
        monkeypatch.setattr(parakeet.ParakeetModel, "_speech_pieces",
                            staticmethod(lambda a: [(0, 16000), (20 * 16000, 21 * 16000)]))

        segments, _ = m.transcribe(audio(parakeet.MAX_PASS_S + 30))

        assert [s.text for s in segments] == ["Hello world", "Hello world"]

    def test_short_audio_is_one_pass(self, monkeypatch):
        monkeypatch.setattr(parakeet.ParakeetModel, "_speech_pieces",
                            staticmethod(lambda a: pytest.fail("must not split")))

        segments, _ = model([" ok"]).transcribe(audio(parakeet.MAX_PASS_S))

        assert text(segments) == "ok"

    def test_speech_pieces_respect_the_pass_limit(self):
        rng = np.random.default_rng(0)
        # Broadband noise that Silero accepts as one long run of speech.
        noise = (rng.standard_normal(16000 * 70) * 0.3).astype(np.float32)

        pieces = parakeet.ParakeetModel._speech_pieces(noise)

        assert all(end - start <= parakeet.MAX_PASS_S * 16000 for start, end in pieces)


class TestSpeechGate:
    def test_no_speech_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(parakeet.ParakeetModel, "_has_speech", staticmethod(lambda a, p: False))

        segments, info = model([" ok"]).transcribe(audio(0.1), vad_filter=True)

        assert segments == []
        assert info.duration == pytest.approx(0.1)

    def test_gate_only_runs_with_vad_filter(self, monkeypatch):
        monkeypatch.setattr(parakeet.ParakeetModel, "_has_speech", staticmethod(lambda a, p: False))

        segments, _ = model([" ok"]).transcribe(audio(0.1))

        assert text(segments) == "ok"


class TestLanguageMask:
    def test_hard_ban_keeps_the_given_alphabet(self):
        m = model([" Привет"], runner_up={0: " Hello"})

        assert text(m.transcribe(audio(0.1), language="en")[0]) == "Hello"
        assert text(m.transcribe(audio(0.1), language="ru")[0]) == "Привет"

    def test_small_penalty_lets_a_clear_winner_through(self):
        # The favourite leads by 1.0, so it survives a 0.5 penalty.
        m = model([" Привет"], runner_up={0: " Hello"}, penalty=0.5)

        assert text(m.transcribe(audio(0.1), language="en")[0]) == "Привет"

    def test_penalty_is_chosen_per_language(self):
        m = object.__new__(parakeet.ParakeetModel)
        m._attach(FakeAsr([" Hello"], runner_up={0: " Привет"}), "fake", {"*": None, "ru": 0.5})
        # ru's own soft penalty lets the clear Latin winner through...
        assert text(m.transcribe(audio(0.1), language="ru")[0]) == "Hello"
        # ...while a ru ban overrides a soft "*" default.
        m._attach(FakeAsr([" Hello"], runner_up={0: " Привет"}), "fake", {"*": 0.5, "ru": None})
        assert text(m.transcribe(audio(0.1), language="ru")[0]) == "Привет"

    def test_zero_turns_the_mask_off(self):
        m = model([" Привет"], runner_up={0: " Hello"}, penalty=0)

        assert text(m.transcribe(audio(0.1), language="en")[0]) == "Привет"

    def test_unknown_language_allows_every_alphabet(self):
        m = model([" Привет"], runner_up={0: " Hello"})

        assert text(m.transcribe(audio(0.1), language="ja")[0]) == "Привет"

    def test_concurrent_calls_keep_their_own_language(self):
        # One ONNX session serves dictation and a meeting at once.
        m = model([" Привет"], runner_up={0: " Hello"})
        out = {}

        def run(i, lang):
            out[i] = text(m.transcribe(audio(0.1), language=lang)[0])

        threads = [threading.Thread(target=run, args=(i, l)) for i, l in enumerate(["en", "ru"] * 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert out == {i: "Hello" if i % 2 == 0 else "Привет" for i in range(8)}

    def test_default_policy_bans_except_cyrillic_languages(self):
        assert parakeet.DEFAULT_SCRIPT_PENALTY["*"] is None
        assert parakeet.DEFAULT_SCRIPT_PENALTY["ru"] == 3.0

    def test_every_supported_language_maps_to_an_alphabet(self):
        assert parakeet.SCRIPT_OF_LANGUAGE["en"] == "LATIN"
        assert parakeet.SCRIPT_OF_LANGUAGE["uk"] == "CYRILLIC"
        assert len(parakeet.SCRIPT_OF_LANGUAGE) == 25


class TestAlphabetDetection:
    """language=None: decode once per allowlist alphabet, keep the best scored."""

    def test_picks_the_confident_alphabet(self):
        # Cyrillic speech: the Latin-only decode can't find anything and stays blank.
        m = model([" Привет", " Мир"], allowlist=["en", "ru"])

        segments, info = m.transcribe(audio(0.2))

        assert text(segments) == "Привет Мир"
        assert info.language == "ru"

    def test_picks_latin_for_latin_speech(self):
        m = model([" Hello", " world"], allowlist=["en", "ru"])

        segments, info = m.transcribe(audio(0.2))

        assert text(segments) == "Hello world"
        assert info.language == "en"

    def test_coverage_beats_a_short_confident_fragment(self):
        # Mean log probability alone prefers the 1-piece fragment (-0.12 vs -0.13).
        assert parakeet.detection_score([3], [-0.12], 50) < parakeet.detection_score([45], [-0.13] * 30, 50)

    def test_empty_decode_never_wins(self):
        assert parakeet.detection_score([], [], 50) == float("-inf")

    def test_same_alphabet_languages_decode_once_and_report_unknown(self):
        m = model([" Hello"], allowlist=["en", "de"])

        assert m._candidates == [("en", None)]
        assert m.transcribe(audio(0.1))[1].language is None

    def test_given_language_skips_detection(self):
        m = model([" Привет"], runner_up={0: " Hello"}, allowlist=["en", "ru"])

        segments, info = m.transcribe(audio(0.1), language="en")

        assert text(segments) == "Hello"
        assert info.language == "en"

    def test_without_allowlist_decodes_once_unmasked(self):
        m = model([" Привет"], runner_up={0: " Hello"})

        segments, info = m.transcribe(audio(0.1))

        assert text(segments) == "Привет"
        assert info.language is None


class TestGlossary:
    def test_bonus_continues_a_started_term(self):
        # After " Ku", the continuation "ber" trails " ok" by 1.0 < bonus 2.0.
        script = [" Ku", " ok", "net", "es"]
        m_plain = model(script, runner_up={1: "ber"})
        m_glossary = model(script, runner_up={1: "ber"}, glossary=["Kubernetes"])

        assert text(m_plain.transcribe(audio(0.4))[0]) == "Ku oknetes"
        assert text(m_glossary.transcribe(audio(0.4))[0]) == "Kubernetes"

    def test_terms_are_not_started_from_common_first_pieces(self):
        # " Ku" is not boosted: it would pull the term into unrelated words.
        m = model([" ok", "ber"], runner_up={0: " Ku"}, glossary=["Kubernetes"])

        assert text(m.transcribe(audio(0.2))[0]) == "okber"

    def test_multi_word_terms_stay_one_sequence(self):
        piece_to_id = {p: i for i, p in enumerate(VOCAB)}
        glossary = parakeet._Glossary(["ML repository"], piece_to_id)

        assert [IDS[" M"], IDS["L"], IDS[" rep"], IDS["osi"], IDS["tor"], IDS["y"]] in glossary.sequences
        # Mid-term continuation needs the whole prefix so far, " M" "L" " rep"...
        assert IDS["osi"] in glossary.boosted([IDS[" M"], IDS["L"], IDS[" rep"]])
        # ...so " rep" starting some other word ("report") gets no push toward "osi".
        assert IDS["osi"] not in glossary.boosted([IDS[" ok"], IDS[" rep"]])


class TestFillers:
    def W(self, word):
        return (parakeet.Word(0.0, 1.0, word, 1.0), [-0.1])

    def words(self, *ws, language=None):
        return [w.word for w, _ in parakeet.drop_fillers([self.W(x) for x in ws], language)]

    def test_english_hesitations_are_removed(self):
        assert self.words("I", "uh", "would", "um,", "ask", "er", language="en") == ["I", "would", "ask"]

    def test_capital_moves_to_the_next_word(self):
        assert self.words("Uh", "you", "may") == ["You", "may"]

    def test_sentence_end_moves_to_the_previous_word(self):
        assert self.words("and", "describe", "uh.") == ["and", "describe."]

    def test_cyrillic_hesitations_are_removed_for_russian(self):
        assert self.words("ну", "э", "да", language="ru") == ["ну", "да"]

    @pytest.mark.parametrize("language, sentence", [
        ("de", ["Er", "sagt,", "dass", "wir", "um", "fünf", "Uhr", "gehen."]),
        ("pt", ["Eu", "comprei", "um", "carro", "novo."]),
        ("da", ["Det", "er", "godt."]),
        (None, ["Er", "is", "een", "probleem."]),
    ])
    def test_real_words_of_other_languages_are_kept(self, language, sentence):
        assert self.words(*sentence, language=language) == sentence

    def test_unknown_language_removes_only_unambiguous_sounds(self):
        assert self.words("so", "uh", "um", "hmm", language=None) == ["so", "um"]

    def test_real_words_are_kept(self):
        assert self.words("Okay,", "um-brella", "hmmm", language="en") == ["Okay,", "um-brella", "hmmm"]


class TestParseScriptPenalty:
    def test_parses_ban_numbers_and_default(self):
        assert parakeet.parse_script_penalty("*:ban, ru:3, UK:2.5") == {"*": None, "ru": 3.0, "uk": 2.5}

    @pytest.mark.parametrize("raw", ["ru", "ru:", ":3", "ru:soft", "ru:-1", "ru:nan"])
    def test_bad_entries_are_rejected(self, raw):
        with pytest.raises(ValueError, match="entry"):
            parakeet.parse_script_penalty(raw)
