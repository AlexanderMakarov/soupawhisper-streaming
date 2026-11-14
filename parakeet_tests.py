"""Unit tests for parakeet.py with a tiny fake onnx-asr model (no download, no ONNX)."""
import threading

import numpy as np
import pytest

import parakeet

BLANK = "<blk>"
# id: piece. A leading space starts a word, as in the real vocabulary.
VOCAB = [" Hello", " world", ".", " Привет", " uh", " Kube", "rnetes", "<|en|>", " ok", BLANK]
IDS = {p: i for i, p in enumerate(VOCAB)}


class FakeAsr:
    """Emits a scripted token per encoder frame unless score edits change the winner.

    Each frame's scores favour `script[t]` by 5.0 and the runner-up by 4.0, so a
    hard ban or a large penalty on the favourite lets the runner-up win.
    """

    window_step = 0.01
    _subsampling_factor = 8
    _max_tokens_per_step = 10

    def __init__(self, script, runner_up=None):
        self._vocab = dict(enumerate(VOCAB))
        self._vocab_size = len(VOCAB)
        self._blank_idx = IDS[BLANK]
        self.script = script
        self.runner_up = runner_up or {}

    def _create_state(self):
        return 0

    def _decode(self, prev_tokens, prev_state, frame):
        t = int(frame[0])
        logits = np.full(self._vocab_size, -10.0, np.float32)
        logits[IDS[self.script[t]]] = 5.0
        if t in self.runner_up:
            logits[IDS[self.runner_up[t]]] = 4.0
        # step 1: one token per frame, like a TDT duration head saying "advance".
        return logits, 1, prev_state + 1

    def _preprocessor(self, audio, lengths):
        return audio, lengths

    def _encode(self, audio, lengths):
        frames = np.arange(len(self.script), dtype=np.float32)[:, None]
        return frames[None], np.array([len(self.script)])


def model(script, runner_up=None, penalty=None):
    """A model whose penalty applies to every language (None = ban)."""
    m = object.__new__(parakeet.ParakeetModel)
    m._attach(FakeAsr(script, runner_up), "fake", {"*": penalty})
    return m


def text(segments):
    return " ".join(s.text for s in segments)


class TestTranscribe:
    def test_words_join_into_sentence_segments(self):
        m = model([" Hello", " world", ".", " ok", BLANK])

        segments, info = m.transcribe(np.zeros(16000, np.float32))

        assert [s.text for s in segments] == ["Hello world.", "ok"]
        assert info.duration == pytest.approx(1.0)

    def test_int16_audio_is_accepted(self):
        segments, _ = model([" ok"]).transcribe(np.zeros(1600, np.int16))

        assert text(segments) == "ok"

    def test_word_times_exclude_the_edge_padding(self):
        # Frames are 80 ms; the 0.3 s pad is subtracted and times clamp to the audio.
        segments, _ = model([BLANK] * 5 + [" Hello", " world"]).transcribe(np.zeros(16000, np.float32))

        first, second = segments[0].words
        assert first.start == pytest.approx(5 * 0.08 - parakeet.EDGE_PAD_S)
        assert second.start == pytest.approx(6 * 0.08 - parakeet.EDGE_PAD_S)
        assert 0.0 <= first.probability <= 1.0

    def test_tags_never_reach_the_text(self):
        segments, _ = model(["<|en|>", " ok"], runner_up={0: BLANK}).transcribe(np.zeros(1600, np.float32))

        assert text(segments) == "ok"

    def test_whisper_only_options_are_ignored(self):
        segments, _ = model([" ok"]).transcribe(
            np.zeros(1600, np.float32), initial_prompt="x", condition_on_previous_text=True,
            without_timestamps=True, temperature=0.2,
        )

        assert text(segments) == "ok"

    def test_has_no_language_detector(self):
        # Callers rely on this to take the language from the keyboard layout.
        assert not hasattr(model([" ok"]), "detect_language")


class TestSpeechGate:
    def test_no_speech_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(parakeet.ParakeetModel, "_has_speech", staticmethod(lambda a, p: False))

        segments, info = model([" ok"]).transcribe(np.zeros(1600, np.float32), vad_filter=True)

        assert segments == []
        assert info.duration == pytest.approx(0.1)

    def test_gate_only_runs_with_vad_filter(self, monkeypatch):
        monkeypatch.setattr(parakeet.ParakeetModel, "_has_speech", staticmethod(lambda a, p: False))

        segments, _ = model([" ok"]).transcribe(np.zeros(1600, np.float32))

        assert text(segments) == "ok"


class TestLanguage:
    def test_hard_ban_keeps_the_layout_alphabet(self):
        m = model([" Привет"], runner_up={0: " Hello"})

        assert text(m.transcribe(np.zeros(1600, np.float32), language="en")[0]) == "Hello"
        assert text(m.transcribe(np.zeros(1600, np.float32), language="ru")[0]) == "Привет"

    def test_small_penalty_lets_a_clear_winner_through(self):
        # The favourite leads by 1.0, so it survives a 0.5 penalty.
        m = model([" Привет"], runner_up={0: " Hello"}, penalty=0.5)

        assert text(m.transcribe(np.zeros(1600, np.float32), language="en")[0]) == "Привет"

    def test_unknown_or_missing_language_allows_every_alphabet(self):
        m = model([" Привет"], runner_up={0: " Hello"})

        assert text(m.transcribe(np.zeros(1600, np.float32), language=None)[0]) == "Привет"
        assert text(m.transcribe(np.zeros(1600, np.float32), language="ja")[0]) == "Привет"

    def test_concurrent_calls_keep_their_own_language(self):
        # One ONNX session serves dictation and a meeting at once.
        m = model([" Привет"], runner_up={0: " Hello"})
        out = {}

        def run(i, lang):
            out[i] = text(m.transcribe(np.zeros(1600, np.float32), language=lang)[0])

        threads = [threading.Thread(target=run, args=(i, l)) for i, l in enumerate(["en", "ru"] * 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert out == {i: "Hello" if i % 2 == 0 else "Привет" for i in range(8)}

    def test_penalty_is_chosen_per_layout_language(self):
        m = object.__new__(parakeet.ParakeetModel)
        m._attach(FakeAsr([" Hello"], runner_up={0: " Привет"}), "fake", {"*": None, "ru": 0.5})

        # ru's own soft penalty lets the clear Latin winner through...
        assert text(m.transcribe(np.zeros(1600, np.float32), language="ru")[0]) == "Hello"
        # ...while a ru ban overrides a soft "*" default.
        m2 = object.__new__(parakeet.ParakeetModel)
        m2._attach(FakeAsr([" Hello"], runner_up={0: " Привет"}), "fake", {"*": 0.5, "ru": None})
        assert text(m2.transcribe(np.zeros(1600, np.float32), language="ru")[0]) == "Привет"

    def test_zero_turns_the_mask_off(self):
        m = model([" Привет"], runner_up={0: " Hello"}, penalty=0)

        assert text(m.transcribe(np.zeros(1600, np.float32), language="en")[0]) == "Привет"

    def test_default_policy_bans_except_cyrillic_languages(self):
        assert parakeet.DEFAULT_SCRIPT_PENALTY["*"] is None
        assert parakeet.DEFAULT_SCRIPT_PENALTY["ru"] == 3.0

    def test_every_supported_language_maps_to_a_script(self):
        assert parakeet.SCRIPT_OF_LANGUAGE["en"] == "LATIN"
        assert parakeet.SCRIPT_OF_LANGUAGE["uk"] == "CYRILLIC"
        assert len(parakeet.SCRIPT_OF_LANGUAGE) == 25


class TestGlossary:
    def test_bonus_lets_a_term_beat_a_close_runner_up(self):
        # Without the glossary " ok" wins frame 0; " Kube" trails by 1.0 < bonus 2.0.
        m = model([" ok", "rnetes"], runner_up={0: " Kube"})

        assert text(m.transcribe(np.zeros(1600, np.float32))[0]) == "okrnetes"
        assert text(m.transcribe(np.zeros(1600, np.float32), hotwords="Kubernetes")[0]) == "Kubernetes"

    def test_short_first_pieces_do_not_start_a_term(self):
        glossary = parakeet._Glossary(["ok"], {" ok": 0}, {0: " ok"})

        assert glossary.starters == frozenset()

    def test_a_started_term_is_continued(self):
        glossary = parakeet._Glossary(["Kubernetes"], {" Kube": 5, "rnetes": 6}, {5: " Kube", 6: "rnetes"})

        assert 6 in glossary.boosted([5])


class TestFillers:
    def W(self, word):
        return (parakeet.Word(0.0, 1.0, word, 1.0), [-0.1])

    def words(self, *ws):
        return [w.word for w, _ in parakeet.drop_fillers([self.W(x) for x in ws])]

    def test_hesitations_are_removed(self):
        assert self.words("I", "uh", "would", "um,", "ask") == ["I", "would", "ask"]

    def test_capital_moves_to_the_next_word(self):
        assert self.words("Uh", "you", "may") == ["You", "may"]

    def test_sentence_end_moves_to_the_previous_word(self):
        assert self.words("and", "describe", "uh.") == ["and", "describe."]

    def test_cyrillic_hesitations_are_removed(self):
        assert self.words("ну", "э", "да") == ["ну", "да"]

    def test_real_words_are_kept(self):
        assert self.words("Okay,", "um-brella", "hmmm") == ["Okay,", "um-brella", "hmmm"]


class TestParseScriptPenalty:
    def test_parses_ban_numbers_and_default(self):
        assert parakeet.parse_script_penalty("*:ban, ru:3, UK:2.5") == {"*": None, "ru": 3.0, "uk": 2.5}

    @pytest.mark.parametrize("raw", ["ru", "ru:", ":3", "ru:soft", "ru:-1", "ru:nan"])
    def test_bad_entries_are_rejected(self, raw):
        with pytest.raises(ValueError, match="entry"):
            parakeet.parse_script_penalty(raw)
