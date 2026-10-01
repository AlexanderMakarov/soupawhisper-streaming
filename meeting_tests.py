#!/usr/bin/env python3
"""
Tests for SoupaWhisper meeting.py (meeting recording mode).
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.timeout(2)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# dictate_tests.py installs a MagicMock for webrtcvad in sys.modules at import time,
# and pytest collects it before this file. dictate.py imports meeting, so by now a
# copy of meeting holding that mock is cached too. Drop both so the import below
# rebuilds meeting against the real VAD -- speech_ratio is only meaningful against
# the real implementation. dictate keeps its own already-bound references, so the
# dictation tests are unaffected.
sys.modules.pop("webrtcvad", None)
sys.modules.pop("meeting", None)

import meeting


class TestParseSrt:
    def test_parses_single_cue_into_start_end_text(self):
        srt = "1\n00:00:04,120 --> 00:00:05,900\nHi, can you hear me?\n\n"

        cues = meeting.parse_srt(srt)

        assert len(cues) == 1
        assert cues[0].start == pytest.approx(4.12)
        assert cues[0].end == pytest.approx(5.9)
        assert cues[0].text == "Hi, can you hear me?"


def cue(start, end, text="word"):
    return meeting.Cue(start, end, text)


class TestMergeTracks:
    def test_orders_blocks_by_time_and_labels_each_speaker(self):
        me = [cue(5.0, 6.0, "Yes, loud and clear.")]
        them = [cue(1.0, 2.0, "Hi, can you hear me?")]

        blocks = meeting.merge_tracks(me, them)

        assert [(b.speaker, b.text) for b in blocks] == [
            ("Them", "Hi, can you hear me?"),
            ("Me", "Yes, loud and clear."),
        ]

    def test_merges_my_consecutive_sentences_into_one_block(self):
        me = [cue(1.0, 2.0, "First sentence."), cue(2.5, 3.5, "Second sentence.")]

        blocks = meeting.merge_tracks(me, [], me_gap_s=5.0)

        assert len(blocks) == 1
        assert blocks[0].text == "First sentence. Second sentence."
        assert blocks[0].start == pytest.approx(1.0)
        assert blocks[0].end == pytest.approx(3.5)

    def test_splits_them_when_silence_exceeds_their_gap(self):
        them = [cue(1.0, 2.0, "Alice speaking."), cue(12.0, 13.0, "Bob speaking.")]

        blocks = meeting.merge_tracks([], them, them_gap_s=8.0)

        assert [b.text for b in blocks] == ["Alice speaking.", "Bob speaking."]

    def test_caps_block_duration_so_a_monologue_keeps_getting_timestamps(self):
        # Continuous speech, no gap ever exceeding them_gap_s.
        them = [cue(float(i), float(i) + 1.0, f"s{i}") for i in range(0, 120)]

        blocks = meeting.merge_tracks([], them, them_gap_s=8.0, them_max_block_s=30.0)

        assert len(blocks) > 1
        assert all(b.end - b.start <= 30.0 for b in blocks)

    def test_caps_them_tighter_than_me_so_participants_get_dense_timestamps(self):
        # Same continuous 120s of speech on each track, never overlapping.
        them = [cue(float(i), float(i) + 1.0, f"t{i}") for i in range(0, 120)]
        me = [cue(200.0 + i, 201.0 + i, f"m{i}") for i in range(0, 120)]

        blocks = meeting.merge_tracks(
            me, them, me_max_block_s=600.0, them_max_block_s=20.0
        )

        them_blocks = [b for b in blocks if b.speaker == "Them"]
        me_blocks = [b for b in blocks if b.speaker == "Me"]
        assert len(me_blocks) == 1
        assert len(them_blocks) >= 6


class TestRenderMarkdown:
    def test_renders_each_block_with_timestamp_and_speaker(self):
        blocks = [
            meeting.Block("Them", 4.0, 5.5, "Hi, can you hear me?"),
            meeting.Block("Me", 6.12, 7.8, "Yes, loud and clear."),
        ]

        out = meeting.render_markdown(blocks)

        assert "**[00:00:04] Them:** Hi, can you hear me?" in out
        assert "**[00:00:06] Me:** Yes, loud and clear." in out


class TestWavUsage:
    def test_reports_total_bytes_and_count_of_wavs_under_root(self, tmp_path):
        (tmp_path / "meetings" / "2026-09-01_1430").mkdir(parents=True)
        (tmp_path / "meetings" / "2026-09-01_1430" / "mic.wav").write_bytes(b"x" * 100)
        (tmp_path / "meetings" / "2026-09-01_1430" / "them.wav").write_bytes(b"x" * 250)
        (tmp_path / "recording_20260107.wav").write_bytes(b"x" * 50)
        (tmp_path / "notes.txt").write_bytes(b"x" * 999)

        usage = meeting.wav_usage(tmp_path)

        assert usage.total_bytes == 400
        assert usage.count == 3

    def test_reports_zero_when_root_does_not_exist(self, tmp_path):
        usage = meeting.wav_usage(tmp_path / "missing")

        assert usage.total_bytes == 0
        assert usage.count == 0


class TestPurgeWavs:
    def test_deletes_wavs_and_reports_freed_bytes(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "mic.wav").write_bytes(b"x" * 300)
        (tmp_path / "keep.txt").write_bytes(b"x" * 10)

        freed = meeting.purge_wavs(tmp_path)

        assert freed == 300
        assert not (tmp_path / "a" / "mic.wav").exists()
        assert (tmp_path / "keep.txt").exists()

    def test_never_deletes_the_session_still_being_recorded(self, tmp_path):
        live = tmp_path / "live"
        live.mkdir()
        (live / "mic.wav").write_bytes(b"x" * 500)
        (tmp_path / "old").mkdir()
        (tmp_path / "old" / "mic.wav").write_bytes(b"x" * 200)

        freed = meeting.purge_wavs(tmp_path, keep=live)

        assert freed == 200
        assert (live / "mic.wav").exists()


class TestSpeechRatio:
    def test_digital_silence_has_no_speech(self):
        silence = np.zeros(16000, dtype=np.int16)

        assert meeting.speech_ratio(silence) == 0.0

    def test_a_real_noise_floor_still_counts_as_silence(self):
        # A live mic never returns digital zero; +/-4 LSB hiss must not read as
        # speech or the auto-stop watchdog would never fire.
        hiss = np.random.default_rng(0).integers(-4, 5, 16000).astype(np.int16)

        assert meeting.speech_ratio(hiss) == 0.0

    def test_loud_broadband_audio_counts_as_speech(self):
        noise = np.random.default_rng(0).normal(0, 6000, 16000)
        noise = noise.clip(-32768, 32767).astype(np.int16)

        assert meeting.speech_ratio(noise) > 0.9


class TestWavTail:
    def _write_wav(self, path, samples):
        import wave as w
        with w.open(str(path), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(samples.tobytes())

    def test_reads_only_the_last_window_of_a_growing_file(self, tmp_path):
        wav = tmp_path / "them.wav"
        # Ramp stays inside int16 so each sample's value identifies its position.
        ramp = (np.arange(16000 * 10) % 30000).astype(np.int16)
        self._write_wav(wav, ramp)

        tail = meeting.wav_tail(wav, seconds=2.0)

        assert len(tail) == 32000
        assert tail[0] == (16000 * 8) % 30000

    def test_returns_whole_file_when_shorter_than_the_window(self, tmp_path):
        wav = tmp_path / "mic.wav"
        self._write_wav(wav, np.zeros(1600, dtype=np.int16))

        assert len(meeting.wav_tail(wav, seconds=60.0)) == 1600

    def test_returns_empty_when_the_file_has_no_frames_yet(self, tmp_path):
        wav = tmp_path / "mic.wav"
        self._write_wav(wav, np.zeros(0, dtype=np.int16))

        assert len(meeting.wav_tail(wav, seconds=60.0)) == 0


class TestRecordCommands:
    def test_mic_track_records_the_default_source(self):
        cmd = meeting.pw_record_cmd(meeting.Path("/tmp/x/mic.wav"), capture_sink=False)

        assert cmd[0] == "pw-record"
        assert "--rate=16000" in cmd and "--channels=1" in cmd and "--format=s16" in cmd
        assert "stream.capture.sink=true" not in " ".join(cmd)
        assert cmd[-1] == "/tmp/x/mic.wav"

    def test_participant_track_follows_the_default_sink_monitor(self):
        cmd = meeting.pw_record_cmd(meeting.Path("/tmp/x/them.wav"), capture_sink=True)

        assert "stream.capture.sink=true" in " ".join(cmd)


class FakeProc:
    def __init__(self, cmd, **kw):
        self.cmd, self.terminated, self._rc = cmd, False, None
    def terminate(self): self.terminated = True; self._rc = 0
    def wait(self, timeout=None): return 0
    def poll(self): return self._rc


class TestRecorderLifecycle:
    def make(self, tmp_path, **kw):
        kw.setdefault("transcript_dir", tmp_path / "transcripts")
        kw.setdefault("tmp_root", tmp_path / "tmp")
        kw.setdefault("spawn", FakeProc)
        return meeting.MeetingRecorder(**kw)

    def test_start_spawns_one_recorder_per_track(self, tmp_path):
        rec = self.make(tmp_path)

        rec.start()

        assert rec.active
        assert len(rec.procs) == 2
        assert rec.session_dir.is_dir()

    def test_stop_terminates_both_tracks_gracefully(self, tmp_path):
        rec = self.make(tmp_path)
        rec.start()
        procs = list(rec.procs)

        rec.stop_recording()

        assert all(p.terminated for p in procs)
        assert not rec.active

    def test_auto_stops_once_max_duration_is_reached(self, tmp_path):
        now = [1000.0]
        rec = self.make(tmp_path, max_duration_s=7200, clock=lambda: now[0])
        rec.start()

        now[0] += 7201
        assert rec.auto_stop_reason() == "max_duration"

    def test_does_not_auto_stop_before_max_duration(self, tmp_path):
        now = [1000.0]
        rec = self.make(tmp_path, max_duration_s=7200, clock=lambda: now[0])
        rec.start()

        now[0] += 3600
        assert rec.auto_stop_reason() is None


class DeadProc(FakeProc):
    def poll(self): return 1


class TestRecorderFailureDetection:
    def make(self, tmp_path, spawn):
        return meeting.MeetingRecorder(
            transcript_dir=tmp_path / "t", tmp_root=tmp_path / "tmp", spawn=spawn
        )

    def test_reports_when_a_track_recorder_dies_mid_meeting(self, tmp_path):
        rec = self.make(tmp_path, DeadProc)
        rec.start()

        assert rec.track_failure() is not None
        assert "mic.wav" in rec.track_failure()

    def test_reports_nothing_while_both_tracks_run(self, tmp_path):
        rec = self.make(tmp_path, FakeProc)
        rec.start()

        assert rec.track_failure() is None

    def test_start_fails_loudly_when_the_session_dir_cannot_be_created(self, tmp_path):
        blocker = tmp_path / "tmp"
        blocker.write_text("not a directory")
        rec = self.make(tmp_path, FakeProc)

        with pytest.raises(OSError):
            rec.start()
        assert not rec.active


class TestRenderHeader:
    def test_separates_header_from_first_block_by_one_blank_line(self):
        out = meeting.render_markdown(
            [meeting.Block("Me", 0.0, 1.0, "Hello.")], header="# Meeting X"
        )

        assert out.startswith("# Meeting X\n\n**[00:00:00] Me:** Hello.")


class FakeSegment:
    def __init__(self, start, end, text, avg_logprob=None):
        self.start, self.end, self.text = start, end, text
        if avg_logprob is not None:
            self.avg_logprob = avg_logprob


class WindowModel:
    """Model stub recording which language each window was decoded with."""

    def __init__(self, detect=("en", 0.9, [("en", 0.9), ("ru", 0.05), ("la", 0.3)])):
        self._detect = detect
        self.calls = []

    def detect_language(self, audio, **kw):
        return self._detect

    def transcribe(self, audio, **kw):
        self.calls.append(kw.get("language"))
        n = len(self.calls)
        return [FakeSegment(1.0, 2.0, f"w{n}")], SimpleNamespace(duration=len(audio) / 16000)


def speechlike(seconds, seed=0):
    """Broadband audio the VAD accepts as speech (silence is now skipped)."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 6000, int(seconds * 16000)).clip(-32768, 32767)
    return (samples / 32768).astype(np.float32)


class ModelDictation:
    """Dictation stand-in exposing the meeting model + allowlist finish_session needs."""

    def __init__(self, model=None, allowlist=None, fail=False):
        self._model = model or WindowModel()
        self.config = {"language_allowlist": allowlist}
        self.fail = fail

    def meeting_model(self):
        if self.fail:
            raise RuntimeError("Model not loaded")
        return self._model


class TestFinishSessionOutputs:
    def session(self, tmp_path, seconds=2.0):
        import wave as w
        d = tmp_path / "tmp" / "meetings" / "2026-09-01_143000"
        d.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(d / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\0" * int(seconds * 16000 * 2))
        return d

    def test_transcript_is_written_next_to_the_audio_as_well(self, tmp_path):
        session = self.session(tmp_path)

        out = meeting.finish_session(ModelDictation(), session, tmp_path / "docs")

        assert out.parent == tmp_path / "docs"
        assert (session / "2026-09-01_143000.md").exists()
        assert (session / "2026-09-01_143000.md").read_text() == out.read_text()

    def test_per_track_srt_files_are_kept_for_seeking(self, tmp_path):
        session = self.session(tmp_path)

        meeting.finish_session(ModelDictation(), session, tmp_path / "docs")

        assert (session / "mic.srt").exists()
        assert (session / "them.srt").exists()

    def test_a_session_log_records_the_transcription_run(self, tmp_path):
        session = self.session(tmp_path)

        meeting.finish_session(ModelDictation(), session, tmp_path / "docs")

        log = (session / "transcribe.log").read_text()
        assert "mic.wav" in log and "them.wav" in log

    def test_audio_is_kept_when_transcription_fails(self, tmp_path):
        session = self.session(tmp_path)

        with pytest.raises(RuntimeError):
            meeting.finish_session(ModelDictation(fail=True), session, tmp_path / "docs")

        assert (session / "mic.wav").exists()


class ProgressDictation(ModelDictation):
    """ModelDictation plus the retention flag and the slot the tray reads progress from."""

    def __init__(self, keep_audio=False, **kw):
        super().__init__(**kw)
        self.config["meeting_keep_audio"] = keep_audio
        self.meeting_progress = None


class TestSessionProgressAndRetention:
    @pytest.fixture(autouse=True)
    def three_speech_runs(self, monkeypatch):
        """The synthetic track is noise, which the real VAD rightly rejects; pin the
        segmentation so these tests exercise finish_session, not the VAD."""
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [
                (s * sample_rate, (s + 5) * sample_rate) for s in (0, 30, 60)
            ],
        )

    def session(self, tmp_path, seconds=90.0):
        import wave as w
        d = tmp_path / "tmp" / "meetings" / "2026-09-01_143000"
        d.mkdir(parents=True)
        pcm = (speechlike(seconds) * 32767).astype(np.int16).tobytes()
        for name in ("mic.wav", "them.wav"):
            with w.open(str(d / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(pcm)
        return d

    def test_log_records_per_run_progress(self, tmp_path):
        session = self.session(tmp_path)

        meeting.finish_session(ProgressDictation(), session, tmp_path / "docs")

        log = (session / "transcribe.log").read_text()
        assert "1/3" in log and "3/3" in log

    def test_progress_is_published_for_the_tray(self, tmp_path):
        session = self.session(tmp_path)
        d = ProgressDictation()

        meeting.finish_session(d, session, tmp_path / "docs")

        # Cleared once finished so the tray stops showing a stale percentage.
        assert d.meeting_progress is None

    def test_audio_is_deleted_by_default(self, tmp_path):
        session = self.session(tmp_path)

        meeting.finish_session(ProgressDictation(), session, tmp_path / "docs")

        assert not (session / "mic.wav").exists()

    def test_audio_is_kept_when_configured(self, tmp_path):
        session = self.session(tmp_path)

        meeting.finish_session(ProgressDictation(keep_audio=True), session, tmp_path / "docs")

        assert (session / "mic.wav").exists()


class ConfidenceModel(WindowModel):
    """Emits segments with caller-supplied confidence."""

    def __init__(self, logprobs):
        super().__init__()
        self.logprobs = logprobs

    def transcribe(self, audio, **kw):
        self.calls.append(kw.get("language"))
        segs = []
        for i, lp in enumerate(self.logprobs):
            s = FakeSegment(float(i), float(i) + 1.0, f"text{i}")
            s.avg_logprob = lp
            segs.append(s)
        return segs, SimpleNamespace(duration=len(audio) / 16000)


class TestLowConfidenceSegmentsDropped:
    """Noise that VAD accepts still makes Whisper invent text, but its own
    avg_logprob separates it cleanly (measured: -2.2 junk vs -0.5 real speech)."""

    def audio(self):
        return np.zeros(30 * 16000, dtype=np.float32)

    def test_keeps_a_short_real_utterance(self):
        # Measured: a real one-word answer ("Всё.") scores -1.09 -- short utterances
        # are naturally less confident, and the old -1.0 floor threw them away.
        model = ConfidenceModel([-1.09])

        cues = meeting.transcribe_runs(model, self.audio(), find_runs=runs_at((1.0, 5.0)))

        assert [c.text for c in cues] == ["text0"]

    def test_drops_segments_below_the_confidence_floor(self):
        model = ConfidenceModel([-2.2, -0.5])

        cues = meeting.transcribe_runs(model, self.audio(), find_runs=runs_at((1.0, 5.0)))

        assert [c.text for c in cues] == ["text1"]

    def test_keeps_everything_when_all_segments_are_confident(self):
        model = ConfidenceModel([-0.5, -0.9])

        cues = meeting.transcribe_runs(model, self.audio(), find_runs=runs_at((1.0, 5.0)))

        assert len(cues) == 2

    def test_segments_without_a_confidence_score_are_kept(self):
        # Never silently drop output just because a backend omits the field.
        model = WindowModel()

        cues = meeting.transcribe_runs(model, self.audio(), find_runs=runs_at((1.0, 5.0)))

        assert len(cues) == 1


class RunModel:
    """Model stub that reports a different language per call, in order."""

    def __init__(self, languages=("en", "ru")):
        self._languages = list(languages)
        self.calls = []
        self.chunk_lengths = []

    def detect_language(self, audio, **kw):
        lang = self._languages[min(len(self.calls), len(self._languages) - 1)]
        return (lang, 0.9, [(lang, 0.9)])

    def transcribe(self, audio, **kw):
        self.calls.append(kw.get("language"))
        self.chunk_lengths.append(len(audio) / 16000)
        return ([FakeSegment(0.5, 1.0, f"w{len(self.calls)}")],
                SimpleNamespace(duration=len(audio) / 16000))


def runs_at(*spans):
    """A find_runs stub: fixed (start, end) sample spans, ignoring the audio."""
    return lambda audio, sample_rate=16000, settings=None: [
        (int(s * sample_rate), int(e * sample_rate)) for s, e in spans
    ]


class TestTranscribeRuns:
    """Regression: a 30s window holding two languages was decoded entirely in the
    louder one, silently dropping the other speaker's phrase, and Whisper stretched
    its segment back over the leading silence so every block was stamped 00:00:00."""

    def test_decodes_each_speech_run_with_its_own_language(self):
        model = RunModel(languages=("en", "ru"))
        audio = np.zeros(20 * 16000, dtype=np.float32)

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((2.0, 4.0), (10.0, 12.0))
        )

        assert model.calls == ["en", "ru"]

    def test_offsets_cues_by_the_run_start_not_the_file_start(self):
        model = RunModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)

        cues = meeting.transcribe_runs(
            model, audio, find_runs=runs_at((2.0, 4.0), (10.0, 12.0))
        )

        # Segment at 0.5s inside runs starting at 2.0s and 10.0s.
        assert [round(c.start, 2) for c in cues] == [2.5, 10.5]

    def test_sends_only_the_run_audio_to_the_model(self):
        model = RunModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)

        meeting.transcribe_runs(model, audio, find_runs=runs_at((2.0, 4.0)))

        assert model.chunk_lengths == [pytest.approx(2.0)]

    def test_splits_a_long_run_so_language_can_change_mid_monologue(self):
        model = RunModel(languages=("en", "en", "ru"))
        audio = np.zeros(80 * 16000, dtype=np.float32)

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((0.0, 75.0)),
            settings=meeting.RunSettings(max_run_s=30.0),
        )

        assert model.calls == ["en", "en", "ru"]

    def test_transcribes_nothing_when_no_speech_was_found(self):
        model = RunModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)

        cues = meeting.transcribe_runs(model, audio, find_runs=runs_at())

        assert model.calls == []
        assert cues == []

    def test_reports_progress_per_run(self):
        model = RunModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)
        seen = []

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((1.0, 2.0), (5.0, 6.0), (9.0, 10.0)),
            progress=lambda done, total: seen.append((done, total)),
        )

        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_allowlist_still_filters_nonsense_language_winners(self):
        model = RunModel()
        model.detect_language = lambda audio, **kw: (
            "la", 0.3, [("la", 0.3), ("ru", 0.2), ("en", 0.1)]
        )
        audio = np.zeros(20 * 16000, dtype=np.float32)

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((1.0, 2.0)), allowlist=["en", "ru"]
        )

        assert model.calls == ["ru"]

    def test_drops_low_confidence_segments_as_hallucination(self):
        class Noisy(RunModel):
            def transcribe(self, audio, **kw):
                self.calls.append(kw.get("language"))
                return ([FakeSegment(0.0, 1.0, "Редактор субтитров А.Семкин",
                                     avg_logprob=-2.2)],
                        SimpleNamespace(duration=1.0))

        model = Noisy()
        audio = np.zeros(20 * 16000, dtype=np.float32)

        cues = meeting.transcribe_runs(model, audio, find_runs=runs_at((1.0, 2.0)))

        assert cues == []


class TestSpeechRuns:
    def test_finds_no_speech_in_silence(self):
        assert meeting.speech_runs(np.zeros(10 * 16000, dtype=np.float32)) == []

    def test_rejects_broadband_noise_that_webrtcvad_accepted(self):
        # webrtcvad scored this 0.9+; the silero VAD used here is the stronger guard.
        assert meeting.speech_runs(speechlike(10)) == []


class TestRunSettings:
    """The speech-run VAD knobs, tunable from [meeting] in config.ini."""

    def test_defaults_come_from_the_module_constants(self):
        s = meeting.RunSettings()

        assert s.min_silence_ms == meeting.RUN_MIN_SILENCE_MS
        assert s.pad_ms == meeting.RUN_PAD_MS
        assert s.vad_threshold == meeting.RUN_VAD_THRESHOLD
        assert s.min_speech_ms == meeting.RUN_MIN_SPEECH_MS
        assert s.max_run_s == meeting.LANGUAGE_WINDOW_S
        assert s.min_avg_logprob == meeting.MIN_AVG_LOGPROB

    def test_reads_every_override_from_the_dictation_config(self):
        s = meeting.RunSettings.from_config({
            "meeting_run_min_silence_ms": 1500,
            "meeting_run_pad_ms": 50,
            "meeting_run_vad_threshold": 0.7,
            "meeting_run_min_speech_ms": 250,
            "meeting_run_max_seconds": 20.0,
            "meeting_min_avg_logprob": -2.0,
        })

        assert (s.min_silence_ms, s.pad_ms, s.vad_threshold) == (1500, 50, 0.7)
        assert (s.min_speech_ms, s.max_run_s, s.min_avg_logprob) == (250, 20.0, -2.0)

    def test_absent_keys_keep_the_defaults(self):
        assert meeting.RunSettings.from_config({}) == meeting.RunSettings()

    def test_tolerates_a_missing_config(self):
        # finish_session runs from a Dictation whose config may not be loaded yet.
        assert meeting.RunSettings.from_config(None) == meeting.RunSettings()


class TestSpeechRunsHonoursSettings:
    def capture(self, monkeypatch):
        seen = {}

        def fake(audio, options):
            seen["options"] = options
            return []

        monkeypatch.setattr(meeting, "get_speech_timestamps", fake)
        return seen

    def test_passes_every_knob_through_to_the_vad(self, monkeypatch):
        seen = self.capture(monkeypatch)

        meeting.speech_runs(
            np.zeros(16000, dtype=np.float32),
            settings=meeting.RunSettings(
                min_silence_ms=1500, pad_ms=50, vad_threshold=0.7, min_speech_ms=250
            ),
        )

        o = seen["options"]
        assert o.min_silence_duration_ms == 1500
        assert o.speech_pad_ms == 50
        assert o.threshold == 0.7
        assert o.min_speech_duration_ms == 250

    def test_defaults_when_no_settings_given(self, monkeypatch):
        seen = self.capture(monkeypatch)

        meeting.speech_runs(np.zeros(16000, dtype=np.float32))

        assert seen["options"].min_silence_duration_ms == meeting.RUN_MIN_SILENCE_MS


class TestTranscribeRunsHonoursSettings:
    def test_confidence_floor_comes_from_settings(self):
        model = ConfidenceModel([-1.8])
        audio = np.zeros(20 * 16000, dtype=np.float32)

        kept = meeting.transcribe_runs(
            model, audio, find_runs=runs_at((1.0, 5.0)),
            settings=meeting.RunSettings(min_avg_logprob=-2.0),
        )
        dropped = meeting.transcribe_runs(
            ConfidenceModel([-1.8]), audio, find_runs=runs_at((1.0, 5.0)),
            settings=meeting.RunSettings(min_avg_logprob=-1.0),
        )

        assert len(kept) == 1 and dropped == []

    def test_run_length_cap_comes_from_settings(self):
        model = RunModel(languages=("en",))
        audio = np.zeros(80 * 16000, dtype=np.float32)

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((0.0, 60.0)),
            settings=meeting.RunSettings(max_run_s=20.0),
        )

        assert len(model.calls) == 3


class TestFinishSessionAppliesConfiguredSettings:
    def test_config_reaches_the_vad(self, tmp_path, monkeypatch):
        import wave as w
        session = tmp_path / "2026-09-03_094213"
        session.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(session / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        seen = {}
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: seen.setdefault("s", settings) and [],
        )
        d = ProgressDictation(keep_audio=True)
        d.config["meeting_run_min_silence_ms"] = 1200

        meeting.finish_session(d, session, tmp_path / "docs")

        assert seen["s"].min_silence_ms == 1200


class GlossaryModel(RunModel):
    """Records the full kwargs each transcribe() call received."""

    def __init__(self):
        super().__init__()
        self.kwargs = []

    def transcribe(self, audio, **kw):
        self.kwargs.append(kw)
        return super().transcribe(audio, **kw)


class TestGlossaryReachesTheModel:
    """Dictation primes Whisper with custom_terms; meeting mode did not, so domain
    words came back wrong ("sync 8 group" for sync.WaitGroup, "cup flow" for Kubeflow)."""

    def test_transcribe_runs_forwards_the_glossary(self):
        model = GlossaryModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)
        glossary = {"initial_prompt": "Glossary: Redis.", "hotwords": "Redis"}

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((1.0, 3.0)), transcribe_kwargs=glossary
        )

        assert model.kwargs[0]["initial_prompt"] == "Glossary: Redis."
        assert model.kwargs[0]["hotwords"] == "Redis"

    def test_language_is_still_decided_per_run(self):
        model = GlossaryModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)

        meeting.transcribe_runs(
            model, audio, find_runs=runs_at((1.0, 3.0), (8.0, 10.0)),
            transcribe_kwargs={"hotwords": "Redis"},
        )

        assert [kw["language"] for kw in model.kwargs] == ["en", "ru"]

    def test_no_glossary_sends_no_extra_kwargs(self):
        model = GlossaryModel()
        audio = np.zeros(20 * 16000, dtype=np.float32)

        meeting.transcribe_runs(model, audio, find_runs=runs_at((1.0, 3.0)))

        assert "hotwords" not in model.kwargs[0]

    def test_finish_session_takes_the_glossary_from_the_dictation(self, tmp_path, monkeypatch):
        import wave as w
        session = tmp_path / "2026-09-03_185547"
        session.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(session / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [(0, 16000)],
        )
        model = GlossaryModel()
        d = ProgressDictation(model=model, keep_audio=True)
        d.custom_terms_kwargs = {"hotwords": "Redis Kubeflow"}

        meeting.finish_session(d, session, tmp_path / "docs")

        assert model.kwargs[0]["hotwords"] == "Redis Kubeflow"


class TestBlockLengthDefaults:
    """A 10-minute cap produced 415-word walls; the reference transcript for the same
    interview capped around 121. 60s reproduces that shape (measured: 110 words)."""

    def test_me_blocks_cap_at_a_readable_length(self):
        assert meeting.ME_MAX_BLOCK_S == 60.0

    def test_finish_session_honours_configured_block_caps(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            meeting, "merge_tracks",
            lambda me, them, **kw: seen.update(kw) or [],
        )
        import wave as w
        session = tmp_path / "2026-09-03_185547"
        session.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(session / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [],
        )
        d = ProgressDictation(keep_audio=True)
        d.config["meeting_me_max_block_seconds"] = 45.0
        d.config["meeting_them_max_block_seconds"] = 15.0

        meeting.finish_session(d, session, tmp_path / "docs")

        assert seen["me_max_block_s"] == 45.0
        assert seen["them_max_block_s"] == 15.0


class TestRejectPhrasesSharedWithDictation:
    """reject_phrases is one [behavior] setting for every mode: the noise it filters
    ("thank you", "um") is Whisper's, not any one mode's."""

    def session(self, tmp_path):
        import wave as w
        d = tmp_path / "2026-09-07_120000"
        d.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(d / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        return d

    def dictation(self, monkeypatch, texts, reject=None):
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [(0, 16000)],
        )

        class Model(RunModel):
            def transcribe(self, audio, **kw):
                self.calls.append(kw.get("language"))
                segs = [FakeSegment(float(i), float(i) + 1.0, t)
                        for i, t in enumerate(texts)]
                return segs, SimpleNamespace(duration=1.0)

        d = ProgressDictation(model=Model(), keep_audio=True)
        if reject is not None:
            d.should_reject_text = lambda t: t.strip().lower().strip(".!") in reject
        return d

    def test_a_cue_matching_a_reject_phrase_is_dropped(self, tmp_path, monkeypatch):
        d = self.dictation(monkeypatch, ["Thank you.", "Real content here."],
                           reject={"thank you"})

        out = meeting.finish_session(d, self.session(tmp_path), tmp_path / "docs")

        assert "Thank you" not in out.read_text()
        assert "Real content here." in out.read_text()

    def test_partial_matches_are_kept(self, tmp_path, monkeypatch):
        d = self.dictation(monkeypatch, ["Thank you for the answer."],
                           reject={"thank you"})

        out = meeting.finish_session(d, self.session(tmp_path), tmp_path / "docs")

        assert "Thank you for the answer." in out.read_text()

    def test_works_without_a_reject_predicate(self, tmp_path, monkeypatch):
        d = self.dictation(monkeypatch, ["Thank you."])

        out = meeting.finish_session(d, self.session(tmp_path), tmp_path / "docs")

        assert "Thank you." in out.read_text()


class TestGlossaryIsTheSharedOne:
    def test_finish_session_reads_the_shared_custom_terms(self, tmp_path, monkeypatch):
        import wave as w
        session = tmp_path / "2026-09-07_120000"
        session.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(session / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [(0, 16000)],
        )
        model = GlossaryModel()
        d = ProgressDictation(model=model, keep_audio=True)
        d.custom_terms_kwargs = {"hotwords": "Redis Kubeflow"}

        meeting.finish_session(d, session, tmp_path / "docs")

        assert model.kwargs[0]["hotwords"] == "Redis Kubeflow"


class WordModel(RunModel):
    """Model stub returning per-word timings, as word_timestamps=True does."""

    def transcribe(self, audio, **kw):
        self.calls.append(kw.get("language"))
        self.kwargs = getattr(self, "kwargs", []) + [kw]
        seg = FakeSegment(0.5, 1.5, "hello world")
        seg.words = [SimpleNamespace(start=0.5, end=0.9, word=" hello", probability=0.98),
                     SimpleNamespace(start=1.0, end=1.5, word=" world", probability=0.91)]
        return [seg], SimpleNamespace(duration=len(audio) / 16000)


class TestWordTimestamps:
    """Per-word timings, for measuring pauses and response latency."""

    def audio(self):
        return np.zeros(20 * 16000, dtype=np.float32)

    def test_off_by_default(self):
        model = WordModel()

        meeting.transcribe_runs(model, self.audio(), find_runs=runs_at((1.0, 3.0)))

        assert model.kwargs[0].get("word_timestamps") in (None, False)

    def test_requested_when_enabled(self):
        model = WordModel()

        meeting.transcribe_runs(
            model, self.audio(), find_runs=runs_at((1.0, 3.0)),
            settings=meeting.RunSettings(word_timestamps=True),
        )

        assert model.kwargs[0]["word_timestamps"] is True

    def test_words_are_offset_onto_the_meeting_clock(self):
        model = WordModel()

        cues = meeting.transcribe_runs(
            model, self.audio(), find_runs=runs_at((10.0, 13.0)),
            settings=meeting.RunSettings(word_timestamps=True),
        )

        # Run starts at 10s, so its 0.5s word sits at 10.5s of the meeting.
        assert [round(w.start, 2) for w in cues[0].words] == [10.5, 11.0]
        assert [w.text for w in cues[0].words] == ["hello", "world"]

    def test_cues_have_no_words_when_disabled(self):
        model = WordModel()

        cues = meeting.transcribe_runs(model, self.audio(), find_runs=runs_at((1.0, 3.0)))

        assert cues[0].words == []


class TestWordTimestampFile:
    def session(self, tmp_path):
        import wave as w
        d = tmp_path / "2026-09-07_120000"
        d.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(d / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        return d

    def run(self, tmp_path, monkeypatch, enabled):
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [(0, 16000)],
        )
        d = ProgressDictation(model=WordModel(), keep_audio=True)
        d.config["meeting_word_timestamps"] = enabled
        session = self.session(tmp_path)
        meeting.finish_session(d, session, tmp_path / "docs")
        return session

    def test_words_json_written_when_enabled(self, tmp_path, monkeypatch):
        import json
        session = self.run(tmp_path, monkeypatch, True)

        data = json.loads((session / "mic.words.json").read_text())

        assert data[0]["text"] == "hello world"
        assert data[0]["words"][0] == {
            "start": 0.5, "end": 0.9, "text": "hello", "probability": 0.98
        }

    def test_no_file_when_disabled(self, tmp_path, monkeypatch):
        session = self.run(tmp_path, monkeypatch, False)

        assert not (session / "mic.words.json").exists()


def block(speaker, start, end, text):
    return meeting.Block(speaker, start, end, text)


class TestConversationMetrics:
    """Frontmatter stats: who talked, how fast, and how long each waited."""

    def test_speaking_ratio_is_me_over_them_by_duration(self):
        blocks = [block("Me", 0.0, 30.0, "a b c"), block("Them", 30.0, 40.0, "d e")]

        m = meeting.conversation_metrics(blocks)

        assert m["me_speaking_s"] == 30.0
        assert m["them_speaking_s"] == 10.0
        assert m["me_them_speaking_ratio"] == 3.0

    def test_ratio_is_none_when_only_one_side_spoke(self):
        m = meeting.conversation_metrics([block("Me", 0.0, 10.0, "a")])

        assert m["me_them_speaking_ratio"] is None
        assert m["them_speaking_s"] == 0.0

    def test_wpm_is_words_over_block_duration(self):
        # 10 words in 30s = 20 wpm; 5 words in 10s = 30 wpm.
        blocks = [block("Me", 0.0, 30.0, " ".join("w" * 10)),
                  block("Me", 40.0, 50.0, " ".join("w" * 5))]

        m = meeting.conversation_metrics(blocks)

        assert m["me_wpm_avg"] == 25.0
        assert m["me_wpm_stdev"] == 5.0

    def test_think_time_only_counts_speaker_changes(self):
        blocks = [
            block("Them", 0.0, 10.0, "question"),
            block("Me", 12.0, 20.0, "answer"),      # 2.0s after Them -> my think time
            block("Me", 21.0, 25.0, "still me"),    # same speaker, not a think gap
            block("Them", 27.0, 30.0, "reply"),     # 2.0s after Me -> their think time
        ]

        m = meeting.conversation_metrics(blocks)

        assert m["me_think_time_avg_s"] == 2.0
        assert m["them_think_time_avg_s"] == 2.0

    def test_overlaps_are_not_counted_as_think_time(self):
        # Interrupting gives a negative gap; averaging it in would understate waiting.
        blocks = [block("Them", 0.0, 10.0, "q"),
                  block("Me", 8.0, 12.0, "interrupting"),
                  block("Them", 20.0, 22.0, "q2"),
                  block("Me", 24.0, 26.0, "a")]

        m = meeting.conversation_metrics(blocks)

        assert m["me_think_time_avg_s"] == 2.0

    def test_empty_transcript_yields_no_crash(self):
        m = meeting.conversation_metrics([])

        assert m["me_them_speaking_ratio"] is None
        assert m["me_wpm_avg"] is None
        assert m["me_think_time_avg_s"] is None


class TestFrontmatter:
    def test_render_markdown_emits_yaml_frontmatter(self):
        blocks = [block("Me", 0.0, 30.0, " ".join("w" * 10)),
                  block("Them", 40.0, 50.0, " ".join("w" * 10))]

        out = meeting.render_markdown(blocks, "# Meeting x",
                                      metrics=meeting.conversation_metrics(blocks))

        assert out.startswith("---\n")
        head = out.split("---")[1]
        assert "me_them_speaking_ratio: 3.0" in head
        assert "me_wpm_avg: 20.0" in head
        assert "# Meeting x" in out

    def test_none_values_render_as_empty(self):
        out = meeting.render_markdown([block("Me", 0.0, 10.0, "hi")], "# x",
                                      metrics=meeting.conversation_metrics(
                                          [block("Me", 0.0, 10.0, "hi")]))

        assert "me_them_speaking_ratio:\n" in out

    def test_no_frontmatter_without_metrics(self):
        out = meeting.render_markdown([block("Me", 0.0, 10.0, "hi")], "# x")

        assert not out.startswith("---")

    def test_finish_session_writes_frontmatter(self, tmp_path, monkeypatch):
        import wave as w
        session = tmp_path / "2026-09-07_120000"
        session.mkdir(parents=True)
        for name in ("mic.wav", "them.wav"):
            with w.open(str(session / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(b"\x00\x00" * 16000)
        monkeypatch.setattr(
            meeting, "speech_runs",
            lambda audio, sample_rate=16000, settings=None: [(0, 16000)],
        )

        out = meeting.finish_session(
            ProgressDictation(keep_audio=True), session, tmp_path / "docs"
        )

        assert out.read_text().startswith("---\n")


class ContextLanguageModel(RunModel):
    """Script detector evidence independently of decoder calls."""

    def __init__(self, evidence):
        super().__init__()
        self.evidence = iter(evidence)
        self.detected_audio = []
        self.decoded_audio = []

    def detect_language(self, audio, **kw):
        self.detected_audio.append(audio.copy())
        result = next(self.evidence)
        if isinstance(result, Exception):
            raise result
        return result

    def transcribe(self, audio, **kw):
        self.decoded_audio.append(audio.copy())
        return super().transcribe(audio, **kw)


def language_scores(language, probability, other_probability):
    other = 'ru' if language == 'en' else 'en'
    return language, probability, [(language, probability), (other, other_probability)]


class TestSurroundingLanguageDetection:
    @pytest.mark.parametrize('language,wrong', [('en', 'ru'), ('ru', 'en')])
    def test_context_repairs_uncertain_run_without_changing_decode_audio_or_offsets(self, language, wrong):
        model = ContextLanguageModel([
            language_scores(language, .9, .05),
            language_scores(wrong, .117, .101),
            language_scores(language, .8, .1),
        ])
        audio = np.arange(10 * 16000, dtype=np.float32)
        seen = []

        cues = meeting.transcribe_runs(
            model, audio, allowlist=['en', 'ru'],
            find_runs=runs_at((1, 4), (5, 7)),
            progress=lambda done, total: seen.append((done, total)),
        )

        assert model.calls == [language, language]
        np.testing.assert_array_equal(model.detected_audio[-1], audio[16000:7 * 16000])
        np.testing.assert_array_equal(model.decoded_audio[-1], audio[5 * 16000:7 * 16000])
        assert [cue.start for cue in cues] == [1.5, 5.5]
        assert seen == [(1, 2), (2, 2)]

    @pytest.mark.parametrize('languages', [('en', 'ru', 'en'), ('ru', 'en', 'ru')])
    def test_confident_short_switches_do_not_consult_surrounding_speech(self, languages):
        model = ContextLanguageModel([language_scores(lang, .9, .05) for lang in languages])
        audio = np.zeros(10 * 16000, dtype=np.float32)

        meeting.transcribe_runs(model, audio, allowlist=['en', 'ru'],
                               find_runs=runs_at((1, 3), (4, 4.5), (5, 7)))

        assert model.calls == list(languages)
        assert [len(a) for a in model.detected_audio] == [32000, 8000, 32000]

    def test_low_absolute_allowlisted_probability_still_uses_context(self):
        model = ContextLanguageModel([
            ('uk', .6, [('uk', .6), ('ru', .148), ('en', .034)]),
            language_scores('en', .8, .1),
        ])

        lang = meeting._detect_window_language(model, np.zeros(100), ['en', 'ru'],
                                               context=np.zeros(1000))

        assert lang == 'en'
        assert len(model.detected_audio) == 2

    def test_close_high_probability_contest_uses_context(self):
        model = ContextLanguageModel([language_scores('ru', .5, .45),
                                      language_scores('en', .8, .1)])
        assert meeting._detect_window_language(model, np.zeros(100), ['en', 'ru'],
                                                context=np.zeros(1000)) == 'en'

    @pytest.mark.parametrize('context_result', [language_scores('en', .2, .15), RuntimeError('context failed')])
    def test_weak_or_failed_context_preserves_local_choice(self, context_result):
        model = ContextLanguageModel([language_scores('ru', .117, .101), context_result])
        assert meeting._detect_window_language(model, np.zeros(100), ['en', 'ru'],
                                                context=np.zeros(1000)) == 'ru'

    def test_failed_local_detection_can_recover_from_context(self):
        model = ContextLanguageModel([RuntimeError('local failed'), language_scores('ru', .8, .1)])
        assert meeting._detect_window_language(model, np.zeros(100), ['en', 'ru'],
                                                context=np.zeros(1000)) == 'ru'

    def test_both_detector_failures_delegate_to_decoder(self):
        model = ContextLanguageModel([RuntimeError('local failed'), RuntimeError('context failed')])
        assert meeting._detect_window_language(model, np.zeros(100), ['en', 'ru'],
                                                context=np.zeros(1000)) is None

    def test_context_also_respects_allowlist(self):
        model = ContextLanguageModel([
            language_scores('en', .2, .15),
            ('uk', .6, [('uk', .6), ('ru', .36), ('en', .04)]),
        ])
        assert meeting._detect_window_language(model, np.zeros(100), ['en', 'ru'],
                                                context=np.zeros(1000)) == 'ru'

    def test_context_works_without_an_allowlist(self):
        model = ContextLanguageModel([language_scores('ru', .2, .15), language_scores('en', .9, .05)])
        assert meeting._detect_window_language(model, np.zeros(100), None,
                                                context=np.zeros(1000)) == 'en'

    def test_isolated_run_does_not_retry_detection(self):
        model = ContextLanguageModel([language_scores('ru', .117, .101)])
        meeting.transcribe_runs(model, np.zeros(20 * 16000, dtype=np.float32),
                               find_runs=runs_at((5, 7)))
        assert model.calls == ['ru']
        assert len(model.detected_audio) == 1

    def test_language_is_not_carried_between_tracks(self):
        model = ContextLanguageModel([language_scores('en', .9, .05),
                                      language_scores('ru', .117, .101)])
        for _ in range(2):
            meeting.transcribe_runs(model, np.zeros(10 * 16000, dtype=np.float32),
                                   find_runs=runs_at((1, 3)))
        assert model.calls == ['en', 'ru']


class TestLanguageContextBounds:
    def test_context_stops_at_long_pauses_on_both_sides(self):
        audio = np.arange(300, dtype=np.float32)
        runs = [(0, 20), (50, 70), (80, 100), (130, 150)]
        # Sample rate 10: three-second gaps must separate clusters.
        np.testing.assert_array_equal(meeting._language_context(audio, runs, 1, 10), audio[50:100])
        np.testing.assert_array_equal(meeting._language_context(audio, runs, 2, 10), audio[50:100])

    def test_context_is_centered_and_capped_at_thirty_seconds(self):
        audio = np.arange(600, dtype=np.float32)
        runs = [(s, s + 10) for s in range(0, 600, 20)]
        np.testing.assert_array_equal(meeting._language_context(audio, runs, 15, 10), audio[160:450])

    @pytest.mark.parametrize('index,expected', [(0, (0, 150)), (29, (440, 590))])
    def test_track_edges_are_clipped_to_speech(self, index, expected):
        audio = np.arange(600, dtype=np.float32)
        runs = [(s, s + 10) for s in range(0, 600, 20)]
        np.testing.assert_array_equal(meeting._language_context(audio, runs, index, 10),
                                      audio[expected[0]:expected[1]])

    def test_full_window_run_needs_no_extra_context(self):
        audio = np.arange(600, dtype=np.float32)
        runs = [(0, 300), (300, 600)]
        np.testing.assert_array_equal(meeting._language_context(audio, runs, 0, 10), audio[:300])
