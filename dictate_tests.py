#!/usr/bin/env python3
"""
Tests for SoupaWhisper dictate.py
"""

import pytest
import logging
import numpy as np
import queue
from unittest.mock import MagicMock, patch
from typing import Any
from types import SimpleNamespace
import threading

# Add 2 second timeout to all tests to prevent infinite loops
pytestmark = pytest.mark.timeout(2)

# Import the modules to test
import sys
import os

# Add the directory containing dictate.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock pynput and streaming audio deps before importing dictate
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
webrtcvad_mock = MagicMock()
vad_instance = MagicMock()
vad_instance.is_speech = MagicMock(return_value=False)
webrtcvad_mock.Vad = MagicMock(return_value=vad_instance)
sys.modules['webrtcvad'] = webrtcvad_mock
streamsad_mock = MagicMock()
streamsad_mock.SAD = MagicMock()
sys.modules['streamsad'] = streamsad_mock

# Now import dictate
import dictate

# Import real Segment and Word from faster_whisper
from faster_whisper.transcribe import Segment, Word


class MockWhisperModel:
    """Mock WhisperModel for testing."""
    def __init__(self, model_name="base.en", device="cpu", compute_type="int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []
        self.transcribe = MagicMock(side_effect=self._transcribe_impl)

    def detect_language(self, audio_path, **kwargs):
        """Stub: Spanish first, then English — used to test language_allowlist."""
        return (
            "es",
            0.9,
            [("es", 0.5), ("en", 0.4), ("ru", 0.05)],
        )
    
    def _transcribe_impl(self, audio_path, **kwargs):
        """Mock transcribe that records calls and returns test data."""
        self.transcribe_calls.append((audio_path, kwargs))
        
        # Return real segments based on audio path or kwargs
        if "word_timestamps" in kwargs and kwargs["word_timestamps"]:
            # Return segments with word timestamps for streaming tests
            words = [
                Word(word="hello", start=0.0, end=0.5, probability=0.9),
                Word(word="world", start=0.6, end=1.0, probability=0.9),
            ]
            segment = Segment(
                id=0, seek=0, start=0.0, end=1.0, text="hello world",
                tokens=[], avg_logprob=0.0, compression_ratio=0.0,
                no_speech_prob=0.0, words=words, temperature=None
            )
            return [segment], {"language": "en"}
        else:
            # Return simple segments for non-streaming tests
            segment = Segment(
                id=0, seek=0, start=0.0, end=1.0, text="test transcription",
                tokens=[], avg_logprob=0.0, compression_ratio=0.0,
                no_speech_prob=0.0, words=None, temperature=None
            )
            return [segment], SimpleNamespace(duration=1.0)


@pytest.fixture
def mock_config(tmp_path, monkeypatch):
    """Create a temporary config file and mock CONFIG_PATH."""
    config_dir = tmp_path / ".config" / "soupawhisper"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.ini"
    
    # Create a test config
    config_content = """[whisper]
model = base.en
device = cpu
compute_type = int8

[hotkey]
key = f10

[behavior]
auto_type = true
notifications = false
default_streaming = false
clipboard = true
tray_icon = false
tray_show_language = true

[streaming]
vad_silence_threshold_seconds = 1.0
vad_sample_rate = 16000
vad_chunk_size_ms = 30
vad_threshold = 0.5
"""
    config_file.write_text(config_content)
    
    # Mock the CONFIG_PATH
    monkeypatch.setattr(dictate, "CONFIG_PATH", config_file)
    return config_file


@pytest.fixture
def mock_whisper_model(monkeypatch):
    """Mock WhisperModel."""
    model = MockWhisperModel()

    def mock_init(model_name, device="cpu", compute_type="int8"):
        model.model_name = model_name
        model.device = device
        model.compute_type = compute_type
        return model

    monkeypatch.setattr(dictate.WhisperModel, "__new__", lambda cls, *args, **kwargs: mock_init(*args, **kwargs))

    return model


@pytest.fixture
def mock_pyaudio_stream(monkeypatch):
    """Mock pyaudio stream."""
    mock_stream = MagicMock()
    mock_stream.read.return_value = b'\x00' * 3200
    mock_stream.stop_stream = MagicMock()
    mock_stream.close = MagicMock()
    return mock_stream


@pytest.fixture
def mock_xdotool(monkeypatch):
    """Mock xdotool."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0)
    
    # Mock subprocess.run for "which" command and other calls
    def mock_run_with_which(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
            result = MagicMock()
            if len(cmd) > 1 and cmd[1] in ["xdotool", "xclip"]:
                result.returncode = 0
            else:
                result.returncode = 1
            return result
        # For other commands (xdotool type, etc.), use the mock
        return mock_run(cmd, **kwargs)
    
    monkeypatch.setattr(dictate.subprocess, "run", mock_run_with_which)
    
    # Store the mock so tests can access it
    mock_run_with_which._mock_run = mock_run
    return mock_run_with_which


class TestTyper:
    """Tests for Typer class."""
    
    def test_typer_init(self, mock_xdotool):
        """Test Typer initialization."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer(delay_ms=20, start_delay_ms=100)
        assert typer.delay_ms == 20
        assert typer.start_delay_ms == 100
        assert typer.enabled is True
    
    def test_typer_type_rewrite_append(self, mock_xdotool):
        """Test typing text (append mode with previous_length=0)."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
            typer.type_rewrite("hello world", 0)
        assert mock_xdotool._mock_run.called

    def test_typer_type_rewrite_incremental(self, mock_xdotool):
        """Test incremental typing using type_rewrite with previous_length=0."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
            # Simulate incremental: calculate suffix and type with previous_length=0
            previous_text = "hello"
            new_text = "hello world"
            suffix = new_text[len(previous_text):]
            typer.type_rewrite(suffix, 0)
        assert mock_xdotool._mock_run.called

    def test_typer_type_rewrite_correction(self, mock_xdotool):
        """Test rewrite typing with character removal."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
            typer.type_rewrite("new text", 5)
        assert mock_xdotool._mock_run.called


class TestTyperMacOS:
    """macOS types through Quartz (pynput), so no Apple Events consent is involved."""

    @staticmethod
    def _typer(**kwargs):
        with patch.object(dictate, "IS_MACOS", True):
            return dictate.Typer(delay_ms=1, start_delay_ms=0, **kwargs)

    def test_types_via_pynput_without_shelling_out(self):
        """AppleScript keystroke needs Automation access and stalls on its consent dialog."""
        typer = self._typer()
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            with patch.object(dictate.subprocess, "run") as run:
                typer.type_rewrite("привет", 0)

        typed = "".join(call.args[0] for call in typer._controller.type.call_args_list)
        assert typed == "привет"
        run.assert_not_called()

    def test_deletes_previous_text_with_backspace(self):
        typer = self._typer()
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            typer.type_rewrite("new", 3)
        assert typer._controller.tap.call_count == 3

    def test_zero_delay_types_the_whole_string_in_one_burst(self):
        """A per-character sleep dominates the cost; 0 means hand it all to pynput at once."""
        typer = self._typer()
        typer.delay_ms = 0
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            with patch.object(dictate.time, "sleep") as sleep:
                typer.type_rewrite("hello world", 0)

        typer._controller.type.assert_called_once_with("hello world")
        sleep.assert_not_called()

    def test_positive_delay_still_paces_characters(self):
        """Kept so typing_delay can still be raised to watch the text appear."""
        typer = self._typer()
        typer.delay_ms = 5
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            with patch.object(dictate.time, "sleep") as sleep:
                typer.type_rewrite("abc", 0)

        assert typer._controller.type.call_count == 3
        assert sleep.call_args_list == [((0.005,),)] * 3

    def test_zero_delay_survives_the_constructor(self):
        with patch.object(dictate, "IS_MACOS", True):
            assert dictate.Typer(delay_ms=0).delay_ms == 0

    def test_no_controller_built_on_linux(self, mock_xdotool):
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
        assert typer._controller is None


class TestStreamingDictation:
    """Tests for StreamingDictation class."""

    def test_streaming_initializes(self, mock_config, mock_whisper_model, mock_xdotool):
        """Basic sanity check that StreamingDictation can be created."""
        config = dictate.load_config()
        config["default_streaming"] = True
        config["auto_type"] = True  # Required for streaming mode
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        assert isinstance(dictation, dictate.StreamingDictation)

    def test_reject_phrases_normalization_ignores_punctuation(self):
        assert dictate._normalize_reject_phrase(" um... ") == "um"
        assert dictate._normalize_reject_phrase("Hmm..") == "hmm"
        assert dictate._normalize_reject_phrase("THANK YOU!!!") == "thank you"

    def test_reject_phrases_disabled_when_empty(self):
        d = dictate.StreamingDictation.__new__(dictate.StreamingDictation)
        d.config = {"reject_phrases": ""}
        d._reject_phrase_set = d._build_reject_phrase_set()
        assert d._reject_phrase_set == frozenset()
        assert d.should_reject_text("thank you") is False

    def test_reject_phrases_exact_whole_chunk_match(self):
        d = dictate.StreamingDictation.__new__(dictate.StreamingDictation)
        d.config = {"reject_phrases": "thank you, um, hmm"}
        d._reject_phrase_set = d._build_reject_phrase_set()

        assert d.should_reject_text("thank you") is True
        assert d.should_reject_text("Thank you!!!") is True
        assert d.should_reject_text("um...") is True
        assert d.should_reject_text("Hmm..") is True

        # Multiple phrases / extra words must NOT be rejected.
        assert d.should_reject_text("thank you thanks") is False
        assert d.should_reject_text("um well") is False


class TestCustomTerms:
    """Tests for custom-terms glossary (initial_prompt + hotwords biasing)."""

    def test_parse_custom_terms_empty(self):
        assert dictate._parse_custom_terms("") == []
        assert dictate._parse_custom_terms("   ") == []
        assert dictate._parse_custom_terms(",,, ,") == []

    def test_parse_custom_terms_comma_separated(self):
        assert dictate._parse_custom_terms("Claude, Kubernetes, GraphQL") == [
            "Claude", "Kubernetes", "GraphQL"
        ]

    def test_parse_custom_terms_newline_separated(self):
        assert dictate._parse_custom_terms("Claude\nKubernetes\nGraphQL") == [
            "Claude", "Kubernetes", "GraphQL"
        ]

    def test_parse_custom_terms_preserves_case_and_multiword(self):
        assert dictate._parse_custom_terms("ML repository, Claude") == [
            "ML repository", "Claude"
        ]

    def test_parse_custom_terms_dedup_preserves_order(self):
        assert dictate._parse_custom_terms("Claude, Kubernetes, Claude, GraphQL") == [
            "Claude", "Kubernetes", "GraphQL"
        ]

    def test_build_custom_terms_kwargs_empty_returns_empty_dict(self):
        assert dictate._build_custom_terms_kwargs([]) == {}

    def test_build_custom_terms_kwargs_populated(self):
        kw = dictate._build_custom_terms_kwargs(["Claude", "ML repository"])
        assert kw == {
            "initial_prompt": "Glossary: Claude, ML repository.",
            "hotwords": "Claude ML repository",
        }

    def test_load_config_custom_terms_default_empty(self, mock_config):
        config = dictate.load_config()
        assert config["custom_terms"] == ""

    def test_load_config_custom_terms_parsed_from_behavior(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "clipboard = true",
                "clipboard = true\ncustom_terms = Claude, Kubernetes, ML repository",
            )
        )
        config = dictate.load_config()
        assert config["custom_terms"] == "Claude, Kubernetes, ML repository"

    def test_dictation_caches_empty_kwargs_when_unset(self, mock_config, mock_whisper_model, mock_xdotool):
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model_loaded.wait(timeout=1.0)
        assert d.custom_terms_kwargs == {}

    def test_dictation_caches_kwargs_when_configured(self, mock_config, mock_whisper_model, mock_xdotool):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "clipboard = true",
                "clipboard = true\ncustom_terms = Claude, Kubernetes",
            )
        )
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model_loaded.wait(timeout=1.0)
        assert d.custom_terms_kwargs == {
            "initial_prompt": "Glossary: Claude, Kubernetes.",
            "hotwords": "Claude Kubernetes",
        }

    def test_custom_terms_passed_to_non_streaming_transcribe(self, mock_config, mock_whisper_model, mock_xdotool):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "clipboard = true",
                "clipboard = true\ncustom_terms = Claude, Kubernetes",
            )
        )
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model = mock_whisper_model
        d.model_error = None
        d.model_loaded.set()

        audio = np.zeros(1600, dtype=np.int16)
        d._transcribe_audio_array(audio)

        _, kwargs = mock_whisper_model.transcribe_calls[-1]
        assert kwargs.get("initial_prompt") == "Glossary: Claude, Kubernetes."
        assert kwargs.get("hotwords") == "Claude Kubernetes"

    def test_custom_terms_absent_when_disabled_non_streaming(self, mock_config, mock_whisper_model, mock_xdotool):
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model = mock_whisper_model
        d.model_error = None
        d.model_loaded.set()

        audio = np.zeros(1600, dtype=np.int16)
        d._transcribe_audio_array(audio)

        _, kwargs = mock_whisper_model.transcribe_calls[-1]
        assert "initial_prompt" not in kwargs
        assert "hotwords" not in kwargs


class TestDictation:
    """Tests for non-streaming Dictation class (backward compatibility)."""
    
    @patch('dictate.pyaudio.PyAudio')
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_non_streaming_mode(self, mock_popen, mock_run, mock_pyaudio, mock_config, mock_whisper_model):
        """Test non-streaming mode (backward compatibility)."""
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_stream.read.return_value = b'\x00' * 3200
        mock_audio_stream.stop_stream = MagicMock()
        mock_audio_stream.close = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        # Ensure non-streaming mode
        config = dictate.load_config()
        config["default_streaming"] = False
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Start recording
        dictation.start_recording()
        assert dictation.recording is True
        
        # Stop recording
        dictation.stop_recording()
        assert dictation.recording is False
    
    def test_config_loading(self, mock_config):
        """Test configuration loading."""
        # Reload config to test loading
        config = dictate.load_config()
        assert "model" in config
        assert "default_streaming" in config
        assert "clipboard" in config
        assert config["model"] == "base.en"
        assert isinstance(config["default_streaming"], bool)
        assert config["clipboard"] is True

    def test_language_defaults_to_en(self, mock_config):
        config = dictate.load_config()
        assert config["language"] == "en"

    def test_language_auto_maps_to_none(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(content.replace("compute_type = int8", "compute_type = int8\nlanguage = auto"))
        config = dictate.load_config()
        assert config["language"] is None

    def test_language_allowlist_parsed(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "compute_type = int8",
                "compute_type = int8\nlanguage = auto\nlanguage_allowlist = en, ru",
            )
        )
        config = dictate.load_config()
        assert config["language"] is None
        assert config["language_allowlist"] == ["en", "ru"]

    def test_resolve_transcription_language_allowlist_picks_best_of_two(self):
        model = MockWhisperModel()
        audio = np.zeros(1600, dtype=np.float32)
        # Whisper "top" is es, but allowlist is en,ru -> should pick ru (0.05) vs ... wait
        # filtered: en 0.4, ru 0.05 -> max is en
        assert (
            dictate.resolve_transcription_language(
                model, audio, None, ["en", "ru"]
            )
            == "en"
        )

    def test_resolve_transcription_language_fixed_skips_allowlist(self):
        model = MockWhisperModel()
        audio = np.zeros(1600, dtype=np.float32)
        assert (
            dictate.resolve_transcription_language(
                model, audio, "ru", ["en", "ru"]
            )
            == "ru"
        )

    def test_resolve_transcription_language_single_allowlist_no_detect_call(self):
        """One candidate needs no detect_language (same as fixed language)."""
        model = MagicMock()
        model.detect_language = MagicMock()
        audio = np.zeros(1600, dtype=np.float32)
        assert dictate.resolve_transcription_language(model, audio, None, ["ru"]) == "ru"
        model.detect_language.assert_not_called()

    def test_layout_language_map_parsing(self):
        parsed = dictate.Dictation._parse_layout_to_language_map(
            "com.apple.keylayout.US:en, com.apple.keylayout.Russian:ru, xkb:de:de"
        )
        assert parsed["com.apple.keylayout.US"] == "en"
        assert parsed["com.apple.keylayout.Russian"] == "ru"
        assert parsed["xkb"] == "de:de"

    def test_language_from_layout_direct_and_heuristic(self):
        m = {"com.apple.keylayout.US": "en"}
        assert dictate.language_from_layout("com.apple.keylayout.US", m) == "en"
        assert dictate.language_from_layout("com.apple.keylayout.Russian", {}) is None

    def test_linux_active_xkb_layout_uses_group_index(self, monkeypatch):
        monkeypatch.setattr(dictate, "_which_ok", lambda name: False)
        monkeypatch.setattr(
            dictate,
            "_linux_xkb_layouts_from_setxkbmap",
            lambda: ["us", "ru", "am"],
        )
        monkeypatch.setattr(dictate, "_linux_xkb_group_index", lambda: 1)
        assert dictate._linux_active_xkb_layout() == "ru"

    def test_parse_setxkbmap_layouts(self):
        assert dictate.parse_setxkbmap_layouts("rules: evdev\nlayout: us,ru,am\n") == [
            "us",
            "ru",
            "am",
        ]
        assert dictate.parse_setxkbmap_layouts("layout: ru\n") == ["ru"]
        assert dictate.parse_setxkbmap_layouts("") is None
        assert dictate.parse_setxkbmap_layouts("model: pc104\n") is None

    def test_linux_active_prefers_xkb_switch(self, monkeypatch):
        monkeypatch.setattr(
            dictate,
            "_which_ok",
            lambda name: name == "xkb-switch",
        )
        monkeypatch.setattr(dictate, "_run_cmd", lambda cmd, timeout_s=0.5: "ru")
        monkeypatch.setattr(
            dictate,
            "_linux_xkb_layouts_from_setxkbmap",
            lambda: (_ for _ in ()).throw(AssertionError("should not fall back")),
        )
        assert dictate._linux_active_xkb_layout() == "ru"

    def test_detect_current_keyboard_language_linux_maps_us(self, monkeypatch):
        monkeypatch.setattr(dictate, "IS_MACOS", False)
        monkeypatch.setattr(dictate, "detect_current_keyboard_layout", lambda: "us,ru,am")
        monkeypatch.setattr(dictate, "_linux_active_xkb_layout", lambda: "us")
        assert dictate.detect_current_keyboard_language({"us": "en", "ru": "ru"}) == "en"

    def test_capture_session_language_uses_active_layout_not_layout_list(
        self, mock_config, monkeypatch
    ):
        """setxkbmap layout_id is often 'us,ru,am'; capture must use active group."""
        content = mock_config.read_text()
        content = content.replace(
            "compute_type = int8",
            "compute_type = int8\nlanguage = auto\nlanguage_allowlist = en, ru",
        )
        content = content.replace(
            "clipboard = true",
            "clipboard = true\n"
            "enforce_language_from_layout = true\n"
            "layout_to_language = us:en, ru:ru",
        )
        mock_config.write_text(content)
        config = dictate.load_config()
        monkeypatch.setattr(dictate, "IS_MACOS", False)
        monkeypatch.setattr(dictate, "detect_current_keyboard_layout", lambda: "us,ru,am")
        monkeypatch.setattr(dictate, "_linux_active_xkb_layout", lambda: "ru")
        d = dictate.Dictation(config)
        d._begin_session_language()
        assert d._session_enforced_language == "ru"
        d._end_session_language()
        assert d._session_enforced_language is None

    def test_stop_recording_clears_session_language(
        self, mock_config, mock_whisper_model, mock_xdotool
    ):
        config = dictate.load_config()
        config["enforce_language_from_layout"] = True
        d = dictate.Dictation(config)
        d.model_loaded.set()
        d.recording = True
        d.audio_data = []
        d.audio_thread = None
        d.audio_stream = None
        d._session_enforced_language = "en"
        with patch.object(d, "_report_audio_problem"):
            d.stop_recording()
        assert d._session_enforced_language is None
        assert d.transcribing is False

    def test_enforce_language_from_layout_overrides_auto(self, mock_config, monkeypatch):
        # Set auto language + allowlist.
        content = mock_config.read_text()
        content = content.replace(
            "compute_type = int8",
            "compute_type = int8\nlanguage = auto\nlanguage_allowlist = en, ru",
        )
        # Inject behavior keys into the existing [behavior] section.
        content = content.replace(
            "clipboard = true",
            "clipboard = true\n"
            "enforce_language_from_layout = true\n"
            "layout_to_language = com.apple.keylayout.Russian:ru",
        )
        mock_config.write_text(content)
        config = dictate.load_config()
        # Pretend we're on macOS to avoid xdotool usage.
        monkeypatch.setattr(dictate, "IS_MACOS", True)
        # Force detector to return Russian layout.
        monkeypatch.setattr(dictate, "detect_current_keyboard_layout", lambda: "com.apple.keylayout.Russian")
        # Avoid depending on real HIToolbox parsing in tests.
        monkeypatch.setattr(dictate, "_macos_input_source_languages_for_id", lambda _id: ["ru"])

        d = dictate.Dictation(config)
        # Don't wait for real model thread; stub model directly.
        model = MockWhisperModel()
        d.model = model
        d.model_error = None
        d.model_loaded.set()

        # Mimic "hotkey pressed" behavior (capture once at session start).
        d._capture_session_enforced_language()

        audio = np.zeros(1600, dtype=np.int16)
        d._transcribe_audio_array(audio)
        # Ensure transcribe was called with language="ru".
        _, kwargs = model.transcribe_calls[-1]
        assert kwargs.get("language") == "ru"

    def test_detect_current_keyboard_language_macos_uses_input_source_languages(self, monkeypatch):
        monkeypatch.setattr(dictate, "IS_MACOS", True)
        # Match the real-world structure: AppleCurrentKeyboardLayoutInputSourceID exists, while
        # AppleSelectedInputSources entries may not contain InputSourceID or InputSourceLanguages.
        monkeypatch.setattr(
            dictate,
            "_macos_hitoolbox_plist",
            lambda: {
                "AppleCurrentKeyboardLayoutInputSourceID": "com.apple.keylayout.ABC",
                "AppleSelectedInputSources": [
                    {"Bundle ID": "com.apple.PressAndHold", "InputSourceKind": "Non Keyboard Input Method"},
                    {"InputSourceKind": "Keyboard Layout", "KeyboardLayout ID": 252, "KeyboardLayout Name": "ABC"},
                ],
            },
        )
        # We still expect language detection to use _macos_input_source_languages_for_id fallback
        # when no explicit mapping exists.
        monkeypatch.setattr(dictate, "_macos_input_source_languages_for_id", lambda _id: ["en"])
        assert dictate.detect_current_keyboard_language({}) == "en"

    def test_detect_current_keyboard_layout_macos_uses_current_layout_id(self, monkeypatch):
        monkeypatch.setattr(dictate, "IS_MACOS", True)
        monkeypatch.setattr(
            dictate,
            "_macos_hitoolbox_plist",
            lambda: {
                "AppleCurrentKeyboardLayoutInputSourceID": "com.apple.keylayout.ABC",
                "AppleSelectedInputSources": [
                    {"Bundle ID": "com.apple.PressAndHold", "InputSourceKind": "Non Keyboard Input Method"},
                    {"InputSourceKind": "Keyboard Layout", "KeyboardLayout ID": 252, "KeyboardLayout Name": "ABC"},
                ],
            },
        )
        assert dictate.detect_current_keyboard_layout() == "com.apple.keylayout.ABC"

    def test_config_clipboard_disabled(self, mock_config):
        """Test that clipboard=false is correctly loaded."""
        # Replace existing clipboard = true with clipboard = false
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        
        config = dictate.load_config()
        assert config["clipboard"] is False


class TestNoAudioNotification:
    """Tests for the once-per-session 'No audio detected' toast (notify_no_audio)."""

    def test_notify_no_audio_defaults_to_true(self, mock_config):
        """Absent from config: one toast per session is the default."""
        config = dictate.load_config()
        assert config["notify_no_audio"] is True

    def test_notify_no_audio_can_be_disabled(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(content.replace("clipboard = true", "clipboard = true\nnotify_no_audio = false"))

        config = dictate.load_config()
        assert config["notify_no_audio"] is False

    def test_report_audio_problem_logs_but_does_not_notify_when_disabled(
        self, mock_config, mock_whisper_model, caplog
    ):
        config = dictate.load_config()
        config["notify_no_audio"] = False
        d = dictate.Dictation(config)

        with caplog.at_level(logging.WARNING, logger="dictate"):
            with patch.object(d, "notify") as mock_notify:
                with patch.object(d, "_get_available_input_devices") as mock_devices:
                    d._report_audio_problem("Audio input is effectively silent (too low amplitude)")

        mock_notify.assert_not_called()
        # Device enumeration is only needed for the notification body.
        mock_devices.assert_not_called()
        assert "Audio input is effectively silent (too low amplitude)" in caplog.text

    def test_report_audio_problem_notifies_once_per_session_by_default(
        self, mock_config, mock_whisper_model
    ):
        config = dictate.load_config()
        assert config["notify_no_audio"] is True
        d = dictate.Dictation(config)

        with patch.object(d, "notify") as mock_notify:
            with patch.object(d, "_get_available_input_devices", return_value=[(0, "test device", 1)]):
                d._report_audio_problem("Audio input contains only zeros")
                d._report_audio_problem("Audio input is effectively silent (too low amplitude)")
                d._report_audio_problem("No audio data recorded")

        assert mock_notify.call_count == 1
        assert mock_notify.call_args[0][0] == "No audio detected - check device"
        assert mock_notify.call_args[0][3] == 5000

    def test_check_valid_audio_input_still_rejects_silent_segments(
        self, mock_config, mock_whisper_model
    ):
        """Suppressing the notification must not change segment filtering."""
        config = dictate.load_config()
        config["notify_no_audio"] = False
        d = dictate.Dictation(config)

        with patch.object(d, "notify") as mock_notify:
            assert d._check_valid_audio_input(np.zeros(1600, dtype=np.int16)) is True
            assert d._check_valid_audio_input(np.full(1600, 10, dtype=np.int16)) is True
            assert d._check_valid_audio_input(np.full(1600, 5000, dtype=np.int16)) is False

        mock_notify.assert_not_called()

    def test_start_recording_resets_notified_flag(
        self, mock_config, mock_whisper_model, mock_xdotool
    ):
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model_loaded.set()
        d._audio_problem_notified = True

        with patch.object(d, "_start_pyaudio_stream", return_value=MagicMock()):
            with patch.object(d, "_audio_recording_worker"):
                with patch.object(d, "notify"):
                    d.start_recording()

        assert d._audio_problem_notified is False
        d.recording = False
        if d.audio_thread:
            d.audio_thread.join(timeout=1.0)


class TestClipboardIntegration:
    """Tests specifically for the clipboard parameter integration."""

    @patch('dictate.pyaudio.PyAudio')
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_dictation_no_clipboard_call(self, mock_popen, mock_run, mock_pyaudio, mock_config, mock_whisper_model):
        """Test that Dictation doesn't call xclip when clipboard is disabled."""
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_stream.read.return_value = b'\x00' * 3200
        mock_audio_stream.stop_stream = MagicMock()
        mock_audio_stream.close = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Mock audio data
        dictation.audio_data = [np.array([0] * 1600, dtype=np.int16)]
        dictation.recording = True
        
        # Mock model return
        segment = Segment(
            id=0, seek=0, start=0.0, end=1.0, text="test text",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=None, temperature=None
        )
        model: Any = dictation.model
        model.transcribe.side_effect = None
        model.transcribe.return_value = ([segment], {})
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    @patch('dictate.pyaudio.PyAudio')
    def test_streaming_dictation_no_clipboard_call(self, mock_pyaudio, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test that StreamingDictation doesn't call xclip when clipboard is disabled."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        config["auto_type"] = True  # Required for streaming mode
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Clear queues to prevent timeout
        while not dictation.transcription_queue.empty():
            try:
                dictation.transcription_queue.get_nowait()
            except queue.Empty:
                break
        while not dictation.typing_queue.empty():
            try:
                dictation.typing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Set up state for stop_recording
        dictation.recording = True
        dictation.accumulated_text = "final text"
        dictation.audio_stream = mock_audio_stream
        dictation.audio_interface = mock_audio_instance
        dictation.audio_thread = None  # No thread to join
        dictation.transcription_thread = None
        dictation.typing_thread = None
        dictation.file_saving_thread = None
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    def test_check_dependencies_clipboard_optional(self, monkeypatch):
        """Test that xclip is optional in check_dependencies if clipboard is disabled."""
        # Force Linux path so we test the xclip/xdotool dependency checks.
        monkeypatch.setattr(dictate, "IS_MACOS", False)
        mock_run = MagicMock()

        # Mock 'which' to return 1 for xclip (missing)
        def side_effect(cmd, **kwargs):
            res = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "xclip":
                res.returncode = 1
            else:
                res.returncode = 0
            return res

        mock_run.side_effect = side_effect
        monkeypatch.setattr(dictate.subprocess, "run", mock_run)

        # Should NOT exit if clipboard is False (webrtcvad and pyaudio are already imported, so import check passes)
        dictate.check_dependencies({"clipboard": False, "auto_type": False, "default_streaming": False})

        # Should exit if clipboard is True (since xclip is missing)
        with pytest.raises(SystemExit):
            dictate.check_dependencies({"clipboard": True, "auto_type": False, "default_streaming": False})


class TestTrayStatus:
    def test_load_config_tray_defaults(self, mock_config):
        config = dictate.load_config()
        assert config["tray_icon"] is False  # mock_config sets false
        assert config["tray_show_language"] is True

    def test_load_config_tray_defaults_when_unset(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.ini"
        config_file.write_text("[whisper]\nmodel = base.en\n[behavior]\nnotifications = false\n")
        monkeypatch.setattr(dictate, "CONFIG_PATH", config_file)
        config = dictate.load_config()
        assert config["tray_icon"] is True
        assert config["tray_show_language"] is True

    def test_derive_state_priority(self):
        import tray_status

        d = SimpleNamespace(
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            recording=False,
            stopping=False,
            transcribing=False,
        )
        assert tray_status.derive_state(d) == "loading"
        d.model_loaded.set()
        assert tray_status.derive_state(d) == "idle"
        d.recording = True
        assert tray_status.derive_state(d) == "recording"
        d.recording = False
        d.stopping = True
        assert tray_status.derive_state(d) == "transcribing"
        d.stopping = False
        d.transcribing = True
        assert tray_status.derive_state(d) == "transcribing"
        d.transcribing = False
        d.model_error = "boom"
        assert tray_status.derive_state(d) == "error"

    def test_language_label_precedence(self):
        import tray_status

        d = SimpleNamespace(
            _session_enforced_language="ru",
            config={"language": "en"},
        )
        assert tray_status.language_label(d) == ("RU", True)
        d._session_enforced_language = None
        assert tray_status.language_label(d) == ("EN", False)
        d.config["language"] = None
        assert tray_status.language_label(d) == ("AUTO", False)

    def test_tooltip_idle_includes_hotkey(self):
        import tray_status

        d = SimpleNamespace(
            config={"model": "base.en"},
            get_hotkey_name=lambda: "f12",
            model_error=None,
            _tray_error=None,
        )
        tip = tray_status.tooltip_for(d, "idle")
        assert "F12" in tip
        assert "idle" in tip.lower()
        # AppIndicator titles are latin-1
        tip.encode("latin-1")

    def test_compose_language_badge_draws_pixels(self):
        import tray_status
        from PIL import Image

        base = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        out = tray_status.compose_language_badge(base, "AUTO")
        assert out.size == (64, 64)
        # Badge should paint some opaque pixels where the chip is
        assert max(px[3] for px in out.getdata()) > 200

    def test_latin1_safe_replaces_non_latin1(self):
        import tray_status

        assert tray_status.latin1_safe("SoupaWhisper - idle") == "SoupaWhisper - idle"
        out = tray_status.latin1_safe("ok — dash")
        out.encode("latin-1")
        assert "—" not in out

    def test_tray_title_includes_language(self):
        import tray_status

        d = SimpleNamespace(
            config={"model": "base", "language": None, "tray_show_language": True},
            get_hotkey_name=lambda: "f12",
            model_error=None,
            _tray_error=None,
            _session_enforced_language="en",
        )
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        assert "[EN]" in tray._title_for("recording")
        d.config["tray_show_language"] = False
        assert "[EN]" not in tray._title_for("recording")

    def test_badge_hidden_when_tray_show_language_false(self):
        import tray_status

        d = SimpleNamespace(
            config={"tray_show_language": False, "language": None},
            _session_enforced_language="ru",
        )
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        assert tray._language_badge_text() == ""

    def test_refresh_skips_update_menu_when_signature_unchanged(self):
        import tray_status

        d = SimpleNamespace(
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            recording=False,
            stopping=False,
            transcribing=False,
            config={"model": "base", "language": None, "tray_show_language": True},
            get_hotkey_name=lambda: "f12",
            _session_enforced_language=None,
        )
        d.model_loaded.set()
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        tray._images = {name: MagicMock() for name in tray_status.STATE_NAMES}
        tray._last_state = "idle"
        tray._last_title = tray._title_for("idle")
        tray._last_badge = "AUTO"
        tray._last_menu_signature = tray._menu_signature()
        tray._icon = None
        icon = MagicMock()
        with patch.object(tray, "_image_for", return_value=MagicMock()):
            tray._refresh(icon)
        icon.update_menu.assert_not_called()

    def test_compose_language_badge_short_labels_match_auto_width(self):
        """EN/RU use the same centered chip as AUTO (same left/right edges)."""
        import tray_status
        from PIL import Image

        base = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        auto = tray_status.compose_language_badge(base, "AUTO")
        en = tray_status.compose_language_badge(base, "EN")

        def chip_bounds(img: Image.Image) -> tuple[int, int]:
            px = img.load()
            left, right = img.width, -1
            for x in range(img.width):
                for y in range(img.height):
                    if px[x, y][3] > 200 and px[x, y][0] < 40:
                        left = min(left, x)
                        right = max(right, x)
            return left, right

        assert chip_bounds(auto) == chip_bounds(en)
        left, right = chip_bounds(auto)
        # Horizontally centered; vertically at the bottom.
        assert abs((left + right) / 2 - 31.5) < 3
        px = auto.load()
        bottom_has_chip = any(
            px[x, auto.height - 1][3] > 200 and px[x, auto.height - 1][0] < 40
            for x in range(auto.width)
        )
        assert bottom_has_chip

    def test_clear_session_language_resets_tray_to_auto(self):
        import tray_status

        d = SimpleNamespace(
            _session_enforced_language="ru",
            config={"language": None},
        )
        assert tray_status.language_label(d) == ("RU", True)
        d._session_enforced_language = None
        assert tray_status.language_label(d) == ("AUTO", False)

    def test_toggle_menu_text_and_action_schedules_immediately(self):
        import tray_status

        calls = []
        d = SimpleNamespace(
            recording=False,
            stopping=False,
            transcribing=False,
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            config={"model": "base.en", "tray_show_language": False},
            start_recording=lambda: calls.append("start"),
            stop_recording=lambda: calls.append("stop"),
            _schedule_hotkey_action=lambda fn, name: calls.append((name, fn)),
        )
        d.model_loaded.set()
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        tray._icon = None
        tray._last_state = None
        tray._last_title = None
        tray._images = {}
        tray._images_16 = {}

        assert tray._toggle_menu_text() == "Start dictation"
        assert tray._toggle_enabled() is True
        tray._on_toggle_dictation(MagicMock(), None)
        assert calls[0][0] == "start_recording"

        d.recording = True
        calls.clear()
        assert tray._toggle_menu_text() == "Stop dictation"
        tray._on_toggle_dictation(MagicMock(), None)
        assert calls[0][0] == "stop_recording"

    def test_missing_assets_raises(self, tmp_path):
        import tray_status

        with pytest.raises(tray_status.TrayStartError):
            tray_status.load_state_images(tmp_path)

    def test_run_with_tray_false_skips_tray(self, mock_config, mock_whisper_model, mock_xdotool):
        config = dictate.load_config()
        config["tray_icon"] = False
        d = dictate.Dictation(config)
        d.model_loaded.set()
        calls = []

        def fake_loop():
            calls.append("supervisor")
            d.running = False

        with patch.object(d, "_run_supervisor_loop", side_effect=fake_loop):
            with patch.dict(sys.modules, {"tray_status": MagicMock()}):
                d.run()
        assert calls == ["supervisor"]

    def test_run_with_tray_true_exits_on_prepare_failure(
        self, mock_config, mock_whisper_model, mock_xdotool
    ):
        import tray_status

        config = dictate.load_config()
        config["tray_icon"] = True
        d = dictate.Dictation(config)
        d.model_loaded.set()

        with patch.object(tray_status.TrayStatus, "prepare", side_effect=tray_status.TrayStartError("no host")):
            with pytest.raises(SystemExit) as ei:
                d.run()
            assert ei.value.code == 1

    def test_compose_empty_badge_returns_base(self):
        import tray_status
        from PIL import Image

        base = Image.new("RGBA", (64, 64), (1, 2, 3, 255))
        assert tray_status.compose_language_badge(base, "") is base

    def test_prepare_builds_icon_with_mocked_pystray(self, tmp_path):
        import tray_status
        from PIL import Image

        assets = tmp_path / "tray"
        assets.mkdir()
        for name in tray_status.STATE_NAMES:
            Image.new("RGBA", (64, 64), (0, 0, 0, 255)).save(assets / f"{name}.png")

        d = SimpleNamespace(
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            recording=False,
            stopping=False,
            transcribing=False,
            config={"model": "base", "language": None, "tray_show_language": True},
            get_hotkey_name=lambda: "f12",
            _session_enforced_language=None,
            running=True,
        )
        d.model_loaded.set()

        fake_icon = MagicMock()
        fake_pystray = MagicMock()
        fake_pystray.Icon.return_value = fake_icon
        fake_menu = MagicMock()
        fake_menu.SEPARATOR = object()
        fake_pystray.Menu = fake_menu
        fake_pystray.MenuItem = MagicMock()

        with patch.dict(sys.modules, {"pystray": fake_pystray}):
            # Re-import Menu inside prepare from pystray — inject via import
            import types

            mod = types.ModuleType("pystray")
            mod.Icon = fake_pystray.Icon
            mod.Menu = fake_menu
            mod.MenuItem = MagicMock(return_value=MagicMock())
            with patch.dict(sys.modules, {"pystray": mod}):
                tray = tray_status.TrayStatus(d, assets_dir=assets)
                tray.prepare()
        assert tray._icon is fake_icon
        mod.Icon.assert_called_once()

    def test_language_menu_text_from_layout(self):
        import tray_status

        d = SimpleNamespace(
            _session_enforced_language="ru",
            config={"language": None},
        )
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        assert tray._language_menu_text() == "Language: RU (from layout)"
        d._session_enforced_language = None
        assert tray._language_menu_text() == "Language: AUTO"


class TestMacOSMenuBarImage:
    """macOS draws its own pixmap: pystray squashes any image to the bar thickness."""

    @staticmethod
    def _mic() -> "Image.Image":
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((22, 6, 42, 40), radius=10, fill=(0, 0, 0, 255))
        draw.rectangle((30, 40, 34, 56), fill=(0, 0, 0, 255))
        return img

    @staticmethod
    def _rows_with_ink(img) -> list[int]:
        px = img.load()
        return [
            y
            for y in range(img.height)
            if any(px[x, y][3] > 40 for x in range(img.width))
        ]

    def test_rendered_at_retina_scale(self):
        import tray_status

        img = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        assert img.height == 44

    def test_glyph_does_not_touch_bar_edges(self):
        """22pt edge-to-edge is why the icon reads as oversized; leave breathing room."""
        import tray_status

        img = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        rows = self._rows_with_ink(img)
        assert rows, "image is empty"
        assert min(rows) >= 2
        assert max(rows) <= img.height - 3

    def test_badge_sits_below_the_glyph(self):
        import tray_status
        from PIL import Image

        # A solid block fills its glyph box exactly, so any ink lower down is the chip.
        block = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
        img = tray_status.macos_menu_bar_image(block, "AUTO", "idle", thickness=22, scale=2)
        pad = round(44 * tray_status.MACOS_EDGE_PADDING_RATIO)
        glyph_bottom = pad + round(44 * tray_status.MACOS_GLYPH_RATIO) - 1
        assert max(self._rows_with_ink(img)) > glyph_bottom

    def test_short_labels_stay_centered(self):
        import tray_status

        auto = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        en = tray_status.macos_menu_bar_image(self._mic(), "EN", "idle", thickness=22, scale=2)
        assert auto.size == en.size

    def test_template_states_render_opaque_mask(self):
        """setTemplate_ keeps only alpha, so badge text must be ink, not a knocked-out chip."""
        import tray_status

        img = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        badge_band = img.crop((0, img.height // 2, img.width, img.height))
        assert max(px[3] for px in badge_band.getdata()) > 200

    @pytest.mark.parametrize("thickness,scale", [(22, 2), (24, 2), (37, 2), (22, 1), (22, 3)])
    def test_layout_scales_with_the_measured_bar(self, thickness, scale):
        """Bar thickness is not a constant across macOS releases and displays."""
        import tray_status

        img = tray_status.macos_menu_bar_image(
            self._mic(), "AUTO", "idle", thickness=thickness, scale=scale
        )
        height = round(thickness * scale)
        assert img.height == height

        rows = self._rows_with_ink(img)
        pad = max(1, round(height * tray_status.MACOS_EDGE_PADDING_RATIO))
        assert min(rows) >= pad, "glyph must not touch the top of the bar"
        assert max(rows) <= height - pad - 1, "chip must not touch the bottom of the bar"

        # The glyph keeps its share of the bar instead of a fixed point size.
        glyph_px = round(height * tray_status.MACOS_GLYPH_RATIO)
        assert max(self._rows_with_ink(img)) > pad + glyph_px - 1, "chip sits below the glyph"

    def test_colored_state_badge_uses_the_state_colour(self):
        """White-on-outline vanishes into a light menu bar; the chip tracks the glyph."""
        import tray_status

        img = tray_status.macos_menu_bar_image(
            self._mic(), "AUTO", "recording", thickness=22, scale=2
        )
        band = img.crop((0, img.height // 2, img.width, img.height))
        colours = {px[:3] for px in band.getdata() if px[3] > 200}
        assert tray_status.MACOS_BADGE_COLORS["recording"][:3] in colours

    def test_macos_install_image_hides_button_title(self):
        """The language chip lives in the pixmap; no second copy beside the icon."""
        import tray_status

        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = SimpleNamespace(config={"tray_show_language": True})
        tray._images = {name: self._mic() for name in tray_status.STATE_NAMES}
        button = MagicMock()
        status_item = MagicMock()
        status_item.button.return_value = button
        tray._icon = SimpleNamespace(_status_item=status_item)

        # AppKit only exists on macOS; the test runs everywhere, so stub the bridge.
        with patch.object(tray_status, "IS_MACOS", True):
            with patch.object(tray_status, "_ns_image_from", return_value="NSImage") as make:
                tray._install_macos_image("idle", "AUTO")

        make.assert_called_once()
        assert make.call_args.kwargs["template"] is True
        button.setImage_.assert_called_once_with("NSImage")
        button.setTitle_.assert_called_once_with("")

    def test_macos_colored_states_are_not_template_images(self):
        """Recording/error carry meaning in their colour; a template mask would erase it."""
        import tray_status

        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = SimpleNamespace(config={"tray_show_language": True})
        tray._images = {name: self._mic() for name in tray_status.STATE_NAMES}
        status_item = MagicMock()
        tray._icon = SimpleNamespace(_status_item=status_item)

        with patch.object(tray_status, "IS_MACOS", True):
            with patch.object(tray_status, "_ns_image_from", return_value="NSImage") as make:
                tray._install_macos_image("recording", "EN")

        assert make.call_args.kwargs["template"] is False

    def test_apply_language_badge_sets_no_title_on_macos(self):
        import tray_status

        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        button = MagicMock()
        status_item = MagicMock()
        status_item.button.return_value = button
        icon = MagicMock()
        icon._appindicator = None
        icon._status_item = status_item
        tray._icon = icon

        with patch.object(tray_status, "IS_MACOS", True):
            tray._apply_language_badge("AUTO")

        button.setTitle_.assert_not_called()

    def test_setup_installs_macos_image_before_polling(self):
        """Template mode was only applied on a state change, so the first icon was raw."""
        import tray_status

        d = SimpleNamespace(running=False)
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        tray._stop_poll = threading.Event()
        tray._stop_poll.set()
        tray._last_state = "idle"
        tray._last_badge = "AUTO"
        icon = MagicMock()

        with patch.object(tray_status, "IS_MACOS", True):
            with patch.object(tray, "_install_macos_image") as install:
                tray._setup(icon)

        install.assert_called_once_with("idle", "AUTO")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMeetingConfig:
    """Config for meeting recording mode."""

    def test_meeting_defaults_when_section_absent(self, mock_config):
        config = dictate.load_config()

        assert config["meeting_transcript_dir"].endswith("/meetings")
        assert config["meeting_max_duration_min"] == 120
        assert config["meeting_silence_stop_min"] == 10
        assert config["meeting_hotkey"] == ""

    def test_meeting_values_read_from_section(self, mock_config):
        mock_config.write_text(
            mock_config.read_text()
            + "\n[meeting]\n"
            "transcript_dir = ~/Documents/calls\n"
            "max_duration_min = 90\n"
            "silence_stop_min = 5\n"
            "hotkey = shift+f12\n"
        )

        config = dictate.load_config()

        assert config["meeting_transcript_dir"].endswith("/Documents/calls")
        assert "~" not in config["meeting_transcript_dir"]
        assert config["meeting_max_duration_min"] == 90
        assert config["meeting_silence_stop_min"] == 5
        assert config["meeting_hotkey"] == "shift+f12"


class TestMeetingRunSettingsConfig:
    """Speech-run VAD knobs, tunable without editing meeting.py."""

    def test_defaults_match_the_meeting_module(self, mock_config):
        config = dictate.load_config()

        assert config["meeting_run_min_silence_ms"] == dictate.meeting_mod.RUN_MIN_SILENCE_MS
        assert config["meeting_run_pad_ms"] == dictate.meeting_mod.RUN_PAD_MS
        assert config["meeting_run_vad_threshold"] == dictate.meeting_mod.RUN_VAD_THRESHOLD
        assert config["meeting_run_min_speech_ms"] == dictate.meeting_mod.RUN_MIN_SPEECH_MS
        assert config["meeting_run_max_seconds"] == dictate.meeting_mod.LANGUAGE_WINDOW_S
        assert config["meeting_min_avg_logprob"] == dictate.meeting_mod.MIN_AVG_LOGPROB

    def test_values_read_from_section(self, mock_config):
        mock_config.write_text(
            mock_config.read_text()
            + "\n[meeting]\n"
            "run_min_silence_ms = 1500\n"
            "run_pad_ms = 50\n"
            "run_vad_threshold = 0.7\n"
            "run_min_speech_ms = 250\n"
            "run_max_seconds = 20\n"
            "min_avg_logprob = -2.0\n"
        )

        config = dictate.load_config()

        assert config["meeting_run_min_silence_ms"] == 1500
        assert config["meeting_run_pad_ms"] == 50
        assert config["meeting_run_vad_threshold"] == 0.7
        assert config["meeting_run_min_speech_ms"] == 250
        assert config["meeting_run_max_seconds"] == 20.0
        assert config["meeting_min_avg_logprob"] == -2.0

    def test_config_builds_the_settings_meeting_uses(self, mock_config):
        mock_config.write_text(
            mock_config.read_text() + "\n[meeting]\nrun_min_silence_ms = 1500\n"
        )

        settings = dictate.meeting_mod.RunSettings.from_config(dictate.load_config())

        assert settings.min_silence_ms == 1500
        assert settings.pad_ms == dictate.meeting_mod.RUN_PAD_MS


class TestMeetingGlossaryConfig:
    """Meeting mode primes Whisper with the same glossary dictation uses, unless
    [meeting] custom_terms overrides it -- interview vocabulary differs from the
    terms you dictate every day."""

    def test_meeting_has_no_glossary_key_of_its_own(self, mock_config):
        # One glossary for every mode: [behavior] custom_terms.
        mock_config.write_text(
            mock_config.read_text() + "\n[meeting]\ncustom_terms = Airflow\n"
        )

        config = dictate.load_config()

        assert "meeting_custom_terms" not in config

    def test_block_caps_have_the_tuned_defaults(self, mock_config):
        config = dictate.load_config()

        assert config["meeting_me_max_block_seconds"] == dictate.meeting_mod.ME_MAX_BLOCK_S
        assert config["meeting_them_max_block_seconds"] == dictate.meeting_mod.THEM_MAX_BLOCK_S

    def test_block_caps_read_from_section(self, mock_config):
        mock_config.write_text(
            mock_config.read_text()
            + "\n[meeting]\nme_max_block_seconds = 45\nthem_max_block_seconds = 15\n"
        )

        config = dictate.load_config()

        assert config["meeting_me_max_block_seconds"] == 45.0
        assert config["meeting_them_max_block_seconds"] == 15.0


class TestParseHotkeySpec:
    """Meeting mode needs a modifier combo; dictation's hotkey is a single key."""

    def test_bare_key_has_no_modifiers(self):
        assert dictate.parse_hotkey_spec("f12") == (frozenset(), "f12")

    def test_shift_modifier_is_split_off(self):
        assert dictate.parse_hotkey_spec("shift+f12") == (frozenset({"shift"}), "f12")

    def test_is_case_and_space_insensitive(self):
        assert dictate.parse_hotkey_spec(" Shift + F12 ") == (frozenset({"shift"}), "f12")

    def test_supports_several_modifiers(self):
        assert dictate.parse_hotkey_spec("ctrl+shift+f12") == (
            frozenset({"ctrl", "shift"}),
            "f12",
        )

    def test_empty_spec_means_no_hotkey(self):
        assert dictate.parse_hotkey_spec("") == (frozenset(), "")


class TestTrayMeetingState:
    """Meeting mode needs its own icon so a live recording is unmistakable."""

    def _dictation(self, **kw):
        base = dict(model_error=None, _tray_error=None, recording=False,
                    stopping=False, transcribing=False, meeting_active=False)
        base["model_loaded"] = SimpleNamespace(is_set=lambda: True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_meeting_state_is_reported_while_recording_a_meeting(self):
        import tray_status

        assert tray_status.derive_state(self._dictation(meeting_active=True)) == "meeting"

    def test_meeting_outranks_dictation_recording(self):
        import tray_status

        d = self._dictation(meeting_active=True, recording=True)

        assert tray_status.derive_state(d) == "meeting"

    def test_errors_still_outrank_meeting(self):
        import tray_status

        d = self._dictation(meeting_active=True, _tray_error="disk full")

        assert tray_status.derive_state(d) == "error"

    def test_meeting_is_a_known_state_with_an_icon(self):
        import tray_status

        assert "meeting" in tray_status.STATE_NAMES
        assert "meeting" in tray_status.load_state_images()


class NoBackgroundModelLoad:
    """Dictation.__init__ spawns a thread building a real WhisperModel (network +
    disk). Tests supply their own model, so stub the loader out."""

    @pytest.fixture(autouse=True)
    def _no_model_load(self, monkeypatch):
        monkeypatch.setattr(dictate.Dictation, "_load_model", lambda self: None)


class TestDictationDisabledDuringMeeting(NoBackgroundModelLoad):
    """Meeting mode must neuter the dictation hotkey.

    If it fired mid-call, transcribed text would be typed into whatever window has
    focus -- the Zoom chat, a shared doc, anything.
    """

    def _dictation(self):
        d = dictate.Dictation(dictate.load_config())
        d.model_loaded.set()
        return d

    def test_hotkey_does_not_start_dictation_while_a_meeting_records(self, mock_config):
        d = self._dictation()
        scheduled = []
        d._schedule_hotkey_action = lambda fn, name: scheduled.append(name)
        d.meeting_active = True

        d.on_press(d.hotkey)

        assert scheduled == []

    def test_hotkey_does_not_stop_dictation_while_a_meeting_records(self, mock_config):
        d = self._dictation()
        scheduled = []
        d._schedule_hotkey_action = lambda fn, name: scheduled.append(name)
        d.meeting_active = True

        d.on_release(d.hotkey)

        assert scheduled == []

    def test_hotkey_works_normally_when_no_meeting_is_recording(self, mock_config):
        d = self._dictation()
        scheduled = []
        d._schedule_hotkey_action = lambda fn, name: scheduled.append(name)

        d.on_press(d.hotkey)

        assert scheduled == ["start_recording"]


class FakeMeetingProc:
    def __init__(self, cmd, **kw): self.cmd = cmd
    def terminate(self): pass
    def wait(self, timeout=None): return 0
    def poll(self): return None


class TestDictationMeetingControl(NoBackgroundModelLoad):
    def _dictation(self, tmp_path):
        d = dictate.Dictation(dictate.load_config())
        d.model_loaded.set()
        d.config["meeting_transcript_dir"] = str(tmp_path / "transcripts")
        d._meeting_tmp_root = tmp_path / "tmp"
        d._meeting_spawn = FakeMeetingProc
        return d

    def test_starting_a_meeting_suspends_dictation(self, mock_config, tmp_path):
        d = self._dictation(tmp_path)

        d.start_meeting()

        assert d.meeting_active is True
        assert d._dictation_suspended() is True

    def test_stopping_a_meeting_resumes_dictation(self, mock_config, tmp_path):
        d = self._dictation(tmp_path)
        d.start_meeting()

        d.stop_meeting(transcribe=False)

        assert d.meeting_active is False
        assert d._dictation_suspended() is False

    def test_starting_twice_does_not_spawn_a_second_meeting(self, mock_config, tmp_path):
        d = self._dictation(tmp_path)
        d.start_meeting()
        first = d.meeting.session_dir

        d.start_meeting()

        assert d.meeting.session_dir == first


class TestTrayMeetingMenu:
    def _tray(self, **kw):
        import tray_status
        base = dict(model_error=None, _tray_error=None, recording=False, stopping=False,
                    transcribing=False, meeting_active=False, config={})
        base["model_loaded"] = SimpleNamespace(is_set=lambda: True)
        base.update(kw)
        return tray_status, tray_status.TrayStatus(SimpleNamespace(**base))

    def test_menu_offers_to_start_a_meeting_when_idle(self):
        _, tray = self._tray()

        assert tray._meeting_menu_text() == "Start meeting recording"

    def test_menu_offers_to_stop_a_meeting_while_recording(self):
        _, tray = self._tray(meeting_active=True)

        assert tray._meeting_menu_text() == "Stop meeting recording"

    def test_disk_line_shows_size_and_count_of_scratch_audio(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "mic.wav").write_bytes(b"x" * 2_000_000)
        _, tray = self._tray()
        tray.dictation._meeting_tmp_root = tmp_path

        assert tray._meeting_disk_text() == "Meeting audio: 2 MB (1 file)"

    def test_disk_line_reads_clean_when_there_is_no_scratch_audio(self, tmp_path):
        _, tray = self._tray()
        tray.dictation._meeting_tmp_root = tmp_path

        assert tray._meeting_disk_text() == "Meeting audio: none"


class TestScratchWavPath:
    """All WAVs this app writes live under one root the tray can size and wipe."""

    def test_debug_recordings_go_under_the_shared_scratch_root(self, tmp_path):
        import meeting

        path = dictate.scratch_wav_path("recording_20260101_120000.wav", root=tmp_path)

        assert path.parent == tmp_path / "recordings"
        assert path.parent.is_dir()

    def test_default_root_is_the_shared_soupawhisper_tmp_dir(self):
        import meeting

        assert str(meeting.TMP_ROOT).endswith("/soupawhisper-streaming")


class TestModifierName:
    def test_recognises_left_and_right_shift_as_shift(self):
        assert dictate.modifier_name(SimpleNamespace(name="shift_l")) == "shift"
        assert dictate.modifier_name(SimpleNamespace(name="shift_r")) == "shift"
        assert dictate.modifier_name(SimpleNamespace(name="shift")) == "shift"

    def test_recognises_ctrl_alt_and_cmd(self):
        assert dictate.modifier_name(SimpleNamespace(name="ctrl_l")) == "ctrl"
        assert dictate.modifier_name(SimpleNamespace(name="alt_gr")) == "alt"
        assert dictate.modifier_name(SimpleNamespace(name="cmd_r")) == "cmd"

    def test_returns_none_for_a_normal_key(self):
        assert dictate.modifier_name(SimpleNamespace(name="f12")) is None

    def test_returns_none_for_a_character_key_without_a_name(self):
        assert dictate.modifier_name(SimpleNamespace(char="a")) is None


class TestMeetingHotkey(NoBackgroundModelLoad):
    def _dictation(self, spec="shift+f12", live_mods=()):
        cfg = dictate.load_config()
        cfg["meeting_hotkey"] = spec
        d = dictate.Dictation(cfg)
        d.model_loaded.set()
        d._modifier_source = lambda: set(live_mods)
        d.scheduled = []
        d._schedule_hotkey_action = lambda fn, name: d.scheduled.append(name)
        return d

    def test_combo_toggles_meeting_mode(self, mock_config):
        d = self._dictation(live_mods={"shift"})

        d.on_press(d._meeting_key)

        assert d.scheduled == ["toggle_meeting"]

    def test_bare_key_without_the_modifier_does_not_toggle_meeting(self, mock_config):
        d = self._dictation()

        d.on_press(d._meeting_key)

        assert "toggle_meeting" not in d.scheduled

    def test_combo_still_works_while_a_meeting_is_recording(self, mock_config):
        d = self._dictation(live_mods={"shift"})
        d.meeting_active = True

        d.on_press(d._meeting_key)

        assert d.scheduled == ["toggle_meeting"]

    def test_no_modifier_held_means_no_combo(self, mock_config):
        d = self._dictation(live_mods=set())

        d.on_press(d._meeting_key)

        assert "toggle_meeting" not in d.scheduled

    def test_no_meeting_hotkey_configured_means_no_combo(self, mock_config):
        d = self._dictation(spec="")

        assert d._meeting_key is None


class TestFileTranscriptionLanguage(NoBackgroundModelLoad):
    """Dictation locks one language per file; meeting mode does NOT (see
    meeting.transcribe_windowed), because a call can switch languages."""

    def _dictation_with_wav(self, tmp_path):
        import wave as w
        d = dictate.Dictation(dictate.load_config())
        d.model = MockWhisperModel()
        d.model_loaded.set()
        wav = tmp_path / "mic.wav"
        with w.open(str(wav), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(b"\0" * 32000)
        return d, wav

    def test_default_still_locks_one_language_for_dictation(self, mock_config, tmp_path):
        d, wav = self._dictation_with_wav(tmp_path)

        d.transcribe_file_to_output(str(wav), str(tmp_path / "out.srt"))

        _, kwargs = d.model.transcribe_calls[-1]
        assert not kwargs.get("multilingual")
        assert kwargs["language"] is not None


class TestFileModeSkipsPynput(NoBackgroundModelLoad):
    """--file / make transcribe must work without an X display (no pynput import)."""

    def test_interactive_false_does_not_import_pynput(self, mock_config, monkeypatch):
        def boom():
            raise AssertionError("pynput must not be imported in file transcription mode")

        monkeypatch.setattr(dictate, "_import_keyboard", boom)
        monkeypatch.setattr(dictate, "get_hotkey", boom)

        d = dictate.Dictation(dictate.load_config(), interactive=False)

        assert d.hotkey is None
        assert d.typer is None
        assert d._interactive is False


class TestMeetingModel(NoBackgroundModelLoad):
    """Meeting transcription gets its own model instance.

    Sharing dictation's model would serialise the two and make dictation wait for a
    long meeting transcription; a separate instance also lets us cap its threads.
    """

    def test_defaults_to_half_the_cores(self, mock_config):
        config = dictate.load_config()

        assert config["meeting_cpu_threads"] == max(1, (os.cpu_count() or 2) // 2)

    def test_thread_count_is_configurable(self, mock_config):
        mock_config.write_text(
            mock_config.read_text() + "\n[meeting]\ncpu_threads = 3\n"
        )

        assert dictate.load_config()["meeting_cpu_threads"] == 3

    def test_meeting_model_is_not_the_dictation_model(self, mock_config, monkeypatch):
        built = {}

        def fake_model(name, device=None, compute_type=None, cpu_threads=None):
            built["cpu_threads"] = cpu_threads
            return MockWhisperModel()

        monkeypatch.setattr(dictate, "WhisperModel", fake_model)
        d = dictate.Dictation(dictate.load_config())
        d.model = MockWhisperModel()

        meeting_model = d.meeting_model()

        assert meeting_model is not d.model
        assert built["cpu_threads"] == max(1, (os.cpu_count() or 2) // 2)

    def test_meeting_model_is_built_once_and_reused(self, mock_config, monkeypatch):
        calls = []
        monkeypatch.setattr(
            dictate, "WhisperModel",
            lambda *a, **kw: (calls.append(1), MockWhisperModel())[1],
        )
        d = dictate.Dictation(dictate.load_config())

        d.meeting_model(); d.meeting_model()

        assert len(calls) == 1


class TestTrayMeetingTranscribingLabel:
    """Dictation is usable during meeting transcription, so the icon must stay idle -
    but the menu should still say the job is running."""

    def _tray(self, **kw):
        import tray_status
        base = dict(model_error=None, _tray_error=None, recording=False, stopping=False,
                    transcribing=False, meeting_active=False, meeting_transcribing=False,
                    config={})
        base["model_loaded"] = SimpleNamespace(is_set=lambda: True)
        base.update(kw)
        return tray_status, tray_status.TrayStatus(SimpleNamespace(**base))

    def test_menu_reports_transcription_in_progress(self):
        _, tray = self._tray(meeting_transcribing=True)

        assert tray._meeting_menu_text() == "Transcribing meeting…"

    def test_menu_item_is_disabled_while_transcribing(self):
        _, tray = self._tray(meeting_transcribing=True)

        assert tray._meeting_enabled() is False

    def test_icon_stays_idle_so_dictation_looks_available(self):
        ts, tray = self._tray(meeting_transcribing=True)

        assert ts.derive_state(tray.dictation) == "idle"

    def test_menu_returns_to_start_when_transcription_finishes(self):
        _, tray = self._tray(meeting_transcribing=False)

        assert tray._meeting_menu_text() == "Start meeting recording"
        assert tray._meeting_enabled() is True


class TestMeetingHotkeyUsesLiveModifiers(NoBackgroundModelLoad):
    """Regression: X11 reports Shift's RELEASE as a different keysym (65032, an ISO
    group-switch) when layout switching is bound to Shift, so press/release tracking
    desynchronises permanently. Modifier state must be read live at trigger time."""

    def _dictation(self, live_mods):
        cfg = dictate.load_config()
        cfg["key"] = "f12"          # user's real setup: F12 dictates, Shift+F12 meets
        cfg["meeting_hotkey"] = "shift+f12"
        d = dictate.Dictation(cfg)
        d.model_loaded.set()
        d._modifier_source = lambda: set(live_mods)
        d.scheduled = []
        d._schedule_hotkey_action = lambda fn, name: d.scheduled.append(name)
        return d

    def test_bare_key_does_not_toggle_meeting_when_shift_is_stale(self, mock_config):
        # The reported bug: F12 alone started meeting mode because 'shift' was stuck.
        d = self._dictation(live_mods=set())
        d._held_modifiers = {"shift"}  # stale, never cleared by a release event

        d.on_press(d._meeting_key)

        assert "toggle_meeting" not in d.scheduled
        assert d.scheduled == ["start_recording"]

    def test_combo_toggles_meeting_when_shift_is_really_held(self, mock_config):
        d = self._dictation(live_mods={"shift"})

        d.on_press(d._meeting_key)

        assert d.scheduled == ["toggle_meeting"]

    def test_extra_stale_modifiers_do_not_block_the_combo(self, mock_config):
        # The other reported bug: Shift+F12 fell through to dictation because a stale
        # ctrl/alt made the tracked set unequal to {"shift"}.
        d = self._dictation(live_mods={"shift"})
        d._held_modifiers = {"shift", "ctrl", "alt"}

        d.on_press(d._meeting_key)

        assert d.scheduled == ["toggle_meeting"]


class TestStreamingListenerTracksReleases:
    def test_streaming_listener_registers_on_release(self, mock_config, monkeypatch):
        monkeypatch.setattr(dictate.Dictation, "_load_model", lambda self: None)
        d = dictate.StreamingDictation(dictate.load_config())

        d._create_hotkey_listener()

        kwargs = dictate._import_keyboard().Listener.call_args.kwargs
        assert "on_release" in kwargs


class TestTrayShowsTranscriptionProgress:
    def _tray(self, **kw):
        import tray_status
        base = dict(model_error=None, _tray_error=None, recording=False, stopping=False,
                    transcribing=False, meeting_active=False, meeting_transcribing=False,
                    meeting_progress=None, config={})
        base["model_loaded"] = SimpleNamespace(is_set=lambda: True)
        base.update(kw)
        return tray_status, tray_status.TrayStatus(SimpleNamespace(**base))

    def test_shows_percentage_while_transcribing(self):
        _, tray = self._tray(meeting_transcribing=True, meeting_progress="mic 45%")

        assert tray._meeting_menu_text() == "Transcribing meeting… mic 45%"

    def test_falls_back_to_plain_label_before_the_first_window(self):
        _, tray = self._tray(meeting_transcribing=True, meeting_progress=None)

        assert tray._meeting_menu_text() == "Transcribing meeting…"


class TestMeetingKeepAudioConfig:
    def test_audio_is_deleted_by_default(self, mock_config):
        assert dictate.load_config()["meeting_keep_audio"] is False

    def test_can_be_enabled_for_debugging(self, mock_config):
        mock_config.write_text(
            mock_config.read_text() + "\n[meeting]\nkeep_audio = true\n"
        )

        assert dictate.load_config()["meeting_keep_audio"] is True


class TestSharedGlossaryAndRejectPhrases(NoBackgroundModelLoad):
    """Both are [behavior] settings that every mode reads off the base Dictation."""

    def test_glossary_is_built_on_the_base_dictation(self, mock_config):
        mock_config.write_text(
            mock_config.read_text().replace(
                "[behavior]", "[behavior]\ncustom_terms = Redis, Kubeflow"
            )
        )

        d = dictate.Dictation(dictate.load_config())

        assert "Redis" in d.custom_terms_kwargs["hotwords"]
        assert "Kubeflow" in d.custom_terms_kwargs["initial_prompt"]

    def test_empty_glossary_yields_no_kwargs(self, mock_config):
        d = dictate.Dictation(dictate.load_config())

        assert d.custom_terms_kwargs == {}

    def test_reject_predicate_is_available_without_streaming(self, mock_config):
        mock_config.write_text(
            mock_config.read_text().replace(
                "[behavior]", "[behavior]\nreject_phrases = thank you, um"
            )
        )

        d = dictate.Dictation(dictate.load_config())

        assert d.should_reject_text("Thank you!!!") is True
        assert d.should_reject_text("thank you for that") is False


class TestStreamingTextContext:
    def make_worker(self, texts, languages=None, allowlist=None, language='en'):
        d = dictate.StreamingDictation.__new__(dictate.StreamingDictation)
        d.config = {'language': language, 'language_allowlist': allowlist}
        d.vad_sample_rate = 16000
        d.recording = d.stopping = d.file_mode = False
        d.transcription_queue = queue.Queue()
        d.typing_queue = queue.Queue()
        d.accumulated_text = ''
        d.custom_terms_kwargs = dictate._build_custom_terms_kwargs(['ELK', 'GitHub Actions'])
        d._check_valid_audio_input = lambda audio: False
        d._maybe_enforced_language_for_field = lambda: None
        d.should_reject_text = lambda text: text == 'thanks'
        responses = iter(texts)
        detected = iter(languages or [])
        d.model = MagicMock()
        d.model.detect_language.side_effect = lambda *a, **kw: next(detected)

        def transcribe(audio, **kw):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return [SimpleNamespace(text=response)], SimpleNamespace(language=kw.get('language') or 'en')

        d.model.transcribe.side_effect = transcribe
        return d

    def run_chunks(self, d, spans):
        for start, end in spans:
            audio = np.full(1600, 5000, dtype=np.int16)
            d.transcription_queue.put(dictate._StreamingChunk(audio, start, end))
        d.transcription_queue.put(None)
        d._transcription_worker()
        return [call.kwargs for call in d.model.transcribe.call_args_list]

    def test_next_chunk_receives_context_without_retyping_prior_output(self):
        d = self.make_worker(['We use ELK.', 'And GitHub Actions.'])
        glossary = dict(d.custom_terms_kwargs)
        calls = self.run_chunks(d, [(0, 1), (2, 3)])

        assert calls[0]['initial_prompt'] == glossary['initial_prompt']
        assert calls[1]['initial_prompt'] == glossary['initial_prompt'] + '\nWe use ELK.'
        assert calls[1]['hotwords'] == glossary['hotwords']
        assert d.custom_terms_kwargs == glossary
        assert d.accumulated_text == 'We use ELK. And GitHub Actions.'
        assert list(d.typing_queue.queue) == [('We use ELK.', 1), (' And GitHub Actions.', 2)]
        assert [len(c.args[0]) for c in d.model.transcribe.call_args_list] == [1600, 1600]

    @pytest.mark.parametrize('gap,keeps_context', [(4.99, True), (5, False), (8, False)])
    def test_silence_reset_uses_speech_gap_including_exact_five_seconds(self, gap, keeps_context):
        d = self.make_worker(['Old sentence.', 'Correction.'])
        calls = self.run_chunks(d, [(0, 1), (1 + gap, 2 + gap)])
        assert ('Old sentence.' in calls[1]['initial_prompt']) is keeps_context
        assert calls[1]['hotwords'] == d.custom_terms_kwargs['hotwords']
        assert d.accumulated_text == 'Old sentence. Correction.'

    def test_slow_decoder_does_not_reset_continuous_speech(self, monkeypatch):
        # Simulate inference taking 100 seconds per chunk, with queued speech only
        # one second apart. Never use these completion times as silence evidence.
        ticks = iter([0, 100, 101, 201])
        monkeypatch.setattr(dictate, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
        d = self.make_worker(['First.', 'Second.'])
        calls = self.run_chunks(d, [(0, 1), (2, 3)])
        assert calls[1]['initial_prompt'].endswith('\nFirst.')

    def test_recent_text_accumulates_but_is_limited_to_fifty_words(self):
        first = ' '.join(f'a{i}' for i in range(40))
        second = ' '.join(f'b{i}' for i in range(30))
        d = self.make_worker([first, second, 'Next.'])
        calls = self.run_chunks(d, [(0, 1), (2, 3), (4, 5)])
        tail = calls[2]['initial_prompt'].split('\n')[1]
        assert tail.split() == (first + ' ' + second).split()[-50:]

    def test_context_also_works_without_a_glossary(self):
        d = self.make_worker(['First.', 'Second.'])
        d.custom_terms_kwargs = {}
        calls = self.run_chunks(d, [(0, 1), (2, 3)])
        assert 'initial_prompt' not in calls[0]
        assert calls[1]['initial_prompt'] == 'First.'

    def test_new_worker_session_starts_with_only_the_glossary(self):
        d = self.make_worker(['First session.', 'New session.'])
        self.run_chunks(d, [(0, 1)])
        d.accumulated_text = ''
        calls = self.run_chunks(d, [(0, 1)])
        assert calls[1]['initial_prompt'] == d.custom_terms_kwargs['initial_prompt']

    @pytest.mark.parametrize('languages', [('en', 'ru', 'ru'), ('ru', 'en', 'en')])
    def test_selected_language_switch_clears_context_before_decoding(self, languages):
        scores = [(lang, .9, [(lang, .9)]) for lang in languages]
        d = self.make_worker(['First.', 'Second.', 'Third.'], scores,
                             allowlist=['en', 'ru'], language=None)
        calls = self.run_chunks(d, [(0, 1), (2, 3), (4, 5)])
        assert calls[1]['initial_prompt'] == d.custom_terms_kwargs['initial_prompt']
        assert calls[2]['initial_prompt'].endswith('\nSecond.')
        assert [c['language'] for c in calls] == list(languages)

    def test_unrestricted_auto_resolves_language_before_using_context(self):
        scores = [('ru', .9, [('ru', .9)]), ('ru', .9, [('ru', .9)])]
        d = self.make_worker(['English.', 'Russian.', 'More Russian.'], scores, language=None)
        calls = self.run_chunks(d, [(0, 1), (2, 3), (4, 5)])
        # First decode auto-detects English; subsequent calls preflight language.
        assert calls[0]['language'] is None
        assert calls[1]['language'] == 'ru'
        assert calls[1]['initial_prompt'] == d.custom_terms_kwargs['initial_prompt']
        assert calls[2]['initial_prompt'].endswith('\nRussian.')
        options = d.model.detect_language.call_args.kwargs['vad_parameters']
        assert isinstance(options, dictate.VadOptions)
        assert options.threshold == .33

    def test_failed_auto_language_preflight_clears_context_and_keeps_decoding(self):
        d = self.make_worker(['First.', 'Second.'], language=None)
        d.model.detect_language.side_effect = RuntimeError('detector failed')
        calls = self.run_chunks(d, [(0, 1), (2, 3)])
        assert calls[1]['initial_prompt'] == d.custom_terms_kwargs['initial_prompt']
        assert calls[1]['language'] is None
        assert d.accumulated_text == 'First. Second.'

    def test_fixed_language_does_not_add_detection_calls(self):
        d = self.make_worker(['First.', 'Second.'])
        self.run_chunks(d, [(0, 1), (2, 3)])
        d.model.detect_language.assert_not_called()

    def test_layout_enforcement_still_takes_priority(self):
        d = self.make_worker(['First.', 'Second.'], language=None, allowlist=['en', 'ru'])
        d._maybe_enforced_language_for_field = lambda: 'ru'
        calls = self.run_chunks(d, [(0, 1), (2, 3)])
        assert [c['language'] for c in calls] == ['ru', 'ru']
        assert calls[1]['initial_prompt'].endswith('\nFirst.')
        d.model.detect_language.assert_not_called()

    def test_rejected_speech_is_not_prompted_but_is_not_counted_as_silence(self):
        d = self.make_worker(['First.', 'thanks', 'Second.'])
        calls = self.run_chunks(d, [(0, 1), (5, 6), (10, 11)])
        assert calls[2]['initial_prompt'].endswith('\nFirst.')
        assert 'thanks' not in calls[2]['initial_prompt']
        assert d.accumulated_text == 'First. Second.'

    @pytest.mark.parametrize('failed', ['', RuntimeError('decode failed')])
    def test_empty_or_failed_decode_clears_untrusted_context(self, failed):
        d = self.make_worker(['First.', failed, 'Second.'])
        calls = self.run_chunks(d, [(0, 1), (2, 3), (4, 5)])
        assert calls[2]['initial_prompt'] == d.custom_terms_kwargs['initial_prompt']
        assert d.accumulated_text == 'First. Second.'


class TestStreamingSpeechTimestamps:
    def make_capture(self):
        d = dictate.StreamingDictation.__new__(dictate.StreamingDictation)
        d.vad_sample_rate = 16000
        d.vad_frame_size = 480
        d.vad_min_speech_chunks = 0
        d.vad_silence_threshold_seconds = .06
        d.config = {'save_recordings': False}
        d.file_mode = False
        d.transcription_queue = queue.Queue()
        d.vad = MagicMock()
        d._reset_speech_mode()
        return d

    @pytest.mark.parametrize('forced', [False, True])
    def test_queued_bounds_exclude_trailing_silence_and_include_first_speech(self, forced):
        d = self.make_capture()
        frame = np.full(480, 5000, dtype=np.int16)
        d.vad.is_speech.return_value = True
        d._process_audio_chunk(frame, 1.0, .03)
        d.vad.is_speech.return_value = False
        d._process_audio_chunk(frame, 1.03, .03)
        if forced:
            d._finalize_segment()
        else:
            d._process_audio_chunk(frame, 1.06, .03)
        chunk = d.transcription_queue.get_nowait()
        assert chunk.speech_start == 1.0
        assert chunk.speech_end == pytest.approx(1.03)
        assert len(chunk.audio) == 960

    def test_capture_timestamps_follow_samples_even_if_worker_scheduling_changes(self):
        d = self.make_capture()
        d.recording = True
        d.audio_stream = MagicMock()
        reads = []

        def read(*args, **kw):
            reads.append(None)
            if len(reads) == 2:
                d.recording = False
            return np.full(480, 5000, dtype=np.int16).tobytes()

        d.audio_stream.read.side_effect = read
        d._process_audio_chunk = MagicMock()
        d._finalize_segment = MagicMock()
        d._continuous_audio_stream_worker(480)
        assert [c.args[1] for c in d._process_audio_chunk.call_args_list] == [0, .03]
        assert [c.args[2] for c in d._process_audio_chunk.call_args_list] == [.03, .03]


class TestStreamingContextConfiguration:
    def test_defaults_preserve_the_original_context_behavior(self, mock_config):
        config = dictate.load_config()
        assert config['streaming_context_words'] == 50
        assert config['streaming_context_reset_seconds'] == 5.0

    def test_reads_both_streaming_overrides(self, mock_config):
        mock_config.write_text(mock_config.read_text().replace(
            '[streaming]', '[streaming]\ncontext_words = 12\ncontext_reset_seconds = 2.5'
        ))
        config = dictate.load_config()
        assert config['streaming_context_words'] == 12
        assert config['streaming_context_reset_seconds'] == 2.5

    @pytest.mark.parametrize('key,value', [
        ('context_words', '-1'), ('context_words', '1.5'),
        ('context_reset_seconds', '-1'), ('context_reset_seconds', 'nan'),
        ('context_reset_seconds', 'inf'),
    ])
    def test_invalid_values_are_rejected(self, mock_config, key, value):
        mock_config.write_text(mock_config.read_text().replace(
            '[streaming]', f'[streaming]\n{key} = {value}'
        ))
        with pytest.raises(ValueError):
            dictate.load_config()

    def test_custom_word_limit_is_used_by_the_worker(self):
        helper = TestStreamingTextContext()
        d = helper.make_worker(['one two three four', 'five six', 'Next.'])
        d.config['streaming_context_words'] = 3
        calls = helper.run_chunks(d, [(0, 1), (2, 3), (4, 5)])
        assert calls[1]['initial_prompt'].endswith('\ntwo three four')
        assert calls[2]['initial_prompt'].endswith('\nfour five six')

    def test_zero_words_disables_history_and_preserves_glossary(self, mock_config):
        mock_config.write_text(mock_config.read_text().replace(
            '[streaming]', '[streaming]\ncontext_words = 0'
        ))
        helper = TestStreamingTextContext()
        d = helper.make_worker(['First.', 'Second.'])
        d.config.update(dictate.load_config())
        calls = helper.run_chunks(d, [(0, 1), (2, 3)])
        assert [c['initial_prompt'] for c in calls] == [d.custom_terms_kwargs['initial_prompt']] * 2
        assert d.accumulated_text == 'First. Second.'

    @pytest.mark.parametrize('reset_seconds,keeps_context', [(2, False), (10, True), (0, False)])
    def test_custom_reset_interval_is_used_by_the_worker(self, mock_config, reset_seconds, keeps_context):
        mock_config.write_text(mock_config.read_text().replace(
            '[streaming]', f'[streaming]\ncontext_reset_seconds = {reset_seconds}'
        ))
        helper = TestStreamingTextContext()
        d = helper.make_worker(['First.', 'Second.'])
        d.config.update(dictate.load_config())
        calls = helper.run_chunks(d, [(0, 1), (4, 5)])
        assert ('First.' in calls[1]['initial_prompt']) is keeps_context
