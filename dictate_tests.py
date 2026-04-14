#!/usr/bin/env python3
"""
Tests for SoupaWhisper dictate.py
"""

import pytest
import numpy as np
import queue
from unittest.mock import MagicMock, patch
from typing import Any
from types import SimpleNamespace

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
        typer = dictate.Typer(delay_ms=20, start_delay_ms=100)
        assert typer.delay_ms == 20
        assert typer.start_delay_ms == 100
        assert typer.enabled is True
    
    def test_typer_type_rewrite_append(self, mock_xdotool):
        """Test typing text (append mode with previous_length=0)."""
        typer = dictate.Typer()
        typer.type_rewrite("hello world", 0)
        assert mock_xdotool._mock_run.called
    
    def test_typer_type_rewrite_incremental(self, mock_xdotool):
        """Test incremental typing using type_rewrite with previous_length=0."""
        typer = dictate.Typer()
        # Simulate incremental: calculate suffix and type with previous_length=0
        previous_text = "hello"
        new_text = "hello world"
        suffix = new_text[len(previous_text):]
        typer.type_rewrite(suffix, 0)
        assert mock_xdotool._mock_run.called
    
    def test_typer_type_rewrite_correction(self, mock_xdotool):
        """Test rewrite typing with character removal."""
        typer = dictate.Typer()
        typer.type_rewrite("new text", 5)
        assert mock_xdotool._mock_run.called


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




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
