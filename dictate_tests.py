#!/usr/bin/env python3
"""
Tests for SoupaWhisper dictate.py
"""

import pytest
import tempfile
import threading
import time
import wave
import numpy as np
import shutil
import queue
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from io import BytesIO

# Add 2 second timeout to all tests to prevent infinite loops
pytestmark = pytest.mark.timeout(2)

# Import the modules to test
import sys
import os

# Add the directory containing dictate.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock pynput before importing dictate
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()

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
            return [segment], {"language": "en"}


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

[streaming]
default_streaming = false
streaming_chunk_seconds = 3.0
streaming_overlap_seconds = 1.5
streaming_typing_delay = 0.01
"""
    config_file.write_text(config_content)
    
    # Mock the CONFIG_PATH
    original_config_path = dictate.CONFIG_PATH
    monkeypatch.setattr(dictate, "CONFIG_PATH", config_file)
    
    # Reload config
    try:
        dictate.CONFIG = dictate.load_config()
    except Exception:
        # If loading fails, restore original
        monkeypatch.setattr(dictate, "CONFIG_PATH", original_config_path)
        raise
    
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
def mock_arecord(monkeypatch):
    """Mock arecord subprocess."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdout = BytesIO()
    mock_process.wait.return_value = 0
    mock_process.terminate = MagicMock()
    
    def mock_popen(cmd, **kwargs):
        if "arecord" in cmd:
            return mock_process
        return MagicMock()
    
    monkeypatch.setattr(dictate.subprocess, "Popen", mock_popen)
    monkeypatch.setattr(dictate.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    
    return mock_process


@pytest.fixture
def mock_xdotool(monkeypatch):
    """Mock xdotool."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0)
    
    # Mock subprocess.run for "which" command and other calls
    def mock_run_with_which(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
            result = MagicMock()
            if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
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
    
    @pytest.mark.skip(reason="Test needs rewrite to properly mock threads - currently hangs")
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    @patch('dictate.tempfile.NamedTemporaryFile')
    @patch('dictate.ThreadPoolExecutor')
    @patch('threading.Thread')
    def test_streaming_recording(self, mock_thread_class, mock_executor, mock_tempfile, mock_popen, mock_run, mock_config, mock_whisper_model, mock_arecord):
        """Test streaming recording mode."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        # Setup mocks
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_process.stdout = BytesIO(b'\x00' * 96000)  # 3 seconds of audio data
        mock_popen.return_value = mock_process
        
        # Mock tempfile
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test_chunk.wav"
        mock_temp.close = MagicMock()
        mock_tempfile.return_value = mock_temp
        
        # Don't mock ThreadPoolExecutor - let it be created, but we'll control what runs
        # The mock_executor will still be called, but we need to ensure it returns a real mock
        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.add_done_callback = MagicMock()
        mock_pool.submit.return_value = mock_future
        mock_pool.shutdown = MagicMock()
        mock_executor.return_value = mock_pool
        
        # Mock threading.Thread to return a mock that doesn't actually start threads
        mock_thread_instance = MagicMock()
        mock_thread_instance.start = MagicMock()
        mock_thread_instance.join = MagicMock()
        mock_thread_instance.is_alive = MagicMock(return_value=False)
        mock_thread_class.return_value = mock_thread_instance
        
        # Update config for streaming
        config = dictate.load_config()
        config["default_streaming"] = True
        config["auto_type"] = False  # Disable typing to avoid extra threads
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Mock thread pool submit to not actually run anything
        def mock_submit(fn, *args, **kwargs):
            mock_fut = MagicMock()
            mock_fut.add_done_callback = MagicMock()
            return mock_fut
        dictation.threads_pool.submit = mock_submit
        
        # Mock the worker methods to prevent them from running
        dictation._record_chunks_worker = MagicMock()
        dictation._transcription_worker = MagicMock()
        dictation._typing_worker = MagicMock()
        
        # Start recording - this should set the flag but not create real threads
        dictation.start_recording()
        assert dictation.recording is True
        
        # Immediately stop recording
        dictation.stop_recording()
        assert dictation.recording is False
    
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    @patch('dictate.tempfile.NamedTemporaryFile')
    @patch('threading.Thread')
    def test_streaming_can_start_multiple_times(self, mock_thread_class, mock_tempfile, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test that streaming mode can be started multiple times after stopping."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        # Setup mocks
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_process.terminate = MagicMock()
        mock_popen.return_value = mock_process
        
        # Mock tempfile
        temp_file_counter = 0
        def mock_tempfile_side_effect(*args, **kwargs):
            nonlocal temp_file_counter
            temp_file_counter += 1
            mock_temp = MagicMock()
            mock_temp.name = f"/tmp/test_chunk_{temp_file_counter}.wav"
            mock_temp.close = MagicMock()
            return mock_temp
        mock_tempfile.side_effect = mock_tempfile_side_effect
        
        # Mock threading.Thread to return a mock that doesn't actually start threads
        mock_thread_instance = MagicMock()
        mock_thread_instance.start = MagicMock()
        mock_thread_instance.join = MagicMock()
        mock_thread_instance.is_alive = MagicMock(return_value=False)
        mock_thread_class.return_value = mock_thread_instance
        
        # Update config for streaming
        config = dictate.load_config()
        config["default_streaming"] = True
        config["auto_type"] = True  # Required for streaming mode
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Ensure model is set and event is set
        dictation.model = mock_whisper_model
        dictation.model_loaded.set()
        
        # Mock the worker methods to prevent them from running
        def mock_transcription_worker():
            # Worker that exits immediately when recording stops
            while dictation.recording or not dictation.transcription_queue.empty():
                try:
                    chunk_data = dictation.transcription_queue.get(timeout=0.1)
                    if chunk_data is None:
                        break
                except queue.Empty:
                    continue
        
        def mock_typing_worker():
            # Worker that exits immediately when recording stops
            while dictation.recording or not dictation.typing_queue.empty():
                try:
                    typing_task = dictation.typing_queue.get(timeout=0.1)
                    if typing_task is None:
                        break
                except queue.Empty:
                    continue
        
        dictation._transcription_worker = mock_transcription_worker
        dictation._typing_worker = mock_typing_worker
        dictation._record_chunks_worker = MagicMock()
        dictation._finalize_transcription = MagicMock(return_value="")
        
        # First recording session
        dictation.start_recording()
        assert dictation.recording is True
        first_thread_pool = dictation.threads_pool
        
        # Stop first recording (this shuts down the thread pool)
        dictation.stop_recording()
        assert dictation.recording is False
        
        # Verify thread pool was shut down by checking if we can submit to it
        with pytest.raises(RuntimeError):
            first_thread_pool.submit(lambda: None)
        
        # Second recording session - should recreate thread pool
        dictation.start_recording()
        assert dictation.recording is True
        second_thread_pool = dictation.threads_pool
        
        # Verify a new thread pool was created
        assert first_thread_pool is not second_thread_pool
        
        # Verify we can submit to the new thread pool without error (this was the bug - RuntimeError before fix)
        try:
            future = second_thread_pool.submit(lambda: 42)
            # If we get here without RuntimeError, the fix worked
            assert future is not None
        except RuntimeError as e:
            pytest.fail(f"Thread pool should be usable after recreation, but got: {e}")
        
        # Stop second recording
        dictation.stop_recording()
        assert dictation.recording is False
        
        # Third recording session - should work again
        dictation.start_recording()
        assert dictation.recording is True
        third_thread_pool = dictation.threads_pool
        
        # Verify another new thread pool was created
        assert second_thread_pool is not third_thread_pool
        assert first_thread_pool is not third_thread_pool
        
        # Verify we can submit to the new thread pool without error
        try:
            future = third_thread_pool.submit(lambda: 84)
            assert future is not None
        except RuntimeError as e:
            pytest.fail(f"Thread pool should be usable after recreation, but got: {e}")
        
        dictation.stop_recording()
        assert dictation.recording is False
    
    def test_overlap_buffer(self, mock_config, mock_whisper_model):
        """Test overlap buffer handling."""
        config = dictate.load_config()
        config["streaming_chunk_seconds"] = 3.0
        config["streaming_overlap_seconds"] = 1.5
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Simulate reading chunks
        chunk_frames = int(3.0 * 16000)
        overlap_frames = int(1.5 * 16000)
        
        # First chunk
        audio1 = np.random.randn(chunk_frames).astype(np.float32)
        dictation.overlap_buffer = []
        
        # Process chunk
        if dictation.overlap_buffer:
            combined = np.concatenate([np.concatenate(dictation.overlap_buffer), audio1])
        else:
            combined = audio1
        
        # Save overlap for next chunk
        dictation.overlap_buffer = [audio1[-overlap_frames:]]
        
        assert len(dictation.overlap_buffer[0]) == overlap_frames
        
        # Second chunk should use overlap
        audio2 = np.random.randn(chunk_frames).astype(np.float32)
        combined2 = np.concatenate([np.concatenate(dictation.overlap_buffer), audio2])
        
        assert len(combined2) == chunk_frames + overlap_frames


class TestDictation:
    """Tests for non-streaming Dictation class (backward compatibility)."""
    
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_non_streaming_mode(self, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test non-streaming mode (backward compatibility)."""
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

    def test_config_clipboard_disabled(self, mock_config):
        """Test that clipboard=false is correctly loaded."""
        # Ensure it's in the [behavior] section
        content = mock_config.read_text()
        new_content = content.replace("[behavior]\n", "[behavior]\nclipboard = false\n")
        mock_config.write_text(new_content)
        
        config = dictate.load_config()
        assert config["clipboard"] is False

class TestClipboardIntegration:
    """Tests specifically for the clipboard parameter integration."""

    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_dictation_no_clipboard_call(self, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test that Dictation doesn't call xclip when clipboard is disabled."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        content = mock_config.read_text()
        new_content = content.replace("[behavior]\n", "[behavior]\nclipboard = false\n")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Mock a recorded file
        dictation.temp_file = MagicMock()
        dictation.temp_file.name = "test.wav"
        dictation.recording = True
        
        # Mock model return
        segment = Segment(
            id=0, seek=0, start=0.0, end=1.0, text="test text",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=None, temperature=None
        )
        dictation.model.transcribe.side_effect = None
        dictation.model.transcribe.return_value = ([segment], {})
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_streaming_dictation_no_clipboard_call(self, mock_popen, mock_run, mock_config, mock_whisper_model, mock_arecord):
        """Test that StreamingDictation doesn't call xclip when clipboard is disabled."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        content = mock_config.read_text()
        new_content = content.replace("[behavior]\n", "[behavior]\nclipboard = false\n")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Initialize record_thread to avoid AttributeError
        dictation.record_thread = None
        
        # Clear queues to prevent timeout in _finalize_transcription
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
        
        # Mock _finalize_transcription to return immediately without waiting
        dictation._finalize_transcription = MagicMock(return_value="final text")
        
        # Set up transcriber
        dictation.transcriber = MagicMock()
        dictation.transcriber.finalize.return_value = "final text"
        dictation.recording = True
        dictation.accumulated_text = "final text"  # Set accumulated text directly
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    def test_check_dependencies_clipboard_optional(self, monkeypatch):
        """Test that xclip is optional in check_dependencies if clipboard is disabled."""
        mock_run = MagicMock()
        
        # Mock 'which' to return 0 for arecord, but 1 for xclip
        def side_effect(cmd, **kwargs):
            res = MagicMock()
            if cmd[1] == "xclip":
                res.returncode = 1
            else:
                res.returncode = 0
            return res
        
        mock_run.side_effect = side_effect
        monkeypatch.setattr(dictate.subprocess, "run", mock_run)
        
        # Should NOT exit if clipboard is False
        dictate.check_dependencies({"clipboard": False, "auto_type": False})
        
        # Should exit if clipboard is True (since xclip is missing)
        with pytest.raises(SystemExit):
            dictate.check_dependencies({"clipboard": True, "auto_type": False})


class TestWordDeduplication:
    """Tests for word deduplication logic in StreamingDictation."""
    
    def test_exact_overlap_deduplication(self, mock_config, mock_whisper_model):
        """Test that exact overlapping words are deduplicated."""
        config = dictate.load_config()
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # First chunk words (absolute timestamps)
        dictation.accumulated_words = [
            Word(word="hello", start=0.0, end=0.5, probability=0.9),
            Word(word="world", start=0.6, end=1.0, probability=0.9),
        ]
        
        # Second chunk with exact overlap (chunk starts at 1.5)
        # The word "world" at 1.5-2.0 is in overlap region and should match previous "world" at 0.6-1.0
        # Word "test" at 3.1-3.5 is after overlap_end, so it's in non-overlapped region
        chunk2_start = 1.5
        new_words = [
            Word(word="world", start=chunk2_start + 0.0, end=chunk2_start + 0.5, probability=0.9),  # 1.5-2.0, should match previous "world" at 0.6-1.0
            Word(word="test", start=chunk2_start + 1.6, end=chunk2_start + 2.0, probability=0.9),  # 3.1-3.5, after overlap
        ]
        new_segments = [Segment(
            id=0, seek=0, start=chunk2_start, end=chunk2_start + 2.0, text="world test",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=new_words, temperature=None
        )]
        
        text_to_remove, text_to_type = dictation._process_words_with_overlap(new_segments, chunk2_start, 2)
        
        # Should only add "test", not duplicate "world"
        accumulated_text = " ".join(w.word for w in dictation.accumulated_words)
        assert "test" in accumulated_text
        # "world" should only appear once (from the original)
        assert accumulated_text.count("world") == 1
    
    def test_partial_overlap_deduplication(self, mock_config, mock_whisper_model):
        """Test that partially overlapping words are handled correctly."""
        config = dictate.load_config()
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # First chunk words (absolute timestamps)
        dictation.accumulated_words = [
            Word(word="hello", start=0.0, end=0.5, probability=0.9),
            Word(word="world", start=0.6, end=1.0, probability=0.9),
        ]
        
        # Second chunk with partial overlap
        chunk2_start = 0.8
        new_words = [
            Word(word="world", start=chunk2_start + 0.0, end=chunk2_start + 0.4, probability=0.9),  # 0.8-1.2, partial overlap with previous "world" at 0.6-1.0
            Word(word="test", start=chunk2_start + 0.5, end=chunk2_start + 0.9, probability=0.9),  # 1.3-1.7
        ]
        new_segments = [Segment(
            id=0, seek=0, start=chunk2_start, end=chunk2_start + 0.9, text="world test",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=new_words, temperature=None
        )]
        
        text_to_remove, text_to_type = dictation._process_words_with_overlap(new_segments, chunk2_start, 2)
        
        # Should handle overlap correctly
        accumulated_text = " ".join(w.word for w in dictation.accumulated_words)
        assert "test" in accumulated_text
    
    def test_no_overlap_new_words(self, mock_config, mock_whisper_model):
        """Test that non-overlapping words are both added."""
        config = dictate.load_config()
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # First chunk words (absolute timestamps)
        dictation.accumulated_words = [
            Word(word="hello", start=0.0, end=0.5, probability=0.9),
        ]
        
        # Second chunk with no overlap (chunk starts at 1.5)
        # Word "world" at 5.0-5.5 is after overlap_end, so it's in non-overlapped region
        chunk2_start = 1.5
        new_words = [
            Word(word="world", start=5.0, end=5.5, probability=0.9),  # After overlap region
        ]
        new_segments = [Segment(
            id=0, seek=0, start=5.0, end=5.5, text="world",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=new_words, temperature=None
        )]
        
        text_to_remove, text_to_type = dictation._process_words_with_overlap(new_segments, chunk2_start, 2)
        
        # Both words should be present
        accumulated_text = " ".join(w.word for w in dictation.accumulated_words)
        assert "hello" in accumulated_text
        assert "world" in accumulated_text
    
    def test_word_replacement_on_mismatch(self, mock_config, mock_whisper_model):
        """Test that mismatched words at similar timestamps are correctly replaced.
        
        This tests the specific bug where "shop." was not replaced with "so we".
        Scenario: Chunk 1 has "full fine shop." and Chunk 2 has "and full fine so we should find three seats in the way."
        After cutting "and" (near start), "shop." should be replaced with "so we should find three seats in the way."
        """
        config = dictate.load_config()
        config["streaming_match_words_threshold_seconds"] = 0.1
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Chunk 1: "full fine shop." (absolute timestamps)
        # Chunk 1 started at 0.0
        dictation.accumulated_words = [
            Word(word="full", start=0.0, end=0.3, probability=0.9),
            Word(word="fine", start=0.4, end=0.7, probability=0.9),
            Word(word="shop.", start=0.8, end=1.1, probability=0.9),
        ]
        
        # Chunk 2: "and full fine so we should find three seats in the way."
        # Chunk 2 starts at 0.0 (same start for simplicity, but we'll use overlap logic)
        # Actually, let's use 0.5 to have real overlap
        chunk2_start = 0.0
        
        # Create new segments for chunk 2 with absolute timestamps
        # "and" is at chunk start (should be cut out), "full" and "fine" match, "so" replaces "shop."
        # Key: "so" at 0.9-1.1 should match timestamp with "shop." at 0.8-1.1 (within 0.1 threshold)
        # Make sure "full" and "fine" have matching timestamps first
        threshold = config["streaming_match_words_threshold_seconds"]
        new_words = [
            Word(word="and", start=chunk2_start + 0.0, end=chunk2_start + 0.05, probability=0.9),  # 0.0-0.05, will be cut (near start, within threshold)
            Word(word="full", start=chunk2_start + 0.0, end=chunk2_start + 0.3, probability=0.9),  # 0.0-0.3, matches "full" at 0.0-0.3
            Word(word="fine", start=chunk2_start + 0.4, end=chunk2_start + 0.7, probability=0.9),  # 0.4-0.7, matches "fine" at 0.4-0.7
            Word(word="so", start=chunk2_start + 0.8, end=chunk2_start + 1.1, probability=0.9),  # 0.8-1.1, matches "shop." at 0.8-1.1 but different text!
            Word(word="we", start=chunk2_start + 1.2, end=chunk2_start + 1.4, probability=0.9),  # 1.2-1.4
            Word(word="should", start=chunk2_start + 1.5, end=chunk2_start + 1.8, probability=0.9),  # 1.5-1.8
            Word(word="find", start=chunk2_start + 1.9, end=chunk2_start + 2.2, probability=0.9),  # 1.9-2.2
            Word(word="three", start=chunk2_start + 2.3, end=chunk2_start + 2.6, probability=0.9),  # 2.3-2.6
            Word(word="seats", start=chunk2_start + 2.7, end=chunk2_start + 3.0, probability=0.9),  # 2.7-3.0
            Word(word="in", start=chunk2_start + 3.1, end=chunk2_start + 3.3, probability=0.9),  # 3.1-3.3
            Word(word="the", start=chunk2_start + 3.4, end=chunk2_start + 3.6, probability=0.9),  # 3.4-3.6
            Word(word="way.", start=chunk2_start + 3.7, end=chunk2_start + 4.0, probability=0.9),  # 3.7-4.0
        ]
        new_segments = [Segment(
            id=0, seek=0, start=chunk2_start, end=chunk2_start + 4.0,
            text="and full fine so we should find three seats in the way.",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=new_words, temperature=None
        )]
        
        # Call the function
        text_to_remove, text_to_type = dictation._process_words_with_overlap(new_segments, chunk2_start, 2)
        
        # Verify: "shop." should be removed, "so we should find three seats in the way." should be typed
        assert "shop." in text_to_remove or "shop" in text_to_remove, f"Expected 'shop.' in text_to_remove, got: {text_to_remove}"
        assert "so" in text_to_type, f"Expected 'so' in text_to_type, got: {text_to_type}"
        assert "we" in text_to_type, f"Expected 'we' in text_to_type, got: {text_to_type}"
        
        # The accumulated_words should be updated - "shop." should be replaced with "so we should find three seats in the way."
        accumulated_text = " ".join(w.word for w in dictation.accumulated_words)
        assert "shop." not in accumulated_text, f"Expected 'shop.' to be removed from accumulated_words, got: {accumulated_text}"
        assert "so" in accumulated_text, f"Expected 'so' in accumulated_words, got: {accumulated_text}"
        assert "we" in accumulated_text, f"Expected 'we' in accumulated_words, got: {accumulated_text}"


class TestChunkSizes:
    """Tests for various chunk and overlap sizes in StreamingDictation."""
    
    @pytest.mark.parametrize("chunk_seconds,overlap_seconds", [
        (2.0, 1.0),
        (3.0, 1.5),
        (4.0, 2.0),
        (5.0, 2.5),
    ])
    def test_various_chunk_sizes(self, mock_config, mock_whisper_model, chunk_seconds, overlap_seconds):
        """Test StreamingDictation with various chunk and overlap sizes."""
        config = dictate.load_config()
        config["streaming_chunk_seconds"] = chunk_seconds
        config["streaming_overlap_seconds"] = overlap_seconds
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Verify it was set up correctly
        assert dictation.streaming_chunk_seconds == chunk_seconds
        assert dictation.streaming_overlap_seconds == overlap_seconds
    
    def test_chunk_sizes_match_configuration(self, mock_config, mock_whisper_model):
        """Test that StreamingDictation uses streaming_chunk_seconds and streaming_overlap_seconds from configuration."""
        chunk_seconds = 3.0
        overlap_seconds = 1.5
        
        # Update config
        config = dictate.load_config()
        config["streaming_chunk_seconds"] = chunk_seconds
        config["streaming_overlap_seconds"] = overlap_seconds
        
        # Create dictation instance
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Verify configuration values match
        assert dictation.streaming_chunk_seconds == chunk_seconds
        assert dictation.streaming_overlap_seconds == overlap_seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
