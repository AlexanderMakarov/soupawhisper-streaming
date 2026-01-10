#!/usr/bin/env python3

import argparse
import configparser
import subprocess
import tempfile
import threading
import signal
import sys
import os
import queue
import time
import wave
import logging
from pathlib import Path
from typing import Optional, Iterable, Tuple
from faster_whisper.transcribe import Segment

import numpy as np
import pyaudio
import webrtcvad

from pynput import keyboard
from faster_whisper import WhisperModel

__version__ = "0.1.0"

# Logger will be configured in main()
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"
DEFAULT_HOTKEY = "f12"


def load_config():
    config = configparser.ConfigParser()

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        # Whisper
        "model": config.get("whisper", "model", fallback="base.en"),
        "device": config.get("whisper", "device", fallback="cpu"),
        "compute_type": config.get("whisper", "compute_type", fallback="int8"),
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
        # Streaming
        "min_speech_length_seconds": config.getfloat("streaming", "min_speech_length_seconds", fallback=1.0),
        "vad_silence_threshold_seconds": config.getfloat("streaming", "vad_silence_threshold_seconds", fallback=1.0),
        "vad_sample_rate": config.getint("streaming", "vad_sample_rate", fallback=16000),
        "vad_chunk_size_ms": config.getfloat("streaming", "vad_chunk_size_ms", fallback=20.0),
        "vad_min_speech_chunks": config.getint("streaming", "vad_min_speech_chunks", fallback=10),
        "vad_threshold": config.getfloat("streaming", "vad_threshold", fallback=0.5),
    }


def get_hotkey(key_name: str) -> keyboard.KeyCode:
    """Map key name to pynput key."""
    key_name = key_name.lower()
    if hasattr(keyboard.Key, key_name):
        return getattr(keyboard.Key, key_name)
    elif len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)
    else:
        logger.warning(f"Unknown key: {key_name}, defaulting to {DEFAULT_HOTKEY}")
        return get_hotkey(DEFAULT_HOTKEY)


class Typer:
    """Types and removes characters in any inputs via xdotool."""

    def __init__(self, delay_ms: int = 10, start_delay_ms: int = 250):
        self.delay_ms = max(1, int(delay_ms))
        self.start_delay_ms = int(start_delay_ms)
        self.enabled = subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0
        if not self.enabled:
            logger.warning("[typer] xdotool not found, typing disabled")

    def type_rewrite(self, text: str, previous_length: int = 0):
        """
        Type text using xdotool.

        Args:
            text: The text to type.
            previous_length: The number of characters to delete before typing the new text.
        """
        if not self.enabled or not text:
            return
        if self.start_delay_ms > 0:
            time.sleep(self.start_delay_ms / 1000.0)
        # Remove previous characters if needed.
        if previous_length > 0:
            subprocess.run(
                ["xdotool", "key", "BackSpace", "--clearmodifiers", "--repeat", str(previous_length)],#, "--repeat-delay", str(self.delay_ms)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        # Type the new text.
        subprocess.run(
            ["xdotool", "type", "--delay", str(self.delay_ms), "--clearmodifiers", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )


class Dictation:
    def __init__(self, config: dict):
        self.config = config
        self.hotkey = get_hotkey(config["key"])
        self.recording = False
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        self.typer: Optional[Typer] = None
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream = None
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_data: list[np.ndarray] = []
        self.sample_rate = 16000
        self.frames_per_buffer = 4096  # ~0.25 seconds of audio

        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100  # Delay to avoid modifiers from hotkey
            )

        # Load model in background.
        logger.debug(f"Loading Whisper model ({config['model']})...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def get_hotkey_name(self) -> str:
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
        """Show available audio devices and instructions when no audio is detected.
        
        Args:
            issue_description: Description of the audio input issue (e.g., "Audio input contains only zeros" or "Audio input has too low amplitude")
        """
        logger.error(f"[record] {issue_description}. Building list of devices...")
        devices = self._get_available_input_devices()
        devices_list = [f"  {i}: {name}" for i, name, _ in devices]
        devices_text = "\n".join(devices_list[:8])
        if len(devices_list) > 8:
            devices_text += f"\n  ... and {len(devices_list) - 8} more"
        config_path = CONFIG_PATH
        message = f"{issue_description}\n\nSet 'audio_input_device' in {config_path}\nSupported (by pyaudio, not on this machine!) devices:\n{devices_text}"
        self.notify("No audio detected - check device", message, logging.WARNING, 10000)

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
        # Transcribe audio.
        segments, info = self.model.transcribe(
            audio_array,
            vad_filter=True,
        )
        # Convert segments to text.
        text = self._segments_to_text(segments, self.config["auto_sentence"])
        return text, info.duration

    def on_press(self, key):
        if key == self.hotkey:
            self.start_recording()

    def on_release(self, key):
        if key == self.hotkey:
            self.stop_recording()

    def stop(self):
        logger.info("\nExiting...")
        self.running = False

    def run(self):
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        ) as listener:
            listener.join()

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

        self.recording = True
        self.audio_data = []

        self.audio_stream = self._start_pyaudio_stream(self.frames_per_buffer)
        if self.audio_stream is None:
            self.recording = False
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
                file_path = os.path.join(tempfile.gettempdir(), f"recording_{timestamp}.wav")
                try:
                    with wave.open(file_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_array.tobytes())
                    logger.info(f"[record] Saved recording to {file_path}")
                except Exception as e:
                    logger.error(f"[record] Failed to save recording: {e}", exc_info=True)

            text, duration = self._transcribe_audio_array(audio_array)
            if text:
                logger.info(f"Transcribed {duration:.2f}s: {text}")
                if self.config["clipboard"]:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE
                    )
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
        finally:
            self.audio_data = []

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
        self.transcription_queue: queue.Queue[np.ndarray | None] = queue.Queue()
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
                start_delay_ms=100
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

    def _finish_model_loading(self):
        logger.info(f"Press [{self.get_hotkey_name().upper()}] to start transcribing, press one more time to stop. Press Ctrl+C to quit.")

    def on_press(self, key):
        if key == self.hotkey:
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

    def run(self):
        with keyboard.Listener(
            on_press=self.on_press
        ) as listener:
            listener.join()

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
        # Reset transcription state.
        self.recording = True
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
                start_delay_ms=100
            )
            self.typing_thread = threading.Thread(target=self._typing_worker, daemon=True)
            self.typing_thread.start()
        if self.config.get("save_recordings", False):
            self.file_saving_thread = threading.Thread(target=self._file_saving_worker, daemon=True)
            self.file_saving_thread.start()
        # Notify about recording start.
        self.notify("Recording...", f"Press {self.get_hotkey_name().upper()} when done", logging.INFO, 1500, "audio-input-microphone")

    def _continuous_audio_stream_worker(self, frames_per_buffer: int):
        chunk_duration = frames_per_buffer / float(self.vad_sample_rate)
        try:
            while self.recording:
                if not self.audio_stream:
                    break
                chunk_start_time = time.monotonic() - self.recording_start_time
                data = self.audio_stream.read(frames_per_buffer, exception_on_overflow=False)
                if not data:
                    continue
                samples = np.frombuffer(data, dtype=np.int16)
                self._process_audio_chunk(samples, chunk_start_time, chunk_duration)
        except Exception as e:
            logger.error(f"[record] Error in audio stream worker: {e}", exc_info=True)
        finally:
            self._finalize_segment()

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
            if self.in_speech:
                # Just add new chunk to the current speech segment.
                self.speech_segment_chunks.append(frame)
                self.speech_end_time = frame_start_time + frame_duration
            else:
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
            # Always add the frame to the current speech segment.
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
                self.transcription_queue.put(segment)
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
            self.transcription_queue.put(segment)
            # Save segment to file if needed.
            if not self.file_mode and self.config.get("save_recordings", False):
                self.file_saving_queue.put((segment, self.vad_sample_rate))
            self._reset_speech_mode()

    def _transcription_worker(self):
        """Transcription worker thread - processes chunks in order."""
        chunk_idx = 0
        # FYI: continue during "stopping" state to process all remaining items before exiting.
        while self.recording or self.stopping or not self.transcription_queue.empty():
            try:
                segment: np.ndarray | None = self.transcription_queue.get(timeout=0.1)
                if segment is None:
                    break
                if self.model is None:
                    logger.error("[transcriber] Model not loaded")
                    continue
                # Check if segment is all zeros or effectively silent (audio input problem)
                if self._check_valid_audio_input(segment):
                    continue
                # Convert int16 to float32 and normalize to [-1.0, 1.0]
                if segment.dtype == np.int16:
                    segment = segment.astype(np.float32) / 32768.0
                trans_start = time.time()
                segments, info = self.model.transcribe(
                    segment,
                    # FYI: don't disable vad_filter, it will cause errors like
                    # "No speech threshold is met (0.620832 > 0.600000)"
                    # if we detected speech incorrectly or cutted with big gaps inside.
                    vad_filter=True,
                    # FYI: don't set temperature=0.0, it will cause errors like
                    # "Log probability threshold is not met with temperature 0.0 (-1.359611 < -1.000000)"
                    # temperature=0.0, 
                    language="en",
                    condition_on_previous_text=True,
                    without_timestamps=True,
                )
                trans_duration = time.time() - trans_start
                text = self._segments_to_text(segments, False)
                if not text:
                    logger.info(f"[transcriber] Empty transcription (transcribed in {trans_duration:.2f}s)")
                    continue
                else:
                    logger.info(f"[transcriber] Transcribed {len(segment) / float(self.vad_sample_rate):.2f}s in {trans_duration:.2f}s: {text}")
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
                file_path = os.path.join(tempfile.gettempdir(), f"stream_chunk_{timestamp}.wav")
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
            self.transcription_thread.join(timeout=5.0)
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
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=final_text.encode())
                logger.info(f"[idle] Pasted to clipboard: {final_text}")
            logger.info(f"[idle] Final text: {final_text}")
            self.notify("Got:", final_text[:100] + ("..." if len(final_text) > 100 else ""), logging.INFO, 3000)
        else:
            logger.info("[idle] No speech detected")
            self.notify("No speech detected", "Try speaking louder or check audio device", logging.WARNING, 2000)

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


def check_dependencies(config: dict):
    """Check that required system commands are available."""
    missing = []
    # System command dependencies.
    required_cmds = []
    if config["clipboard"]:
        required_cmds.append("xclip")
    if config["auto_type"]:
        required_cmds.append("xdotool")
    for cmd in required_cmds:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            missing.append((cmd, cmd))
    # Check for missing dependencies.
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
    args = parser.parse_args()
    check_dependencies(config)

    # Apply arguments.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    # Handle file transcription mode.
    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"WAV file not found: {args.file}")
        dictation.model_loaded.wait()
        if dictation.model_error or dictation.model is None:
            logger.error("Cannot transcribe file: model failed to load")
            sys.exit(1)
        try:
            dictation.transcribe_file(args.file)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}", exc_info=True)
            sys.exit(1)

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # Start dictation.
    dictation.run()


if __name__ == "__main__":
    main()
