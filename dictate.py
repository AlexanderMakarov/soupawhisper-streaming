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
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

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

    # Defaults
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "key": DEFAULT_HOTKEY,
        "default_streaming": "true",
        "notifications": "true",
        "clipboard": "true",
        "auto_type": "true",
        "typing_delay": "0.01",
        "streaming_chunk_seconds": "3.0",
        "streaming_overlap_seconds": "1.5",
        "streaming_match_words_threshold_seconds": "0.1",
    }

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        # Whisper
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        # Hotkey
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        # Behavior
        "default_streaming": config.getboolean("behavior", "default_streaming", fallback=defaults["default_streaming"] == "true"),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
        "clipboard": config.getboolean("behavior", "clipboard", fallback=True),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "typing_delay": config.getboolean("behavior", "typing_delay", fallback=float(defaults["typing_delay"])),
        # Streaming
        "streaming_chunk_seconds": config.getfloat("streaming", "streaming_chunk_seconds", fallback=float(defaults["streaming_chunk_seconds"])),
        "streaming_overlap_seconds": config.getfloat("streaming", "streaming_overlap_seconds", fallback=float(defaults["streaming_overlap_seconds"])),
        "streaming_match_words_threshold_seconds": config.getfloat("streaming", "streaming_match_words_threshold_seconds", fallback=float(defaults["streaming_match_words_threshold_seconds"])),
    }


def get_hotkey(key_name):
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
        # Clear any modifier keys that might be stuck from hotkey press
        # subprocess.run(
        #     ["xdotool", "keyup", "ctrl", "alt", "shift", "super"],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        #     timeout=2
        # )
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
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        self.typer: Optional[Typer] = None

        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100  # Delay to avoid modifiers from hotkey
            )

        # Load model in background
        logger.info(f"Loading Whisper model ({config['model']})...")
        threading.Thread(target=lambda: self._load_model(""), daemon=True).start()

    def get_hotkey_name(self):
        return getattr(self.hotkey, 'name', None) or getattr(self.hotkey, 'char', DEFAULT_HOTKEY)

    def _load_model(self, log_suffix: str = ""):
        try:
            self.model = WhisperModel(self.config["model"], device=self.config["device"], compute_type=self.config["compute_type"])
            self.model_loaded.set()
            dictation_type = f"streaming {log_suffix}".strip() if log_suffix else "dictation"
            logger.info(f"Model loaded. Ready for {dictation_type}!")
            logger.info(f"Hold [{self.get_hotkey_name()}] to record, release to transcribe.")
            logger.info("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            logger.error(f"Failed to load model: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                logger.info("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification."""
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
    
    def on_press(self, key):
        if key == self.hotkey:
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

    def stop(self):
        logger.info("\nExiting...")
        self.running = False
        os._exit(0)

    def run(self):
        with keyboard.Listener(
            on_press=self.on_press
        ) as listener:
            listener.join()

    def start_record_process(self, output_file: str, duration: Optional[float] = None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        """
        Start arecord process with shared parameters, writing WAV file.
        
        Args:
            output_file: Path to write WAV file to
            duration: Optional duration in seconds (if None, records until terminated)
            stdout: Where to redirect stdout (default: DEVNULL)
            stderr: Where to redirect stderr (default: DEVNULL)
        
        Returns:
            subprocess.Popen instance
        """
        cmd = [
            "arecord",
            "-f", "S16_LE",  # Format: 16-bit little-endian
            "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
            "-c", "1",       # Mono
            "-t", "wav",
        ]
        
        if duration is not None:
            cmd.extend(["-d", str(int(duration))])
        
        cmd.append(output_file)
        
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr
        )

    def start_recording(self):
        if self.recording or self.model_error:
            return

        self.recording = True
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        # Record using arecord (ALSA) - works on most Linux systems
        self.record_process = self.start_record_process(self.temp_file.name)
        logger.info("[record] Recording...")
        self.notify("Recording...", f"Press {self.get_hotkey_name().upper()} again to stop", "audio-input-microphone", 30000)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        logger.info("[idle] Recording stopped, transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 1500)

        # Wait for model if not loaded yet
        self.model_loaded.wait()

        if self.model_error:
            logger.error("[idle] Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        # Transcribe
        if self.model is None:
            logger.error("[idle] Cannot transcribe: model not loaded")
            return
        
        try:
            temp_file_name = self.temp_file.name if self.temp_file else None
            if not temp_file_name or not os.path.exists(temp_file_name):
                return
            
            # Read WAV file as numpy array
            with wave.open(temp_file_name, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()
                audio_bytes = wf.readframes(num_frames)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            trans_start = time.time()
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,
                vad_filter=True,
            )
            trans_duration = time.time() - trans_start

            text = " ".join(segment.text.strip() for segment in segments)

            if text:
                logger.info(f"[idle] Transcribed {info.duration:.2f}s in {trans_duration:.2f}s: \"{text}\"")
                # Copy to clipboard using xclip
                if self.config["clipboard"]:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE
                    )
                    process.communicate(input=text.encode())
                    logger.info(f"[idle] Pasted to clipboard: {text}")

                # Type it into the active input field
                if self.config["auto_type"] and self.typer:
                    self.typer.type_rewrite(text, 0)

                self.notify("Transcribed:", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                logger.info("[idle] No speech detected")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            logger.error(f"[idle] {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            # Cleanup temp file
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)


class StreamingDictation(Dictation):
    """Streaming dictation mode with incremental transcription.

    Inside there is thread pool of 4 threads:
    - 2 for recording
    - 1 for transcribing
    - 1 for typing

    Thread of transcribing should take tasks from queue in order of chunks.
    Thread of typing should work in the same order (order of chunks).

    Strategy of recording and transcription:
    - Create temp file for chunk 1, start recording for streaming_chunk_seconds in a thread pool
    - Wait streaming_overlap_seconds, then start chunk 2 (while chunk 1 is still recording) in a thread pool. Repeat this for next chunks

    Strategy of transcription:
    - After getting transcription for chunk remove temporal file and check it took less than streaming_overlap_seconds seconds
    - If transcription took streaming_overlap_seconds seconds or more than produce log and desktop notification about too slow transcription but continue working (transcription would be lagging)
    - If it is first chunk then just print transcription as is
    - If not first chunk then process new words with overlaps handling

    Overlaps handling strategy:
    - Chunks will overlap by streaming_overlap_seconds
    - In overlapped interval need to match words between previous chunk and current chunk by timestamps with streaming_match_words_threshold_seconds threshold
    - Words from the current chunk have priority over words from the previous chunk except the first word if it is timestamped withing streaming_match_words_threshold_seconds of the chunk start (first word in chunk could be just a part of the word so could be transcribed wrongly)
    - On any difference need to remove old text including first wrong word and type new text starting from this difference, inlcuding transcription of not overlapped part of chunk
    - If no difference found just print words from not overlapped interval of current chunk
    """

    def __init__(self, config: dict):
        # Initialize base class (sets up config, hotkey, model loading, etc.)
        super().__init__(config)
        self.streaming_chunk_seconds = config["streaming_chunk_seconds"]
        self.streaming_overlap_seconds = config["streaming_overlap_seconds"]
        self.streaming_match_words_threshold_seconds = config["streaming_match_words_threshold_seconds"]

        # Thread pool: 2 for recording, 1 for transcribing, 1 for typing
        self.threads_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="streaming")
        
        # Queues for ordered processing
        self.transcription_queue = queue.Queue()  # FIFO queue for transcription tasks
        self.typing_queue = queue.Queue()  # FIFO queue for typing tasks
        
        # Recording state
        self.chunk_index = 0
        self.recording_start_time: Optional[float] = None
        self.active_recordings: dict[int, tuple[str, subprocess.Popen, float]] = {}  # chunk_index -> (file_path, process, start_time)
        self.record_thread: Optional[threading.Thread] = None
        
        # Transcription state
        self.previous_chunk_words: list[tuple[float, float, str]] = []  # Words from previous chunk for overlap matching
        self.accumulated_text = ""
        self.last_transcribed_chunk_index = 0
        
        # Typing state
        self.last_typed_text = ""
        
        # Create typer once in constructor
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100
            )
        else:
            self.typer = None

    def _load_model(self, log_suffix: str = ""):
        """Override to call parent with streaming suffix."""
        super()._load_model("streaming")

    def _record_single_chunk(self, chunk_idx: int, chunk_start_time: float) -> tuple[str, float]:
        """Record a single chunk in thread pool. Returns (file_path, actual_start_time)."""
        chunk_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk_file.close()

        # Start recording this chunk (with duration limit)
        chunk_process = self.start_record_process(chunk_file.name, duration=self.streaming_chunk_seconds)
        self.active_recordings[chunk_idx] = (chunk_file.name, chunk_process, chunk_start_time)
        logger.info(f"[record] Started recording chunk {chunk_idx} to {chunk_file.name}")
        # Wait for recording to finish
        chunk_process.wait()
        logger.info(f"[record] Chunk {chunk_idx} recording finished")
        # Remove from active recordings
        if chunk_idx in self.active_recordings:
            del self.active_recordings[chunk_idx]
        return (chunk_file.name, chunk_start_time)
    
    def _record_chunks_coordinator(self):
        """Coordinates chunk recording: starts chunks with proper timing."""
        try:
            while self.recording:
                current_time = time.time()
                
                # Start new chunk
                self.chunk_index += 1
                chunk_start_time = current_time if self.recording_start_time is None else self.recording_start_time + (self.chunk_index - 1) * self.streaming_overlap_seconds
                if self.recording_start_time is None:
                    self.recording_start_time = chunk_start_time

                # Submit recording task to thread pool (2 recording threads)
                future = self.threads_pool.submit(self._record_single_chunk, self.chunk_index, chunk_start_time)

                # When recording finishes, queue for transcription
                def on_recording_done(fut):
                    try:
                        file_path, start_time = fut.result()
                        self.transcription_queue.put((file_path, start_time, self.chunk_index))
                    except Exception as e:
                        logger.error(f"[record] Error in recording chunk {self.chunk_index}: {e}")

                future.add_done_callback(on_recording_done)

                # Calculate exact time when next chunk should start
                next_chunk_start_time = self.recording_start_time + self.chunk_index * self.streaming_overlap_seconds
                # Measure time after task submission to account for execution time
                time_after_submission = time.time()
                sleep_duration = next_chunk_start_time - time_after_submission
                # Sleep until exact next chunk start time (or continue immediately if already past)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        except Exception as e:
            logger.error(f"[record] Error in chunk coordinator: {e}")

    def start_recording(self):
        if self.recording or self.model_error:
            return
        
        self.recording = True
        self.chunk_index = 0
        self.last_typed_text = ""
        self.recording_start_time = None
        self.active_recordings = {}
        
        # Reset transcription state
        self.previous_chunk_words = []
        self.accumulated_text = ""
        self.last_transcribed_chunk_index = 0
        
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
        
        self.model_loaded.wait()
        if self.model_error:
            return
        
        if self.model is None:
            return
        
        # Recreate thread pool (it may have been shut down from previous recording)
        try:
            self.threads_pool.shutdown(wait=False)
        except RuntimeError:
            # Already shut down, ignore
            pass
        self.threads_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="streaming")
        
        # Start transcription worker (1 thread in pool)
        self.threads_pool.submit(self._transcription_worker)
        
        # Start typing worker (1 thread in pool)
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100
            )
            self.threads_pool.submit(self._typing_worker)
        
        # Start recording coordinator thread
        self.record_thread = threading.Thread(target=self._record_chunks_coordinator, daemon=True)
        self.record_thread.start()

        logger.info("[record] Recording (streaming mode)...")
        self.notify("Recording...", f"Release {self.get_hotkey_name().upper()} when done", "audio-input-microphone", 1500)
    
    def _transcription_worker(self):
        """Transcription worker thread - processes chunks in order."""
        while self.recording or not self.transcription_queue.empty():
            try:
                chunk_data = self.transcription_queue.get(timeout=0.1)
                if chunk_data is None:
                    break
                chunk_file_path, chunk_absolute_start_time, chunk_idx = chunk_data
                # Transcribe chunk
                trans_start = time.time()
                new_words = self._transcribe_chunk(chunk_file_path, chunk_absolute_start_time)
                trans_duration = time.time() - trans_start
                # Remove temp file
                try:
                    os.unlink(chunk_file_path)
                except Exception as e:
                    logger.warning(f"[record] Failed to delete chunk file {chunk_file_path}: {e}")

                # Check if transcription took too long
                if trans_duration >= self.streaming_overlap_seconds:
                    logger.warning(f"[transcriber] Chunk {chunk_idx} transcription took {trans_duration:.2f}s (>= {self.streaming_overlap_seconds}s) - transcription is lagging")
                    if self.config["notifications"]:
                        self.notify("Transcription too slow", f"Chunk {chunk_idx} took {trans_duration:.2f}s. Transcription is lagging.", "dialog-warning", 3000)

                # Process words based on chunk index
                if chunk_idx == 1:
                    # First chunk - just print transcription as is
                    if new_words:
                        text_to_type = " ".join(word for _, _, word in new_words)
                        self.accumulated_text = text_to_type
                        self.typing_queue.put((text_to_type, 0, chunk_idx))
                else:
                    # Not first chunk - process with overlap handling
                    if new_words:
                        self._process_words_with_overlap(new_words, chunk_absolute_start_time, chunk_idx)
                self.last_transcribed_chunk_index = chunk_idx
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[transcriber] {e}")

    def _typing_worker(self):
        """Typing worker thread - types text in order of chunks."""
        while self.recording or not self.typing_queue.empty():
            try:
                typing_task = self.typing_queue.get(timeout=0.1)
                if typing_task is None:
                    break
                text_to_type, chars_to_remove, chunk_idx = typing_task
                if self.typer:
                    try:
                        self.typer.type_rewrite(text_to_type, chars_to_remove)
                        self.last_typed_text = self.accumulated_text
                    except Exception as e:
                        logger.error(f"[typer] Typing failed: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[typer] {e}")

    def _transcribe_chunk(self, chunk_file_path: str, chunk_absolute_start_time: float) -> list[tuple[float, float, str]]:
        """
        Transcribe a single audio chunk file using faster-whisper with word timestamps.
        Returns list of (absolute_start, absolute_end, word_text) tuples.
        """
        try:
            if self.model is None:
                logger.error("[transcriber] Model not loaded")
                return []
            # Pass file path to faster-whisper for transcription
            segments, info = self.model.transcribe(
                chunk_file_path,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )
            new_words: list[tuple[float, float, str]] = []
            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip()
                        if not word_text:
                            continue

                        # Convert relative timestamps to absolute by adding chunk start time
                        absolute_start = word.start + chunk_absolute_start_time
                        absolute_end = word.end + chunk_absolute_start_time
                        new_words.append((absolute_start, absolute_end, word_text))
            return new_words
        except Exception as e:
            logger.error(f"[record] [transcriber] Failed to transcribe chunk: {e}")
            return []

    def _process_words_with_overlap(self, new_words: list[tuple[float, float, str]], chunk_absolute_start_time: float, chunk_idx: int):
        """
        Process new words with overlap handling according to docstring strategy.
        
        Overlaps handling strategy:
        - Match words between previous chunk and current chunk by timestamps with threshold
        - Words from current chunk have priority except first word if timestamped within threshold of chunk start
        - On any difference: remove old text including first wrong word and type new text starting from difference
        - If no difference found: just print words from not overlapped interval of current chunk
        """
        if not new_words:
            return
        
        chunk_start = chunk_absolute_start_time
        chunk_end = chunk_absolute_start_time + self.streaming_chunk_seconds
        overlap_start = chunk_start
        overlap_end = chunk_start + self.streaming_overlap_seconds
        
        # Get words from previous chunk that are in overlap region
        previous_overlap_words = [
            (start, end, word) for start, end, word in self.previous_chunk_words
            if start < overlap_end and end > overlap_start
        ]
        
        # Match words by timestamps with threshold
        threshold = self.streaming_match_words_threshold_seconds
        matched_previous_indices = set()
        matched_new_indices = set()
        
        # Check first word of new chunk - special handling if within threshold of chunk start
        first_word_start, first_word_end, first_word_text = new_words[0]
        first_word_near_start = (first_word_start - chunk_start) <= threshold
        
        # Match words in overlap region
        for i, (new_start, new_end, new_word) in enumerate(new_words):
            if new_start >= overlap_end:
                break  # Beyond overlap region
            
            for j, (prev_start, prev_end, prev_word) in enumerate(previous_overlap_words):
                if j in matched_previous_indices:
                    continue
                
                # Check if timestamps match within threshold
                if abs(new_start - prev_start) <= threshold or abs(new_end - prev_end) <= threshold:
                    matched_previous_indices.add(j)
                    matched_new_indices.add(i)
                    break
        
        # Determine if there's a difference
        has_difference = False
        if not first_word_near_start:
            # First word is not near start, check if it matches
            if 0 not in matched_new_indices:
                has_difference = True
        else:
            # First word is near start - check if any previous words don't match
            if len(matched_previous_indices) < len(previous_overlap_words):
                has_difference = True
        
        # Get non-overlapped words from current chunk
        non_overlapped_words = [
            (start, end, word) for start, end, word in new_words
            if start >= overlap_end
        ]
        
        if has_difference:
            # Remove old text including first wrong word and type new text starting from difference
            # Find where difference starts in accumulated text
            # For simplicity, remove all text from overlap start and retype everything from there
            text_to_remove = self.accumulated_text
            new_text_parts = []
            
            # Get words from new chunk (current chunk has priority)
            for i, (start, end, word) in enumerate(new_words):
                if i in matched_new_indices and not (i == 0 and first_word_near_start):
                    continue  # Skip matched words (except first if near start)
                new_text_parts.append(word)
            
            new_text = " ".join(new_text_parts)
            
            # Update accumulated text
            # Remove text from overlap start, add new text
            # For simplicity, rebuild from scratch
            all_words = []
            # Keep words from previous chunk before overlap
            for start, end, word in self.previous_chunk_words:
                if end <= overlap_start:
                    all_words.append((start, end, word))
            
            # Add all words from new chunk (current has priority)
            all_words.extend(new_words)
            all_words.sort(key=lambda x: x[0])
            
            self.accumulated_text = " ".join(word for _, _, word in all_words)
            
            # Calculate chars to remove
            chars_to_remove = len(text_to_remove)
            text_to_type = self.accumulated_text
            
            self.typing_queue.put((text_to_type, chars_to_remove, chunk_idx))
        else:
            # No difference - just print words from not overlapped interval
            if non_overlapped_words:
                text_to_type = " ".join(word for _, _, word in non_overlapped_words)
                self.accumulated_text += " " + text_to_type if self.accumulated_text else text_to_type
                self.typing_queue.put((text_to_type, 0, chunk_idx))
        
        # Update previous chunk words for next iteration
        self.previous_chunk_words = new_words
    
    def _finalize_transcription(self) -> str:
        """Finalize transcription and return complete text."""
        # Wait for all queues to empty (with timeout to prevent infinite loops)
        max_wait_time = 5.0  # Maximum 5 seconds
        wait_time = 0.0
        while (not self.transcription_queue.empty() or not self.typing_queue.empty()) and wait_time < max_wait_time:
            time.sleep(0.1)
            wait_time += 0.1
        return self.accumulated_text
    
    def stop_recording(self):
        if not self.recording:
            return
        
        self.recording = False
        
        # Wait for all active recordings to finish and queue them
        for chunk_idx, (file_path, process, start_time) in list(self.active_recordings.items()):
            if process.poll() is None:
                process.wait()
            if os.path.exists(file_path):
                self.transcription_queue.put((file_path, start_time, chunk_idx))
        
        self.active_recordings.clear()
        
        # Signal workers to stop
        self.transcription_queue.put(None)
        if self.config["auto_type"]:
            self.typing_queue.put(None)
        
        # Wait for threads
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        
        # Shutdown thread pool (if not already shut down)
        try:
            self.threads_pool.shutdown(wait=True)
        except RuntimeError:
            # Already shut down, ignore
            pass

        logger.info("[idle] Recording stopped, transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 1500)
        
        # Finalize any remaining chunks
        final_text = self._finalize_transcription()
        
        if final_text:
            if self.config["clipboard"]:
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=final_text.encode())
                logger.info(f"[idle] Pasted to clipboard: {final_text}")
            
            if self.config["auto_type"] and self.typer and final_text != self.last_typed_text:
                if final_text.startswith(self.last_typed_text):
                    suffix = final_text[len(self.last_typed_text):]
                    if suffix:
                        self.typer.type_rewrite(suffix, 0)
                else:
                    self.typer.type_rewrite(final_text, 0)
                logger.info(f"[idle] Typed: {final_text}")

            self.notify("Got:", final_text[:100] + ("..." if len(final_text) > 100 else ""), "emblem-ok-symbolic", 3000)
        else:
            logger.info("[idle] No speech detected")
            self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)
    
    def stop(self):
        """Override to stop recording before exiting."""
        logger.info("\nExiting...")
        self.running = False
        if self.recording:
            self.stop_recording()
        os._exit(0)


def check_dependencies(config: dict):
    """Check that required system commands are available."""
    missing = []

    required_cmds = ["arecord"]
    if config["clipboard"]:
        required_cmds.append("xclip")

    for cmd in required_cmds:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            pkg = "alsa-utils" if cmd == "arecord" else cmd
            missing.append((cmd, pkg))

    if config["auto_type"]:
        if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
            missing.append(("xdotool", "xdotool"))

    if missing:
        logger.error("Missing dependencies:")
        for cmd, pkg in missing:
            logger.error(f"  {cmd} - install with: sudo apt install {pkg}")
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

    model_name = config["model"]
    cache_path = get_model_cache_path()
    
    description = f"""SoupaWhisper - Push-to-talk voice dictation.

Config file: {CONFIG_PATH}
Streaming mode: {config["default_streaming"]}
Current model: {model_name}
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
        help="Disable streaming transcription mode (force non-streaming)"
    )
    args = parser.parse_args()

    logger.info(f"SoupaWhisper v{__version__}")
    logger.info(f"Config: {CONFIG_PATH}")

    check_dependencies(config)

    use_streaming = config["default_streaming"]
    if args.streaming:
        use_streaming = True
    elif args.no_streaming:
        use_streaming = False

    if use_streaming:
        dictation = StreamingDictation(config)
    else:
        dictation = Dictation(config)

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
