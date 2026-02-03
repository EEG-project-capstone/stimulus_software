# lib/audio_stream_manager.py

"""
Audio stream management for the EEG Stimulus Package.
Provides simplified, thread-safe audio playback.
"""

import logging
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from lib.constants import AudioParams
from lib.exceptions import AudioDeviceError, AudioPlaybackError

logger = logging.getLogger('eeg_stimulus.audio_stream')


class AudioStreamManager:
    """Manages audio stream lifecycle with proper cleanup."""

    def __init__(self):
        """Initialize audio stream manager."""
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._buffer: Optional[np.ndarray] = None
        self._buffer_position = 0
        self._stopping = False  # Flag to skip callbacks during intentional stop
        logger.info("AudioStreamManager initialized")
    
    def play(self,
             samples: np.ndarray,
             sample_rate: int = AudioParams.SAMPLE_RATE,
             on_finish: Optional[Callable[[], None]] = None) -> None:
        """Play audio samples with automatic cleanup.

        Args:
            samples: Audio samples (int16, shape: (n_samples, n_channels))
            sample_rate: Sample rate in Hz
            on_finish: Callback to execute when playback finishes

        Raises:
            AudioDeviceError: If audio device cannot be accessed
            AudioPlaybackError: If playback fails
        """
        # Validate samples
        samples = self._validate_samples(samples)

        # Check if stream is currently active before stopping
        was_active = False
        with self._lock:
            # Reset stopping flag for new playback
            self._stopping = False
            if self._stream is not None and self._stream.active:
                was_active = True
                logger.warning("New play() called while previous stream active, stopping previous")

        # Stop any existing stream
        logger.debug(f"play() calling stop() (was_active={was_active})")
        self.stop()
        
        # Set up buffer
        with self._lock:
            self._buffer = samples.copy()
            self._buffer_position = 0
        
        # Create callback
        def stream_callback(outdata, frames, time_info, status):
            if status:
                logger.warning(f"Audio stream status: {status}")

            with self._lock:
                if self._buffer is None or self._buffer_position >= len(self._buffer):
                    outdata.fill(0)
                    raise sd.CallbackStop

                available = len(self._buffer) - self._buffer_position
                chunk_size = min(frames, available)
                chunk = self._buffer[self._buffer_position:self._buffer_position + chunk_size]
                self._buffer_position += chunk_size

            outdata[:chunk_size] = chunk
            if chunk_size < frames:
                outdata[chunk_size:] = 0
        
        # Create finished callback
        def finished_callback():
            finish_time = time.time()
            logger.debug(f"Audio playback finished at {finish_time:.3f}")

            # Check if we're in an intentional stop - skip callback if so
            with self._lock:
                if self._stopping:
                    logger.debug("Skipping on_finish callback during intentional stop")
                    self._buffer = None
                    self._buffer_position = 0
                    self._stream = None
                    return

                # Track if this is being called after stream was already cleared
                was_already_none = self._stream is None
                self._buffer = None
                self._buffer_position = 0
                self._stream = None

            if on_finish is not None:
                try:
                    logger.debug(f"Calling on_finish callback (stream_was_none={was_already_none})")
                    on_finish()
                except Exception as e:
                    logger.error(f"Error in finish callback: {e}", exc_info=True)
        
        # Create and start stream
        try:
            logger.debug(f"AudioParams.STREAM_LATENCY value: {AudioParams.STREAM_LATENCY}")
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=samples.shape[1],
                dtype=AudioParams.BUFFER_DTYPE,
                callback=stream_callback,
                finished_callback=finished_callback,
                latency=max(float(AudioParams.STREAM_LATENCY), 0.1)  # Ensure STREAM_LATENCY is a float
            )
            
            logger.debug(f"Audio stream configured with latency={stream.latency} and buffer size={len(samples)}")
            
            with self._lock:
                self._stream = stream
            
            stream.start()
            logger.debug(f"Audio stream started: {len(samples)} samples at {sample_rate}Hz, "
                        f"{samples.shape[1]} channels")
            
        except sd.PortAudioError as e:
            error_msg = f"Audio device error: {e}"
            logger.error(error_msg)
            raise AudioDeviceError(error_msg) from e
        except Exception as e:
            error_msg = f"Audio playback failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise AudioPlaybackError(error_msg) from e
    
    def stop(self) -> None:
        """Stop audio playback immediately with cleanup."""
        # Get stream reference while holding lock
        with self._lock:
            stream_to_close = self._stream
            # Set stopping flag to prevent callbacks from executing
            self._stopping = True
            # Clear state immediately so no new operations use this stream
            self._stream = None
            self._buffer = None
            self._buffer_position = 0

        # Close stream OUTSIDE the lock to avoid deadlock with finished_callback
        if stream_to_close is not None:
            try:
                if stream_to_close.active:
                    logger.debug("Aborting active audio stream")
                    stream_to_close.abort()  # Immediate stop
                stream_to_close.close()  # Wait for termination (may call finished_callback)
                logger.debug("Audio stream closed")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}", exc_info=True)

        # Clear stopping flag after stream is fully closed
        with self._lock:
            self._stopping = False
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing.
        
        Returns:
            True if stream is active
        """
        with self._lock:
            return self._stream is not None and self._stream.active
    
    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Validate and reshape audio samples.
        
        Args:
            samples: Input samples
            
        Returns:
            Validated samples (n_samples, n_channels)
            
        Raises:
            AudioPlaybackError: If samples are invalid
        """
        if not isinstance(samples, np.ndarray):
            raise AudioPlaybackError("Samples must be a numpy array")
        
        # Ensure correct dtype
        if samples.dtype != np.int16:
            logger.warning(f"Converting samples from {samples.dtype} to int16")
            if np.issubdtype(samples.dtype, np.floating):
                # Convert float to int16
                samples = np.clip(samples, -1.0, 1.0)
                samples = (samples * AudioParams.MAX_AMPLITUDE).astype(np.int16)
            else:
                samples = samples.astype(np.int16)
        
        # Reshape to (n_samples, n_channels)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        elif samples.ndim != 2:
            raise AudioPlaybackError(f"Samples must be 1D or 2D, got {samples.ndim}D")
        
        # Validate not empty
        if len(samples) == 0:
            raise AudioPlaybackError("Cannot play empty audio samples")
        
        return samples
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
