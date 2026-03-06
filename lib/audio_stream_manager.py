# lib/audio_stream_manager.py

"""
Audio stream management for the EEG Stimulus Package.

Uses a single persistent OutputStream that runs for the lifetime of the app.
play() swaps the active buffer; stop() clears it.  Between clips the stream
outputs silence, keeping the audio device active and preventing the low-power
sleep that causes finished_callback to silently drop on ChromeOS / macOS.
"""

import logging
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from lib.constants import AudioParams
from lib.exceptions import AudioDeviceError, AudioPlaybackError

logger = logging.getLogger('eeg_stimulus.audio_stream')


class AudioStreamManager:
    """Persistent-stream audio playback.

    One OutputStream is opened at startup and never closed until shutdown().
    Playback is controlled by swapping self._buffer:
      - buffer set  → stream_callback reads from it and advances position
      - buffer None → stream_callback outputs silence
    Completion is detected when the buffer is exhausted inside stream_callback,
    which fires on_finish exactly once (guarded by _finish_fired).
    """

    def __init__(self):
        """Open the persistent audio stream."""
        self._lock = threading.Lock()
        self._buffer: Optional[np.ndarray] = None
        self._buffer_position: int = 0
        self._on_finish: Optional[Callable[[], None]] = None
        self._on_onset: Optional[Callable[[float], None]] = None
        self._finish_fired: bool = False
        self._stream: Optional[sd.OutputStream] = None
        self._channels: int = 2  # Fixed stereo; mono inputs are upmixed
        self._start_persistent_stream()
        logger.info("AudioStreamManager initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self,
             samples: np.ndarray,
             sample_rate: int = AudioParams.SAMPLE_RATE,
             on_finish: Optional[Callable[[], None]] = None,
             on_onset: Optional[Callable[[float], None]] = None) -> None:
        """Queue audio samples for playback.

        Args:
            samples: Audio samples (int16, shape: (n_samples, n_channels))
            sample_rate: Sample rate in Hz (must match stream rate, 44100)
            on_finish: Callback executed when the buffer is exhausted

        Raises:
            AudioPlaybackError: If samples are invalid or stream unavailable
        """
        samples = self._validate_samples(samples)

        # Upmix mono to stereo to match the persistent stream channel count
        if samples.shape[1] == 1 and self._channels == 2:
            samples = np.column_stack([samples, samples])

        if sample_rate != AudioParams.SAMPLE_RATE:
            logger.warning(
                f"Sample rate {sample_rate}Hz differs from stream rate "
                f"{AudioParams.SAMPLE_RATE}Hz; audio may play at wrong speed"
            )

        with self._lock:
            if self._stream is None:
                raise AudioPlaybackError("Persistent stream is not available")
            was_playing = (
                self._buffer is not None and
                self._buffer_position < len(self._buffer)
            )
            self._buffer = samples.copy()
            self._buffer_position = 0
            self._on_finish = on_finish
            self._on_onset = on_onset
            self._finish_fired = False

        if was_playing:
            logger.warning("play() called while previous buffer active, replacing buffer")

        logger.debug(
            f"Audio stream configured with buffer size={len(samples)} "
            f"samples at {sample_rate}Hz"
        )

    def stop(self) -> None:
        """Stop current playback; persistent stream continues outputting silence."""
        with self._lock:
            self._buffer = None
            self._buffer_position = 0
            self._on_finish = None
            self._on_onset = None
            self._finish_fired = True   # Block any in-flight completion
        logger.debug("Audio playback stopped")

    def shutdown(self) -> None:
        """Close the persistent stream. Call once on application exit."""
        self.stop()
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
                logger.info("Persistent audio stream closed")
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}", exc_info=True)

    def is_playing(self) -> bool:
        """Return True while the buffer has unread samples."""
        with self._lock:
            return (
                self._buffer is not None and
                self._buffer_position < len(self._buffer)
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_persistent_stream(self) -> None:
        """Open the OutputStream and start it."""

        def stream_callback(outdata, frames, time_info, status):
            if status:
                logger.warning(f"Audio stream status: {status}")

            fire_finish = None
            fire_onset = None

            with self._lock:
                if self._buffer is None or self._buffer_position >= len(self._buffer):
                    # No audio queued — output silence
                    outdata.fill(0)
                    # First arrival at exhaustion (not a deliberate stop)
                    if self._on_finish is not None and not self._finish_fired:
                        self._finish_fired = True
                        fire_finish = self._on_finish
                        self._on_finish = None
                else:
                    # First chunk of a new buffer — capture DAC onset time
                    if self._buffer_position == 0 and self._on_onset is not None:
                        fire_onset = self._on_onset
                        self._on_onset = None

                    available = len(self._buffer) - self._buffer_position
                    chunk_size = min(frames, available)
                    chunk = self._buffer[
                        self._buffer_position:self._buffer_position + chunk_size
                    ]
                    self._buffer_position += chunk_size

                    outdata[:chunk_size] = chunk
                    if chunk_size < frames:
                        # Buffer exhausted within this chunk
                        outdata[chunk_size:].fill(0)
                        if self._on_finish is not None and not self._finish_fired:
                            self._finish_fired = True
                            fire_finish = self._on_finish
                            self._on_finish = None

            # Call outside the lock: avoids holding it during external code
            if fire_onset is not None:
                try:
                    dac_time = time_info.outputBufferDacTime
                    current_time = time_info.currentTime
                    logger.debug(f"outputBufferDacTime={dac_time:.6f}s "
                                 f"currentTime={current_time:.6f}s "
                                 f"latency={dac_time - current_time:.6f}s")
                    if dac_time <= current_time:
                        logger.warning(
                            f"outputBufferDacTime ({dac_time:.6f}s) <= currentTime "
                            f"({current_time:.6f}s); output latency reported as zero or "
                            "negative. DAC timestamps will be missing in CSV. "
                            "This likely indicates broken latency reporting via CRAS."
                        )
                        fire_onset(None)
                    else:
                        fire_onset(dac_time)
                except Exception as e:
                    logger.error(f"Error in on_onset callback: {e}", exc_info=True)
            if fire_finish is not None:
                try:
                    fire_finish()
                except Exception as e:
                    logger.error(f"Error in on_finish callback: {e}", exc_info=True)

        try:
            stream = sd.OutputStream(
                samplerate=AudioParams.SAMPLE_RATE,
                channels=self._channels,
                dtype=AudioParams.BUFFER_DTYPE,
                callback=stream_callback,
                latency=max(float(AudioParams.STREAM_LATENCY), 0.1),
            )
            stream.start()
            with self._lock:
                self._stream = stream
            logger.info(
                f"Persistent audio stream started "
                f"({AudioParams.SAMPLE_RATE}Hz, {self._channels}ch)"
            )
        except sd.PortAudioError as e:
            error_msg = f"Audio device error: {e}"
            logger.error(error_msg)
            raise AudioDeviceError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to start persistent audio stream: {e}"
            logger.error(error_msg, exc_info=True)
            raise AudioDeviceError(error_msg) from e

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

        if samples.dtype != np.int16:
            logger.warning(f"Converting samples from {samples.dtype} to int16")
            if np.issubdtype(samples.dtype, np.floating):
                samples = np.clip(samples, -1.0, 1.0)
                samples = (samples * AudioParams.MAX_AMPLITUDE).astype(np.int16)
            else:
                samples = samples.astype(np.int16)

        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        elif samples.ndim != 2:
            raise AudioPlaybackError(f"Samples must be 1D or 2D, got {samples.ndim}D")

        if len(samples) == 0:
            raise AudioPlaybackError("Cannot play empty audio samples")

        return samples

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
