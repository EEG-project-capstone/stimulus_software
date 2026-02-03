# tests/test_audio_stream_manager.py
"""Tests for audio_stream_manager.py - Thread-safe audio playback."""

import pytest
import sys
import numpy as np
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.audio_stream_manager import AudioStreamManager
from lib.exceptions import AudioPlaybackError, AudioDeviceError
from lib.constants import AudioParams


class TestAudioStreamManagerInit:
    """Tests for AudioStreamManager initialization."""

    def test_initialization(self):
        """AudioStreamManager should initialize with empty state."""
        manager = AudioStreamManager()
        assert manager._stream is None
        assert manager._buffer is None
        assert manager._buffer_position == 0

    def test_is_playing_returns_false_initially(self):
        """is_playing should return False when no stream is active."""
        manager = AudioStreamManager()
        assert manager.is_playing() is False


class TestSampleValidation:
    """Tests for _validate_samples method."""

    def test_validates_1d_array(self):
        """1D arrays should be reshaped to (n, 1)."""
        manager = AudioStreamManager()
        samples = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        result = manager._validate_samples(samples)
        assert result.shape == (5, 1)

    def test_validates_2d_array(self):
        """2D arrays should pass through."""
        manager = AudioStreamManager()
        samples = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
        result = manager._validate_samples(samples)
        assert result.shape == (3, 2)

    def test_converts_float_to_int16(self):
        """Float samples should be converted to int16."""
        manager = AudioStreamManager()
        samples = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        result = manager._validate_samples(samples)
        assert result.dtype == np.int16

    def test_clips_float_values(self):
        """Float values outside [-1, 1] should be clipped."""
        manager = AudioStreamManager()
        samples = np.array([2.0, -2.0], dtype=np.float32)
        result = manager._validate_samples(samples)
        # Should be clipped to max int16 values
        assert result[0, 0] == AudioParams.MAX_AMPLITUDE
        assert result[1, 0] == -AudioParams.MAX_AMPLITUDE

    def test_rejects_non_array(self):
        """Non-array input should raise AudioPlaybackError."""
        manager = AudioStreamManager()
        with pytest.raises(AudioPlaybackError, match="must be a numpy array"):
            manager._validate_samples([1, 2, 3])

    def test_rejects_3d_array(self):
        """3D arrays should raise AudioPlaybackError."""
        manager = AudioStreamManager()
        samples = np.zeros((2, 3, 4), dtype=np.int16)
        with pytest.raises(AudioPlaybackError, match="must be 1D or 2D"):
            manager._validate_samples(samples)

    def test_rejects_empty_array(self):
        """Empty arrays should raise AudioPlaybackError."""
        manager = AudioStreamManager()
        samples = np.array([], dtype=np.int16)
        with pytest.raises(AudioPlaybackError, match="Cannot play empty"):
            manager._validate_samples(samples)

    def test_converts_int32_to_int16(self):
        """int32 samples should be converted to int16."""
        manager = AudioStreamManager()
        samples = np.array([100, 200, 300], dtype=np.int32)
        result = manager._validate_samples(samples)
        assert result.dtype == np.int16


class TestPlayMethod:
    """Tests for play method with mocked sounddevice."""

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_play_creates_stream(self, mock_output_stream):
        """play should create an OutputStream."""
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1, 2], [3, 4]], dtype=np.int16)

        manager.play(samples)

        mock_output_stream.assert_called_once()
        mock_stream.start.assert_called_once()

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_play_uses_correct_sample_rate(self, mock_output_stream):
        """play should use specified sample rate."""
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1], [2]], dtype=np.int16)

        manager.play(samples, sample_rate=48000)

        call_kwargs = mock_output_stream.call_args[1]
        assert call_kwargs['samplerate'] == 48000

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_play_stops_existing_stream(self, mock_output_stream):
        """play should stop any existing stream before starting new one."""
        mock_stream1 = MagicMock()
        mock_stream1.active = True
        mock_stream2 = MagicMock()
        mock_stream2.active = False
        mock_output_stream.side_effect = [mock_stream1, mock_stream2]

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)

        # First play
        manager.play(samples)
        # Second play should stop first stream
        manager.play(samples)

        # First stream should have been stopped
        mock_stream1.abort.assert_called()

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_play_sets_buffer(self, mock_output_stream):
        """play should copy samples to internal buffer."""
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[100, 200], [300, 400]], dtype=np.int16)

        manager.play(samples)

        assert manager._buffer is not None
        assert np.array_equal(manager._buffer, samples)
        assert manager._buffer_position == 0


class TestStopMethod:
    """Tests for stop method."""

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_stop_clears_state(self, mock_output_stream):
        """stop should clear buffer and stream state."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples)

        manager.stop()

        assert manager._stream is None
        assert manager._buffer is None
        assert manager._buffer_position == 0

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_stop_aborts_active_stream(self, mock_output_stream):
        """stop should abort active stream."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples)

        manager.stop()

        mock_stream.abort.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_stop_with_no_stream(self):
        """stop should handle case when no stream exists."""
        manager = AudioStreamManager()
        # Should not raise
        manager.stop()
        assert manager._stream is None


class TestIsPlayingMethod:
    """Tests for is_playing method."""

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_is_playing_when_stream_active(self, mock_output_stream):
        """is_playing should return True when stream is active."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples)

        assert manager.is_playing() is True

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_is_playing_when_stream_inactive(self, mock_output_stream):
        """is_playing should return False when stream is not active."""
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples)

        assert manager.is_playing() is False


class TestOnFinishCallback:
    """Tests for on_finish callback functionality."""

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_on_finish_callback_stored(self, mock_output_stream):
        """on_finish callback should be usable in finished_callback."""
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        callback_called = [False]

        def on_finish():
            callback_called[0] = True

        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples, on_finish=on_finish)

        # Get the finished_callback that was passed to OutputStream
        call_kwargs = mock_output_stream.call_args[1]
        finished_callback = call_kwargs['finished_callback']

        # Simulate stream finishing
        finished_callback()

        assert callback_called[0] is True


class TestErrorHandling:
    """Tests for error handling."""

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_port_audio_error_raises_device_error(self, mock_output_stream):
        """PortAudioError should raise AudioDeviceError."""
        import sounddevice as sd
        # Create a proper PortAudioError with a valid error code
        # Use a mock that behaves like PortAudioError for string conversion
        error = sd.PortAudioError(-9999)
        # Override __str__ to return a proper string
        error.args = ("Mocked audio device error",)
        mock_output_stream.side_effect = error

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)

        with pytest.raises(AudioDeviceError):
            manager.play(samples)

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_generic_error_raises_playback_error(self, mock_output_stream):
        """Generic errors should raise AudioPlaybackError."""
        mock_output_stream.side_effect = RuntimeError("Unknown error")

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)

        with pytest.raises(AudioPlaybackError):
            manager.play(samples)


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_stop_calls(self):
        """Multiple concurrent stop calls should not raise errors."""
        manager = AudioStreamManager()

        def stop_thread():
            for _ in range(10):
                manager.stop()
                time.sleep(0.001)

        threads = [threading.Thread(target=stop_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert manager._stream is None

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_concurrent_is_playing_calls(self, mock_output_stream):
        """Concurrent is_playing calls should not raise errors."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples)

        results = []

        def check_playing():
            for _ in range(10):
                results.append(manager.is_playing())
                time.sleep(0.001)

        threads = [threading.Thread(target=check_playing) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be boolean
        assert all(isinstance(r, bool) for r in results)


class TestCleanup:
    """Tests for cleanup behavior."""

    @patch('lib.audio_stream_manager.sd.OutputStream')
    def test_del_calls_stop(self, mock_output_stream):
        """__del__ should call stop to clean up resources."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream.return_value = mock_stream

        manager = AudioStreamManager()
        samples = np.array([[1]], dtype=np.int16)
        manager.play(samples)

        # Manually call __del__
        manager.__del__()

        # Stream should have been stopped
        mock_stream.abort.assert_called()
