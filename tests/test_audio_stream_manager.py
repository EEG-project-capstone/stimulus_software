# tests/test_audio_stream_manager.py
"""Tests for audio_stream_manager.py - Persistent-stream audio playback."""

import pytest
import numpy as np
import threading
import time
from unittest.mock import MagicMock, patch

from lib.audio_stream_manager import AudioStreamManager
from lib.exceptions import AudioPlaybackError, AudioDeviceError
from lib.constants import AudioParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asm():
    """Return (manager, mock_stream, stream_callback).

    Creates an AudioStreamManager with sd.OutputStream mocked so no real
    audio device is needed.  The patch only needs to be active during
    __init__, after which the manager holds a reference to mock_stream.
    """
    mock_stream = MagicMock()
    mock_stream.active = True
    with patch('lib.audio_stream_manager.sd.OutputStream') as mock_cls:
        mock_cls.return_value = mock_stream
        manager = AudioStreamManager()
        stream_callback = mock_cls.call_args[1]['callback']
    return manager, mock_stream, stream_callback


def _outdata(frames, channels=2):
    """Blank output buffer of the right shape."""
    return np.zeros((frames, channels), dtype=np.int16)


def _stereo(n, value=1000):
    """n-frame stereo int16 buffer filled with value."""
    return np.full((n, 2), value, dtype=np.int16)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:

    def test_creates_persistent_stream_on_init(self):
        mock_stream = MagicMock()
        with patch('lib.audio_stream_manager.sd.OutputStream') as mock_cls:
            mock_cls.return_value = mock_stream
            AudioStreamManager()
        mock_cls.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_stream_uses_correct_sample_rate(self):
        with patch('lib.audio_stream_manager.sd.OutputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            AudioStreamManager()
        assert mock_cls.call_args[1]['samplerate'] == AudioParams.SAMPLE_RATE

    def test_stream_uses_stereo(self):
        with patch('lib.audio_stream_manager.sd.OutputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            AudioStreamManager()
        assert mock_cls.call_args[1]['channels'] == 2

    def test_initial_state_no_buffer(self):
        manager, _, _ = _make_asm()
        assert manager._buffer is None
        assert manager._buffer_position == 0

    def test_initial_is_playing_false(self):
        manager, _, _ = _make_asm()
        assert manager.is_playing() is False

    def test_port_audio_error_raises_device_error(self):
        import sounddevice as sd
        err = sd.PortAudioError(-9999)
        err.args = ("mock device error",)
        with patch('lib.audio_stream_manager.sd.OutputStream', side_effect=err):
            with pytest.raises(AudioDeviceError):
                AudioStreamManager()

    def test_generic_error_raises_device_error(self):
        with patch('lib.audio_stream_manager.sd.OutputStream',
                   side_effect=RuntimeError("unknown")):
            with pytest.raises(AudioDeviceError):
                AudioStreamManager()


# ---------------------------------------------------------------------------
# play()
# ---------------------------------------------------------------------------

class TestPlay:

    def test_sets_buffer_and_position(self):
        manager, _, _ = _make_asm()
        samples = _stereo(100)
        manager.play(samples)
        assert manager._buffer is not None
        assert len(manager._buffer) == 100
        assert manager._buffer_position == 0

    def test_is_playing_true_after_play(self):
        manager, _, _ = _make_asm()
        manager.play(_stereo(100))
        assert manager.is_playing() is True

    def test_upmixes_mono_to_stereo(self):
        manager, _, _ = _make_asm()
        mono = np.ones((50, 1), dtype=np.int16) * 500
        manager.play(mono)
        assert manager._buffer.shape == (50, 2)

    def test_play_resets_finish_fired(self):
        manager, _, _ = _make_asm()
        manager._finish_fired = True
        manager.play(_stereo(10))
        assert manager._finish_fired is False

    def test_play_stores_on_finish(self):
        manager, _, _ = _make_asm()
        cb = MagicMock()
        manager.play(_stereo(10), on_finish=cb)
        assert manager._on_finish is cb

    def test_play_raises_if_stream_unavailable(self):
        manager, _, _ = _make_asm()
        manager._stream = None
        with pytest.raises(AudioPlaybackError):
            manager.play(_stereo(10))

    def test_sample_rate_mismatch_logs_warning(self, caplog):
        import logging
        manager, _, _ = _make_asm()
        with caplog.at_level(logging.WARNING, logger='eeg_stimulus.audio_stream'):
            manager.play(_stereo(10), sample_rate=48000)
        assert any('wrong speed' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

class TestStop:

    def test_clears_buffer(self):
        manager, _, _ = _make_asm()
        manager.play(_stereo(100))
        manager.stop()
        assert manager._buffer is None
        assert manager._buffer_position == 0

    def test_is_playing_false_after_stop(self):
        manager, _, _ = _make_asm()
        manager.play(_stereo(100))
        manager.stop()
        assert manager.is_playing() is False

    def test_stream_still_open_after_stop(self):
        manager, mock_stream, _ = _make_asm()
        manager.play(_stereo(10))
        manager.stop()
        # Persistent stream must NOT be closed
        mock_stream.stop.assert_not_called()
        mock_stream.close.assert_not_called()
        assert manager._stream is mock_stream

    def test_stop_sets_finish_fired_to_prevent_callback(self):
        manager, _, _ = _make_asm()
        cb = MagicMock()
        manager.play(_stereo(10), on_finish=cb)
        manager.stop()
        assert manager._finish_fired is True
        assert manager._on_finish is None

    def test_stop_with_no_buffer_is_safe(self):
        manager, _, _ = _make_asm()
        manager.stop()  # should not raise


# ---------------------------------------------------------------------------
# stream_callback behaviour
# ---------------------------------------------------------------------------

class TestStreamCallback:

    def test_outputs_buffer_data(self):
        manager, _, cb = _make_asm()
        samples = _stereo(100, value=1000)
        manager.play(samples)

        outdata = _outdata(50)
        cb(outdata, 50, None, None)

        assert np.all(outdata == 1000)
        assert manager._buffer_position == 50

    def test_outputs_silence_when_no_buffer(self):
        manager, _, cb = _make_asm()
        outdata = _outdata(64)
        outdata[:] = 999
        cb(outdata, 64, None, None)
        assert np.all(outdata == 0)

    def test_fires_on_finish_when_buffer_exhausted(self):
        manager, _, cb = _make_asm()
        fired = []
        manager.play(_stereo(50), on_finish=lambda: fired.append(1))

        cb(_outdata(50), 50, None, None)   # exact fit — exhausted
        cb(_outdata(50), 50, None, None)   # next call: silence

        assert len(fired) == 1

    def test_fires_on_finish_on_partial_final_chunk(self):
        manager, _, cb = _make_asm()
        fired = []
        manager.play(_stereo(30), on_finish=lambda: fired.append(1))

        cb(_outdata(50), 50, None, None)   # 30 samples read, 20 frames silence

        assert len(fired) == 1

    def test_on_finish_fires_exactly_once(self):
        manager, _, cb = _make_asm()
        fired = []
        manager.play(_stereo(50), on_finish=lambda: fired.append(1))

        for _ in range(5):
            cb(_outdata(50), 50, None, None)

        assert len(fired) == 1

    def test_stop_prevents_on_finish(self):
        manager, _, cb = _make_asm()
        cb_mock = MagicMock()
        manager.play(_stereo(50), on_finish=cb_mock)
        manager.stop()

        cb(_outdata(50), 50, None, None)

        cb_mock.assert_not_called()

    def test_partial_chunk_pads_with_silence(self):
        manager, _, cb = _make_asm()
        manager.play(_stereo(30, value=500))

        outdata = _outdata(50)
        cb(outdata, 50, None, None)

        # First 30 rows from buffer, rest silence
        assert np.all(outdata[:30] == 500)
        assert np.all(outdata[30:] == 0)

    def test_advances_buffer_position_correctly(self):
        manager, _, cb = _make_asm()
        manager.play(_stereo(100))

        cb(_outdata(30), 30, None, None)
        assert manager._buffer_position == 30
        cb(_outdata(30), 30, None, None)
        assert manager._buffer_position == 60


# ---------------------------------------------------------------------------
# Sample validation
# ---------------------------------------------------------------------------

class TestSampleValidation:

    def test_validates_1d_array(self):
        manager, _, _ = _make_asm()
        result = manager._validate_samples(np.array([1, 2, 3], dtype=np.int16))
        assert result.shape == (3, 1)

    def test_validates_2d_array(self):
        manager, _, _ = _make_asm()
        samples = np.ones((3, 2), dtype=np.int16)
        assert manager._validate_samples(samples).shape == (3, 2)

    def test_converts_float_to_int16(self):
        manager, _, _ = _make_asm()
        result = manager._validate_samples(np.array([0.5, -0.5], dtype=np.float32))
        assert result.dtype == np.int16

    def test_clips_float_values(self):
        manager, _, _ = _make_asm()
        result = manager._validate_samples(np.array([2.0, -2.0], dtype=np.float32))
        assert result[0, 0] == AudioParams.MAX_AMPLITUDE
        assert result[1, 0] == -AudioParams.MAX_AMPLITUDE

    def test_converts_int32_to_int16(self):
        manager, _, _ = _make_asm()
        result = manager._validate_samples(np.array([1, 2, 3], dtype=np.int32))
        assert result.dtype == np.int16

    def test_rejects_non_array(self):
        manager, _, _ = _make_asm()
        with pytest.raises(AudioPlaybackError, match="must be a numpy array"):
            manager._validate_samples([1, 2, 3])

    def test_rejects_3d_array(self):
        manager, _, _ = _make_asm()
        with pytest.raises(AudioPlaybackError, match="must be 1D or 2D"):
            manager._validate_samples(np.zeros((2, 3, 4), dtype=np.int16))

    def test_rejects_empty_array(self):
        manager, _, _ = _make_asm()
        with pytest.raises(AudioPlaybackError, match="Cannot play empty"):
            manager._validate_samples(np.array([], dtype=np.int16))


# ---------------------------------------------------------------------------
# shutdown() / __del__
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_shutdown_closes_stream(self):
        manager, mock_stream, _ = _make_asm()
        manager.shutdown()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_shutdown_clears_stream_reference(self):
        manager, _, _ = _make_asm()
        manager.shutdown()
        assert manager._stream is None

    def test_del_calls_shutdown(self):
        manager, mock_stream, _ = _make_asm()
        manager.__del__()
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()

    def test_double_shutdown_is_safe(self):
        manager, _, _ = _make_asm()
        manager.shutdown()
        manager.shutdown()   # should not raise


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_stop_calls(self):
        manager, _, _ = _make_asm()
        manager.play(_stereo(10000))

        def do_stop():
            for _ in range(10):
                manager.stop()
                time.sleep(0.001)

        threads = [threading.Thread(target=do_stop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert manager._buffer is None

    def test_concurrent_is_playing_calls(self):
        manager, _, _ = _make_asm()
        manager.play(_stereo(10000))
        results = []

        def check():
            for _ in range(20):
                results.append(manager.is_playing())
                time.sleep(0.001)

        threads = [threading.Thread(target=check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(isinstance(r, bool) for r in results)
