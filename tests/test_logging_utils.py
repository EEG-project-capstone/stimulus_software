# tests/test_logging_utils.py
"""Tests for logging_utils.py - Logging context managers."""

import pytest
import sys
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.logging_utils import log_operation, log_phase


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_logs_start_message(self, caplog):
        """log_operation should log 'Starting:' message."""
        with caplog.at_level(logging.INFO):
            with log_operation("test_operation"):
                pass

        assert any("Starting: test_operation" in record.message for record in caplog.records)

    def test_logs_completed_message(self, caplog):
        """log_operation should log 'Completed:' message on success."""
        with caplog.at_level(logging.INFO):
            with log_operation("test_operation"):
                pass

        assert any("Completed: test_operation" in record.message for record in caplog.records)

    def test_logs_timing_info(self, caplog):
        """log_operation should include timing in completion message."""
        with caplog.at_level(logging.INFO):
            with log_operation("test_operation"):
                time.sleep(0.1)  # Sleep for measurable time

        # Find the completed message
        completed_messages = [r.message for r in caplog.records if "Completed:" in r.message]
        assert len(completed_messages) == 1
        # Check it contains seconds format
        assert "s)" in completed_messages[0]

    def test_logs_failed_message_on_exception(self, caplog):
        """log_operation should log 'Failed:' message on exception."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                with log_operation("failing_operation"):
                    raise ValueError("test error")

        assert any("Failed: failing_operation" in record.message for record in caplog.records)

    def test_exception_is_propagated(self):
        """log_operation should re-raise exceptions."""
        with pytest.raises(RuntimeError, match="propagated error"):
            with log_operation("error_operation"):
                raise RuntimeError("propagated error")

    def test_custom_log_level(self, caplog):
        """log_operation should use specified log level."""
        with caplog.at_level(logging.DEBUG):
            with log_operation("debug_operation", level=logging.DEBUG):
                pass

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) >= 2  # Start and complete messages

    def test_default_log_level_is_info(self, caplog):
        """log_operation should default to INFO level."""
        with caplog.at_level(logging.INFO):
            with log_operation("info_operation"):
                pass

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 2  # Start and complete messages

    def test_nested_operations(self, caplog):
        """Nested log_operations should work correctly."""
        with caplog.at_level(logging.INFO):
            with log_operation("outer"):
                with log_operation("inner"):
                    pass

        messages = [r.message for r in caplog.records]
        assert any("Starting: outer" in m for m in messages)
        assert any("Starting: inner" in m for m in messages)
        assert any("Completed: inner" in m for m in messages)
        assert any("Completed: outer" in m for m in messages)


class TestLogPhase:
    """Tests for log_phase context manager."""

    def test_logs_phase_started(self, caplog):
        """log_phase should log 'Phase started:' message."""
        with caplog.at_level(logging.DEBUG):
            with log_phase("test_phase"):
                pass

        assert any("Phase started: test_phase" in record.message for record in caplog.records)

    def test_logs_phase_completed(self, caplog):
        """log_phase should log 'Phase completed:' message on success."""
        with caplog.at_level(logging.DEBUG):
            with log_phase("test_phase"):
                pass

        assert any("Phase completed: test_phase" in record.message for record in caplog.records)

    def test_logs_phase_failed_on_exception(self, caplog):
        """log_phase should log 'Phase failed:' on exception."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                with log_phase("failing_phase"):
                    raise ValueError("test error")

        assert any("Phase failed: failing_phase" in record.message for record in caplog.records)

    def test_exception_is_propagated(self):
        """log_phase should re-raise exceptions."""
        with pytest.raises(KeyError, match="missing_key"):
            with log_phase("error_phase"):
                raise KeyError("missing_key")

    def test_uses_custom_logger(self, caplog):
        """log_phase should use provided logger instance."""
        custom_logger = logging.getLogger("custom_test_logger")

        with caplog.at_level(logging.DEBUG, logger="custom_test_logger"):
            with log_phase("custom_phase", logger_instance=custom_logger):
                pass

        custom_records = [r for r in caplog.records if r.name == "custom_test_logger"]
        assert len(custom_records) >= 2  # Started and completed

    def test_default_logger_is_eeg_stimulus(self, caplog):
        """log_phase should default to eeg_stimulus logger."""
        with caplog.at_level(logging.DEBUG, logger="eeg_stimulus"):
            with log_phase("default_phase"):
                pass

        eeg_records = [r for r in caplog.records if r.name == "eeg_stimulus"]
        assert len(eeg_records) >= 2

    def test_nested_phases(self, caplog):
        """Nested log_phase calls should work correctly."""
        with caplog.at_level(logging.DEBUG):
            with log_phase("outer_phase"):
                with log_phase("inner_phase"):
                    pass

        messages = [r.message for r in caplog.records]
        assert any("Phase started: outer_phase" in m for m in messages)
        assert any("Phase started: inner_phase" in m for m in messages)
        assert any("Phase completed: inner_phase" in m for m in messages)
        assert any("Phase completed: outer_phase" in m for m in messages)


class TestLogOperationEdgeCases:
    """Edge case tests for log_operation."""

    def test_empty_operation_name(self, caplog):
        """log_operation should handle empty operation name."""
        with caplog.at_level(logging.INFO):
            with log_operation(""):
                pass

        assert any("Starting: " in record.message for record in caplog.records)

    def test_special_characters_in_name(self, caplog):
        """log_operation should handle special characters in name."""
        with caplog.at_level(logging.INFO):
            with log_operation("test/operation:with-special_chars"):
                pass

        assert any("test/operation:with-special_chars" in record.message for record in caplog.records)

    def test_very_fast_operation(self, caplog):
        """log_operation should handle very fast operations."""
        with caplog.at_level(logging.INFO):
            with log_operation("instant_operation"):
                pass  # No sleep, instant completion

        completed = [r for r in caplog.records if "Completed:" in r.message]
        assert len(completed) == 1
        # Should show 0.00s or similar
        assert "0." in completed[0].message


class TestLogPhaseEdgeCases:
    """Edge case tests for log_phase."""

    def test_empty_phase_name(self, caplog):
        """log_phase should handle empty phase name."""
        with caplog.at_level(logging.DEBUG):
            with log_phase(""):
                pass

        assert any("Phase started: " in record.message for record in caplog.records)

    def test_none_logger_uses_default(self, caplog):
        """log_phase with None logger should use default."""
        with caplog.at_level(logging.DEBUG):
            with log_phase("test_phase", logger_instance=None):
                pass

        assert any("Phase started: test_phase" in record.message for record in caplog.records)
