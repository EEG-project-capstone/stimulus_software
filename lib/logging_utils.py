# lib/logging_utils.py

"""
Logging utilities for the EEG Stimulus Package.
Provides context managers and helpers for cleaner logging.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger('eeg_stimulus')


@contextmanager
def log_operation(operation_name: str, level: int = logging.INFO):
    """Context manager for logging operations with timing.

    Args:
        operation_name: Name of the operation
        level: Logging level for success messages

    Example:
        with log_operation("stimulus_preparation"):
            prepare_stimuli()
    """
    logger.log(level, f"Starting: {operation_name}")
    start_time = time.time()

    try:
        yield
        duration = time.time() - start_time
        logger.log(level, f"Completed: {operation_name} ({duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed: {operation_name} ({duration:.2f}s) - {e}")
        raise


@contextmanager
def log_phase(phase_name: str, logger_instance: Optional[logging.Logger] = None):
    """Context manager for logging phases of operation.

    Args:
        phase_name: Name of the phase
        logger_instance: Specific logger to use (default: root eeg_stimulus logger)

    Example:
        with log_phase("audio_loading"):
            load_audio_files()
    """
    log = logger_instance or logger
    log.debug(f"Phase started: {phase_name}")

    try:
        yield
        log.debug(f"Phase completed: {phase_name}")
    except Exception as e:
        log.error(f"Phase failed: {phase_name} - {e}")
        raise
