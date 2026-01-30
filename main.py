# main.py

import tkinter as tk
from lib.app import TkApp
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

def shutdown_logging():
    logger = logging.getLogger('eeg_stimulus')
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

def setup_logging():
    logger_name = 'eeg_stimulus'
    app_logger = logging.getLogger(logger_name)

    if app_logger.hasHandlers():
        return

    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logs directory if it doesn't exist
    script_dir = os.path.dirname(__file__)
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Try to use logs dir; fall back to temp if not writable
    log_file = os.path.join(logs_dir, 'eeg_stimulus_app.log')
    try:
        with open(log_file, 'a'):
            pass
    except (OSError, IOError):
        import tempfile
        log_file = os.path.join(tempfile.gettempdir(), 'eeg_stimulus_app.log')

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in terminal

    app_logger.addHandler(file_handler)
    app_logger.addHandler(console_handler)

    # Suppress noisy loggers
    for noisy in ['matplotlib.font_manager', 'matplotlib.pyplot', 'PIL']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    app_logger.info("=== EEG Stimulus App Started ===")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger('eeg_stimulus')

    def handle_tk_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Application interrupted by user.")
            root.quit()
        else:
            logger.error("Unhandled Tkinter callback exception", exc_info=(exc_type, exc_value, exc_traceback))

    try:
        root = tk.Tk()
        root.report_callback_exception = handle_tk_exception

        def on_closing():
            logger.info("Application closing via window.")
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        app = TkApp(root)
        root.mainloop()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (main thread).")
    except Exception:
        logger.exception("Unhandled exception in main thread:")
        raise
    finally:
        shutdown_logging()