# lib/config.py

"""
Runtime configuration for the EEG Stimulus Package.

Handles per-session state (current date, output paths) and ensures output
directories exist. All file paths are defined in lib/constants.FilePaths.
"""

from pathlib import Path
import time
import logging
from typing import Optional, Dict, Any

from lib.constants import FilePaths

logger = logging.getLogger('eeg_stimulus.config')


class Config:
    """Runtime configuration: output paths and per-session state."""

    def __init__(self):
        self.current_date = time.strftime("%Y-%m-%d")
        self._create_directories()
        logger.info("Config initialized")

    def _create_directories(self):
        """Create output directories if they don't exist."""
        for dir_path in [FilePaths.RESULTS_DIR, FilePaths.EDFS_DIR]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
            except Exception as e:
                logger.warning(f"Could not create directory {dir_path}: {e}")

    def get_results_path(self, patient_id: str, date: Optional[str] = None) -> Path:
        """Generate the results CSV path for a patient.

        Args:
            patient_id: Patient identifier
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to results CSV file
        """
        if not patient_id:
            raise ValueError("Patient ID cannot be empty")
        if date is None:
            date = self.current_date
        filename = f"{patient_id}_{date}_stimulus_results.csv"
        return FilePaths.RESULTS_DIR / filename

    def get_edf_path(self, patient_id: str, date: Optional[str] = None) -> Path:
        """Generate the EDF file path for a patient.

        Args:
            patient_id: Patient identifier
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to EDF file
        """
        if not patient_id:
            raise ValueError("Patient ID cannot be empty")
        if date is None:
            date = self.current_date
        filename = f"{patient_id}_{date}.edf"
        return FilePaths.EDFS_DIR / filename

    def get_config_summary(self) -> Dict[str, Any]:
        """Return a summary of the current configuration."""
        return {
            'current_date': self.current_date,
            'result_dir': str(FilePaths.RESULTS_DIR),
            'edf_dir': str(FilePaths.EDFS_DIR),
            'sentences_available': len(list(FilePaths.SENTENCES_DIR.glob('*.wav')))
                                   if FilePaths.SENTENCES_DIR.exists() else 0
        }
