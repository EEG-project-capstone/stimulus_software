# lib/edf_parser.py

import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd

from lib.exceptions import EDFFileError

logger = logging.getLogger('eeg_stimulus')


class EDFParser:
    def __init__(self, edf_path: str) -> None:
        self.edf_path = Path(edf_path)
        self.raw = None
        self.sync_sample = None  # Store the detected sync point
        self.sync_time = None    # Store the time of the sync point
        self._header_info = None  # Cache for raw header info

    def validate_file(self) -> dict:
        """Validate the EDF file exists and has valid format.

        Returns:
            dict with validation results: {'valid': bool, 'error': str or None, 'header': dict or None}
        """
        result = {'valid': False, 'error': None, 'header': None}

        # Check file exists
        if not self.edf_path.exists():
            result['error'] = f"EDF file not found: {self.edf_path}"
            logger.error(result['error'])
            return result

        # Check file extension
        if self.edf_path.suffix.lower() not in ['.edf', '.bdf']:
            result['error'] = f"Invalid file extension: {self.edf_path.suffix}. Expected .edf or .bdf"
            logger.warning(result['error'])
            # Continue anyway - might still be valid

        # Check file is readable and has minimum size for EDF header (256 bytes)
        try:
            file_size = self.edf_path.stat().st_size
            if file_size < 256:
                result['error'] = f"File too small to be valid EDF: {file_size} bytes"
                logger.error(result['error'])
                return result
        except OSError as e:
            result['error'] = f"Cannot read file: {e}"
            logger.error(result['error'])
            return result

        # Parse EDF header manually to validate format
        try:
            header = self._read_edf_header()
            result['header'] = header
            result['valid'] = True
            logger.info(f"EDF validation successful: {header.get('n_channels', '?')} channels, "
                       f"version: {header.get('version', 'unknown')}")
        except Exception as e:
            result['error'] = f"Invalid EDF format: {e}"
            logger.error(result['error'])
            return result

        return result

    def _read_edf_header(self) -> dict:
        """Read and parse the EDF header without loading full data.

        Returns:
            dict with header information
        """
        with open(self.edf_path, 'rb') as f:
            # Fixed header (256 bytes)
            version = f.read(8).decode('ascii').strip()
            patient_id = f.read(80).decode('ascii', errors='replace').strip()
            recording_id = f.read(80).decode('ascii', errors='replace').strip()
            start_date = f.read(8).decode('ascii').strip()
            start_time = f.read(8).decode('ascii').strip()
            header_bytes = int(f.read(8).decode('ascii').strip())
            reserved = f.read(44).decode('ascii', errors='replace').strip()
            n_records = int(f.read(8).decode('ascii').strip())
            record_duration = float(f.read(8).decode('ascii').strip())
            n_channels = int(f.read(4).decode('ascii').strip())

            # Read channel labels (16 bytes each)
            channel_labels = []
            for _ in range(n_channels):
                label = f.read(16).decode('ascii', errors='replace').strip()
                channel_labels.append(label)

        # Calculate duration
        total_duration = n_records * record_duration

        self._header_info = {
            'version': version,
            'patient_id': patient_id,
            'recording_id': recording_id,
            'start_date': start_date,
            'start_time': start_time,
            'header_bytes': header_bytes,
            'reserved': reserved,
            'n_records': n_records,
            'record_duration': record_duration,
            'n_channels': n_channels,
            'channel_labels': channel_labels,
            'total_duration_sec': total_duration,
        }

        return self._header_info

    def load_edf(self) -> None:
        """Load the EDF file with validation."""
        logger.info(f"Loading EDF file: {self.edf_path}")

        # Validate first
        validation = self.validate_file()
        if not validation['valid']:
            raise EDFFileError(validation['error'])

        try:
            # Suppress MNE's verbose output
            with mne.utils.use_log_level('WARNING'):
                self.raw = mne.io.read_raw_edf(str(self.edf_path), preload=True)
            logger.info(f"Loaded raw data with {len(self.raw.ch_names)} channels, "
                       f"sfreq={self.raw.info['sfreq']} Hz")
        except Exception as e:
            raise EDFFileError(f"Failed to load EDF with MNE: {e}") from e

    def get_info_summary(self) -> dict:
        """Get comprehensive info about the loaded EDF.

        Returns:
            dict with EDF information including channels, duration, subject info, etc.
        """
        if self.raw is None:
            raise RuntimeError("EDF data not loaded. Call load_edf() first.")

        info = self.raw.info
        duration_sec = len(self.raw.times) / info['sfreq']

        summary = {
            # Basic info
            'ch_names': self.raw.ch_names,
            'n_channels': len(self.raw.ch_names),
            'sfreq': info['sfreq'],
            'n_times': len(self.raw.times),
            'duration': duration_sec,
            'duration_formatted': self._format_duration(duration_sec),
            'meas_date': info.get('meas_date'),

            # Subject info (if available)
            'subject_info': self._extract_subject_info(),

            # Recording info from header
            'header_info': self._header_info if self._header_info else {},

            # Channel types summary
            'channel_types': self._get_channel_type_summary(),

            # Data statistics (quick summary)
            'highpass': info.get('highpass', None),
            'lowpass': info.get('lowpass', None),
        }

        return summary

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.1f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.1f}s"
        else:
            return f"{secs:.1f}s"

    def _extract_subject_info(self) -> dict:
        """Extract subject information from MNE info structure."""
        if self.raw is None:
            logger.warning("_extract_subject_info called but EDF data not loaded")
            return {}
        info = self.raw.info
        subject = {}

        # Try to get subject info from MNE's info structure
        if info.get('subject_info'):
            subj_info = info['subject_info']
            subject['id'] = subj_info.get('id', subj_info.get('his_id', None))
            subject['sex'] = subj_info.get('sex', None)
            subject['birthday'] = subj_info.get('birthday', None)
            subject['hand'] = subj_info.get('hand', None)

        # Also try to extract from header if available
        if self._header_info:
            if not subject.get('id') and self._header_info.get('patient_id'):
                subject['id'] = self._header_info['patient_id']
            subject['recording_id'] = self._header_info.get('recording_id', None)
            subject['start_date'] = self._header_info.get('start_date', None)
            subject['start_time'] = self._header_info.get('start_time', None)

        return subject

    def _get_channel_type_summary(self) -> dict:
        """Get summary of channel types."""
        if self.raw is None:
            logger.warning("_get_channel_type_summary called but EDF data not loaded")
            return {}
        try:
            types_list = self.raw.get_channel_types()
            unique_types, counts = np.unique(types_list, return_counts=True)
            return dict(zip(unique_types, counts.tolist()))
        except Exception as e:
            logger.warning(f"Failed to get channel types: {e}")
            return {}

    def get_data_segment(self, start_sec: float, duration_sec: float, ch_names: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray]:        
        """Get a segment of EEG data. 
        Args:
            start_sec: Start time in seconds **from EDF start**.
            duration_sec: Duration to extract (may be truncated at EDF end).
            ch_names: List of channel names to extract. If None, returns all channels.  
        Returns:
            data: Array of shape (n_channels, n_times)
            times: Time vector **relative to EDF start** (not relative to start_sec!).
        """
        if self.raw is None:
            raise RuntimeError("EDF data not loaded. Call load_edf() first.")
        start_sample = int(start_sec * self.raw.info['sfreq'])
        end_sample = start_sample + int(duration_sec * self.raw.info['sfreq'])
        # Ensure indices are within bounds
        end_sample = min(end_sample, len(self.raw.times))
        start_sample = max(0, start_sample)

        if ch_names:
            # Ensure requested channels exist
            ch_names = [ch for ch in ch_names if ch in self.raw.ch_names]
            sel = mne.pick_channels(self.raw.ch_names, include=ch_names)
        else:
            sel = slice(None) # All channels

        data, times = self.raw[sel, start_sample:end_sample]

        return data, times

    def find_sync_point(self, stimulus_csv_path: str, threshold_std: float = 3, search_duration: float = 300) -> None:        
        """
        Finds the approximate start of the first command trial by detecting an audio artifact.
        Parameters:
        - stimulus_csv_path: Path to the stimulus CSV file.
        - threshold_std: Threshold for artifact detection (multiplier for std).
        - search_duration: Max time (seconds) from EDF start to search for the artifact.
        """
        if self.raw is None:
            raise RuntimeError("EDF data not loaded. Call load_edf() first.")

        logger.info(f"Attempting to find sync point using stimulus CSV: {stimulus_csv_path}")

        try:
            df = pd.read_csv(stimulus_csv_path)
        except FileNotFoundError:
            logger.error(f"Stimulus CSV file not found: {stimulus_csv_path}")
            return
        except Exception as e:
            logger.error(f"Error reading stimulus CSV: {e}")
            return

        # Find the first command trial (in the order the rows appear in the CSV)
        cmd_stim_types = ['right_command', 'right_command+p', 'left_command', 'left_command+p']
        first_cmd_trial = df[df['stim_type'].isin(cmd_stim_types)].iloc[0] if not df[df['stim_type'].isin(cmd_stim_types)].empty else None

        if first_cmd_trial is None:
            logger.warning("No command trials found in the stimulus CSV.")
            self.sync_time = None
            return
        
        original_index = df[df['stim_type'].isin(cmd_stim_types)].index.tolist()[0]
        csv_row_number = original_index + 2 

        if csv_row_number is not None:
            logger.info(
                f"This corresponds to the estimated start of the first command trial: "
                f"{first_cmd_trial['stim_type']} (CSV row {csv_row_number})."
            )

        # Search for the artifact in the EEG data from the beginning of the EDF
        search_start_sec = 0
        search_end_sec = min(search_duration, len(self.raw.times) / self.raw.info['sfreq'])

        logger.info(f"Searching for audio artifact in EDF from {search_start_sec}s to {search_end_sec}s.")

        # Use channels likely to pick up audio artifacts (e.g., Fp1, Fp2)
        frontal_chs = [ch for ch in ['Fp1', 'Fp2'] if ch in self.raw.ch_names]
        if not frontal_chs:
            frontal_chs = self.raw.ch_names[:3]  # Fallback
            logger.warning(f"No Fp1/Fp2 found. Using first 3 channels for artifact detection: {frontal_chs}")

        # Get data for the search window
        start_sample_search = int(search_start_sec * self.raw.info['sfreq'])
        end_sample_search = int(search_end_sec * self.raw.info['sfreq'])
        data_search, times_search = self.raw[frontal_chs, start_sample_search:end_sample_search]

        # Calculate amplitude envelope (mean absolute value across selected channels)
        amp_env = np.abs(data_search).mean(axis=0)

        # Calculate baseline noise level (e.g., first 10% of the search window)
        baseline_len = max(1, int(0.1 * len(amp_env)))
        baseline_amp = amp_env[:baseline_len]
        baseline_mean = np.mean(baseline_amp)
        baseline_std = np.std(baseline_amp)

        # Define threshold
        threshold = baseline_mean + threshold_std * baseline_std
        logger.debug(f"Baseline mean: {baseline_mean:.2f}, std: {baseline_std:.2f}, threshold: {threshold:.2f}")

        # Find points exceeding threshold
        above_threshold = np.where(amp_env > threshold)[0]

        if len(above_threshold) == 0:
            logger.warning(f"No audio artifact detected above threshold ({threshold:.2f}) in the search window.")
            self.sync_time = None
            return

        # Take the first point exceeding the threshold as the sync point
        peak_idx_in_search = above_threshold[0]
        self.sync_sample = start_sample_search + peak_idx_in_search
        self.sync_time = self.sync_sample / self.raw.info['sfreq']

        logger.info(f"Sync point detected at sample {self.sync_sample}, time {self.sync_time:.3f}s in EDF.")
        logger.info(
            f"This corresponds to the estimated start of the first command trial: "
            f"{first_cmd_trial['stim_type']} (CSV row {csv_row_number})."
        )