# lib/edf_parser.py

import mne
import pandas as pd
import numpy as np
from typing import Union
import logging

logger = logging.getLogger('eeg_stimulus')

class EDFParser:
    def __init__(self, edf_path: str) -> None:
        self.edf_path = edf_path
        self.raw = None
        self.sync_sample = None # Store the detected sync point
        self.sync_time = None   # Store the time of the sync point

    def load_edf(self) -> None:
        """Load the EDF file."""
        logger.info(f"Loading EDF file: {self.edf_path}")
        self.raw = mne.io.read_raw_edf(self.edf_path, preload=True)
        logger.info(f"Loaded raw data with {len(self.raw.ch_names)} channels, sfreq={self.raw.info['sfreq']} Hz")

    def get_info_summary(self) -> dict:
        """Get basic info about the loaded EDF."""
        if self.raw is None:
            raise RuntimeError("EDF data not loaded. Call load_edf() first.")
        return {
            'ch_names': self.raw.ch_names,
            'sfreq': self.raw.info['sfreq'],
            'n_times': len(self.raw.times),
            'duration': len(self.raw.times) / self.raw.info['sfreq'],
            'meas_date': self.raw.info['meas_date']
        }

    # def get_channel_types(self) -> dict:
    #     """Get a summary of channel types."""
    #     if self.raw is None:
    #         raise RuntimeError("EDF data not loaded. Call load_edf() first.")
    #     # get_channel_types() returns a list of strings for each channel
    #     types_list = self.raw.get_channel_types()
    #     # Count occurrences of each type
    #     unique_types, counts = np.unique(types_list, return_counts=True)
    #     # Return as a dictionary
    #     return dict(zip(unique_types, counts))


    def get_data_segment(self, start_sec: float, duration_sec: float, ch_names: Union[list[str], None] = None) -> tuple[np.ndarray, np.ndarray]:        
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
        cmd_trial_types = ['right_command', 'right_command+p', 'left_command', 'left_command+p']
        first_cmd_trial = df[df['trial_type'].isin(cmd_trial_types)].iloc[0] if not df[df['trial_type'].isin(cmd_trial_types)].empty else None

        if first_cmd_trial is None:
            logger.warning("No command trials found in the stimulus CSV.")
            self.sync_time = None
            return
        
        original_index = df[df['trial_type'].isin(cmd_trial_types)].index[0]
        csv_row_number = original_index + 2 

        if csv_row_number is not None:
            logger.info(
                f"This corresponds to the estimated start of the first command trial: "
                f"{first_cmd_trial['trial_type']} (CSV row {csv_row_number})."
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
            f"{first_cmd_trial['trial_type']} (CSV row {csv_row_number})."
        )