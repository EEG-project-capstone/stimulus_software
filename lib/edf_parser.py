# lib/edf_parser.py
import mne
import numpy as np
import pandas as pd

class EDFParser:
    def __init__(self, edf_path):
        self.edf_path = edf_path
        self.raw = None
        self.info = {}
        self.annotations = None

    def load_edf(self):
        """Load the EDF file using MNE."""
        try:
            self.raw = mne.io.read_raw_edf(self.edf_path, preload=True)
            self.annotations = self.raw.annotations
            self._extract_info()
        except Exception as e:
            raise RuntimeError(f"Failed to load EDF file: {e}")

    def _extract_info(self):
        """Extract basic information from the loaded raw object."""
        if self.raw is None:
            raise RuntimeError("EDF file not loaded. Call load_edf() first.")

        self.info = {
            'n_channels': self.raw.info['nchan'],
            'ch_names': self.raw.ch_names,
            'sfreq': self.raw.info['sfreq'],
            'meas_date': self.raw.info.get('meas_date', 'Unknown'),
            'n_times': len(self.raw.times),
            'duration': len(self.raw.times) / self.raw.info['sfreq'], # in seconds
            'annotations_count': len(self.annotations) if self.annotations is not None else 0
        }

    def get_channel_names(self):
        """Return the list of channel names."""
        return self.info.get('ch_names', [])

    def get_info_summary(self):
        """Return a dictionary with basic info."""
        return self.info

    def get_data_segment(self, start_sec=0, duration_sec=10, ch_names=None):
        """
        Get a segment of the raw data.
        :param start_sec: Start time in seconds.
        :param duration_sec: Duration of the segment in seconds.
        :param ch_names: List of channel names to extract. If None, uses all.
        :return: Tuple of (data, times) as numpy arrays.
        """
        if self.raw is None:
            raise RuntimeError("EDF file not loaded. Call load_edf() first.")

        start_sample = int(start_sec * self.raw.info['sfreq'])
        end_sample = int((start_sec + duration_sec) * self.raw.info['sfreq'])

        # Ensure indices are within bounds
        end_sample = min(end_sample, len(self.raw.times))
        start_sample = max(0, start_sample)

        if ch_names is None:
            ch_names = self.get_channel_names()

        # Pick channels and get data
        picks = mne.pick_channels(self.raw.ch_names, ch_names)
        data, times = self.raw.get_data(picks=picks, start=start_sample, stop=end_sample, return_times=True)
        return data, times

    def get_annotations(self):
        """Return the annotations object (if any)."""
        return self.annotations

    def get_channel_types(self):
        """Return the channel types."""
        if self.raw is None:
            raise RuntimeError("EDF file not loaded. Call load_edf() first.")
        return self.raw.get_channel_types()

    def get_raw_object(self):
        """Return the raw MNE object. Use cautiously."""
        return self.raw

    # Optional: Add methods for filtering, resampling, etc., later if needed.
    