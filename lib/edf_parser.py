# lib/edf_parser.py

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.exceptions import EDFFileError

logger = logging.getLogger('eeg_stimulus')


class EDFParser:
    def __init__(self, edf_path: str) -> None:
        self.edf_path = Path(edf_path)
        self.raw: Optional[Any] = None
        self._header_info: Optional[dict[str, Any]] = None  # Cache for raw header info

    def validate_file(self) -> dict[str, Any]:
        """Validate the EDF file exists and has valid format.

        Returns:
            dict with validation results: {'valid': bool, 'error': str or None, 'header': dict or None}
        """
        result: dict[str, Any] = {'valid': False, 'error': None, 'header': None}

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

    def _read_edf_header(self) -> dict[str, Any]:
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
            import mne
            import warnings
            # preload=False memory-maps the file — samples are read from disk on demand
            # rather than loading the entire recording into RAM upfront.
            with mne.utils.use_log_level('ERROR'), \
                 warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='Omitted.*annotation.*outside data range',
                    category=RuntimeWarning)
                self.raw = mne.io.read_raw_edf(str(self.edf_path), preload=False)
            logger.info(f"Loaded raw data with {len(self.raw.ch_names)} channels, "
                       f"sampling rate={self.raw.info['sfreq']} Hz")
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

    def get_annotations(self) -> list[dict]:
        """Return EDF annotations as a list of dicts with onset, duration, description."""
        if self.raw is None:
            raise RuntimeError("EDF data not loaded. Call load_edf() first.")
        return [
            {'onset': float(ann['onset']),
             'duration': float(ann['duration']),
             'description': str(ann['description'])}
            for ann in self.raw.annotations
        ]

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
            import mne
            # Ensure requested channels exist
            ch_names = [ch for ch in ch_names if ch in self.raw.ch_names]
            sel = mne.pick_channels(self.raw.ch_names, include=ch_names)
        else:
            sel = slice(None) # All channels

        data, times = self.raw[sel, start_sample:end_sample]

        return data, times

    def detect_sync_pulse(self, ch_name: str = 'DC7',
                          threshold_std: float = 5.0) -> Optional[dict]:
        """Detect the sync pulse in a single channel by amplitude envelope.

        The sync pulse is a 100 Hz square wave, ~1 second long (SyncPulseParams),
        played through the audio output and picked up on a DC channel (typically DC7).

        Algorithm:
          1. Remove the DC offset using the first 30 s of signal as a reference.
             This is critical: DC channels carry a large (~5 mV) standing offset
             that would otherwise dwarf the baseline standard deviation and make a
             weak pulse nearly invisible to a std-multiplier threshold.
          2. Compute a 50 ms boxcar amplitude envelope on the DC-removed signal.
          3. Threshold = baseline_std * threshold_std  (baseline mean ≈ 0 after DC removal).
          4. Find the first run of samples above threshold that lasts at least
             40 % of SyncPulseParams.DURATION_MS — this rejects transient spikes.

        Args:
            ch_name: Channel to search (default 'DC7').
            threshold_std: Multiplier on the DC-removed baseline std (default 3.0).

        Returns:
            dict with keys 'start_sec', 'end_sec', 'channel' if found, else None.
        """
        if self.raw is None:
            raise RuntimeError("EDF data not loaded. Call load_edf() first.")

        from lib.constants import SyncPulseParams

        # ── Resolve channel ───────────────────────────────────────────────
        ch_names_lower = [c.lower() for c in self.raw.ch_names]
        if ch_name in self.raw.ch_names:
            target = ch_name
        elif ch_name.lower() in ch_names_lower:
            target = self.raw.ch_names[ch_names_lower.index(ch_name.lower())]
        else:
            dc_chs = [c for c in self.raw.ch_names if c.upper().startswith('DC')]
            if not dc_chs:
                logger.warning(f"Channel '{ch_name}' not found and no DC channels available.")
                return None
            target = dc_chs[0]
            logger.warning(f"Channel '{ch_name}' not found — using '{target}' instead.")

        import mne
        sfreq = self.raw.info['sfreq']
        sel = mne.pick_channels(self.raw.ch_names, include=[target])
        data, times = self.raw[sel, :]
        signal = data[0]  # 1-D

        # ── DC removal ────────────────────────────────────────────────────
        # First 30 s (or 5 % of recording, whichever is shorter).
        # Sync pulse always occurs well after the recording starts.
        baseline_samples = min(int(30 * sfreq),
                               max(int(0.05 * len(signal)), int(2 * sfreq)))
        dc_offset = signal[:baseline_samples].mean()
        centered = signal - dc_offset

        # ── Amplitude envelope (50 ms boxcar on DC-removed abs signal) ───
        win = max(1, int(0.05 * sfreq))
        envelope = np.convolve(np.abs(centered), np.ones(win) / win, mode='same')

        baseline_env = envelope[:baseline_samples]
        # After DC removal the baseline mean is near 0; use only std for threshold
        threshold = baseline_env.std() * threshold_std

        logger.debug(
            f"Sync detection on '{target}': DC offset={dc_offset*1e6:.1f} µV, "
            f"baseline std={baseline_env.std()*1e6:.1f} µV, "
            f"threshold={threshold*1e6:.1f} µV ({threshold_std}× std)"
        )

        above = envelope > threshold
        if not above.any():
            logger.warning(
                f"No signal above threshold ({threshold*1e6:.1f} µV) on '{target}'.")
            return None

        # ── Collect all sustained runs above threshold ────────────────────
        # Require each run to last at least 40 % of the pulse duration.
        # DC channels can carry brief noise bursts of similar amplitude to a
        # weak sync pulse, so we first gather ALL qualifying runs, then keep
        # only those whose peak is within 30 % of the strongest qualifying run
        # (the sync pulse is designed to be the dominant signal on this channel).
        min_run = int(0.2 * (SyncPulseParams.DURATION_MS / 1000.0) * sfreq)

        diffs = np.diff(above.astype(np.int8))
        run_starts = np.where(diffs == 1)[0] + 1
        run_ends   = np.where(diffs == -1)[0] + 1
        if above[0]:
            run_starts = np.concatenate([[0], run_starts])
        if above[-1]:
            run_ends = np.concatenate([run_ends, [len(above)]])

        qualifying = [(int(s), int(e))
                      for s, e in zip(run_starts, run_ends)
                      if (e - s) >= min_run]

        if not qualifying:
            logger.warning(f"No sustained sync pulse found on '{target}'.")
            return None

        # Among qualifying runs, keep those whose peak is ≥ 5 % of the strongest.
        # The duration gate (min_run) already filters noise — any run that sustains
        # above threshold for 40 % of the pulse duration is almost certainly a real
        # sync pulse, even if stunted. The 5 % floor just excludes near-zero outliers.
        peaks      = [float(envelope[s:e].max()) for s, e in qualifying]
        global_max = max(peaks)
        strong     = [(s, e) for (s, e), p in zip(qualifying, peaks)
                      if p >= 0.05 * global_max]

        if not strong:
            logger.warning(f"No high-amplitude sustained event found on '{target}'.")
            return None

        candidates = []
        for s, e in strong:
            t_start = float(times[s])
            candidates.append({
                'start_sec': t_start,
                'end_sec': t_start + SyncPulseParams.DURATION_MS / 1000.0,
                'channel': target,
            })

        logger.info(
            f"Sync pulse detected on '{target}': {candidates[0]['start_sec']:.3f}s – {candidates[0]['end_sec']:.3f}s "
            f"(peak={global_max*1e6:.0f} µV, threshold={threshold*1e6:.0f} µV, "
            f"{len(qualifying)} qualifying runs, {len(strong)} above 30 % of peak)"
        )
        return {'start_sec': candidates[0]['start_sec'], 'end_sec': candidates[0]['end_sec'],
                'channel': target, 'candidates': candidates}