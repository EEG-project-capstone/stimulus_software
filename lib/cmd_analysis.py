# lib/cmd_analysis.py

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_multitaper
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from typing import Dict, Any
from tqdm import tqdm
import logging
import os

# Prevent OpenMP threading conflicts (especially on macOS)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

logger = logging.getLogger('eeg_stimulus')


class CMDAnalyzer:
    def __init__(self, eeg_path: str, stimulus_csv_path: str, bad_channels=None, eog_channels=None):
        """
        Initialize CMD analyzer with paths to EDF and stimulus CSV.

        Parameters:
        - eeg_path: Path to .edf file
        - stimulus_csv_path: Path to {patient}_{date}_stimulus_results.csv
        - bad_channels: List of channels to exclude (e.g., ['T7', 'Fp1'])
        - eog_channels: List of EOG channels (not used in CMD, but for consistency)
        """
        self.eeg_path = eeg_path
        self.stimulus_csv_path = stimulus_csv_path
        self.bad_channels = set(bad_channels or [])
        self.eog_channels = eog_channels or []
        self.raw = None
        self.epochs = None
        self.psd_data = None
        self.metadata = None
        self.events = None
        self.sync_time = None  # EDF time (seconds) corresponding to session start (first command trial)

    def set_sync_time(self, sync_time: float) -> None:
        """Set the EDF time (in seconds) corresponding to session start."""
        self.sync_time = sync_time
        logger.info(f"CMDAnalyzer: Sync time set to {sync_time:.3f} seconds (EDF time of session start).")

    def load_and_preprocess_eeg(self) -> None:
        """Load EDF, resample to 512 Hz, bandpass 1–30 Hz, and keep only EEG channels."""
        logger.info(f"Loading EDF file: {self.eeg_path}")
        self.raw = mne.io.read_raw_edf(self.eeg_path, preload=True)
        logger.info(f"Loaded raw data with {len(self.raw.ch_names)} channels, sfreq={self.raw.info['sfreq']:.1f} Hz")

        # Resample to 512 Hz if needed
        if self.raw.info['sfreq'] != 512:
            logger.info(f"Resampling from {self.raw.info['sfreq']:.1f} Hz to 512 Hz")
            self.raw.resample(512)

        # Apply 1–30 Hz bandpass filter
        logger.info("Applying 1–30 Hz bandpass filter")
        self.raw.filter(l_freq=1, h_freq=30, fir_design='firwin')

        # Auto-detect EEG channels
        eeg_picks = mne.pick_types(self.raw.info, eeg=True, eog=False, stim=False, misc=False)
        if len(eeg_picks) > 0:
            eeg_channels = [self.raw.ch_names[i] for i in eeg_picks]
            logger.info(f"Auto-detected {len(eeg_channels)} EEG channels.")
        else:
            # Fallback to standard 10–20 names if auto-detection fails
            standard_eeg_names = {
                'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
                'Cz', 'C3', 'C4', 'T7', 'T8',
                'Pz', 'P3', 'P4', 'P7', 'P8',
                'O1', 'O2', 'Fpz', 'FC1', 'FC2', 'FC5', 'FC6',
                'CP1', 'CP2', 'CP5', 'CP6', 'TP9', 'TP10',
                'PO3', 'PO4', 'POz', 'AF3', 'AF4', 'AF7', 'AF8',
                'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8'
            }
            eeg_channels = [ch for ch in self.raw.ch_names if ch in standard_eeg_names]
            if not eeg_channels:
                # Last resort: take first 21 channels
                eeg_channels = self.raw.ch_names[:min(21, len(self.raw.ch_names))]
                logger.warning(f"No EEG channels detected. Using first {len(eeg_channels)} channels: {eeg_channels}")
            else:
                logger.info(f"Fallback: found {len(eeg_channels)} standard EEG channels.")

        # Remove bad channels
        eeg_channels = [ch for ch in eeg_channels if ch not in self.bad_channels]
        if not eeg_channels:
            raise ValueError("No valid EEG channels remain after applying bad_channels filter.")

        self.raw.pick(eeg_channels)
        try:
            self.raw.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn')
            logger.info("Standard 10–20 montage applied.")
        except Exception as e:
            logger.warning(f"Could not apply standard montage: {e}")

    def load_stimulus_events(self) -> None:
        """Create events for 'keep' (move=1) and 'stop' (move=0) instructions from command trials."""
        if self.raw is None:
            raise RuntimeError("EEG data not loaded. Call load_and_preprocess_eeg() first.")
        if self.sync_time is None:
            raise RuntimeError("Sync time not set. Call set_sync_time() first.")

        logger.info(f"Loading stimulus events from CSV: {self.stimulus_csv_path}")
        df = pd.read_csv(self.stimulus_csv_path)

        required_cols = {'trial_type', 'start_time'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Identify command trials
        cmd_types = {'right_command', 'right_command+p', 'left_command', 'left_command+p'}
        cmd_trials = df[df['trial_type'].isin(cmd_types)].copy()
        if cmd_trials.empty:
            raise ValueError("No command trials found in CSV.")

        cmd_trials = cmd_trials.sort_values('start_time').reset_index(drop=True)
        logger.info(f"Found {len(cmd_trials)} command trials.")

        # Define session start as the first command trial's Unix timestamp
        first_trial_unix = float(cmd_trials.iloc[0]['start_time'])
        logger.info(f"Session start Unix timestamp: {first_trial_unix}")
        logger.info(f"Sync time (EDF time of session start): {self.sync_time:.3f} s")

        eeg_duration = self.raw.times[-1]  # in seconds

        events = []
        metadata_rows = []
        instr_id = 0

        for trial_idx, trial in cmd_trials.iterrows():
            try:
                trial_unix = float(trial['start_time'])
            except (ValueError, TypeError):
                logger.warning(f"Skipping trial with invalid start_time: {trial['start_time']}")
                continue

            # Align trial to EDF timeline
            seconds_since_session_start = trial_unix - first_trial_unix
            trial_start_edf = self.sync_time + seconds_since_session_start

            # Determine if prompt was shown (+p suffix)
            has_prompt = '+p' in trial['trial_type']
            prompt_dur = 2.0 if has_prompt else 0.0

            # Generate 8 keep/stop cycles per command trial
            for cycle in range(8):
                # KEEP event (10s after prompt/cycle start)
                keep_time_trial = prompt_dur + cycle * 20.0
                keep_time_edf = trial_start_edf + keep_time_trial
                keep_sample = int(np.round(keep_time_edf * self.raw.info['sfreq']))

                # STOP event (10s after KEEP)
                stop_time_edf = keep_time_edf + 10.0
                stop_sample = int(np.round(stop_time_edf * self.raw.info['sfreq']))

                # Add KEEP event
                if 0 <= keep_time_edf <= eeg_duration:
                    events.append([keep_sample, 0, 1])
                    metadata_rows.append({
                        'instruction_id': instr_id,
                        'command_trial_id': trial_idx,  # ← critical for proper grouping
                        'cycle': cycle,
                        'move': 1,
                        'instruction_type': 'keep',
                        'time_sample': keep_sample
                    })
                    instr_id += 1

                # Add STOP event
                if 0 <= stop_time_edf <= eeg_duration:
                    events.append([stop_sample, 0, 2])
                    metadata_rows.append({
                        'instruction_id': instr_id,
                        'command_trial_id': trial_idx,
                        'cycle': cycle,
                        'move': 0,
                        'instruction_type': 'stop',
                        'time_sample': stop_sample
                    })
                    instr_id += 1

        if not events:
            raise ValueError("No valid command events could be created within EDF time bounds.")

        self.events = np.array(events, dtype=int)
        self.metadata = pd.DataFrame(metadata_rows)
        logger.info(f"Created {len(events)} instruction events from {len(cmd_trials)} command trials.")

    def create_epochs(self) -> None:
        """Extract 2-second epochs starting at each instruction onset."""
        if self.raw is None:
            raise RuntimeError("EEG data not loaded.")
        if self.events is None or len(self.events) == 0:
            raise RuntimeError("Events not created. Call load_stimulus_events() first.")

        picks = mne.pick_types(self.raw.info, eeg=True, exclude='bads')
        self.epochs = mne.Epochs(
            self.raw,
            self.events,
            event_id={'keep': 1, 'stop': 2},
            tmin=0.0,
            tmax=2.0,
            picks=picks,
            metadata=self.metadata,
            baseline=None,
            preload=True,
            proj=False,
            reject_by_annotation=False,
            verbose=False
        )
        logger.info(f"Created {len(self.epochs)} epochs (t=0 to 2s post-instruction).")

    def compute_psd_features(self) -> None:
        """Compute multitaper PSD in 4 canonical bands and flatten per epoch."""
        if self.epochs is None:
            raise RuntimeError("Epochs not created. Call create_epochs() first.")
        if self.raw is None:
            raise RuntimeError("Raw not found.")

        data = self.epochs.get_data()
        if data.size == 0:
            raise ValueError("No epochs to process.")

        sfreq = self.raw.info['sfreq']
        bands = [(1, 3), (4, 7), (8, 13), (14, 30)]  # delta, theta, alpha, beta
        psd_list = []

        for epoch in data:
            results = psd_array_multitaper(
                epoch, sfreq=sfreq, fmin=1, fmax=30,
                adaptive=True, normalization='full', verbose=False
            )
            psds, freqs = results[0], results[1]
            band_features = []
            for fmin, fmax in bands:
                mask = (freqs >= fmin) & (freqs <= fmax)
                if np.any(mask):
                    mean_psd = psds[:, mask].mean(axis=1)
                else:
                    mean_psd = np.zeros(psds.shape[0])
                band_features.append(mean_psd)
            psd_list.append(np.concatenate(band_features))

        self.psd_data = np.array(psd_list)
        logger.info(f"PSD features computed. Shape: {self.psd_data.shape} (n_epochs, n_chans × 4 bands)")

    def run_analysis(self, n_permutations: int = 100) -> Dict[str, Any]:
        """Run decoding analysis with cross-validation and permutation test."""
        if self.metadata is None or self.psd_data is None:
            raise RuntimeError("Data not prepared. Run event loading and PSD computation first.")

        y = np.array(self.metadata['move'].values)
        groups = np.array(self.metadata['command_trial_id'].values)

        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError(f"At least 2 command trials are required for cross-validation. Found {n_groups}.")

        logger.info(f"Running decoding on {len(y)} samples from {n_groups} command trials.")

        # Classifier pipeline
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='liblinear'))

        # Observed performance
        cv = LeaveOneGroupOut()
        scores = cross_val_score(clf, self.psd_data, y, cv=cv, groups=groups, scoring='roc_auc')
        observed_auc = float(scores.mean())

        # Permutation test
        perm_scores = []
        for _ in tqdm(range(n_permutations), desc="Permutation test"):
            y_shuffled = np.random.permutation(y)
            score = cross_val_score(clf, self.psd_data, y_shuffled, cv=cv, groups=groups, scoring='roc_auc').mean()
            perm_scores.append(float(score))

        perm_scores = np.array(perm_scores)
        p_value = (np.sum(perm_scores >= observed_auc) + 1) / (n_permutations + 1)
        is_cmd = observed_auc > 0.5 and p_value < 0.05

        logger.info(f"Analysis complete → AUC: {observed_auc:.3f} ± {scores.std():.3f}, p = {p_value:.3f}, CMD: {is_cmd}")

        return {
            'auc': observed_auc,
            'auc_std': float(scores.std()),
            'p_value': float(p_value),
            'is_cmd': bool(is_cmd),
            'n_permutations': n_permutations,
            'permutation_scores': perm_scores.tolist(),
            'cv_scores': scores.tolist(),
            'n_samples': len(y),
            'n_command_trials': n_groups
        }