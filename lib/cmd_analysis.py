# lib/cmd_analysis.py

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_multitaper
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from tqdm import tqdm
import logging

# --- Configure logging ---
# You can configure this globally in your main app if desired
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CMDAnalyzer:
    def __init__(self, eeg_path, stimulus_csv_path, bad_channels=None, eog_channels=None):
        """
        Initialize CMD analyzer with paths to EDF and stimulus CSV.
        
        Parameters:
        - eeg_path: path to .edf file
        - stimulus_csv_path: path to {patient}_{date}_stimulus_results.csv
        - bad_channels: list of channels to exclude (e.g., ['T7', 'Fp1'])
        - eog_channels: list of EOG channels (not used in CMD, but for consistency)
        """
        self.eeg_path = eeg_path
        self.stimulus_csv_path = stimulus_csv_path
        self.bad_channels = bad_channels or []
        self.eog_channels = eog_channels or []
        self.raw = None
        self.epochs = None
        self.psd_data = None
        self.metadata = None
        
    def load_and_preprocess_eeg(self):
        """Load EDF and apply 1-30 Hz bandpass filter, keeping only EEG channels."""
        logger.info(f"Loading EDF file: {self.eeg_path}")
        self.raw = mne.io.read_raw_edf(self.eeg_path, preload=True)
        logger.info(f"Loaded raw data with {len(self.raw.ch_names)} channels, sfreq={self.raw.info['sfreq']} Hz")

        # Ensure sampling rate is 512 Hz (Claassen used 512 Hz)
        original_sfreq = self.raw.info['sfreq']
        if original_sfreq != 512:
            logger.info(f"Resampling from {original_sfreq} Hz to 512 Hz")
            self.raw.resample(512)

        # Filter 1-30 Hz
        logger.info("Applying 1-30 Hz bandpass filter")
        self.raw.filter(l_freq=1, h_freq=30, fir_design='firwin')
        
        # Identify EEG channels (standard 10-20 names)
        eeg_ch_names = [
            'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
            'Cz', 'C3', 'C4', 'T7', 'T8',  # Note: T7/T8 = T3/T4 in some systems
            'Pz', 'P3', 'P4', 'P7', 'P8',  # P7/P8 = T5/T6
            'O1', 'O2', 'Fpz', 'FC1', 'FC2', 'FC5', 'FC6',
            'CP1', 'CP2', 'CP5', 'CP6', 'TP9', 'TP10',
            'PO3', 'PO4', 'POz', 'AF3', 'AF4', 'AF7', 'AF8',
            'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8'
        ]
        
        # Find intersection: channels that exist in both raw data AND standard EEG list
        eeg_channels = [ch for ch in self.raw.ch_names if ch in eeg_ch_names]
        logger.info(f"Found {len(eeg_channels)} standard EEG channels: {eeg_channels}")
        
        if not eeg_channels:
            # Fallback: assume first 19-21 channels are EEG (common in clinical EDFs)
            fallback_num = min(21, len(self.raw.ch_names))
            eeg_channels = self.raw.ch_names[:fallback_num]
            logger.warning(f"No standard EEG channels found. Using first {len(eeg_channels)} channels: {eeg_channels}")
            
        # Pick only EEG channels
        self.raw.pick(eeg_channels)
        logger.info(f"Picked {len(self.raw.ch_names)} EEG channels")
        
        # Set standard montage (now all channels are EEG)
        try:
            self.raw.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn')
            logger.info("Standard montage set.")
        except Exception as e:
            logger.warning(f"Could not set standard montage: {e}")

    def load_stimulus_events(self):
        logger.info(f"Loading stimulus events from CSV: {self.stimulus_csv_path}")
        if self.raw is None:
            raise RuntimeError("EEG data not loaded.")
        if self.raw.info['meas_date'] is None:
            raise ValueError("EEG file missing measurement date. Cannot align timing.")
        
        try:
            df = pd.read_csv(self.stimulus_csv_path)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.stimulus_csv_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {self.stimulus_csv_path}")
            raise ValueError("Stimulus CSV file is empty.")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

        logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Filter ONLY command trials (ignore language/oddball)
        cmd_trial_types = [
            'right_command', 'right_command+p',
            'left_command', 'left_command+p'
        ]
        cmd_trials = df[df['trial_type'].isin(cmd_trial_types)].copy()
        
        logger.info(f"Found {len(cmd_trials)} command trials: {cmd_trial_types}")
        if len(cmd_trials) == 0:
             logger.warning("No command trials found in the CSV file. Check trial_type column values.")
             raise ValueError("No command trials found.")

        # --- Log details of command trials ---
        logger.info("Details of command trials:")
        for idx, row in cmd_trials.iterrows():
            logger.info(f"  Row {idx}: trial_type='{row['trial_type']}', start_time={row.get('start_time', 'N/A')}, duration={row.get('duration', 'N/A')}s")

        events = []
        metadata_rows = []
        instr_id = 0
        
        for idx, trial in cmd_trials.iterrows():
            logger.info(f"Processing command trial {idx}: type='{trial['trial_type']}', start_time={trial.get('start_time', 'N/A')}, duration={trial.get('duration', 'N/A')}s")
            
            # Determine if prompt was used based on trial type
            has_prompt = '+p' in trial['trial_type']
            prompt_duration = 2.0 if has_prompt else 0.0 # Estimated prompt duration

            # Get trial start time (assuming it's a Unix timestamp)
            trial_start_timestamp_str = trial.get('start_time')
            if pd.isna(trial_start_timestamp_str) or trial_start_timestamp_str == '':
                 logger.warning(f"Trial {idx} has no or empty 'start_time'. Skipping.")
                 continue
            try:
                 trial_start_timestamp = float(trial_start_timestamp_str)
            except (ValueError, TypeError) as e:
                 logger.warning(f"Could not parse 'start_time' '{trial_start_timestamp_str}' for trial {idx}: {e}. Skipping.")
                 continue

            # Align with EEG measurement time
            # raw.info['meas_date'] is a datetime object or a float (timestamp)
            meas_date_timestamp = self.raw.info['meas_date'].timestamp() if hasattr(self.raw.info['meas_date'], 'timestamp') else self.raw.info['meas_date']
            trial_start_rel_to_eeg = trial_start_timestamp - meas_date_timestamp
            logger.debug(f"  Trial start (timestamp): {trial_start_timestamp}, rel to EEG: {trial_start_rel_to_eeg:.2f}s")

            # --- Calculate Event Times based on known structure ---
            # Each cycle is 20 seconds (10s keep+pause + 10s stop+pause)
            # Total trial duration should be approx: prompt_dur + (8 cycles * 20s)
            # However, we rely on the known timing structure, not the 'duration' column for precise event timing.
            num_cycles = 8
            cycle_duration = 20.0 # seconds

            for cycle_idx in range(num_cycles):
                # Calculate start time of the current cycle within the trial
                # Time after prompt, then add cycle offset
                cycle_start_in_trial = prompt_duration + (cycle_idx * cycle_duration)

                # --- Keep Event ---
                # Keep occurs at the very beginning of the cycle (relative to cycle start)
                keep_time_in_trial = cycle_start_in_trial
                keep_time_abs = trial_start_rel_to_eeg + keep_time_in_trial
                keep_sample = int(keep_time_abs * self.raw.info['sfreq'])
                eeg_duration = len(self.raw.times) / self.raw.info['sfreq']

                if 0 <= keep_time_abs <= eeg_duration:
                    events.append([keep_sample, 0, 1]) # Event ID 1 for 'keep'
                    metadata_rows.append({
                        'instruction_id': instr_id,
                        'trial': cycle_idx, # Use cycle index as trial number for this analysis
                        'move': 1, # Keep
                        'instruction_type': 'keep',
                        'time_sample': keep_sample
                    })
                    instr_id += 1
                    logger.debug(f"    Added KEEP event for cycle {cycle_idx} at {keep_time_abs:.2f}s (sample {keep_sample})")
                else:
                    logger.warning(f"    KEEP event for cycle {cycle_idx} at {keep_time_abs:.2f}s (sample {keep_sample}) is out of EEG range [0, {eeg_duration:.2f}]. Skipping.")

                # --- Stop Event ---
                # Stop occurs 10 seconds after the keep (relative to cycle start)
                stop_time_in_trial = cycle_start_in_trial + 10.0
                stop_time_abs = trial_start_rel_to_eeg + stop_time_in_trial
                stop_sample = int(stop_time_abs * self.raw.info['sfreq'])

                if 0 <= stop_time_abs <= eeg_duration:
                    events.append([stop_sample, 0, 2]) # Event ID 2 for 'stop'
                    metadata_rows.append({
                        'instruction_id': instr_id,
                        'trial': cycle_idx, # Use cycle index as trial number for this analysis
                        'move': 0, # Stop
                        'instruction_type': 'stop',
                        'time_sample': stop_sample
                    })
                    instr_id += 1
                    logger.debug(f"    Added STOP event for cycle {cycle_idx} at {stop_time_abs:.2f}s (sample {stop_sample})")
                else:
                    logger.warning(f"    STOP event for cycle {cycle_idx} at {stop_time_abs:.2f}s (sample {stop_sample}) is out of EEG range [0, {eeg_duration:.2f}]. Skipping.")

        logger.info(f"Final list of events: {len(events)} events created.")
        if not events:
            logger.error("No valid command events found after processing all command trials.")
            raise ValueError("No valid command events found.")
            
        self.events = np.array(events, dtype=int)
        self.metadata = pd.DataFrame(metadata_rows)
        logger.info(f"Events and metadata created. Events shape: {self.events.shape}, Metadata shape: {self.metadata.shape}")

    def create_epochs(self):
        """Segment EEG into 2-second epochs following each instruction."""
        logger.info("Creating epochs...")
        if self.raw is None:
            raise RuntimeError("EEG data not loaded. Call load_and_preprocess_eeg() first.")

        picks = mne.pick_types(self.raw.info, eeg=True, exclude='bads')
        self.epochs = mne.Epochs(
            self.raw,
            events=self.events,
            tmin=0, tmax=2,
            picks=picks,
            metadata=self.metadata,
            preload=True,
            baseline=None,
            proj=False
        )
        logger.info(f"Created {len(self.epochs)} epochs.")

    def compute_psd_features(self):
        """Compute PSD in 4 frequency bands and vectorize."""
        logger.info("Computing PSD features...")
        if self.raw is None:
            raise RuntimeError("EEG data not loaded. Call load_and_preprocess_eeg() first.")
        if self.epochs is None:
            raise RuntimeError("epochs not set. Call create_epochs() first.")
        
        data = self.epochs.get_data()
        n_epochs, n_chans, n_times = data.shape
        
        if n_epochs == 0:
            raise ValueError("No epochs to process.")
        
        # Compute PSD for first epoch to get freqs (same for all epochs)
        first_epoch = data[0]
        result = psd_array_multitaper(
            first_epoch, sfreq=self.raw.info['sfreq'],
            fmin=1, fmax=30, verbose=False
        )
        # Handle both MNE <1.0 (2 values) and MNE >=1.0 (3 values)
        psds_first = result[0]
        freqs = result[1]
        
        # Pre-allocate PSD array
        psds_all = np.empty((n_epochs, n_chans, len(freqs)))
        psds_all[0] = psds_first
        
        # Compute PSD for remaining epochs
        for i in range(1, n_epochs):
            result = psd_array_multitaper(
                data[i], sfreq=self.raw.info['sfreq'],
                fmin=1, fmax=30, verbose=False
            )
            psds_all[i] = result[0]  # PSD is always first element
        
        # Average within bands
        bands = [(1, 3), (4, 7), (8, 13), (14, 30)]
        psd_data = np.zeros((n_epochs, n_chans, len(bands)))
        for i, (fmin, fmax) in enumerate(bands):
            freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            psd_data[:, :, i] = psds_all[:, :, freq_idx].mean(axis=2)
            
        self.psd_data = psd_data.reshape(n_epochs, n_chans * len(bands))
        logger.info(f"PSD features computed. Shape: {self.psd_data.shape}")

    def run_analysis(self, n_permutations=500):
        """Run full CMD analysis and return results."""
        logger.info("Running CMD analysis...")
        if self.metadata is None:
            raise RuntimeError("metadata not set. Call load_stimulus_events() first.")
        if self.psd_data is None:
            raise RuntimeError("psd_data not set. Call compute_psd_features() first.")
        
        # Prepare data - CONVERT TO NUMPY ARRAYS
        y = np.asarray(self.metadata['move'].values)   
        groups = np.asarray(self.metadata['trial'].values)    

        logger.info(f"Using {len(y)} samples, {len(np.unique(groups))} groups for cross-validation.")
        
        # Cross-validated AUC
        clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
        cv = LeaveOneGroupOut()
        scores = cross_val_score(clf, self.psd_data, y, cv=cv, groups=groups, scoring='roc_auc')
        observed_auc = scores.mean()
        logger.info(f"Observed AUC: {observed_auc:.3f} (+/- {scores.std():.3f})")
        
        # Permutation test
        perm_scores = []
        for _ in tqdm(range(n_permutations), desc="Permutation test"):
            y_shuffled = y.copy()
            np.random.shuffle(y_shuffled)
            perm_score = cross_val_score(
                clf, self.psd_data, y_shuffled, 
                cv=cv, groups=groups,  # ← Now a numpy array
                scoring='roc_auc'
            ).mean()
            perm_scores.append(perm_score)
            
        p_value = (np.sum(np.array(perm_scores) >= observed_auc) + 1) / (n_permutations + 1)
        is_cmd = observed_auc > 0.5 and p_value < 0.05

        logger.info(f"Analysis complete. AUC: {observed_auc:.3f}, P-value: {p_value:.3f}, Is CMD: {is_cmd}")
        
        return {
            'auc': observed_auc,
            'auc_std': scores.std(),
            'p_value': p_value,
            'is_cmd': is_cmd,
            'n_permutations': n_permutations,
            'permutation_scores': perm_scores,
            'scores': scores
        }
    
    def detect_signal_start(self, raw, approx_start_sec, window_sec=10, threshold_std=3):
        """
        Detect actual command onset via audio artifact in EEG.
        
        Parameters:
        - raw: MNE Raw object
        - approx_start_sec: approximate start time in seconds
        - window_sec: search window duration
        - threshold_std: number of standard deviations above baseline for detection
        """
        sfreq = raw.info['sfreq']
        start_sample = int((approx_start_sec - 2) * sfreq)  # Search 2s before expected
        end_sample = int((approx_start_sec + window_sec) * sfreq)
        
        # Validate indices
        if start_sample < 0:
            start_sample = 0
        if end_sample > len(raw.times):
            end_sample = len(raw.times)
        
        # Use frontal channels (Fp1, Fp2) for audio artifact detection
        frontal_chs = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names]
        if not frontal_chs:
            frontal_chs = raw.ch_names[:3]  # Fallback to first 3 channels
            
        data, _ = raw[frontal_chs, start_sample:end_sample]
        amp = np.abs(data).mean(axis=0)
        
        # Compute baseline noise level
        baseline_window_len = int(1 * sfreq) # First 1 second as baseline
        if len(amp) <= baseline_window_len:
             # If the window is too small, use a smaller baseline or raise an error
             # For now, let's just use the first few samples if possible
             if len(amp) > 1:
                 baseline_amp = amp[:max(1, len(amp)//2)]
             else:
                 raise ValueError("Audio detection window is too small.")
        else:
            baseline_amp = amp[:baseline_window_len]
        
        baseline_mean = np.mean(baseline_amp)
        baseline_std = np.std(baseline_amp)
        
        # Find first point exceeding threshold
        threshold = baseline_mean + threshold_std * baseline_std
        above_threshold = np.where(amp > threshold)[0]
        
        if len(above_threshold) == 0:
            raise ValueError("No audio artifact detected above threshold")
        
        peak_idx = above_threshold[0]
        actual_start_sample = start_sample + peak_idx
        actual_start_sec = actual_start_sample / sfreq
        
        return actual_start_sample, actual_start_sec