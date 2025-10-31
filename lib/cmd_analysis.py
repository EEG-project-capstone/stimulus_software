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
        self.raw = mne.io.read_raw_edf(self.eeg_path, preload=True)

        # Ensure sampling rate is 512 Hz (Claassen used 512 Hz)
        if self.raw.info['sfreq'] != 512:
            self.raw.resample(512)

        # Filter 1-30 Hz
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
        
        if not eeg_channels:
            # Fallback: assume first 19-21 channels are EEG (common in clinical EDFs)
            eeg_channels = self.raw.ch_names[:21]  # Adjust number as needed
            
        # Pick only EEG channels
        self.raw.pick(eeg_channels)
        
        # Set standard montage (now all channels are EEG)
        self.raw.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn')

    def load_stimulus_events(self):
        if self.raw is None:
            raise RuntimeError("EEG data not loaded.")
        if self.raw.info['meas_date'] is None:
            raise ValueError("EEG file missing measurement date. Cannot align timing.")
        
        df = pd.read_csv(self.stimulus_csv_path)
        
        # Filter ONLY command trials (ignore language/oddball)
        cmd_trials = df[df['trial_type'].isin([
            'right_command', 'right_command+p',
            'left_command', 'left_command+p'
        ])].copy()
        
        if cmd_trials.empty:
            raise ValueError("No command trials found.")
    
        events = []
        metadata_rows = []
        instr_id = 0
        
        for _, trial in cmd_trials.iterrows():
            # Get approximate start from stimulus log
            approx_start_sec = trial['start_time'] - self.raw.info['meas_date'].timestamp()
            
            # Detect ACTUAL start via audio artifact
            try:
                actual_start_sample, actual_start_sec = self.detect_signal_start(
                    self.raw, approx_start_sec
                )
            except (ValueError, IndexError, RuntimeError) as e:
                print(f"Warning: Audio detection failed for trial {trial.name}: {e}")
                print("Falling back to approximate timing")
                actual_start_sec = approx_start_sec
                actual_start_sample = int(actual_start_sec * self.raw.info['sfreq'])
            
            # Extract keep/stop onsets from sentences
            try:
                sentences = eval(trial['sentences']) if isinstance(trial['sentences'], str) else []
            except:
                sentences = []
                print(f"Warning: Could not parse sentences for trial {trial.name}")

            keep_onsets = [s['onset_time'] for s in sentences if 'keep' in s.get('event', '')]
            stop_onsets = [s['onset_time'] for s in sentences if 'stop' in s.get('event', '')]
        
            # Align keep/stop times relative to detected start
            if keep_onsets and len(keep_onsets) == len(stop_onsets):
                # Get relative onset times within trial (seconds from first keep)
                first_keep_abs = keep_onsets[0]
                keep_rel_times = [k - first_keep_abs for k in keep_onsets]
                stop_rel_times = [s - first_keep_abs for s in stop_onsets]
                
                for cycle_idx, (keep_rel, stop_rel) in enumerate(zip(keep_rel_times, stop_rel_times)):
                    # Each cycle = 1 trial (keep + stop)
                    trial_id = cycle_idx
                    
                    # Keep instruction
                    keep_sec = actual_start_sec + keep_rel
                    if 0 <= keep_sec <= (len(self.raw) / self.raw.info['sfreq']):
                        keep_sample = int(keep_sec * self.raw.info['sfreq'])
                        events.append([keep_sample, 0, 1])
                        metadata_rows.append({
                            'instruction_id': instr_id,
                            'trial': trial_id,
                            'move': 1,
                            'instruction_type': 'keep',
                            'time_sample': keep_sample
                        })
                        instr_id += 1
                        
                    # Stop instruction  
                    stop_sec = actual_start_sec + stop_rel
                    if 0 <= stop_sec <= (len(self.raw) / self.raw.info['sfreq']):
                        stop_sample = int(stop_sec * self.raw.info['sfreq'])
                        events.append([stop_sample, 0, 2])
                        metadata_rows.append({
                            'instruction_id': instr_id,
                            'trial': trial_id,
                            'move': 0,
                            'instruction_type': 'stop',
                            'time_sample': stop_sample
                        })
                        instr_id += 1
        
        if not events:
            raise ValueError("No valid command events found.")
            
        self.events = np.array(events, dtype=int)
        self.metadata = pd.DataFrame(metadata_rows)

    def create_epochs(self):
        """Segment EEG into 2-second epochs following each instruction."""
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
        
    def compute_psd_features(self):
        """Compute PSD in 4 frequency bands and vectorize."""
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
        
    def run_analysis(self, n_permutations=500):
        """Run full CMD analysis and return results."""
        if self.metadata is None:
            raise RuntimeError("metadata not set. Call load_stimulus_events() first.")
        if self.psd_data is None:
            raise RuntimeError("psd_data not set. Call compute_psd_features() first.")
        
        # Prepare data - CONVERT TO NUMPY ARRAYS
        y = np.asarray(self.metadata['move'].values)   
        groups = np.asarray(self.metadata['trial'].values)    
        
        # Cross-validated AUC
        clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
        cv = LeaveOneGroupOut()
        scores = cross_val_score(clf, self.psd_data, y, cv=cv, groups=groups, scoring='roc_auc')
        observed_auc = scores.mean()
        
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
        baseline_amp = amp[:int(1 * sfreq)]  # First second as baseline
        baseline_mean = np.mean(baseline_amp)
        baseline_std = np.std(baseline_amp)
        
        # Find first point exceeding threshold
        threshold = baseline_mean + threshold_std * baseline_std  # ← Now threshold_std is defined
        above_threshold = np.where(amp > threshold)[0]
        
        if len(above_threshold) == 0:
            raise ValueError("No audio artifact detected above threshold")
        
        peak_idx = above_threshold[0]
        actual_start_sample = start_sample + peak_idx
        actual_start_sec = actual_start_sample / sfreq
        
        return actual_start_sample, actual_start_sec