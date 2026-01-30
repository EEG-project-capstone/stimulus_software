# lib/results_manager.py

"""
Results management for the EEG Stimulus Package.
Handles thread-safe CSV writing and result storage.
"""

import json
import logging
import threading
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from lib.exceptions import ResultsFileError

logger = logging.getLogger('eeg_stimulus.results_manager')


class ResultsManager:
    """Manages thread-safe writing of stimulus results to CSV files."""
    
    # Standard CSV schema
    COLUMNS = [
        'patient_id',
        'date',
        'stim_type',
        'notes',
        'start_time',
        'end_time',
        'duration'
    ]
    
    def __init__(self, config):
        """Initialize results manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._write_lock = threading.Lock()
        logger.info("ResultsManager initialized")
    
    def append_result(self, 
                     patient_id: str, 
                     result_type: str, 
                     data: Dict[str, Any]) -> Path:
        """Thread-safe result writing with validation.
        
        Args:
            patient_id: Patient identifier
            result_type: Type of result (e.g., 'language', 'right_command')
            data: Dictionary containing result data
            
        Returns:
            Path to the results file
            
        Raises:
            ResultsFileError: If writing fails
        """
        try:
            filepath = self.config.get_results_path(patient_id)
            
            # Standardize schema
            row = self._create_row(patient_id, result_type, data)
            
            # Thread-safe write
            with self._write_lock:
                self._write_row(filepath, row)
            
            logger.debug(f"Result appended: {result_type} for {patient_id}")
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to append result for {patient_id}: {e}"
            logger.error(error_msg, exc_info=True)
            raise ResultsFileError(error_msg) from e
    
    def _create_row(self, patient_id: str, result_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized row dictionary.

        Args:
            patient_id: Patient identifier
            result_type: Type of result
            data: Result data

        Returns:
            Standardized row dictionary
        """
        # Combine events and text notes into single notes field
        events = data.get('events', [])
        text_notes = data.get('notes', '')

        # Create combined notes value
        if events and text_notes:
            # Both events and text notes exist
            notes_value = f"{json.dumps(events)} | {text_notes}"
        elif events:
            # Only events
            notes_value = json.dumps(events)
        else:
            # Only text notes (or empty)
            notes_value = text_notes

        return {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'stim_type': result_type,
            'notes': notes_value,
            'start_time': data.get('start_time', ''),
            'end_time': data.get('end_time', ''),
            'duration': data.get('duration', '')
        }
    
    def _write_row(self, filepath: Path, row: Dict[str, Any]):
        """Write a single row to CSV file.
        
        Args:
            filepath: Path to CSV file
            row: Row data to write
            
        Raises:
            ResultsFileError: If write fails
        """
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists to determine if we need header
            file_exists = filepath.exists()
            
            # Create DataFrame and write
            df = pd.DataFrame([row])
            df.to_csv(
                filepath,
                mode='a',
                header=not file_exists,
                index=False
            )
            
        except Exception as e:
            raise ResultsFileError(f"Failed to write to {filepath}: {e}") from e
    
    def append_note(self, patient_id: str, note: str) -> Path:
        """Append a session note.
        
        Args:
            patient_id: Patient identifier
            note: Note text
            
        Returns:
            Path to results file
        """
        data = {'notes': note}
        return self.append_result(patient_id, 'session_note', data)
    
    def append_sync_pulse(self, patient_id: str, sync_time: float) -> Path:
        """Append a manual sync pulse event.

        Args:
            patient_id: Patient identifier
            sync_time: Time of sync pulse

        Returns:
            Path to results file
        """
        import time

        # Simplified - just store time in columns, not in events array
        data = {
            'notes': f'Manual sync pulse at {time.strftime("%H:%M:%S", time.localtime(sync_time))}',
            'start_time': sync_time,
            'end_time': sync_time + 0.2,  # 200ms duration
            'duration': 0.2
        }
        return self.append_result(patient_id, 'manual_sync_pulse', data)
    
    def read_results(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Read results from CSV file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            DataFrame of results, or None if file doesn't exist
        """
        if not filepath.exists():
            logger.warning(f"Results file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            logger.debug(f"Read {len(df)} results from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to read results from {filepath}: {e}")
            return None
    
    def get_stimulus_sequence(self, filepath: Path) -> List[Dict[str, Any]]:
        """Extract stimulus sequence from results file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            List of stimulus dictionaries
        """
        df = self.read_results(filepath)
        if df is None:
            return []
        
        # Filter for actual stimuli (not notes or sync events)
        stimulus_types = {
            'language', 'right_command', 'right_command+p',
            'left_command', 'left_command+p', 'oddball', 'oddball+p',
            'loved_one_voice', 'control'
        }
        
        stimuli = []
        for _, row in df.iterrows():
            if row['stim_type'] in stimulus_types:
                stimuli.append({
                    'type': row['stim_type'],
                    'start_time': row['start_time'],
                    'duration': row['duration']
                })
        
        return stimuli
    
    def get_session_log(self, filepath: Path) -> List[str]:
        """Extract session log entries (notes, sync events, etc.).
        
        Args:
            filepath: Path to results file
            
        Returns:
            List of formatted log entries
        """
        df = self.read_results(filepath)
        if df is None:
            return []
        
        # Filter for log-type entries
        log_types = {'session_note', 'manual_sync_pulse', 'sync_detection'}
        
        logs = []
        for _, row in df.iterrows():
            notes_value = row['notes']
            has_notes = notes_value is not None and str(notes_value).strip() != ''
            if row['stim_type'] in log_types or has_notes:
                date = row.get('date', 'Unknown')
                notes = str(row.get('notes', '')).strip()
                if notes:
                    logs.append(f"[{date}] {notes}")
        
        return logs
