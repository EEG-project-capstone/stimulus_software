# lib/config.py

from pathlib import Path
import yaml
import time
import logging
from typing import Optional

logger = logging.getLogger('eeg_stimulus')

class Config:
    def __init__(self, config_path: str = 'config.yml'):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.file = yaml.safe_load(f)
        
        # Convert all paths to Path objects
        self._convert_paths_to_pathlib()
        
        # Create necessary directories
        self._create_directories()
        
        # Set up UI options
        self._setup_ui_options()
        
        self.current_date = time.strftime("%Y-%m-%d")
        logger.info(f"Configuration loaded from {config_path}")
    
    def _convert_paths_to_pathlib(self):
        """Convert all file path strings to Path objects."""
        path_keys = [
            'edf_dir', 'result_dir', 'sentences_path',
            'right_keep_path', 'right_stop_path', 'left_keep_path', 'left_stop_path',
            'beep_path', 'loved_one_path', 'male_control_path', 'female_control_path',
            'motor_prompt_path', 'oddball_prompt_path'
        ]
        
        for key in path_keys:
            if key in self.file:
                self.file[key] = Path(self.file[key])
                logger.debug(f"Converted {key} to Path object: {self.file[key]}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = ['result_dir', 'edf_dir']
        
        for dir_key in dirs_to_create:
            if dir_key in self.file:
                dir_path = self.file[dir_key]
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
    
    def _setup_ui_options(self):
        """Set up UI dropdown options."""
        self.cpc_scale = [
            "",
            "CPC 1: No neurological deficit",
            "CPC 2: Mild to moderate dysfunction",
            "CPC 3: Severe dysfunction",
            "CPC 4: Coma",
            "CPC 5: Brain death",
        ]
        self.gose_scale = [
            "",
            "GOSE 1: Dead",
            "GOSE 2: Vegetative state",
            "GOSE 3: Lower severe disability",
            "GOSE 4: Upper severe disability",
            "GOSE 5: Lower moderate disability",
            "GOSE 6: Upper moderate disability",
            "GOSE 7: Lower good recovery",
            "GOSE 8: Upper good recovery",
        ]
        self.graphs = ["", "CMD", "Language Tracking"]
    
    def get_path(self, key: str) -> Path:
        """Get a path from configuration.
        
        Args:
            key: Configuration key for the path
            
        Returns:
            Path object
            
        Raises:
            KeyError: If key doesn't exist in configuration
        """
        if key not in self.file:
            raise KeyError(f"Path key '{key}' not found in configuration")
        return self.file[key]
    
    def get_results_path(self, patient_id: str, date: Optional[str] = None) -> Path:
        """Generate results file path for a patient.
        
        Args:
            patient_id: Patient identifier
            date: Date string (YYYY-MM-DD), defaults to current date
            
        Returns:
            Path to results CSV file
        """
        if date is None:
            date = self.current_date
        filename = f"{patient_id}_{date}_stimulus_results.csv"
        return self.file['result_dir'] / filename
    
    def get_edf_path(self, patient_id: str, date: Optional[str] = None) -> Path:
        """Generate EDF file path for a patient.
        
        Args:
            patient_id: Patient identifier
            date: Date string (YYYY-MM-DD), defaults to current date
            
        Returns:
            Path to EDF file
        """
        if date is None:
            date = self.current_date
        filename = f"{patient_id}_{date}.edf"
        return self.file['edf_dir'] / filename