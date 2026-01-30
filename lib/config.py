# lib/config_improved.py

"""
Improved configuration management with validation.
Replaces lib/config.py with enhanced error checking and setup.
"""

from pathlib import Path
import yaml
import time
import logging
from typing import Optional, Dict, Any, List
from lib.exceptions import ConfigFileError, ConfigValidationError
from lib.constants import CPC_SCALE, GOSE_SCALE

logger = logging.getLogger('eeg_stimulus.config')


class Config:
    """Enhanced configuration management with validation."""
    
    # Required configuration keys
    REQUIRED_KEYS = [
        'edf_dir',
        'result_dir',
        'sentences_path',
        'right_keep_path',
        'right_stop_path',
        'left_keep_path',
        'left_stop_path',
        'beep_path',
        'motor_prompt_path',
        'oddball_prompt_path'
    ]
    
    # Optional configuration keys with defaults
    OPTIONAL_KEYS = {
        'loved_one_path': 'audio_data/static/',
        'male_control_path': 'audio_data/static/ControlStatement_male.wav',
        'female_control_path': 'audio_data/static/ControlStatement_female.wav',
        'sync_pulse_frequency': 1000,
        'sync_pulse_duration_ms': 200
    }
    
    def __init__(self, config_path: str = 'config.yml'):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Raises:
            ConfigFileError: If config file is missing or invalid
            ConfigValidationError: If configuration is invalid
        """
        self.config_path = Path(config_path)
        self.file: Dict[str, Any] = {}
        self.current_date = time.strftime("%Y-%m-%d")
        
        self._load_config()
        self._apply_defaults()
        self._convert_paths()
        self._validate_config()
        self._create_directories()
        self._setup_ui_options()
        
        logger.info(f"Configuration loaded and validated from {config_path}")
    
    def _load_config(self):
        """Load configuration from YAML file.
        
        Raises:
            ConfigFileError: If file cannot be loaded
        """
        if not self.config_path.exists():
            raise ConfigFileError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.file = yaml.safe_load(f)
            
            if not isinstance(self.file, dict):
                raise ConfigFileError("Config file must contain a dictionary")
            
            logger.debug(f"Loaded config with {len(self.file)} keys")
            
        except yaml.YAMLError as e:
            raise ConfigFileError(f"Invalid YAML in config file: {e}") from e
        except Exception as e:
            raise ConfigFileError(f"Failed to load config: {e}") from e
    
    def _apply_defaults(self):
        """Apply default values for optional keys."""
        for key, default_value in self.OPTIONAL_KEYS.items():
            if key not in self.file:
                self.file[key] = default_value
                logger.debug(f"Applied default for {key}: {default_value}")
    
    def _convert_paths(self):
        """Convert all file path strings to Path objects."""
        path_keys = [
            'edf_dir', 'result_dir', 'sentences_path',
            'right_keep_path', 'right_stop_path', 'left_keep_path', 'left_stop_path',
            'beep_path', 'loved_one_path', 'male_control_path', 'female_control_path',
            'motor_prompt_path', 'oddball_prompt_path'
        ]
        
        for key in path_keys:
            if key in self.file and isinstance(self.file[key], str):
                self.file[key] = Path(self.file[key])
                logger.debug(f"Converted {key} to Path")
    
    def _validate_config(self):
        """Validate configuration completeness and correctness.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []
        
        # Check required keys exist
        for key in self.REQUIRED_KEYS:
            if key not in self.file:
                errors.append(f"Missing required key: {key}")
        
        if errors:
            raise ConfigValidationError("Configuration validation failed:\n" + "\n".join(errors))
        
        # Validate specific paths exist
        self._validate_paths(errors)
        
        if errors:
            logger.warning("Configuration warnings:\n" + "\n".join(errors))
    
    def _validate_paths(self, errors: List[str]):
        """Validate that required paths exist.
        
        Args:
            errors: List to append errors to
        """
        # Check that sentences directory exists
        sentences_path = self.file.get('sentences_path')
        if sentences_path and not sentences_path.exists():
            errors.append(f"Sentences directory not found: {sentences_path}")
        elif sentences_path:
            # Check for .wav files
            wav_files = list(sentences_path.glob('*.wav'))
            if not wav_files:
                errors.append(f"No .wav files found in sentences directory: {sentences_path}")
        
        # Check required audio files exist
        required_audio_files = [
            'right_keep_path', 'right_stop_path', 'left_keep_path', 
            'left_stop_path', 'motor_prompt_path', 'oddball_prompt_path'
        ]
        
        for key in required_audio_files:
            path = self.file.get(key)
            if path and not path.exists():
                errors.append(f"Required audio file not found ({key}): {path}")
        
        # Check control files exist (warnings only, not critical)
        for key in ['male_control_path', 'female_control_path']:
            path = self.file.get(key)
            if path and not path.exists():
                logger.warning(f"Control audio file not found ({key}): {path}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = ['result_dir', 'edf_dir']
        
        for dir_key in dirs_to_create:
            if dir_key in self.file:
                dir_path = self.file[dir_key]
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not create directory {dir_path}: {e}")
    
    def _setup_ui_options(self):
        """Set up UI dropdown options."""
        self.cpc_scale = CPC_SCALE
        self.gose_scale = GOSE_SCALE
        self.graphs = ["", "CMD", "Language Tracking"]
    
    def get_path(self, key: str) -> Path:
        """Get a path from configuration.
        
        Args:
            key: Configuration key for the path
            
        Returns:
            Path object
            
        Raises:
            ConfigValidationError: If key doesn't exist
        """
        if key not in self.file:
            raise ConfigValidationError(f"Path key '{key}' not found in configuration")
        return self.file[key]
    
    def get_results_path(self, patient_id: str, date: Optional[str] = None) -> Path:
        """Generate results file path for a patient.
        
        Args:
            patient_id: Patient identifier
            date: Date string (YYYY-MM-DD), defaults to current date
            
        Returns:
            Path to results CSV file
        """
        if not patient_id:
            raise ValueError("Patient ID cannot be empty")
        
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
        if not patient_id:
            raise ValueError("Patient ID cannot be empty")
        
        if date is None:
            date = self.current_date
        
        filename = f"{patient_id}_{date}.edf"
        return self.file['edf_dir'] / filename
    
    def validate_loved_one_setup(self, gender: str, voice_file: Optional[str] = None) -> bool:
        """Validate loved one stimulus configuration.
        
        Args:
            gender: Gender ('Male' or 'Female')
            voice_file: Optional voice file path
            
        Returns:
            True if valid
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        if gender not in ['Male', 'Female']:
            raise ConfigValidationError(f"Invalid gender: {gender}. Must be 'Male' or 'Female'")
        
        # Check control file exists
        control_key = 'male_control_path' if gender == 'Male' else 'female_control_path'
        control_path = self.file.get(control_key)
        
        if not control_path or not control_path.exists():
            raise ConfigValidationError(
                f"Control audio file for {gender} not found: {control_path}"
            )
        
        # Check voice file if provided
        if voice_file:
            voice_path = Path(voice_file)
            if not voice_path.exists():
                # Try relative to loved_one_path
                voice_path = self.file['loved_one_path'] / voice_file
                if not voice_path.exists():
                    raise ConfigValidationError(f"Voice file not found: {voice_file}")
        
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'config_file': str(self.config_path),
            'current_date': self.current_date,
            'result_dir': str(self.file.get('result_dir')),
            'edf_dir': str(self.file.get('edf_dir')),
            'sentences_available': len(list(self.file['sentences_path'].glob('*.wav'))) 
                                  if self.file.get('sentences_path') else 0
        }
