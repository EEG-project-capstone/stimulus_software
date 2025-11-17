# lib/trials.py

import numpy as np
import os
import random
import logging
from pydub import AudioSegment

logger = logging.getLogger('eeg_stimulus.trials')

class Trials:

    def __init__(self, gui_callback):
        self.config = gui_callback.config
        self.trial_dictionary = []
        self.current_trial_index = None

        self.lang_audio = []
        self.lang_trials_ids = []

        self.right_keep_audio = None
        self.right_stop_audio = None
        self.left_keep_audio = None
        self.left_stop_audio = None

        self.loved_one_file = ""
        self.loved_one_gender = ""
        self.loved_one_voice_audio = None
        self.control_voice_audio = None

        self.motor_prompt_audio = None
        self.oddball_prompt_audio = None

        self.sample_rate = 44100
        
        logger.info("Trials initialized")
 
    def generate_trials(self, num_of_each_trials):
        """Generate trial sequence with proper validation"""
        logger.info(f"Generating trials: {num_of_each_trials}")
        
        # Clear existing trials before generating new ones
        self.trial_dictionary = []      

        # Validate loved one trials requirements
        if num_of_each_trials.get("loved", 0) > 0:
            if not self.loved_one_file:
                error_msg = "Loved one trials requested but no audio file specified"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not self.loved_one_gender or self.loved_one_gender not in ['Male', 'Female']:
                error_msg = f"Loved one trials requested but gender not properly set: {self.loved_one_gender}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # We'll collect blocks: each block is a list of trials of the same type
        blocks = []

        # Language trials
        if num_of_each_trials.get("lang", 0) > 0:
            logger.info(f"Generating {num_of_each_trials['lang']} language trials")
            self._generate_language_stimuli(num_of_each_trials["lang"])
            lang_block = []
            for i in range(num_of_each_trials["lang"]):
                lang_block.append({
                    "type": "language",
                    "subtype": f"lang_{i}",
                    "audio_index": i,
                    "status": "pending"
                })
            blocks.append(lang_block)
            logger.debug(f"Added {len(lang_block)} language trials to blocks")

        # Right command (no prompt)
        if num_of_each_trials.get("rcmd", 0) > 0:
            logger.info(f"Loading right command audio for {num_of_each_trials['rcmd']} trials")
            self.right_keep_audio = AudioSegment.from_mp3(self.config.file['right_keep_path'])
            self.right_stop_audio = AudioSegment.from_mp3(self.config.file['right_stop_path'])
            rcmd_block = [{"type": "right_command", "status": "pending"} for _ in range(num_of_each_trials["rcmd"])]
            blocks.append(rcmd_block)
            logger.debug(f"Added {len(rcmd_block)} right command trials")

        # Right command + prompt
        if num_of_each_trials.get("rcmd+p", 0) > 0:
            logger.info(f"Loading right command with prompt audio for {num_of_each_trials['rcmd+p']} trials")
            self.motor_prompt_audio = AudioSegment.from_wav(self.config.file['motor_prompt_path'])
            self.right_keep_audio = AudioSegment.from_mp3(self.config.file['right_keep_path'])
            self.right_stop_audio = AudioSegment.from_mp3(self.config.file['right_stop_path'])
            rcmd_p_block = [{"type": "right_command+p", "status": "pending"} for _ in range(num_of_each_trials["rcmd+p"])]
            blocks.append(rcmd_p_block)
            logger.debug(f"Added {len(rcmd_p_block)} right command+prompt trials")

        # Left command (no prompt)
        if num_of_each_trials.get("lcmd", 0) > 0:
            logger.info(f"Loading left command audio for {num_of_each_trials['lcmd']} trials")
            self.left_keep_audio = AudioSegment.from_mp3(self.config.file['left_keep_path'])
            self.left_stop_audio = AudioSegment.from_mp3(self.config.file['left_stop_path'])
            lcmd_block = [{"type": "left_command", "status": "pending"} for _ in range(num_of_each_trials["lcmd"])]
            blocks.append(lcmd_block)
            logger.debug(f"Added {len(lcmd_block)} left command trials")

        # Left command + prompt
        if num_of_each_trials.get("lcmd+p", 0) > 0:
            logger.info(f"Loading left command with prompt audio for {num_of_each_trials['lcmd+p']} trials")
            self.motor_prompt_audio = AudioSegment.from_wav(self.config.file['motor_prompt_path'])
            self.left_keep_audio = AudioSegment.from_mp3(self.config.file['left_keep_path'])
            self.left_stop_audio = AudioSegment.from_mp3(self.config.file['left_stop_path'])
            lcmd_p_block = [{"type": "left_command+p", "status": "pending"} for _ in range(num_of_each_trials["lcmd+p"])]
            blocks.append(lcmd_p_block)
            logger.debug(f"Added {len(lcmd_p_block)} left command+prompt trials")

        # Oddball (no prompt)
        if num_of_each_trials.get("odd", 0) > 0:
            logger.info(f"Creating {num_of_each_trials['odd']} oddball trials")
            odd_block = [{"type": "oddball", "status": "pending"} for _ in range(num_of_each_trials["odd"])]
            blocks.append(odd_block)
            logger.debug(f"Added {len(odd_block)} oddball trials")

        # Oddball + prompt
        if num_of_each_trials.get("odd+p", 0) > 0:
            logger.info(f"Loading oddball prompt audio for {num_of_each_trials['odd+p']} trials")
            self.oddball_prompt_audio = AudioSegment.from_wav(self.config.file['oddball_prompt_path'])
            odd_p_block = [{"type": "oddball+p", "status": "pending"} for _ in range(num_of_each_trials["odd+p"])]
            blocks.append(odd_p_block)
            logger.debug(f"Added {len(odd_p_block)} oddball+prompt trials")
                
        # Loved one trials
        if num_of_each_trials.get("loved", 0) > 0:
            logger.info(f"Loading loved one voice audio for {num_of_each_trials['loved']} trial pairs")
            lof = self.loved_one_file
            temp_path = lof if os.path.isabs(lof) else os.path.join(self.config.file['loved_one_path'], lof)
            
            # Validate file exists
            if not os.path.exists(temp_path):
                error_msg = f"Loved one audio file not found: {temp_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.debug(f"Loading loved one audio from: {temp_path}")
            self.loved_one_voice_audio = self._load_audio_as_int16(temp_path)
            
            if self.loved_one_gender == 'Male':
                control_path = self.config.file['male_control_path']
            elif self.loved_one_gender == 'Female':
                control_path = self.config.file['female_control_path']
            else:
                error_msg = f"Invalid gender: {self.loved_one_gender}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not os.path.exists(control_path):
                error_msg = f"Control audio file not found: {control_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.debug(f"Loading control audio from: {control_path}")
            self.control_voice_audio = self._load_audio_as_int16(control_path)
            
            loved_block = []
            for i in range(num_of_each_trials["loved"]):
                loved_block.append({"type": "control", "voice_type": "control", "status": "pending"})
                loved_block.append({"type": "loved_one_voice", "voice_type": "loved_one", "status": "pending"})
            blocks.append(loved_block)
            logger.debug(f"Added {len(loved_block)} voice trials (control + loved one pairs)")

        # Randomize the order of blocks
        logger.info(f"Randomizing {len(blocks)} trial blocks")
        random.shuffle(blocks)

        # Flatten blocks into final trial list
        for block in blocks:
            self.trial_dictionary.extend(block)
        
        logger.info(f"Trial generation complete: {len(self.trial_dictionary)} total trials")
        
        # Log trial type summary
        trial_summary = {}
        for trial in self.trial_dictionary:
            trial_type = trial['type']
            trial_summary[trial_type] = trial_summary.get(trial_type, 0) + 1
        logger.info(f"Trial type summary: {trial_summary}")
    
    def _generate_language_stimuli(self, num_of_lang_trials):
        """Generate the specified number of language stimuli"""
        logger.info(f"Generating {num_of_lang_trials} language stimuli")
        
        for i in range(num_of_lang_trials):
            try:
                self._random_lang_stim()
                if (i + 1) % 5 == 0 or i == num_of_lang_trials - 1:
                    percent = int((i + 1) / num_of_lang_trials * 100)
                    logger.info(f"Language stimuli generation: {percent}% complete ({i + 1}/{num_of_lang_trials})")
            except Exception as e:
                logger.error(f"Error generating language stimulus {i}: {e}", exc_info=True)
                raise

    def _load_audio_as_int16(self, path):
        """
        Load audio file and convert to int16 numpy array for sounddevice playback.
        This ensures consistency with the rest of the audio pipeline.
        """
        logger.debug(f"Loading audio file: {path}")
        
        try:
            if path.endswith('.mp3'):
                audio_segment = AudioSegment.from_mp3(path)
            elif path.endswith('.wav'):
                audio_segment = AudioSegment.from_wav(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
        
            # Log original audio properties
            logger.debug(f"Audio properties: {audio_segment.frame_rate}Hz, "
                        f"{audio_segment.channels} channels, "
                        f"{len(audio_segment)}ms duration")
            
            # Resample to 44100 Hz to standardize
            if audio_segment.frame_rate != 44100:
                logger.debug(f"Resampling from {audio_segment.frame_rate}Hz to 44100Hz")
                audio_segment = audio_segment.set_frame_rate(44100)
            
            # Convert to numpy array (already int16 from pydub)
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            
            # Reshape for proper channel handling
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
                logger.debug(f"Loaded stereo audio: shape={samples.shape}")
            else:
                # Mono audio as (n_samples, 1) for consistency
                samples = samples.reshape((-1, 1))
                logger.debug(f"Loaded mono audio: shape={samples.shape}")
            
            logger.info(f"Successfully loaded audio: {path} ({samples.shape})")
            return samples

        except Exception as e:
            logger.error(f"Error loading audio file {path}: {e}", exc_info=True)
            raise

    def _random_lang_stim(self, num_sentence=12):
        """Create a random language stimulus from available sentence files"""
        sentence_path = self.config.file['sentences_path']
        logger.debug(f"Creating language stimulus from: {sentence_path}")
        
        if not os.path.exists(sentence_path):
            error_msg = f"Sentences directory not found: {sentence_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        sentence_files = os.listdir(sentence_path)

        # Filter out non-wav files
        wav_files = [file for file in sentence_files if file.endswith('.wav')]
        logger.debug(f"Found {len(wav_files)} wav files in sentences directory")

        # Ensure num_sentence does not exceed available wav files
        if num_sentence > len(wav_files):
            error_msg = f"Requested {num_sentence} sentences, but only {len(wav_files)} available"
            logger.error(error_msg)
            raise ValueError(error_msg)

        selected_ids = set()  # To keep track of already selected IDs
        combined = AudioSegment.empty()
        sample_ids = []

        while len(sample_ids) < num_sentence:
            # Randomly choose an ID
            id = random.choice(range(len(wav_files)))
            if id in selected_ids:
                continue  # Skip if this ID was already selected
            file = os.path.join(sentence_path, f'lang{id}.wav')
            if os.path.exists(file):
                # If the file exists, add its ID to sample_ids and selected_ids
                sample_ids.append(id)
                selected_ids.add(id)

                # Read and concatenate the audio
                try:
                    audio = AudioSegment.from_wav(file)
                    combined += audio
                except Exception as e:
                    logger.warning(f"Error loading sentence file {file}: {e}")
                    selected_ids.remove(id)
                    sample_ids.remove(id)
                    continue
            else:
                logger.warning(f"Expected sentence file not found: {file}")
                continue
        
        # save audio segment
        self.lang_audio.append(combined)
        # save sample IDs
        self.lang_trials_ids.append(sample_ids)
        
        logger.debug(f"Language stimulus created: {len(sample_ids)} sentences, "
                    f"{len(combined)}ms total duration, IDs={sample_ids}")