# lib/stims.py

import numpy as np
import random
import logging
from pydub import AudioSegment
from pathlib import Path

from lib.constants import FilePaths, LanguageStimParams, MALE_CONTROL_VOICES, FEMALE_CONTROL_VOICES

logger = logging.getLogger('eeg_stimulus.stims')

class Stims:

    def __init__(self):
        self.stim_dictionary = []
        self.current_stim_index = None

        self.lang_audio = []
        self.lang_stims_ids = []

        self.right_keep_audio = None
        self.right_stop_audio = None
        self.left_keep_audio = None
        self.left_stop_audio = None

        self.familiar_file = ""
        self.familiar_gender = ""
        self.familiar_voice_audio = None
        self.unfamiliar_voices_audio = []

        self.motor_prompt_audio = None
        self.oddball_prompt_audio = None

        self.sample_rate = 44100

        logger.info("Stims initialized")

    def generate_stims(self, num_of_each_stims):
        """Generate stimulus sequence with proper validation"""
        logger.info(f"Generating stimuli: {num_of_each_stims}")

        # Clear existing stimuli before generating new ones
        self.stim_dictionary = []
        self.lang_audio = []
        self.lang_stims_ids = []
        self.unfamiliar_voices_audio = []

        # Validate familiar voice stimuli requirements
        if num_of_each_stims.get("familiar_voice", 0) > 0:
            if not self.familiar_file:
                error_msg = "Familiar voice stimuli requested but no audio file specified"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not self.familiar_gender or self.familiar_gender not in ['Male', 'Female']:
                error_msg = f"Familiar voice stimuli requested but gender not properly set: {self.familiar_gender}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # We'll collect blocks: each block is a list of stimuli of the same type
        blocks = []

        # Language stimuli
        if num_of_each_stims.get("lang", 0) > 0:
            logger.info(f"Generating {num_of_each_stims['lang']} language stimuli")
            self._generate_language_stimuli(num_of_each_stims["lang"])
            lang_block = []
            for i in range(num_of_each_stims["lang"]):
                lang_block.append({
                    "type": "language",
                    "subtype": f"lang_{i}",
                    "audio_index": i,
                    "status": "pending"
                })
            blocks.append(lang_block)
            logger.debug(f"Added {len(lang_block)} language stimuli to blocks")

        # Right command (no prompt)
        if num_of_each_stims.get("rcmd", 0) > 0:
            logger.info(f"Loading right command audio for {num_of_each_stims['rcmd']} stimuli")
            self.right_keep_audio = AudioSegment.from_mp3(FilePaths.RIGHT_KEEP_AUDIO)
            self.right_stop_audio = AudioSegment.from_mp3(FilePaths.RIGHT_STOP_AUDIO)
            rcmd_block = [{"type": "right_command", "status": "pending"} for _ in range(num_of_each_stims["rcmd"])]
            blocks.append(rcmd_block)
            logger.debug(f"Added {len(rcmd_block)} right command stimuli")

        # Right command + prompt
        if num_of_each_stims.get("rcmd+p", 0) > 0:
            logger.info(f"Loading right command with prompt audio for {num_of_each_stims['rcmd+p']} stimuli")
            self.motor_prompt_audio = AudioSegment.from_wav(FilePaths.MOTOR_PROMPT)
            self.right_keep_audio = AudioSegment.from_mp3(FilePaths.RIGHT_KEEP_AUDIO)
            self.right_stop_audio = AudioSegment.from_mp3(FilePaths.RIGHT_STOP_AUDIO)
            rcmd_p_block = [{"type": "right_command+p", "status": "pending"} for _ in range(num_of_each_stims["rcmd+p"])]
            blocks.append(rcmd_p_block)
            logger.debug(f"Added {len(rcmd_p_block)} right command+prompt stimuli")

        # Left command (no prompt)
        if num_of_each_stims.get("lcmd", 0) > 0:
            logger.info(f"Loading left command audio for {num_of_each_stims['lcmd']} stimuli")
            self.left_keep_audio = AudioSegment.from_mp3(FilePaths.LEFT_KEEP_AUDIO)
            self.left_stop_audio = AudioSegment.from_mp3(FilePaths.LEFT_STOP_AUDIO)
            lcmd_block = [{"type": "left_command", "status": "pending"} for _ in range(num_of_each_stims["lcmd"])]
            blocks.append(lcmd_block)
            logger.debug(f"Added {len(lcmd_block)} left command stimuli")

        # Left command + prompt
        if num_of_each_stims.get("lcmd+p", 0) > 0:
            logger.info(f"Loading left command with prompt audio for {num_of_each_stims['lcmd+p']} stimuli")
            self.motor_prompt_audio = AudioSegment.from_wav(FilePaths.MOTOR_PROMPT)
            self.left_keep_audio = AudioSegment.from_mp3(FilePaths.LEFT_KEEP_AUDIO)
            self.left_stop_audio = AudioSegment.from_mp3(FilePaths.LEFT_STOP_AUDIO)
            lcmd_p_block = [{"type": "left_command+p", "status": "pending"} for _ in range(num_of_each_stims["lcmd+p"])]
            blocks.append(lcmd_p_block)
            logger.debug(f"Added {len(lcmd_p_block)} left command+prompt stimuli")

        # Oddball (no prompt)
        if num_of_each_stims.get("odd", 0) > 0:
            logger.info(f"Creating {num_of_each_stims['odd']} oddball stimuli")
            odd_block = [{"type": "oddball", "status": "pending"} for _ in range(num_of_each_stims["odd"])]
            blocks.append(odd_block)
            logger.debug(f"Added {len(odd_block)} oddball stimuli")

        # Oddball + prompt
        if num_of_each_stims.get("odd+p", 0) > 0:
            logger.info(f"Loading oddball prompt audio for {num_of_each_stims['odd+p']} stimuli")
            self.oddball_prompt_audio = AudioSegment.from_wav(FilePaths.ODDBALL_PROMPT)
            odd_p_block = [{"type": "oddball+p", "status": "pending"} for _ in range(num_of_each_stims["odd+p"])]
            blocks.append(odd_p_block)
            logger.debug(f"Added {len(odd_p_block)} oddball+prompt stimuli")

        # Familiar voice stimuli — 50% familiar, 50% unfamiliar (gender-matched)
        if num_of_each_stims.get("familiar_voice", 0) > 0:
            n = num_of_each_stims["familiar_voice"]
            logger.info(f"Loading familiar voice audio for {n} trials (50% familiar / 50% unfamiliar)")

            # Load familiar voice
            lof = self.familiar_file
            temp_path = Path(lof) if Path(lof).is_absolute() else FilePaths.FAMILIAR_DIR / lof
            if not temp_path.exists():
                error_msg = f"Familiar voice audio file not found: {temp_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            logger.debug(f"Loading familiar voice audio from: {temp_path}")
            self.familiar_voice_audio = self._load_audio_as_int16(temp_path)

            # Load all gender-matched unfamiliar control statements
            if self.familiar_gender == 'Male':
                voice_names = MALE_CONTROL_VOICES
            elif self.familiar_gender == 'Female':
                voice_names = FEMALE_CONTROL_VOICES
            else:
                error_msg = f"Invalid gender: {self.familiar_gender}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            control_dir = FilePaths.CONTROL_STATEMENTS_DIR
            self.unfamiliar_voices_audio = []
            for name in voice_names:
                path = control_dir / f"{name}_normalized.wav"
                if not path.exists():
                    error_msg = f"Unfamiliar voice audio not found: {path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                logger.debug(f"Loading unfamiliar voice from: {path}")
                self.unfamiliar_voices_audio.append(self._load_audio_as_int16(path))

            # Build balanced 50/50 block: n_familiar familiar + n_unfamiliar unfamiliar
            n_familiar = n // 2
            n_unfamiliar = n - n_familiar

            voice_block = []
            for _ in range(n_familiar):
                voice_block.append({"type": "familiar", "status": "pending"})
            for _ in range(n_unfamiliar):
                voice_index = random.randrange(len(self.unfamiliar_voices_audio))
                voice_block.append({"type": "unfamiliar", "voice_index": voice_index, "status": "pending"})
            random.shuffle(voice_block)
            blocks.append(voice_block)
            logger.debug(f"Added {len(voice_block)} voice stimuli ({n_familiar} familiar, {n_unfamiliar} unfamiliar)")

        # Randomize the order of blocks
        logger.info(f"Randomizing {len(blocks)} stimulus blocks")
        random.shuffle(blocks)

        # Flatten blocks into final stimulus list
        for block in blocks:
            self.stim_dictionary.extend(block)

        logger.info(f"Stimulus generation complete: {len(self.stim_dictionary)} total stimuli")

        # Log stimulus type summary
        stim_summary = {}
        for stim in self.stim_dictionary:
            stim_type = stim['type']
            stim_summary[stim_type] = stim_summary.get(stim_type, 0) + 1
        logger.info(f"Stimulus type summary: {stim_summary}")

    def _generate_language_stimuli(self, num_of_lang_stims):
        """Generate the specified number of language stimuli"""
        logger.info(f"Generating {num_of_lang_stims} language stimuli")

        for i in range(num_of_lang_stims):
            try:
                self._random_lang_stim()
                if (i + 1) % 5 == 0 or i == num_of_lang_stims - 1:
                    percent = int((i + 1) / num_of_lang_stims * 100)
                    logger.info(f"Language stimuli generation: {percent}% complete ({i + 1}/{num_of_lang_stims})")
            except Exception as e:
                logger.error(f"Error generating language stimulus {i}: {e}", exc_info=True)
                raise

    def _load_audio_as_int16(self, path: Path):
        """
        Load audio file and convert to int16 numpy array for sounddevice playback.
        This ensures consistency with the rest of the audio pipeline.
        """
        logger.debug(f"Loading audio file: {path}")

        try:
            if path.suffix == '.mp3':
                audio_segment = AudioSegment.from_mp3(path)
            elif path.suffix == '.wav':
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

    def _random_lang_stim(self, num_sentence=LanguageStimParams.SENTENCES_PER_STIMULUS):
        """Create a random language stimulus from available sentence files"""
        sentence_path = FilePaths.SENTENCES_DIR
        logger.debug(f"Creating language stimulus from: {sentence_path}")

        if not sentence_path.exists():
            error_msg = f"Sentences directory not found: {sentence_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Use pathlib for file listing
        wav_files = list(sentence_path.glob('*.wav'))
        logger.debug(f"Found {len(wav_files)} wav files in sentences directory")

        # Ensure num_sentence does not exceed available wav files
        if num_sentence > len(wav_files):
            error_msg = f"Requested {num_sentence} sentences, but only {len(wav_files)} available"
            logger.error(error_msg)
            raise ValueError(error_msg)

        combined = AudioSegment.empty()
        sample_ids = []

        chosen_files = random.sample(wav_files, num_sentence)
        for file in chosen_files:
            try:
                audio = AudioSegment.from_wav(file)
                combined += audio
                sample_ids.append(file.stem)
            except Exception as e:
                logger.error(f"Error loading sentence file {file}: {e}", exc_info=True)
                raise

        # save audio segment
        self.lang_audio.append(combined)
        # save sample IDs
        self.lang_stims_ids.append(sample_ids)

        logger.debug(f"Language stimulus created: {len(sample_ids)} sentences, "
                    f"{len(combined)}ms total duration, IDs={sample_ids}")
