# lib/auditory_stimulator.py
import os
import random
import yaml
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

class AuditoryStimulator:
     
    def __init__(self):
        """Initialize the auditory stimulator with configuration"""

        # Initialize attributes
        config_file_path = 'config.yml'
        with open(config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)

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

        self.sample_rate = 44100

    def generate_trials(self, num_of_each_trials):      
        # generate random set of trials
        trial_names = []

        # interate through each trial type
        for key in num_of_each_trials:
            # add trial names for for each trial type save gather audio data
            if num_of_each_trials[key] > 0:
                # prepare language stims
                if key == "lang":
                    self._generate_language_stimuli(num_of_each_trials[key])
                    for i in range(num_of_each_trials[key]):
                        trial_names.append(f"lang_{i}")
                # prepare right hand commands
                elif key == "rcmd":
                    self.right_keep_audio = AudioSegment.from_mp3(self.config['right_keep_path'])
                    self.right_stop_audio = AudioSegment.from_mp3(self.config['right_stop_path'])
                    for i in range(num_of_each_trials[key]):
                        trial_names.append(key)
                # prepare left hand commands
                elif key == "lcmd":
                    self.left_keep_audio = AudioSegment.from_mp3(self.config['left_keep_path'])
                    self.left_stop_audio = AudioSegment.from_mp3(self.config['left_stop_path'])
                    for i in range(num_of_each_trials[key]):
                        trial_names.append(key)
                # prepare oddball trials
                elif key == "odd":
                    for i in range(num_of_each_trials[key]):
                        trial_names.append('oddball')
                
                elif key == "loved":
                    # path to loved ones voice recording
                    temp_path = os.path.join(self.config['loved_one_path'], self.loved_one_file)
                    # add loved ones voice recording
                    self.loved_one_voice_audio = self._load_audio(temp_path)
                    # add a gendered control voice recording
                    if self.loved_one_gender == 'Male':
                        self.control_voice_audio = self._load_audio(self.config['male_control_path'])
                    elif self.loved_one_gender == 'Female':
                        self.control_voice_audio = self._load_audio(self.config['female_control_path'])
                    else:
                        raise ValueError(f"No gender selected")
                    # add loved ones or control voice recording
                    trial_names += ['control'] * num_of_each_trials[key]
                    trial_names += ['loved_one'] * num_of_each_trials[key]

        # shuffle the trials to be random
        random.shuffle(trial_names)

        return trial_names

    def _load_audio(self, path):
        """Load audio file and convert to numpy array for sounddevice playback"""
        try:
            if path.endswith('.mp3'):
                audio_segment = AudioSegment.from_mp3(path)
            elif path.endswith('.wav'):
                audio_segment = AudioSegment.from_wav(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
        
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Reshape stereo audio to (n_samples, 2)
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
                
            # Normalize to [-1, 1] range
            return samples.astype(np.float32) / (2**15 - 1)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _generate_language_stimuli(self, num_of_lang_trials):
        """Generate the specified number of language stimuli"""
        for i in range(num_of_lang_trials):
            self._random_lang_stim()
            percent = int(i/num_of_lang_trials*100)
            print(f"Generating language stimuli: {percent}% complete") # Consider using logging

    def _random_lang_stim(self, num_sentence=12):
        """Create a random language stimulus from available sentence files"""

        sentence_files = os.listdir(self.config['sentences_path'])

        # Filter out non-wav files
        wav_files = [file for file in sentence_files if file.endswith('.wav')]

        # Ensure num_sentence does not exceed available wav files
        if num_sentence > len(wav_files):
            raise ValueError(f"Requested {num_sentence} files, but only {len(wav_files)} available.")

        selected_ids = set()  # To keep track of already selected IDs
        combined = AudioSegment.empty()
        sample_ids = []

        while len(sample_ids) < num_sentence:
            # Randomly choose an ID
            id = random.choice(range(len(wav_files)))
            if id in selected_ids:
                continue  # Skip if this ID was already selected
            file = os.path.join(self.config['sentences_path'], f'lang{id}.wav')
            if os.path.exists(file):
                # If the file exists, add its ID to sample_ids and selected_ids
                sample_ids.append(id)
                selected_ids.add(id)

                # Read and concatenate the audio
                audio = AudioSegment.from_wav(file)
                combined += audio
            else:
                continue # This case should ideally not happen if file list is accurate
                
        # save audio segement
        self.lang_audio.append(combined)
        # save sample IDs
        self.lang_trials_ids.append(sample_ids)
