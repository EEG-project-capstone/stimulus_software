import time
import os
import random
import yaml
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import sounddevice as sd

class AuditoryStimulator:
     
    def __init__(self, config_file_path='config.yml'):
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

        self.beep_audio = None

        self.loved_one_file = ""
        self.loved_one_gender = ""
        self.loved_one_audio = None
        self.control_audio = None

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
                # prepare beep
                elif key == "beep":
                    self.beep_audio = AudioSegment.from_mp3(self.config['beep_path'])
                    for i in range(num_of_each_trials[key]):
                        trial_names.append(key)
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
        try:
            if path.endswith('.mp3'):
                audio_segment = AudioSegment.from_mp3(path)
            elif path.endswith('.wav'):
                audio_segment = AudioSegment.from_wav(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
        
            # print(audio_segment.frame_rate)

            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Reshape stereo audio to (n_samples, 2)
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
                
            # Normalize to [-1, 1] range
            return samples.astype(np.float32) / (2**15 - 1)

            # return np.array(audio_segment.get_array_of_samples())

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def play_stimuli(self, trial):

        # add sentences list for lang and oddball orders
        sentences = []

        # administered correct 
        if trial[:4] == "lang":
            start_time, end_time, sentences = self._administer_lang(trial)
        elif trial == "rcmd":
            start_time, end_time = self._administer_right_cmd()
        elif trial == "lcmd":
            start_time, end_time = self._administer_left_cmd()
        elif trial == "beep":
            start_time, end_time = self._administer_beep()
        elif trial == "oddball":
            start_time, end_time, sentences = self._administer_oddball()
        elif trial == "control":          
            start_time, end_time = self._administer_control()
        elif trial == "loved_one": 
            start_time, end_time = self._administer_loved_one()
        else:
            # Default values if trial type is unknown
            start_time = time.time()
            end_time = start_time
            sentences = []

        print(f"Successfully administered {trial}")

        time.sleep(random.uniform(1.2, 2.2)) 

        return start_time, end_time, sentences

    def _generate_language_stimuli(self, num_of_lang_trials):
        for i in range(num_of_lang_trials):
            self._random_lang_stim()
            percent = int(i/num_of_lang_trials*100)
            print(f"Generating language stimuli: {percent}% complete")

    def _random_lang_stim(self, num_sentence=12):

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
                # raise ValueError(f"Requested lang{id}.wav file, but lang{id}.wav does not exist.")
                continue
        # save audio segement
        self.lang_audio.append(combined)
        # save sample IDs
        self.lang_trials_ids.append(sample_ids)

    def _administer_lang(self, trial):

        sentences = []
        n = int(trial[5:])
        start_time = time.time()

        # Debug info
        print(f"Trial: {trial}, index: {n}")
        # print(f"lang_trials_ids length: {len(self.lang_trials_ids)}")
        # print(f"lang_audio length: {len(self.lang_audio)}")
        
        # Safely access data
        if 0 <= n < len(self.lang_trials_ids):
            sentences = self.lang_trials_ids[n]
            print(f"Playing the following lang_trials_ids: {sentences}")
    

        if 0 <= n < len(self.lang_audio):
            print("Playing audio")
            play(self.lang_audio[n])

        # print("Audio playback complete")

        end_time = time.time()

        return start_time, end_time, sentences

    def _administer_right_cmd(self):
        start_time = time.time()
        for _ in range(8):
            play(self.right_keep_audio)
            time.sleep(10)
            play(self.right_stop_audio)
            time.sleep(10)
        end_time = time.time()
        return start_time, end_time

    def _administer_left_cmd(self):
        start_time = time.time()
        for _ in range(8):
            play(self.left_keep_audio)
            time.sleep(10)
            play(self.left_stop_audio)
            time.sleep(10)
        end_time = time.time()
        return start_time, end_time

    def _administer_beep(self):
        start_time = time.time()
        play(self.beep_audio)
        time.sleep(10)
        end_time = time.time()
        return start_time, end_time

    def _administer_oddball(self):

        sentences = []
        start_time = time.time()

        # play 5 standard tones
        for _ in range(5):
            audio_segment = Sine(1000).to_audio_segment(duration=100)
            samples = audio_segment.get_array_of_samples()
            sd.play(samples, audio_segment.frame_rate)
            sd.wait()
            sentences.append('standard')
            time.sleep(1)

        # play 20 standard or rare tones 
        for i in range(20):
            # play rare tone with 20% probability
            if random.random() < 0.2:
                audio_segment = Sine(2000).to_audio_segment(duration=100)
                samples = audio_segment.get_array_of_samples()
                sd.play(samples, audio_segment.frame_rate)
                sd.wait()
                sentences.append('rare')
            # else play standard tone 
            else:
                audio_segment = Sine(1000).to_audio_segment(duration=100)
                samples = audio_segment.get_array_of_samples()
                sd.play(samples, audio_segment.frame_rate)
                sd.wait()
                sentences.append('standard')
            # pause one second after each beep
            time.sleep(1) 
                    
        end_time = time.time()

        # print out sentences: standard or rare
        print(f"Playing the following frequencies: {sentences}")

        return start_time, end_time, sentences

    def _administer_control(self):
        print(f"Playing control recording")
        start_time = time.time()
        self._play_audio_segment(self.control_voice_audio)
        end_time = time.time()
        return start_time, end_time

    def _administer_loved_one(self):
        print(f"Playing loved one recording")        
        start_time = time.time()
        self._play_audio_segment(self.loved_one_voice_audio)
        end_time = time.time()
        return start_time, end_time
    
    def _play_audio_segment(self, audio_segment):
        try:
            if audio_segment is None:
                print("Error: No audio data to play")
                return False
            sd.play(audio_segment, self.sample_rate, blocking=True)
            sd.wait()
        except Exception as e:
            print(f"Warning: Could not play {audio_segment}: {e}")