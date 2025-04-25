import time
import os
import random
import yaml
import time
import streamlit as st
import pydub
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play


config_file_path = 'config.yml'  # Replace with the actual path to your config file
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)


def jittered_delay():
    time.sleep(random.uniform(1.2, 2.2))

def play_mp3(mp3_path, verbose=True):
    if verbose:
        print(f"Playing {mp3_path}...")
    if config['os'].lower() == 'windows':
        import mpv
        player = mpv.MPV(input_default_bindings=True, input_vo_keyboard=True, osc=True)

        @player.property_observer('time-pos')
        def on_time_pos_change(_name, value):
            """Print video start and end times"""
            if value == 0 or value is None:
                start_time = time.time()
                print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        pydub.AudioSegment.converter = os.path.join(os.getcwd(), 'ffmpeg', 'bin', "ffmpeg.exe")               
        pydub.AudioSegment.ffprobe   = os.path.join(os.getcwd(), 'ffmpeg', 'bin', "ffprobe.exe")
        print(f"pydub.AudioSegment.converter: {pydub.AudioSegment.converter}")
        print(f"pydub.AudioSegment.ffprobe: {pydub.AudioSegment.ffprobe}")

        player.play(mp3_path)
        player.wait_for_playback()
    
    elif config['os'].lower() == 'linux':
        audio = AudioSegment.from_mp3(mp3_path)
        play(audio)
    
    else:
        from playsound import playsound
        playsound(mp3_path)

def speed_up_audio(input_path, output_path, speed_factor=1.5):
    audio = AudioSegment.from_mp3(input_path)
    audio = audio.speedup(playback_speed=speed_factor)
    audio.export(output_path, format="mp3")

def random_lang_stim(output_path, num_sentence=12):

    sentence_files = os.listdir(config['sentences_path'])

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

        file = os.path.join(config['sentences_path'], f'lang{id}.wav')

        if os.path.exists(file):
            # If the file exists, add its ID to sample_ids and selected_ids
            sample_ids.append(id)
            selected_ids.add(id)

            # Read and concatenate the audio
            audio = AudioSegment.from_wav(file)
            combined += audio

    # Export the processed audio
    combined.export(output_path, format="mp3")
    
    return sample_ids

def administer_lang(input_path, test_run=False):
    start_time = time.time()
    if not test_run:
        play_mp3(input_path)
        time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def administer_right_cmd(test_run=False):
    start_time = time.time()
    if not test_run:
        for _ in range(8):
            play_mp3(config['right_keep_path'])
            time.sleep(10)
            play_mp3(config['right_stop_path'])
            time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def administer_left_cmd(test_run=False):
    start_time = time.time()
    if not test_run:
        for _ in range(8):
            play_mp3(config['left_keep_path'])
            time.sleep(10)
            play_mp3(config['left_stop_path'])
            time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def administer_beep(test_run=False):
    start_time = time.time()
    if not test_run:
        play_mp3(config['beep_path'])
        time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def administer_oddball(test_run=False):
    start_time = time.time()
    if not test_run:
        play_oddball_stimulus()
        time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def randomize_trials(num_of_each_trial):
    trial_types = []
     
    for key in num_of_each_trial:
        for i in range(num_of_each_trial[key]):
            if key == "lang":
                trial = f"lang_{i}"
            else:
                trial = key
            trial_types.append(trial)

    random.shuffle(trial_types)
    return trial_types

def generate_stimuli(trial_types):
    gen_bar = st.progress(0, text="0")
    n = len(trial_types)
    lang_trials_ids = []
    for i in range(n):
        trial = trial_types[i]
        if trial[:4] == "lang":
            output_path = os.path.join(config['stimuli_dir'], f"{trial}.mp3")
            sample_ids = random_lang_stim(output_path)
            percent = int(i/n*100)
            gen_bar.progress(percent, text=f"{percent}%")
            lang_trials_ids.append(sample_ids)
            print(f"{i}: {output_path}")
        else:
            lang_trials_ids.append([])
    gen_bar.progress(100, text=f"Done")
    return lang_trials_ids

def play_stimuli(trial, test_run=False):
    if trial[:4] == "lang":
        output_path = os.path.join(config['stimuli_dir'], f"{trial}.mp3")
        start_time, end_time = administer_lang(output_path, test_run)
        jittered_delay()
    elif trial == "rcmd":
        start_time, end_time = administer_right_cmd(test_run)
        jittered_delay()
    elif trial == "lcmd":
        start_time, end_time = administer_left_cmd(test_run)
        jittered_delay()
    elif trial == "beep":
        start_time, end_time = administer_beep(test_run)
        jittered_delay()
    else:
        start_time, end_time = administer_oddball(test_run)
        jittered_delay()
    
    return start_time, end_time

def play_oddball_stimulus(
    n_tones=20,            # number of distict tones 
    freq_standard=1000,     # Default standard tone (Hz)
    freq_rare=2000,         # Default rare tone (Hz)
    prob_rare=0.2,          # 20% rare tone probability
):
    """oddball stimulus player"""

    tones = []

    # Create 1-second silent gaps (1000ms)
    silent_gap = AudioSegment.silent(duration=1000)  

    for _ in range(n_tones):
        # Generate tone (standard or rare)
        if random.random() < prob_rare:
            tones.append(Sine(freq_rare).to_audio_segment(duration=100))
        else:
            tones.append(Sine(freq_standard).to_audio_segment(duration=100))
        # add gap between tones
        tones.append(silent_gap)
        
    # Append tone and gap to combined audio
    combined_audio = sum(tones)

    # Play the complete sequence
    play(combined_audio)            

def send_trigger(
    freq=2000,     # Frequency in Hz (default 2000 = high pitch)
    duration=100,  # Duration in milliseconds (default 100ms)
    volume=-10     # Volume in dB (default -10 for safety)
):
    """Play a pure tone beep"""

    beep = Sine(freq).to_audio_segment(duration=duration)
    beep = beep.apply_gain(volume)  # Prevent loud surprises!
    play(beep)