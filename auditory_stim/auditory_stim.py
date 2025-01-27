import time
import os
import random
import yaml
import time
import streamlit as st
from pydub import AudioSegment
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
        player.play(mp3_path)
        player.wait_for_playback()
    elif config['os'].lower() == 'linux':
        audio = AudioSegment.from_mp3(mp3_path)
        play(audio)
    else:
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

def play_lang_stim(output_path):
    start_time = time.time()
    play_mp3(output_path)
    end_time = time.time()
    return start_time, end_time

def right_cmd_stim():
    start_time = time.time()
    for _ in range(8):
        play_mp3(config['right_keep_path'])
        time.sleep(10)

        play_mp3(config['right_stop_path'])
        time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def left_cmd_stim():
    start_time = time.time()
    for _ in range(8):
        play_mp3(config['left_keep_path'])
        time.sleep(10)

        play_mp3(config['left_stop_path'])
        time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def administer_beep():
    start_time = time.time()
    time.sleep(10)
    play_mp3(config['beep_path'])
    time.sleep(10)
    end_time = time.time()
    return start_time, end_time

def randomize_trials(language_stim=72, right_cmd_stim=3, left_cmd_stim=3, beep_stim=6):
    trial_types = []
    num_trials = {
        "lang": language_stim,
        "rcmd": right_cmd_stim,
        "lcmd": left_cmd_stim,
        "beep": beep_stim
    }
    for key in num_trials:
        for i in range(num_trials[key]):
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

def play_stimuli(trial):
    if trial[:4] == "lang":
        output_path = os.path.join(config['stimuli_dir'], f"{trial}.mp3")
        start_time, end_time = play_lang_stim(output_path)
        jittered_delay()
    elif trial == "rcmd":
        start_time, end_time = right_cmd_stim()
        jittered_delay()
    elif trial == "lcmd":
        start_time, end_time = left_cmd_stim()
        jittered_delay()
    else:
        start_time, end_time = administer_beep()
        jittered_delay()
    return start_time, end_time
