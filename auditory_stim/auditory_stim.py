import time
import os
import random
import pandas as pd
from gtts import gTTS
import pyttsx3
import psychtoolbox as ptb
from playsound import playsound
import mpv
import yaml
import time
import streamlit as st
from pydub import AudioSegment
import numpy as np
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub import AudioSegment, effects 
from scipy.io.wavfile import read, write

config_file_path = 'config.yml'  # Replace with the actual path to your config file
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

player = mpv.MPV(input_default_bindings=True, input_vo_keyboard=True, osc=True)

def play_mp3(mp3_path, verbose=True):
    if verbose:
        print(f"Playing {mp3_path}...")
    if config['os'].lower() == 'windows':
        player.play(mp3_path)
        player.wait_for_playback()
    elif config['os'].lower() == 'streamlit':
        st.audio(mp3_path, format="audio/mpeg", autoplay=True)
    else:
        playsound(mp3_path)

# Event handler for when the playback position changes
@player.property_observer('time-pos')
def on_time_pos_change(_name, value):
    """Print video start and end times"""
    if value == 0:
        start_time = time.time()
        print(f"Video started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    if value is None:
        end_time = time.time()
        print(f"Video ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

def remove_silence(input_path, output_path, silence_thresh=-40, min_silence_len=500, padding=55):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_path)
    
    # Detect non-silent parts
    nonsilent_parts = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Add padding around non-silent parts and concatenate them
    segments = []
    for start, end in nonsilent_parts:
        start = max(0, start - padding)
        end = min(len(audio), end + padding)
        segments.append(audio[start:end])
    
    # Combine all non-silent segments
    combined = AudioSegment.empty()
    for segment in segments:
        combined += segment
    
    # Export the processed audio
    combined.export(output_path, format="mp3")

def speed_up_audio(input_path, output_path, speed_factor=1.5):
    audio = AudioSegment.from_mp3(input_path)
    audio = audio.speedup(playback_speed=speed_factor)
    audio.export(output_path, format="mp3")

def gen_lang_stim(output_file_path, num_sentence=12):
    sentence_list = ['cold homes need heat', 'black dog bit thieves', 'smart guys fix things', 'red cat ate rats',
    'fast car hit walls', 'sweet boys kiss girls', 'nice dad held sons', 'good minds save lives', 'dry fur rubs skin',
    'great goat climbs hills', 'hot tea burns tongues', 'wise teen read books', 'poor men want work', 'clear words make sense',
    'slow wolf stole eggs', 'brave votes help towns', 'mad wife broke plates', 'fat kid likes food', 'big chimps throw fruits',
    'young fans cheer stars', 'loud sound hurts ears', 'sharp knife cuts ropes', 'hard rocks smash heads', 'fine chefs cook meals',
    'bright queens wear crowns', 'sick tramp drank wine', 'ill trees lost leafs', 'thick smoke kills bees', 'bored mum did stuff',
    'dark lord threw spells', 'rich counts own lands', 'long walks strain legs', 'grey ship fires bombs', 'large planes cross clouds',
    'small boat fought waves', 'old king loves deer', 'kind host gave beers', 'cool bands play songs', 'fun jokes please crowds',
    'pure gas lights lamps', 'skilled smiths craft steel', 'new staff broke rules', 'strict law had flaws', 'weak birds built nests',
    'green bug seeks holes', 'tall slaves pour tea', 'bold cop beats crime', 'brown bears feed cubs', 'cute pug felt pain',
    'tough scene shows blood', 'tired aunt bakes cakes', 'warm rain melt snow', 'flat feet cause aches', 'starved hounds chase trucks',
    'huge spoon brought soup', 'strong wind shuts doors', 'odd clown sang tales', 'best friends end stress', 'short dwarfs forge swords',
    'white shark scares fish', 'thin guards swipe cards', 'blue pen leaks ink', 'high heels squeeze toes', 'blunt axe chops wood',
    'worn toys wound hands', 'quick fox caught hens', 'grand branch blocks streets', 'deep rock hid gold', 'wet dirt soil socks',
    'pink squid sinks rafts', 'vast space lacks air', 'slick crooks steal rings']

    # Randomly select num_sentence sentences for 1 trial
    sample_id = random.sample(range(len(sentence_list)), num_sentence)
    selected_sentences = [sentence_list[i] for i in sample_id]
    joined_sentences = ' '.join(selected_sentences)

    tts = gTTS(text=joined_sentences, lang="en")
    tts.save(output_file_path)   
    remove_silence(output_file_path,output_file_path)    

    return sample_id

def right_cmd_stim():
    for _ in range(8):
        play_mp3(config['right_keep_path'])
        time.sleep(10)

        play_mp3(config['right_stop_path'])
        time.sleep(10)
    return None

def left_cmd_stim():
    for _ in range(8):
        play_mp3(config['left_keep_path'])
        time.sleep(10)

        play_mp3(config['left_stop_path'])
        time.sleep(10)
    return None

def administer_beep():
    # Get timestamp of when beep will play
    start_time = time.time()
    play_mp3(config['beep_path'])
    end_time = time.time()
    duration = end_time - start_time
    # Break until 45secs
    time.sleep(45-duration)

    return "BEEP", start_time, duration

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
    print(f"trial_types: {trial_types}")
    for trial in trial_types:
        if trial[:4] == "lang":
            output_path = os.path.join(config['stimuli_dir'], f"{trial}.mp3")
            print(f"output_path: {output_path}")
            sample_ids = gen_lang_stim(output_path)
    return sample_ids

def play_stimuli(trial_types):
    for trial in trial_types:
        if trial[:4] == "lang":
            output_path = os.path.join(config['stimuli_dir'], f"{trial}.mp3")
            play_mp3(output_path)
        elif trial == "rcmd":
            right_cmd_stim()
        elif trial == "lcmd":
            left_cmd_stim()
        else:
            administer_beep()
    return None

def generate_and_play_stimuli(patient_id="patient0"):
    current_date = time.strftime("%Y-%m-%d")

    if os.path.exists(config['patient_note_path']):
        patient_df = pd.read_csv(config['patient_note_path'])
    else:
        patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
                                    'sentences', 'start_time', 'duration', 'order'])
        
    administered_stimuli = []

    if ((patient_df['patient_id'] == patient_id) & (patient_df['date'] == current_date)).any():
        trial_types = None
        print("Patient has already been administered stimulus protocol today")
        return
    else:
        trial_types = randomize_trials()

    with st.spinner('Generating stimuli...'):
        time.sleep(5)
        # generate_stimuli(trial_types)
    play_stimuli(trial_types)

    return None

# def generate_and_play_stimuli(patient_id="patient0"):

#     current_date = time.strftime("%Y-%m-%d")

#     if os.path.exists(config['patient_note_path']):
#         patient_df = pd.read_csv(config['patient_note_path'])
#     else:
#         patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
#                                     'sentences', 'start_time', 'duration', 'order'])
        
#     administered_stimuli = []

#     if ((patient_df['patient_id'] == patient_id) & (patient_df['date'] == current_date)).any():
#         print("Patient has already been administered stimulus protocol today")
#         return
#     else:
#         trial_types = randomize_trials()

#         for trial in trial_types:
#             if trial == "lang":
#                 sentences_played, start_time,  duration = language_stim()
#                 administered_stimuli.append({
#                     'patient_id': patient_id,
#                     'date': current_date,
#                     'trial_type': trial,
#                     'sentences': sentences_played,
#                     'start_time': start_time,
#                     'duration': duration,
#                     'order': trial_types
#                 })
#             elif trial == "rcmd":
#                 sentences, start_time, duration = right_cmd_stim()
#                 administered_stimuli.append({
#                     'patient_id': patient_id,
#                     'date': current_date,
#                     'trial_type': trial,
#                     'sentences': sentences,
#                     'start_time': start_time,
#                     'duration': duration,
#                     'order': trial_types
#                 })
#             elif trial == "lcmd":
#                 sentences, start_time, duration = left_cmd_stim()
#                 administered_stimuli.append({
#                     'patient_id': patient_id,
#                     'date': current_date,
#                     'trial_type': trial,
#                     'sentences': sentences,
#                     'start_time': start_time,
#                     'duration': duration,
#                     'order': trial_types
#                 })
#             else:
#                 _, start_time, duration = administer_beep()
#                 administered_stimuli.append({
#                     'patient_id': patient_id,
#                     'date': current_date,
#                     'trial_type': trial,
#                     'sentences': 'BEEP',
#                     'start_time': start_time,
#                     'duration': duration,
#                     'order': trial_types
#                 })
#         pd.DataFrame(administered_stimuli)
#         administered_stimuli_df = pd.concat([patient_df, pd.DataFrame(administered_stimuli)], ignore_index=True)
#         administered_stimuli_df.to_csv(config['patient_note_path'], index=False)