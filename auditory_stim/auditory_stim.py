import time
import os
import random
import pandas as pd
from gtts import gTTS
import psychtoolbox as ptb
from playsound import playsound
import mpv
import yaml
import time

config_file_path = 'config.yml'  # Replace with the actual path to your config file
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

player = mpv.MPV(input_default_bindings=True, input_vo_keyboard=True, osc=True)

def play_mp3(mp3_path):
    if config['os'].lower() == 'windows':
        player.play(mp3_path)
        player.wait_for_playback()
    else:
        playsound(mp3_path)
        

def language_stim(num_sentence=12):
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
    selected_sentences = random.sample(sentence_list, num_sentence)

    joined_sentences = ' '.join(selected_sentences)

    # Create empty list for recording sentences played
    sentences_played = []
    # Start the audio timing (for quality check) for 1 trial
    overall_start_time = time.time()
    # Initialize Google-Text-to-Speech gTTS
    tts = gTTS(text=joined_sentences, lang="en")
    # Temporarily save as mp3 file
    tts.save(config['lang_stim_path'])

    # Get timestamp when sentence is played
    start_time = time.time()  # UTC time        

    # Load and play an MP3 file
    print("Playing sentence: ", joined_sentences)
    play_mp3(config['lang_stim_path'])

    # Get timestamp when is done playing 1 sentence
    end_time = time.time()  # UTC time
    # Duration of one sentence
    sentence_duration = end_time - start_time

    # Record sentence played, time start, and duration of each sentence
    sentences_played.append({
        "stimulus" : joined_sentences,
        "start-time" : start_time,
        "duration" : sentence_duration,
    })
    # Add a 2-second break after 1 trial done
    time.sleep(2)

    # Delete intermediate mp3 file
    if os.path.exists(config['lang_stim_path']):
        os.remove(config['lang_stim_path'])
    
    overall_end_time = time.time()  # for 1 trial
    overall_duration = overall_end_time - overall_start_time  # for 1 trial
    
    return sentences_played, overall_start_time, overall_duration

def right_cmd_stim():

    right_cmd_list = [
        "keep opening and closing your right hand",
        "stop opening and closing your right hand"
    ]
    overall_start_time = time.time()
    for i in range(8):
        for cmd in right_cmd_list:
            # Initialize Google text to speech
            tts = gTTS(text=cmd, lang="en")
            # Temporarily save as mp3 file
            tts.save(config['right_cmd_path'])
            # Get timestamp when sentence is played
            # Play sentence
            play_mp3(config['right_cmd_path'])
            time.sleep(10)
    
    # Delete intermediate mp3 file
    if os.path.exists(config['right_cmd_path']):
        os.remove(config['right_cmd_path'])

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    return right_cmd_list, overall_start_time, overall_duration

def left_cmd_stim():

    left_cmd_list = [
        "keep opening and closing your left hand",
        "stop opening and closing your left hand"
    ]
    overall_start_time = time.time()
    for i in range(8):
        for cmd in left_cmd_list:
            # Initialize Google text to speech
            tts = gTTS(text=cmd, lang="en")
            # Temporarily save as mp3 file
            tts.save(config['left_cmd_path'])
            # Get timestamp when sentence is played
            # Play sentence
            play_mp3(config['left_cmd_path'])
            time.sleep(10)
    
    # Delete intermediate mp3 file
    if os.path.exists(config['left_cmd_path']):
        os.remove(config['left_cmd_path'])

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    return left_cmd_list, overall_start_time, overall_duration

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
            trial_types.append(key)

    random.shuffle(trial_types)
    return trial_types

def generate_and_play_stimuli(patient_id="patient0"):

    current_date = time.strftime("%Y-%m-%d")

    if os.path.exists(config['patient_note_path']):
        patient_df = pd.read_csv(config['patient_note_path'])
    else:
        patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
                                    'sentences', 'start_time', 'duration', 'order'])
        
    administered_stimuli = []

    if ((patient_df['patient_id'] == patient_id) & (patient_df['date'] == current_date)).any():
        print("Patient has already been administered stimulus protocol today")
        return
    else:
        trial_types = randomize_trials()

        for trial in trial_types:
            if trial == "lang":
                sentences_played, start_time,  duration = language_stim()
                administered_stimuli.append({
                    'patient_id': patient_id,
                    'date': current_date,
                    'trial_type': trial,
                    'sentences': sentences_played,
                    'start_time': start_time,
                    'duration': duration,
                    'order': trial_types
                })
            elif trial == "rcmd":
                sentences, start_time, duration = right_cmd_stim()
                administered_stimuli.append({
                    'patient_id': patient_id,
                    'date': current_date,
                    'trial_type': trial,
                    'sentences': sentences,
                    'start_time': start_time,
                    'duration': duration,
                    'order': trial_types
                })
            elif trial == "lcmd":
                sentences, start_time, duration = left_cmd_stim()
                administered_stimuli.append({
                    'patient_id': patient_id,
                    'date': current_date,
                    'trial_type': trial,
                    'sentences': sentences,
                    'start_time': start_time,
                    'duration': duration,
                    'order': trial_types
                })
            else:
                _, start_time, duration = administer_beep()
                administered_stimuli.append({
                    'patient_id': patient_id,
                    'date': current_date,
                    'trial_type': trial,
                    'sentences': 'BEEP',
                    'start_time': start_time,
                    'duration': duration,
                    'order': trial_types
                })
        pd.DataFrame(administered_stimuli)
        administered_stimuli_df = pd.concat([patient_df, pd.DataFrame(administered_stimuli)], ignore_index=True)
        administered_stimuli_df.to_csv(config['patient_note_path'], index=False)