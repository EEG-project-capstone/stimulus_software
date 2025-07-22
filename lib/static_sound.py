from gtts import gTTS
import yaml

# Read config file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

right_cmd_list = [
    "keep opening and closing your right hand",
    "stop opening and closing your right hand"
]

# Initialize Google text to speech
tts = gTTS(text=right_cmd_list[0], lang="en")
# Temporarily save as mp3 file
tts.save(config['right_keep_path'])

# Initialize Google text to speech
tts = gTTS(text=right_cmd_list[1], lang="en")
# Temporarily save as mp3 file
tts.save(config['right_stop_path'])

left_cmd_list = [
        "keep opening and closing your left hand",
        "stop opening and closing your left hand"
    ]

# Initialize Google text to speech
tts = gTTS(text=left_cmd_list[0], lang="en")
# Temporarily save as mp3 file
tts.save(config['left_keep_path'])

# Initialize Google text to speech
tts = gTTS(text=left_cmd_list[1], lang="en")
# Temporarily save as mp3 file
tts.save(config['left_stop_path'])
