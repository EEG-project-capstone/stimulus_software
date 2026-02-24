from gtts import gTTS
from lib.constants import FilePaths

right_cmd_list = [
    "keep opening and closing your right hand",
    "stop opening and closing your right hand"
]

tts = gTTS(text=right_cmd_list[0], lang="en")
tts.save(FilePaths.RIGHT_KEEP_AUDIO)

tts = gTTS(text=right_cmd_list[1], lang="en")
tts.save(FilePaths.RIGHT_STOP_AUDIO)

left_cmd_list = [
    "keep opening and closing your left hand",
    "stop opening and closing your left hand"
]

tts = gTTS(text=left_cmd_list[0], lang="en")
tts.save(FilePaths.LEFT_KEEP_AUDIO)

tts = gTTS(text=left_cmd_list[1], lang="en")
tts.save(FilePaths.LEFT_STOP_AUDIO)
