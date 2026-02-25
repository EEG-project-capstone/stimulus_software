"""
Utility script to regenerate the motor command audio files using gTTS.

The generated files are already committed to the repository under audio_data/static/.
Only run this if you need to recreate them (e.g. to change the command phrasing).

Usage:
    pip install gtts
    python scripts/generate_command_audio.py
"""

import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from gtts import gTTS
except ImportError:
    print("gtts is required: pip install gtts")
    sys.exit(1)

from lib.constants import FilePaths

COMMAND_AUDIO = {
    FilePaths.RIGHT_KEEP_AUDIO: "keep opening and closing your right hand",
    FilePaths.RIGHT_STOP_AUDIO: "stop opening and closing your right hand",
    FilePaths.LEFT_KEEP_AUDIO:  "keep opening and closing your left hand",
    FilePaths.LEFT_STOP_AUDIO:  "stop opening and closing your left hand",
}

if __name__ == "__main__":
    for path, text in COMMAND_AUDIO.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tts = gTTS(text=text, lang="en")
        tts.save(path)
        print(f"Saved: {path}")
