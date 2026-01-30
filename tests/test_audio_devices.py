import sounddevice as sd
import numpy as np

def list_audio_devices():
    print("Available audio devices:")
    print(sd.query_devices())

def test_audio_output():
    print("Testing default audio output device...")
    duration = 1.0  # seconds
    frequency = 440.0  # Hz (A4 tone)
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    try:
        sd.play(tone, samplerate=sample_rate)
        sd.wait()
        print("Audio output test completed successfully.")
    except Exception as e:
        print(f"Error during audio output test: {e}")

if __name__ == "__main__":
    list_audio_devices()
    test_audio_output()