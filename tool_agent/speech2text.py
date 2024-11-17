import datetime
import struct
import wave
import vs
import requests
from pvrecorder import PvRecorder
import os
from tool_agent.agents import OPENAI_API_KEY, PROJECT_ROOT
# for index, device in enumerate(PvRecorder.get_available_devices()):
#     print(f"[{index}] {device}")

AUDIO_PATH = os.path.join(PROJECT_ROOT, "data", "test.wav")

def recording():
    recorder = PvRecorder(device_index=-1, frame_length=512)
    audio = []
    path = AUDIO_PATH
    try:
        print("Recording start")
        # vs.AlrtDialog("Recording start...")
        recorder.start()
        start_time = datetime.datetime.now()

        # while loop for 10 seconds
        while (datetime.datetime.now() - start_time).seconds < 10:
            frame = recorder.read()
            audio.extend(frame)
        
        recorder.stop()
        print("Recording finished")
        # vs.AlrtDialog("Recording finished!")

        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
    except Exception as e:
        print(e)
        vs.AlrtDialog(f"Error while recording: {e}")
    finally:
        recorder.delete()

def openai_speech2text():
    file_path = AUDIO_PATH
    url = 'https://api.openai.com/v1/audio/transcriptions'
    env_file = os.path.join(PROJECT_ROOT, ".env")
    with open(env_file, 'r') as file:
        for line in file:
            if line.startswith('OPENAI_API_KEY'):
                api_key = line.strip().split('=')[1]  # Get the value part after '='
    if not api_key:
        api_key = OPENAI_API_KEY
    
    headers = {
        'Authorization': f'Bearer {api_key}', # somehow the dotenv doesn't work here
    }

    with open(file_path, 'rb') as file:
        files = {
            'file': ('test.wav', file, 'audio/mpeg'),
            'model': (None, 'whisper-1'),
        }
        # POST request
        response = requests.post(url, headers=headers, files=files)

    # Check response status
    if response.ok:
        return response.json()["text"]
    else:
        raise Exception('Error:', response.status_code, response.text)
    
def speech2text_endpoint():
    recording()
    text = openai_speech2text()
    return text

if __name__ == "__main__":
    recording()
    text = openai_speech2text()
    print(text)