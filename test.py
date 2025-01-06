import torch
import torchaudio
import whisper
from utils import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
ichigo_name = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth"
model_size = "merge-medium-vi-2d-2560c-dim64"
whisper_model_name = "medium"
language = "demo"

whisper_model = whisper.load_model(whisper_model_name)
whisper_model.to(device)

ichigo_model = load_model(ref=ichigo_name, size=model_size)
ichigo_model.ensure_whisper(device, language)
ichigo_model.to(device)

def audio_to_sound_tokens_whisperspeech(audio_path):

    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = ichigo_model.encode_audio(wav.to('cuda'))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
def audio_to_sound_tokens_whisperspeech_transcribe(audio_path):
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = ichigo_model.encode_audio(wav.to('cuda'))
        codes = codes[0].cpu().tolist()
