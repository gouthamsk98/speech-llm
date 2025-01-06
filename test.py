import torch
import torchaudio
from whisperspeech.vq_stoks import RQBottleneckTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import hf_hub_download
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists("whisper-vq-stoks-medium-en+pl-fixed.model"):
    hf_hub_download(
        repo_id="homebrewltd/Ichigo-whisper-v0.1",
        filename="merge-medium-vi-2d-2560c-dim64.pth",
        local_dir=".",
    )
vq_model = RQBottleneckTransformer.load_model(
        "merge-medium-vi-2d-2560c-dim64.pth"
    ).to(device)
vq_model.ensure_whisper(device)
def audio_to_sound_tokens(audio_path, target_bandwidth=1.5, device=device):

    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vq_model.encode_audio(wav.to(device))
        codes = codes[0].cpu().tolist()

    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
