import torch
import torchaudio
import whisper
from transformers import pipeline

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
