# SPEECH LLM

## About

This is a derivative work from Ichigo-whisper [Ichigo v0.5 family](https://github.com/janhq/ichigo). It demonstrates how to convert WAV audio to speech tokens, which are then used by the model for inference.

## Get Started

### Installation

1. Create virtual enviroment (venv/conda)

   ```bash
   # venv
   python -m venv speech-llm
   source speech-llm/bin/activate

   # conda
   conda create -n speech-llm python=3.11
   conda activate speech-llm

   ```

2. Clone the repository and install requirement packages

```bash
    git clone https://github.com/gouthamsk98/speech-llm.git
    cd speech-llm
    pip install -r requirements.txt
```

    Ensure that your PyTorch version (2.5.1) is compatible with your CUDA version. For this project, CUDA 12.1 is used.

## Author

[Goutham S Krishna](https://www.linkedin.com/in/goutham-s-krishna-21ab151a0/)

# Acknowledgement

- **[ICHIGO-WHISPER](https://github.com/janhq/WhisperSpeech.git)**

- **[WhisperSpeech](https://github.com/collabora/WhisperSpeech)**

- **[Whisper](https://github.com/openai/whisper)**

- **[Vivoice](https://huggingface.co/datasets/capleaf/viVoice)**

- **[LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_filtered)**
