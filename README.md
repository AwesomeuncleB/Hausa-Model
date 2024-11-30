# Hausa Speech Transcription 

This repository contains a Streamlit application for transcribing Hausa audio files into text using a fine-tuned [Whisper model](https://huggingface.co/therealbee/whisper-small-ha-bible-tts). The model is trained specifically for the Hausa language, ensuring accurate transcription (Not a 100%) for this underrepresented language in NLP tasks.

---

## Features
- **Hausa Speech-to-Text**: Upload audio files in Hausa and get accurate transcriptions.
- **Supports Multiple Audio Formats**: WAV, MP3, OGG, and more.
- **Streamlit Web App**: An easy-to-use interface for seamless transcription.
- **Efficient Processing**: Built with optimized audio handling and resampling.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Install the required dependencies:
  ```bash
  pip install streamlit torch transformers librosa numpy

  git clone https://github.com/AwesomeuncleB/hausa-whisper-transcription.git
cd hausa-whisper-transcription
streamlit run app.py


