import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np
import os

# Page configuration
st.set_page_config(page_title="Hausa Speech Transcription", page_icon="🎙️")

# Load model and processor
@st.cache_resource
def load_model():
    st.info("Loading the transcription model, please wait...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "therealbee/whisper-small-ha-bible-tts",
        ignore_mismatched_sizes=True
    )
    processor = WhisperProcessor.from_pretrained("therealbee/whisper-small-ha-bible-tts")
    return model, processor

# Transcription function
def transcribe_audio(audio_path, model, processor):
    # Load and resample audio
    audio, sampling_rate = librosa.load(audio_path, sr=None)
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    # Prepare inputs
    inputs = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt", 
        language="ha"
    )

    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(inputs.input_features, task="transcribe")

    # Decode transcription
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

# Streamlit app
def main():
    st.title("Hausa Speech Transcription")
    st.write("Upload a Hausa language audio file for transcription.")

    # Load the model and processor
    model, processor = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'ogg'],
        help="Upload a Hausa language audio file."
    )

    if uploaded_file is not None:
        # Get the file extension
        file_extension = uploaded_file.name.split('.')[-1]
        temp_audio_path = f"temp_audio_file.{file_extension}"

        # Save the uploaded file
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the audio player
        st.audio(temp_audio_path)

        # Transcription button
        if st.button("Transcribe"):
            with st.spinner("Transcribing audio..."):
                try:
                    transcription = transcribe_audio(temp_audio_path, model, processor)
                    st.success("Transcription complete!")
                    st.write(transcription)
                except FileNotFoundError:
                    st.error("Audio file not found. Please try uploading again.")
                except ValueError as ve:
                    st.error(f"Value error: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                finally:
                    # Clean up temporary file
                    os.remove(temp_audio_path)

# Run the app
if __name__ == "__main__":
    main()
