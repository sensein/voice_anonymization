####
# Paths
base_path = "../../data/v1/kennedy_james"  # Original data folder path
new_base_path = "../../data/v2/kennedy_james"  # New data structure path
model_id = "openai/whisper-base"
####

import os
import csv
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import soundfile as sf
import time

# Start the timer
start_time = time.time()

# Set the device and data type for processing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load processor for tokenization and feature extraction
processor = AutoProcessor.from_pretrained(model_id)

# Set up the ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Create the new base path directory if it doesn't exist
os.makedirs(new_base_path, exist_ok=True)

# Define paths for metadata files
participants_metadata_path = os.path.join(new_base_path, "participants_metadata.csv")
audio_metadata_path = os.path.join(new_base_path, "audio_metadata.csv")

# Initialize lists to store metadata
participants_metadata = []
audio_metadata = []

def restructure_and_transcribe():
    # Loop through the session types
    for session in ["item_repetition", "free_speech_sentences"]:
        session_path = os.path.join(base_path, session)
        for participant_folder in os.listdir(session_path):
            if participant_folder.startswith("."):  # Skip hidden files
                continue
            
            # Extract participant details
            participant_id, sex_status, native_status = participant_folder.split("_")
            new_participant_path = os.path.join(new_base_path, participant_id)

            # Loop through microphone folders (item_repetition) or directly handle files (free_speech)
            for mic_folder in os.listdir(os.path.join(session_path, participant_folder)):
                if mic_folder.startswith("."):  # Skip hidden files
                    continue
                mic_path = os.path.join(session_path, participant_folder, mic_folder)

                if session == "free_speech_sentences":
                    new_session_path = os.path.join(new_participant_path, session)
                    os.makedirs(new_session_path, exist_ok=True)
                    new_mic_path = os.path.join(new_session_path, mic_folder)
                    os.makedirs(new_mic_path, exist_ok=True)

                    # Process each .wav file
                    for item in os.listdir(mic_path):
                        if item.endswith(".wav"):
                            process_audio_file(
                                mic_path, item, new_mic_path, participant_id, session
                            )

                elif session == "item_repetition":
                    for item in os.listdir(mic_path):
                        if item.startswith("."):  # Skip hidden files
                            continue
                        new_session_path = os.path.join(new_participant_path, f"{item}_repetition")
                        os.makedirs(new_session_path, exist_ok=True)
                        new_mic_path = os.path.join(new_session_path, mic_folder)
                        os.makedirs(new_mic_path, exist_ok=True)

                        # Process each .wav file
                        for file in os.listdir(os.path.join(mic_path, item)):
                            if file.endswith(".wav"):
                                process_audio_file(
                                    os.path.join(mic_path, item), file, new_mic_path, participant_id, f"{item}_repetition"
                                )

            # Update participants metadata (avoid duplicates)
            if participant_id not in [p['participant_id'] for p in participants_metadata]:
                participants_metadata.append({
                    "participant_id": participant_id,
                    "sex_status": sex_status,
                    "age": None,
                    "english_native_status": native_status,
                })

def process_audio_file(source_path, file_name, dest_path, participant_id, session):
    """
    Helper function to process and transcribe audio files.
    """
    new_file_path = os.path.join(dest_path, file_name)

    # Load and save the audio file as mono
    audio_sample, sample_rate = librosa.load(os.path.join(source_path, file_name), sr=None, mono=True)
    sf.write(new_file_path, audio_sample, sample_rate)

    # Resample to target sample rate if needed
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        audio_sample = librosa.resample(audio_sample, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    # Perform transcription and capture timestamps
    result = pipe(audio_sample, return_timestamps=True, generate_kwargs={"language": "en"})
    transcript = file_name.split(".wav")[0].replace("_", " ").lower().strip()
    
    # Update audio metadata
    relative_path = os.path.relpath(new_file_path, new_base_path)
    audio_metadata.append({
        "participant_id": participant_id,
        "session": session,
        "relative_path": relative_path,
        "transcript": transcript,
        "start": result['chunks'][0]['timestamp'][0] if 'chunks' in result and result['chunks'] is not None and result['chunks'][0] is not None and result['chunks'][0]['timestamp'][0] is not None else 0.0,
        "end": result['chunks'][-1]['timestamp'][1] if 'chunks' in result and result['chunks'] is not None and result['chunks'][-1] is not None and result['chunks'][-1]['timestamp'][1] is not None else librosa.get_duration(y=audio_sample, sr=sample_rate),
    })


# Write metadata to CSV files
def write_metadata():
    # Write participants metadata to CSV
    with open(participants_metadata_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["participant_id", "sex_status", "age", "english_native_status"], delimiter='\t')
        writer.writeheader()
        writer.writerows(participants_metadata)

    # Write audio metadata to CSV
    with open(audio_metadata_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["participant_id", "session", "relative_path", "transcript", "start", "end"], delimiter='\t')
        writer.writeheader()
        writer.writerows(audio_metadata)

# Run the restructuring and transcription process
restructure_and_transcribe()
write_metadata()

print(f"Data restructuring and transcription completed in {time.time() - start_time:.2f} seconds.")
