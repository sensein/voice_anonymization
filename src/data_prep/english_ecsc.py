####
# Paths for original and new data
base_path = "../../data/v1/English_ECSC/children"  # Original data folder path
new_base_path = "../../data/v2/English_ECSC"  # New data structure path
model_id = "openai/whisper-base"
####

import os
import csv
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import soundfile as sf
import time
import re

# Start the timer
start_time = time.time()

# Set the device and data type for processing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define model and processor details
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

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

# Loop through the age folders
def restructure_and_transcribe():
    for age_folder in os.listdir(base_path):
        if age_folder.startswith(".") or age_folder == "metadata.xlsx":  # Skip hidden files and metadata
            continue
        
        age_path = os.path.join(base_path, age_folder)

        # Loop through participant files
        for participant_file in os.listdir(age_path):
            if participant_file.startswith("."):  # Skip hidden files
                continue

            # Extract participant details from filename
            # Example: 95M_4023_YR1.cha -> 95M, 4023, YR1
            filename_parts = participant_file.split("_")
            age_sex = filename_parts[0]  # e.g., 95M
            participant_id = filename_parts[1]  # e.g., 4023

            # Split the age and sex (e.g., 95M -> 95, M)
            age_months = int(age_sex[:-1])  # e.g., 95 (convert to integer)
            sex = age_sex[-1]  # e.g., M

            # Convert months to years as a float
            age = round(age_months / 12, 2)  # Convert months to years and round to 2 decimals

            new_participant_path = os.path.join(new_base_path, participant_id)
            
            # Create the participant's directory if it doesn't exist
            os.makedirs(new_participant_path, exist_ok=True)

            # Process both .wav and .cha files, saving directly under the participant's folder
            if participant_file.endswith(".wav") or participant_file.endswith(".cha"):
                process_audio_file(
                    age_path, participant_file, new_participant_path, participant_id
                )

            # Update participants metadata (avoid duplicates)
            if participant_id not in [p['participant_id'] for p in participants_metadata]:
                participants_metadata.append({
                    "participant_id": participant_id,
                    "sex_status": sex,
                    "age": age,  # Now store age as a float (years)
                })

def extract_transcript(cha_file_path, participant_id):
    transcript = []

    # Regular expression to match lines of the target participant
    participant_regex = re.compile(rf"\*{re.escape(participant_id)}:")

    try:
        with open(cha_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # If the line starts with the participant ID, extract the transcript
                if participant_regex.match(line):
                    # Remove the participant ID prefix and any extra whitespaces
                    transcript_line = re.sub(rf"\*{re.escape(participant_id)}:\s*", "", line).lower().strip()
                    # Remove any extra spaces between words
                    transcript_line = " ".join(transcript_line.split())
                    # Remove any " character
                    transcript_line = transcript_line.replace('"', '')
                    # Remove any . or y character
                    transcript_line = transcript_line.replace('.', '').replace(',', '')
                    # Append the cleaned transcript line
                    transcript.append(transcript_line)
        
        # Join all lines into a single string without extra spaces
        return "".join(transcript)
    
    except FileNotFoundError:
        return f"File not found: {cha_file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_audio_file(source_path, file_name, dest_path, participant_id):
    """
    Helper function to process and transcribe audio files.
    """
    new_file_path = os.path.join(dest_path, file_name)

    # Load and save the audio file as mono
    if file_name.endswith(".wav"):
        audio_sample, sample_rate = librosa.load(os.path.join(source_path, file_name), sr=None, mono=True)
        sf.write(new_file_path, audio_sample, sample_rate)

        # Resample to target sample rate if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            audio_sample = librosa.resample(audio_sample, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        # Perform transcription and capture timestamps
        result = pipe(audio_sample, return_timestamps=True, generate_kwargs={"language": "en"})
        transcript = extract_transcript(os.path.join(source_path, file_name).replace(".wav", ".cha"), "CHI")

        # Update audio metadata
        relative_path = os.path.relpath(new_file_path, new_base_path)
        audio_metadata.append({
            "participant_id": participant_id,
            "relative_path": relative_path,
            "transcript": transcript or None,
            "start": result['chunks'][0]['timestamp'][0] if 'chunks' in result and result['chunks'] is not None and result['chunks'][0] is not None and result['chunks'][0]['timestamp'][0] is not None else 0.0,
            "end": result['chunks'][-1]['timestamp'][1] if 'chunks' in result and result['chunks'] is not None and result['chunks'][-1] is not None and result['chunks'][-1]['timestamp'][1] is not None else librosa.get_duration(y=audio_sample, sr=sample_rate),
        })

# Write metadata to CSV files
def write_metadata():
    # Write participants metadata to CSV
    with open(participants_metadata_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["participant_id", "sex_status", "age"], delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(participants_metadata)

    # Write audio metadata to CSV without quotes around text fields
    with open(audio_metadata_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["participant_id", "relative_path", "transcript", "start", "end"], delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(audio_metadata)

# Run the restructuring and transcription process
restructure_and_transcribe()
write_metadata()

print(f"Data restructuring and transcription completed in {time.time() - start_time:.2f} seconds.")
