import torch
import torchaudio
import pyloudnorm as pyln
import librosa
from datasets import Dataset, Audio
import os
from huggingface_hub import HfFolder, HfApi
import tqdm
import shutil
from pathlib import Path
import re
from datetime import datetime
from src.utils import yield_filtered_files, CODEBASE_DIR
import pandas as pd

def resample_audio(audio_path: str, target_sample_rate: int=16000) -> tuple[torch.Tensor, int]:
    """
    Resamples the audio file at the given audio_path to the target_sample_rate.

    Args:
        audio_path (str): The path to the audio file.
        target_sample_rate (int, optional): The desired sample rate for the resampled audio. Defaults to 16000.

    Returns:
        resampled_waveform (torch.Tensor): The resampled audio waveform.
        target_sample_rate (int): The sample rate of the resampled audio.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    resampled_waveform = resampler(waveform)
    return resampled_waveform, target_sample_rate

def stereo_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert a stereo audio waveform to mono.

    Args:
        waveform (torch.Tensor): The input stereo audio waveform.

    Returns:
        torch.Tensor: The converted mono audio waveform.

    References:
        https://github.com/pytorch/audio/issues/363#issuecomment-637131351
    """
    if waveform.shape[0] == 2:  # Check if the audio is stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def normalize_loudness(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, bool]:
    """
    Normalize the loudness of an audio waveform.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        sample_rate (int): The sample rate of the audio waveform.

    Returns:
        torch.Tensor: The loudness-normalized audio waveform.
        bool: True if the waveform was successfully normalized, False if it wasn't.

    References:
        https://github.com/csteinmetz1/pyloudnorm?tab=readme-ov-file#loudness-normalize-and-peak-normalize-audio-files
    """
    # [1, 24220]
    # Ensure waveform is in [samples, channels] format for pyloudnorm
    if waveform.ndim == 2 and waveform.shape[0] == 1:  # Mono audio in [1, samples]
        waveform_np = waveform.squeeze(0).numpy()  # Convert to [samples,]
    elif waveform.ndim == 2: # Stereo audio in [2, samples]
        waveform_np = waveform.transpose(0, 1).numpy()  # Convert to [samples, 2]
    else:
        raise ValueError("Unsupported waveform shape for loudness normalization.")
    
    meter = pyln.Meter(sample_rate)
    try: # [x, 1]
        loudness = meter.integrated_loudness(waveform_np)
        loudness_normalized_audio = pyln.normalize.loudness(waveform_np, loudness, -23.0)  # Target loudness -23 LUFS
        normalized_waveform = torch.tensor(loudness_normalized_audio, dtype=waveform.dtype)
        return normalized_waveform, True
    except ValueError as e:
        print(f"Skipping normalization due to error: {e}")
        return waveform, False

def load_transcript(transcript_path):
    """Load transcript text from a file."""
    with open(transcript_path, 'r', encoding='utf-8') as file:
        return file.read().strip()
    
def create_audio_dataset(dataset_path: str, transcript_path_pattern: str = "{base_name}.txt", 
                         speaker_id_pattern: str = r"(\d+)_", file_extensions: list[str] = ['.wav'],
                         speaker_info_path: str = None, speaker_id_column: str = None, gender_column: str = None) -> Dataset:
    """
    Loads audio files from a specified directory into a Hugging Face `datasets` dataset, including transcripts. 
    The function dynamically finds transcripts and extracts speaker IDs based on a provided pattern, accommodating various audio file extensions.
    Optionally, it maps speaker IDs to genders based on a provided CSV file.

    Args:
        dataset_path (str): Path to the directory containing audio files.
        transcript_path_pattern (str, optional): Format string to locate transcript files, where "{base_name}" 
    will be replaced by the audio file's base name without extension.
        speaker_id_pattern (str, optional): Regex pattern to extract the speaker ID from the file name.
        file_extensions (list[str], optional): List of audio file extensions to include in the dataset.
        speaker_info_path (str, optional): Path to the CSV file containing speaker information.
        speaker_id_column (str, optional): Name of the column containing speaker IDs in the CSV file. Must be provided if speaker_info_path is provided.
        gender_column (str, optional): Name of the column containing gender information in the CSV file. Must be provided if speaker_info_path is provided.

    Returns:
        datasets.Dataset: A dataset object containing audio files and their transcripts, speaker ids, and optionally genders.

    Example:
    --------
    To load a dataset where audio files named 'audio123.wav' have transcripts named 'audio123_transcript.txt':
    >>> dataset = create_audio_dataset("/path/to/files", "{base_name}_transcript.txt")
    """
    speaker_gender_map = {}
    if speaker_info_path:
        if not speaker_id_column or not gender_column:
            raise ValueError("Both speaker_id_column and gender_column must be provided if speaker_info_path is provided.")
        speaker_data = pd.read_csv(speaker_info_path)
        speaker_data[speaker_id_column] = speaker_data[speaker_id_column].astype(str).str.strip()
        speaker_data[gender_column] = speaker_data[gender_column].str.strip()
        speaker_gender_map = dict(zip(speaker_data[speaker_id_column].astype(str), speaker_data[gender_column]))

    audio_files = []
    transcripts = []
    speaker_ids = []
    genders = []
    
    for root, file_name in yield_filtered_files(dataset_path, lambda name: any(name.endswith(ext) for ext in file_extensions)):        
        base_name = os.path.splitext(file_name)[0]  # remove extension from file name
        transcript_path = os.path.join(root, transcript_path_pattern.format(base_name=base_name))
        speaker_id_match = re.match(speaker_id_pattern, base_name)
        
        if not os.path.exists(transcript_path):
            print(f"Transcript not found for file name {file_name}. Skipping audio file.")
            continue

        if speaker_id_match is None:
            print(f"Speaker ID not found in file name {file_name}. Skipping audio file.")
            continue

        audio_path = os.path.join(root, file_name)
        transcript = load_transcript(transcript_path)
        speaker_id = speaker_id_match.group(1)
        gender = speaker_gender_map.get(speaker_id, 'Unknown') if speaker_info_path else 'Not Provided'

        audio_files.append(audio_path)
        transcripts.append(transcript)
        speaker_ids.append(speaker_id)
        genders.append(gender)

    audio_dataset = Dataset.from_dict({"audio": audio_files, "transcript": transcripts, "speaker_id": speaker_ids, "gender": genders}).cast_column("audio", Audio())
    return audio_dataset

def upload_to_huggingface(audio_dataset: Dataset, dataset_name: str, split: str | None = None, is_private: bool = True) -> None:
    """
    Uploads the processed dataset to Hugging Face Hub. Assumes already logged in to Hugging Face.

    Args:
        audio_dataset (Dataset): A dataset object containing audio files and their transcripts.
        dataset_name (str): The desired Hugging Face dataset repository name.
        split (str, optional): The dataset split (e.g. "train", "dev", "test").
    """
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f'Attempting to upload to Hugging Face: {dataset_name}, version: {date}...')      
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Make sure you're logged in using `huggingface-cli login`.")
    else:
        print(f"Hugging Face token found. Uploading dataset...")

    audio_dataset.push_to_hub(
        repo_id=dataset_name, 
        token=token, 
        split=split,
        commit_message=f"Add {split} split at date {date}",
        private=is_private,
        ) # TODO: This hangs on "Creating parquet from Arrow format" for some reason
    print(f'Dataset uploaded to Hugging Face as {dataset_name}.')


# def copy_transcripts_flat(raw_dir, processed_dir, transcript_suffix=".original.txt"):
#     """
#     Copy transcript files from the raw data directory to the top-level of the processed data directory,
#     ignoring the original subdirectory structure.

#     Args:
#     - raw_dir (str): Path to the directory containing the raw dataset and transcripts.
#     - processed_dir (str): Path to the directory where processed files are stored.
#     - transcript_suffix (str): Suffix of transcript files to identify and copy them.
#     """
#     raw_path = Path(raw_dir)
#     processed_path = Path(processed_dir)
#     for transcript_file in raw_path.rglob(f"*{transcript_suffix}"):
#         dest_file = processed_path / transcript_file.name  # Use only filename for destination
#         shutil.copy(transcript_file, dest_file)

def preprocess(raw_data_path: str, processed_data_path: str, transcript_path_pattern: str, 
               speaker_id_pattern: str, file_extensions: list[str] = ['wav'], target_sample_rate: int = 16000, 
               num_samples: int = None, speaker_info_path: str = None, speaker_id_column: str = None, 
               gender_column: str = None) -> Dataset:
    """
    Preprocesses raw audio data into a standardized format, including audio files, transcripts, speaker IDs, and optionally genders.
    The function dynamically finds transcripts, extracts speaker IDs based on a provided pattern, resamples audio files, and truncates them to a fixed number of samples.

    Args:
        raw_data_path (str): Path to the directory containing raw audio files.
        processed_data_path (str): Path to save the processed dataset.
        transcript_path_pattern (str): Format string to locate transcript files, where "{base_name}" will be replaced by the audio file's base name without extension.
        speaker_id_pattern (str): Regex pattern to extract the speaker ID from the file name.
        file_extensions (list[str], optional): List of audio file extensions to include in the dataset. Defaults to ['wav'].
        target_sample_rate (int, optional): Target sample rate for resampling the audio files. Defaults to 16000.
        num_samples (int, optional): Number of samples to truncate or pad the audio files to. Defaults to all samples. If truncated, samples are selected randomly.
        speaker_info_path (str, optional): Path to the CSV file containing speaker information. Defaults to None.
        speaker_id_column (str, optional): Name of the column containing speaker IDs in the CSV file. Must be provided if `speaker_info_path` is provided.
        gender_column (str, optional): Name of the column containing gender information in the CSV file. Must be provided if `speaker_info_path` is provided.

    Returns:
        datasets.Dataset: A dataset object containing processed audio files, their transcripts, speaker IDs, and optionally genders.

    Raises:
        ValueError: If `speaker_info_path` is provided but `speaker_id_column` or `gender_column` are not provided.

    Example:
    --------
    To preprocess a dataset where audio files named 'audio123.wav' have transcripts named 'audio123_transcript.txt', 
    resample to 16kHz, and truncate to 100 samples:
    >>> dataset = preprocess(
            raw_data_path="/path/to/raw/files",
            processed_data_path="/path/to/save/processed/files",
            transcript_path_pattern="{base_name}_transcript.txt",
            speaker_id_pattern=r"(\d+)_",
            file_extensions=['.wav'],
            target_sample_rate=16000,
            num_samples=100,
            speaker_info_path="speakers.csv",
            speaker_id_column="speaker_id",
            gender_column="gender"
        )
    """
    print(f'processing files for {raw_data_path}...')
    os.makedirs(processed_data_path, exist_ok=True)

    for root, file_name in yield_filtered_files(raw_data_path, lambda file_name: any(file_name.endswith(ext) for ext in file_extensions)):
        # skip already processed file
        processed_audio_path = os.path.join(processed_data_path, file_name)
        if os.path.exists(processed_audio_path): continue 

        # copy transcript file to processed path if it exists
        audio_path = os.path.join(root, file_name)
        base_name, ext = os.path.splitext(file_name)
        transcript_name = transcript_path_pattern.format(base_name=base_name)
        transcript_path = os.path.join(root, transcript_name)
        if os.path.exists(transcript_path):
            dest_transcript_path = os.path.join(processed_data_path, transcript_name)
            shutil.copy(transcript_path, dest_transcript_path)

        # process audio file
        waveform, sr = resample_audio(audio_path, target_sample_rate)
        waveform = stereo_to_mono(waveform)
        waveform, normalized = normalize_loudness(waveform, target_sample_rate)
        if not normalized: 
            print('didnt normalize loudness for', file_name)
            continue
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        assert len(waveform) != 0, f"Empty waveform for {file_name}"
        torchaudio.save(processed_audio_path, waveform, sr)

    # Upload to Hugging Face
    audio_dataset = create_audio_dataset(processed_data_path, transcript_path_pattern, speaker_id_pattern, 
                                         file_extensions, speaker_info_path, speaker_id_column, gender_column)
    if num_samples: audio_dataset = audio_dataset.shuffle(seed=42).select(range(num_samples))
    print('processed files successfully')
    return audio_dataset

def main():
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ###############################################################################
    # TODO: Set the params here before running the script
    raw_data_path = f'{CODEBASE_DIR}/data/raw/LibriTTS/dev-clean'
    processed_data_path = f'{CODEBASE_DIR}/data/processed/LibriTTS-dev-clean-16khz-mono-loudnorm'
    transcript_path_pattern = "{base_name}.original.txt" # wont work if no relation with audio file name
    speaker_id_pattern = r"(\d+)_" # get all digits before first underscore in filename
    target_sample_rate = 16000
    num_samples = 100
    dataset_name = f"LibriTTS-dev-clean-16khz-mono-loudnorm-{num_samples}-random-samples-{date}"
    split = 'dev'
    file_extensions = ['.wav']
    
    ###############################################################################
    audio_dataset = preprocess(
        raw_data_path, processed_data_path, transcript_path_pattern, speaker_id_pattern,
        target_sample_rate, num_samples, dataset_name, split, file_extensions
        )
    upload_to_huggingface(audio_dataset, dataset_name, split)

if __name__ == "__main__":
    main()