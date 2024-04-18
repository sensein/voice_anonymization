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
from utils import yield_filtered_files

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
        waveform_np = waveform.transpose(0, 1).numpy()  # Convert to [samples, channels]
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
                         speaker_id_pattern: str = r"(\d+)_", file_extensions: list[str] = ['.wav']) -> Dataset:
    """
    Loads audio files from a specified directory into a Hugging Face `datasets` dataset, including transcripts. 
    The function dynamically finds transcripts and extracts speaker IDs based on a provided pattern, accommodating various audio file extensions.

    Args:
        dataset_path (str): Path to the directory containing audio files.
        transcript_path_pattern (str, optional): Format string to locate transcript files, where "{base_name}" 
    will be replaced by the audio file's base name without extension.
        speaker_id_pattern (str, optional): Regex pattern to extract the speaker ID from the file name.
        file_extensions (list[str], optional): List of audio file extensions to include in the dataset.

    Returns:
        datasets.Dataset: A dataset object containing audio files and their transcripts and speaker ids.

    Example:
    --------
    To load a dataset where audio files named 'audio123.wav' have transcripts named 'audio123_transcript.txt':
    >>> dataset = create_audio_dataset("/path/to/files", "{base_name}_transcript.txt")
    """
    print(f'loading files from {dataset_path}...')
    audio_files = []
    transcripts = []
    speaker_ids = []
    for root, file_name in yield_filtered_files(dataset_path, lambda name: any(name.endswith(ext) for ext in file_extensions)):
        # print(root, file_name)
        audio_path = os.path.join(root, file_name)
        base_name = os.path.splitext(file_name)[0]  # Removes extension from file name

        # Construct transcript path using the pattern, replacing {base_name} placeholder
        transcript_path = os.path.join(root, transcript_path_pattern.format(base_name=base_name)) # todo could try regex+glob for transcript (may be inefficient)
        if os.path.exists(transcript_path):
            transcript = load_transcript(transcript_path)
        else:
            print(f"Transcript not found for {audio_path}. Skipping audio file.")
            continue
        
        speaker_id_match = re.match(speaker_id_pattern, base_name)
        if speaker_id_match:
            speaker_id = speaker_id_match.group(1)
        else:
            print(f"Speaker ID not found in file name {file_name}. Skipping audio file.")
            continue

        audio_files.append(audio_path)
        transcripts.append(transcript)
        speaker_ids.append(speaker_id)

    audio_dataset = Dataset.from_dict({"audio": audio_files, "transcript": transcripts, "speaker_id": speaker_ids}).cast_column("audio", Audio())
    print('done creating audio dataset')
    return audio_dataset

def upload_to_huggingface(audio_dataset: Dataset, dataset_name: str, split: str | None = None) -> None:
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

    # todo add unique string specifying version of dataset (i.e. commit id) "revision=date.date()" or "=after x changes"
    audio_dataset.push_to_hub(
        repo_id=dataset_name, 
        token=token, 
        split=split,
        commit_message=f"Add {split} split at date {date}",
        private=False,
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


def main():
    ###############################################################################
    # TODO: Set the params here before running the script
    codebase_dir = '/om2/user/azain/code/voice_anonymization'
    raw_data_path = f'{codebase_dir}/data/raw/LibriTTS/dev-clean'
    transcript_path_pattern = "{base_name}.original.txt" # wont work if no relation with audio file name
    speaker_id_pattern = r"(\d+)_" # get all digits before first underscore in filename
    target_sample_rate = 16000
    num_samples = 100
    dataset_name = f"azain/LibriTTS-dev-clean-16khz-mono-loudnorm-{num_samples}-random-samples"
    split = 'dev'
    processed_data_path = f"{codebase_dir}/tmp/LibriTTS-processed"
    file_extensions = ['.wav']
    ###############################################################################
    
    os.makedirs(processed_data_path, exist_ok=True)

    # Process Logic
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
    audio_dataset = create_audio_dataset(processed_data_path, transcript_path_pattern, speaker_id_pattern, file_extensions)
    audio_dataset = audio_dataset.shuffle(seed=42).select(range(num_samples))
    upload_to_huggingface(audio_dataset, dataset_name, split)

if __name__ == "__main__":
    main()