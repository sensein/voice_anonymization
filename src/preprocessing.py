import torch
import torchaudio
import pyloudnorm as pyln
import librosa
from datasets import Dataset, Audio
import os
from huggingface_hub import HfFolder, HfApi
from tqdm import tqdm
import shutil
from pathlib import Path

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
    # elif waveform.ndim == 2: # todo check 
    #     waveform_np = waveform.transpose(0, 1).numpy()  # Convert to [samples, channels]
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
    
def load_audio_dataset(dataset_path: str, transcript_path_pattern: str = "{base_name}.txt", file_extensions: list[str] = ['.wav']) -> Dataset:
    """
    Loads audio files from a specified directory into a Hugging Face `datasets` dataset, including transcripts. 
    The function dynamically finds transcripts based on a provided pattern, accommodating various audio file extensions.

    Args:
        dataset_path (str): Path to the directory containing audio files.
        transcript_path_pattern (str): Format string to locate transcript files, where "{base_name}" 
    will be replaced by the audio file's base name without extension.
        file_extensions (list[str], optional): List of audio file extensions to include in the dataset.

    Returns:
        datasets.Dataset: A dataset object containing audio files and their transcripts. # todo add speaker ID

    Example:
    --------
    To load a dataset where audio files named 'audio123.wav' have transcripts named 'audio123.txt':
    >>> dataset = load_audio_dataset("/path/to/audio_files", "{base_name}.txt")
    """
    print(f'loading audio dataset at {dataset_path}...')
    audio_files = []
    transcripts = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                audio_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]  # Removes extension from file name
                # Construct transcript path using the pattern, replacing {base_name} placeholder
                transcript_path = os.path.join(root, transcript_path_pattern.format(base_name=base_name))
                
                if os.path.exists(transcript_path):
                    transcript = load_transcript(transcript_path)
                else:
                    print(f"Transcript not found for {audio_path}. Skipping audio file.")
                    continue
                
                audio_files.append(audio_path)
                transcripts.append(transcript)

    audio_dataset = Dataset.from_dict({"audio": audio_files, "transcript": transcripts}).cast_column("audio", Audio())
    print('done loading audio dataset')
    return audio_dataset

def upload_to_huggingface(audio_dataset: Dataset, hf_dataset_name: str) -> None:
    """
    Uploads the processed dataset to Hugging Face Hub. Assumes already logged in to Hugging Face.

    Args:
        audio_dataset (Dataset): A dataset object containing audio files and their transcripts.
        hf_dataset_name (str): The desired Hugging Face dataset repository name.
    """
    print('attempting to upload to huggingface...')
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Make sure you're logged in using `huggingface-cli login`.")
    else:
        print(f"Hugging Face token found. Uploading dataset...")
    # api = HfApi()
    # api.create_repo(repo_id=hf_dataset_name, token=token, repo_type="dataset", exist_ok=True)
    # todo add unique string specifying version of dataset (i.e. commit id) "revision=date.date()" or "=after x changes"
    audio_dataset.push_to_hub(hf_dataset_name) # TODO: This hangs on "Creating parquet from Arrow format" for some reason
    print(f'Dataset uploaded to Hugging Face as {hf_dataset_name}.')


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
    transcript_path_pattern = "{base_name}.original.txt"
    target_sample_rate = 16000
    hf_dataset_name = "azain/LibriTTS"
    processed_data_path = f"{codebase_dir}/tmp/LibriTTS-processed"
    file_extensions = ['.wav']
    ###############################################################################
    
    os.makedirs(processed_data_path, exist_ok=True)

    # Process Logic
    directories = [x[0] for x in os.walk(raw_data_path)]
    for root in tqdm(directories, desc="Directories"): # did this way to show progress
        _, _, audio_files = next(os.walk(root))
        for audio_file in audio_files:
            if any(audio_file.endswith(ext) for ext in file_extensions):
                processed_audio_path = os.path.join(processed_data_path, audio_file)
                if os.path.exists(processed_audio_path): continue

                audio_path = os.path.join(root, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                transcript_name = transcript_path_pattern.format(base_name=base_name)
                transcript_path = os.path.join(root, transcript_name)
                if os.path.exists(transcript_path):
                    dest_transcript_path = os.path.join(processed_data_path, transcript_name)
                    shutil.copy(transcript_path, dest_transcript_path)
                waveform, sr = resample_audio(audio_path, target_sample_rate)
                waveform = stereo_to_mono(waveform)
                waveform, normalized = normalize_loudness(waveform, target_sample_rate)
                if not normalized: continue
                if waveform.ndim == 1: waveform = waveform.unsqueeze(0)

                torchaudio.save(processed_audio_path, waveform, sr)

    # Upload to Hugging Face
    audio_dataset = load_audio_dataset(processed_data_path, transcript_path_pattern) # todo load original dataset too.
    upload_to_huggingface(audio_dataset, hf_dataset_name)

if __name__ == "__main__":
    main()