from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
import soundfile as sf
from tqdm import tqdm
from src.freevc import VoiceAnonymizer
import numpy as np
anonymizer = VoiceAnonymizer()

def anonymize_audio(item, target_item) -> dict:
    """
    Processes audio data from the dataset using source and target waveforms to generate an anonymized waveform that mimics the acoustic properties of the target.

    Args:
        item (dict): A dictionary for the source audio containing keys 'audio' with subkeys 'array' and 'sample_rate'.
        target_item (dict): A dictionary for the target audio, structured like the source with keys 'audio'.

    Returns:
        dict: The source item enhanced with keys 'converted_audio_waveform' for the anonymized audio.
    """

    waveform = anonymizer.anonymize(
        item['audio']['array'], 
        target_item['audio']['array'],
    )
    item['audio']['array'] = waveform
    # item['converted_audio_waveform'] = waveform

    return item

def voice_convert(audio_dataset: Dataset, target_speaker: int) -> Dataset:
    """
    Voice converts given audio dataset using anonymize_audio.

    Args:
        audio_dataset (Dataset): The dataset containing audio files and their corresponding metadata.
        target_speaker (int): The target speaker id to use as baseline for conversion. Must be present in the dataset.

    Returns:
        Dataset: A new Dataset object containing the original metadata but with audio arrays replaced by anonymized versions.
    """
    print(f"Converting audio dataset using target speaker {target_speaker}")
    target_item = [item for item in audio_dataset if item['speaker_id'] == target_speaker][0] 
    updated_dataset = audio_dataset.map(anonymize_audio, fn_kwargs={'target_item': target_item})
    print("Audio dataset converted.")
    return updated_dataset

def main():

    ###############################################################################
    # TODO: Set the params here before running the script
    dataset_name = f"azain/LibriTTS-dev-clean-16khz-mono-loudnorm-100-random-samples-2024-04-18-17-34-39"
    ###############################################################################

    audio_dataset = load_dataset(dataset_name)
    converted_dataset = voice_convert(audio_dataset)
    converted_dataset.push_to_hub(dataset_name, commit_message="after voice conversion")

if __name__ == "__main__":
    main()
