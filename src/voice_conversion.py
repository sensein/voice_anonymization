from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
import soundfile as sf
from tqdm import tqdm
from src.freevc import VoiceAnonymizer

anonymizer = VoiceAnonymizer()

def anonymize_audio(item, target_item):
    """
    Processes audio data from the dataset using source and target waveforms to generate an anonymized waveform that mimics the acoustic properties of the target.

    Args:
        item (dict): A dictionary for the source audio containing keys 'audio' with subkeys 'array' and 'sample_rate'.
        target_item (dict): A dictionary for the target audio, structured like the source with keys 'audio'.

    Returns:
        dict: The source item enhanced with keys 'converted_audio_waveform' for the anonymized audio.
    """
    # Assuming anonymizer is an instance of a class that supports anonymization using a target waveform
    waveform = anonymizer.anonymize(
        item['audio']['array'], 
        target_item['audio']['array'],
    )

    # item['converted_audio_waveform'] = waveform # TODO do i modify rest of pipeline to support this new column
    item['audio']['array'] = waveform

    return item

def voice_convert(audio_dataset: Dataset, target_index: int) -> Dataset:
    """
    Voice converts given audio dataset using anonymize_audio.

    Args:
        audio_dataset (Dataset): The dataset containing audio files and their corresponding metadata.
        target_index (int): The index of the target audio file to use for conversion.

    Returns:
        Dataset: A new Dataset object containing the original metadata but with audio arrays replaced by anonymized versions.
    """
    target_item = audio_dataset[target_index] # todo can get a random one too
    updated_dataset = audio_dataset.map(anonymize_audio, fn_kwargs={'target_item': target_item})
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
