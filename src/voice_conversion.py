from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
from TTS.api import TTS # todo swap for freevc directly https://github.com/OlaWod/FreeVC
import soundfile as sf
from tqdm import tqdm
#add issue about push_to_hub, maybe opening ports to internet
#use freevc instead of tts (check fabio code history)
#order pipeline all together
def load_and_convert_audio(audio_dataset: Dataset, target_voice_model: str = "voice_conversion_models/multilingual/vctk/freevc24") -> Dataset:
    """
    Loads audio data and converts it using a specified TTS model. Adds the converted audio waveform to the dataset.

    Args:
        audio_dataset (Dataset): The dataset containing audio files and their corresponding metadata.
        target_voice_model (str): Path to the Coqui TTS model for voice conversion, defaults to 'voice_conversion_models/multilingual/vctk/freevc24'.

    Returns:
        Dataset: A new Dataset object containing the original audio, transcript, speaker_id, and the converted audio waveform.
    """
    tts = TTS(target_voice_model)
    processed_data = {
        'original_audio': [],
        'converted_audio_waveform': [],
        'converted_audio_sample_rate': [],
        'transcript': [],
        'speaker_id': []
    }

    print('Starting voice conversion process')
    for item in tqdm(audio_dataset):
        assert 'audio' in item and 'transcript' in item, "Each item must have 'audio' and 'transcript' keys"
        
        audio_data = item['audio']
        transcript = item['transcript']
        waveform, sample_rate = tts.tts(transcript)

        processed_data['original_audio'].append(audio_data)
        processed_data['converted_audio_waveform'].append(waveform)
        processed_data['converted_audio_sample_rate'].append(sample_rate)
        processed_data['transcript'].append(transcript)
        processed_data['speaker_id'].append(item['speaker_id'])

    print('Voice conversion completed')
    return Dataset.from_dict(processed_data)

def main():

    ###############################################################################
    # TODO: Set the params here before running the script
    # hf_dataset_name = "azain/LibriTTS-processed"
    converted_dataset_path = f"{CODEBASE_DIR}/tmp/LibriTTS-processed-with-embeddings-converted"
    ###############################################################################

    # audio_dataset = load_dataset(hf_dataset_name)
    ### TODO temporarily do this since I cant push datasets to HuggingFace
    processed_data_path = f"{CODEBASE_DIR}/tmp/LibriTTS-processed"
    transcript_path_pattern = "{base_name}.original.txt"
    from preprocessing import create_audio_dataset
    audio_dataset = create_audio_dataset(processed_data_path, transcript_path_pattern)
    audio_dataset = audio_dataset.shuffle(seed=42) # shuffle to get random samples
    audio_dataset = audio_dataset.select(range(100)) # small dataset for testing
    ###

    converted_dataset = load_and_convert_audio(audio_dataset)
    converted_dataset.save_to_disk(converted_dataset_path)  # todo push to hub whenever that works

if __name__ == "__main__":
    main()
