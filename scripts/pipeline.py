import argparse
import logging
import yaml
import json
import pandas as pd
from src.preprocessing import preprocess
from src.voice_conversion import voice_convert
from src.automatic_speech_recognition import asr
from src.speaker_verification import process_data_to_embeddings, compute_metrics
from src.utils import CODEBASE_DIR
from src.results import visualize_metrics
import os
import numpy as np
from datasets import load_from_disk
logging.basicConfig(level=logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.ERROR)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def replace_codebase_dir(config, codebase_dir):
    for key, value in config.items():
        if isinstance(value, str) and '{CODEBASE_DIR}' in value:
            config[key] = value.replace('{CODEBASE_DIR}', codebase_dir)
    return config

def handle_speaker_data(speaker_data_path):
    ### Creating speaker_data csv before preprocessing. LOGIC VARIES FOR EACH DATASET.
    speaker_data = pd.read_csv(
        f'{CODEBASE_DIR}/data/raw/LibriTTS/SPEAKERS copy.txt', # improperly formatted speaker data
        delimiter='|', 
        comment=';', 
        skipinitialspace=True
    )
    header = [col.strip() for col in "ID  |SEX| SUBSET           |MINUTES| NAME".split("|")]
    speaker_data.to_csv(speaker_data_path, header=header, index=False, sep=",")
    print("Speaker data formatted and saved.")

def add_anon_columns(anon_dataset, target_dataset):
    audio_waveform_data = anon_dataset.map(lambda example: {'converted_audio_waveform': example['audio']['array']})
    anon_transcription = anon_dataset.map(lambda example: {'asr_transcription_anon': example['asr_transcription']})
    assert len(audio_waveform_data) == len(target_dataset), "Source and target datasets must be of the same length."
    target_dataset = target_dataset.add_column('converted_audio_waveform', audio_waveform_data['converted_audio_waveform'])
    target_dataset = target_dataset.add_column('asr_transcription_anon', anon_transcription['asr_transcription_anon'])
    return target_dataset

def main(config):
    if not os.path.exists(config['speaker_data_path']):
        handle_speaker_data(config['speaker_data_path'])
    
    # 1. Preprocess data
    if os.path.exists(f"{config['processed_data_path']}-dataset"):
        dataset = load_from_disk(f"{config['processed_data_path']}-dataset")
    else:
        dataset = preprocess(
            config['raw_data_path'], config['processed_data_path'], config['transcript_path_pattern'], 
            config['speaker_id_pattern'], config['file_extensions'], config['target_sample_rate'], 
            config['num_samples'], config['speaker_data_path'], config['speaker_id_column'], config['gender_column']
        )
        dataset.save_to_disk(f"{config['processed_data_path']}-dataset")

    # 2. Voice conversion
    if os.path.exists(f"{config['processed_data_path']}-dataset-converted"):
        converted_dataset = load_from_disk(f"{config['processed_data_path']}-dataset-converted")
    else:
        converted_dataset = voice_convert(dataset, config['target_speaker'])
        converted_dataset.save_to_disk(f"{config['processed_data_path']}-dataset-converted")
    
    # 3. ASR
    if os.path.exists(f"{config['processed_data_path']}-dataset-asr-orig"):
        orig_dataset_after_asr = load_from_disk(f"{config['processed_data_path']}-dataset-asr-orig")
    else:
        orig_dataset_after_asr = asr(config['asr_model_id'], dataset, config['split'])
        orig_dataset_after_asr.save_to_disk(f"{config['processed_data_path']}-dataset-asr-orig")
    if os.path.exists(f"{config['processed_data_path']}-dataset-asr-anon"):
        converted_dataset_after_asr = load_from_disk(f"{config['processed_data_path']}-dataset-asr-anon")
    else:
        converted_dataset_after_asr = asr(config['asr_model_id'], converted_dataset, config['split'])
        converted_dataset_after_asr.save_to_disk(f"{config['processed_data_path']}-dataset-asr-anon")

    # 4. Combine original & anon datasets
    if os.path.exists(f"{config['processed_data_path']}-dataset-combined"):
        combined_dataset = load_from_disk(f"{config['processed_data_path']}-dataset-combined")
    else:
        combined_dataset = add_anon_columns(converted_dataset_after_asr, orig_dataset_after_asr)
        combined_dataset.save_to_disk(f"{config['processed_data_path']}-dataset-combined")
    
    # 5. Speaker verification
    if os.path.exists(f"{config['processed_data_path']}-dataset-combined-embeddings"):
        embeddings = load_from_disk(f"{config['processed_data_path']}-dataset-combined-embeddings")
    else:
        embeddings = process_data_to_embeddings(combined_dataset)
        embeddings.save_to_disk(f"{config['processed_data_path']}-dataset-combined-embeddings")
        
    metrics = compute_metrics(embeddings)
    
    with open(f"{CODEBASE_DIR}/results/{config['dataset_name']}-metrics.json", "w") as f:
        json.dump({k: (v.item() if isinstance(v, np.ndarray) else v) for k, v in metrics.items() if k != 'similarities'}, f, indent=4)

    visualize_metrics(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Voice Anonymization Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")

    args = parser.parse_args()
    config = load_config(args.config)
    config = replace_codebase_dir(config, CODEBASE_DIR)
    main(config)
# python scripts/pipeline.py --config config.yaml