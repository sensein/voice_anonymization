from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.metric_stats import EER
from tqdm import tqdm
from torchmetrics.text import WordErrorRate
import huggingface_hub
from datasets import Dataset, Audio, load_from_disk
import json

def add_embeddings(item: dict) -> dict:
    """
    Adds embeddings to each dataset item using a pre-trained model.

    Args:
        item (dict): Contains the 'audio' key with subkey 'array'.

    Returns:
        dict: The item with new 'embeddings' and 'anonymized_embeddings' keys holding the computed embeddings as numpy arrays.
    """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./models/spkrec-ecapa-voxceleb")
    waveform = torch.tensor(item['audio']['array']).squeeze()
    embeddings = classifier.encode_batch(waveform.unsqueeze(0)).squeeze()
    item['embeddings'] = embeddings.numpy()

    # Add anonymized embeddings if 'converted_audio_waveform' exists (i.e. ran voice conversion first)
    if 'converted_audio_waveform' in item:
        anon_waveform = torch.tensor(item['converted_audio_waveform']).squeeze()
        anon_embeddings = classifier.encode_batch(anon_waveform.unsqueeze(0)).squeeze()
        item['anonymized_embeddings'] = anon_embeddings.numpy()

    return item

def process_data_to_embeddings(audio_dataset: Dataset) -> Dataset:
    """
    Enhances an audio dataset by computing and adding embeddings for each entry.

    Args:
        audio_dataset (datasets.Dataset): Dataset containing audio data.

    Returns:
        datasets.Dataset: The dataset with an added 'embeddings' column.
    """
    updated_dataset = audio_dataset.map(add_embeddings, batched=False)
    return updated_dataset

def compute_similarity_score(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float: # todo could try ECAPA too
    """
    Computes the similarity score between two embeddings using cosine similarity.

    References:
        https://github.com/sensein/fab/blob/main/tutorials/voice_anonymization/voice_anonymization.ipynb
    """
    cos = torch.nn.CosineSimilarity(dim=-1)
    return cos(embedding1, embedding2).item()

def compute_metrics(dataset: Dataset) -> dict:
    """
    Computes pairwise embeddings similarity and evaluates EER and WER for the dataset.

    Args:
        dataset (Dataset): Dataset including columns ['speaker_id', 'embeddings', 
        'anonymized_embeddings', 'transcript', 'asr_transcription', 'asr_transcription_anon'].

    Returns:
        dict: A dictionary containing:
            'similarities': Dataset with columns ['same_speaker', 'orig_orig_similarity_score', 'orig_anon_similarity_score', 'anon_anon_similarity_score']
            'orig_wer_score': WER (Word Error Rate) score computed from original ASR transcriptions
            'anon_wer_score': WER (Word Error Rate) score computed from anonymized ASR transcriptions
            'eer_scores': Dictionary with EER (Equal Error Rate) scores and thresholds for:
                'orig_orig': original vs original embeddings
                'orig_anon': original vs anonymized embeddings
                'anon_anon': anonymized vs anonymized embeddings
    """
    print(f'computing metrics and similarity for {dataset}')
    
    # Check if dataset has all the required columns
    required_columns = ['speaker_id', 'embeddings', 'transcript', 'asr_transcription', 'anonymized_embeddings']
    missing_columns = set(required_columns) - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Dataset is missing the following columns: {', '.join(missing_columns)}")
    
    [
        same_speakers,
        orig_orig_similarity_scores, orig_anon_similarity_scores, anon_anon_similarity_scores, 
        orig_orig_positive_scores, orig_orig_negative_scores,
        orig_anon_positive_scores, orig_anon_negative_scores, 
        anon_anon_positive_scores, anon_anon_negative_scores, 
        orig_transcripts, anon_transcripts, predictions

    ] = [[] for _ in range(13)] 


    for i in tqdm(range(len(dataset))):
        orig_transcripts.append(dataset[i]['transcript'])
        anon_transcripts.append(dataset[i]['asr_transcription_anon']['text']) # Anonymized audio that was transcribed
        predictions.append(dataset[i]['asr_transcription']['text'])
        for j in range(i+1, len(dataset)):
            speaker_id1, speaker_id2 = dataset[i]['speaker_id'], dataset[j]['speaker_id']
            original_embedding1, original_embedding2 = torch.tensor(dataset[i]['embeddings']), torch.tensor(dataset[j]['embeddings'])
            anonymized_embedding1, anonymized_embedding2 = torch.tensor(dataset[i]['anonymized_embeddings']), torch.tensor(dataset[j]['anonymized_embeddings'])

            same_speakers.append(speaker_id1 == speaker_id2)
            orig_orig_similarity_score = compute_similarity_score(original_embedding1, original_embedding2)
            orig_anon_similarity_score = compute_similarity_score(original_embedding1, anonymized_embedding2)
            anon_anon_similarity_score = compute_similarity_score(anonymized_embedding1, anonymized_embedding2)
            orig_orig_similarity_scores.append(orig_orig_similarity_score)
            orig_anon_similarity_scores.append(orig_anon_similarity_score)
            anon_anon_similarity_scores.append(anon_anon_similarity_score)
            (orig_orig_positive_scores if speaker_id1 == speaker_id2 else orig_orig_negative_scores).append(orig_orig_similarity_score)
            (orig_anon_positive_scores if speaker_id1 == speaker_id2 else orig_anon_negative_scores).append(orig_anon_similarity_score)
            (anon_anon_positive_scores if speaker_id1 == speaker_id2 else anon_anon_negative_scores).append(anon_anon_similarity_score)

    orig_orig_eer_score, orig_orig_threshold = EER(torch.tensor(orig_orig_positive_scores), torch.tensor(orig_orig_negative_scores))
    orig_anon_eer_score, orig_anon_threshold = EER(torch.tensor(orig_anon_positive_scores), torch.tensor(orig_anon_negative_scores))
    anon_anon_eer_score, anon_anon_threshold = EER(torch.tensor(anon_anon_positive_scores), torch.tensor(anon_anon_negative_scores))

    wer = WordErrorRate()
    orig_wer_score = wer(predictions, orig_transcripts).item()
    anon_wer_score = wer(predictions, anon_transcripts).item()

    similarities = Dataset.from_dict({
        'same_speaker': same_speakers,
        'orig_orig_similarity_score': orig_orig_similarity_scores,
        'orig_anon_similarity_score': orig_anon_similarity_scores,
        'anon_anon_similarity_score': anon_anon_similarity_scores
    })

    metrics = {
        'similarities': similarities,
        'orig_wer_score': orig_wer_score,
        'anon_wer_score': anon_wer_score,
        'eer_scores': {
            'orig_orig': (orig_orig_eer_score, orig_orig_threshold),
            'orig_anon': (orig_anon_eer_score, orig_anon_threshold),
            'anon_anon': (anon_anon_eer_score, anon_anon_threshold)
        }
    }
    
    print('done computing metrics and pairwise similarity')
    return metrics

def speaker_verification(path, from_disk=False, split=None) -> dict:
    """
    Perform speaker verification on the given dataset as described in the pipeline.

    Args:
        path (str): Path to the dataset.
        from_disk (bool): Whether to load dataset from local disk. Loads from Hugging Face if False.
        split (str): Split of the dataset to use. Must be given if from_disk is false.

    Returns:
        dict: A dictionary containing computed metrics and the Dataset of cosine similarities.
    """
    if from_disk:
        audio_dataset = load_from_disk(path)
    else:
        assert split is not None, "Split must be given if loading from Hugging Face."
        audio_dataset = load_dataset(path, split=split, cache_dir=f"{CODEBASE_DIR}/tmp")
    
    embeddings_dataset = process_data_to_embeddings(audio_dataset)
    output = compute_metrics(embeddings_dataset)
    return output

def main():    

    ###############################################################################
    # TODO: Set the params here before running the script
    dataset_path = f"{CODEBASE_DIR}/data/processed/LibriTTS-dev-100-samples-anon"
    from_disk = True
    split='dev'
    ###############################################################################
    metrics = speaker_verification(dataset_path, from_disk, split)
    metrics['similarities'].save_to_disk(f"{CODEBASE_DIR}/data/processed/LibriTTS-dev-100-samples-similarities")
    with open(f"{CODEBASE_DIR}/data/processed/LibriTTS-dev-100-samples-metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != 'similarities'}, f, indent=4)
    print(metrics)

if __name__ == "__main__":
    main()