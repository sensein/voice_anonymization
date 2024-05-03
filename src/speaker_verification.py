from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.metric_stats import EER
from tqdm import tqdm
from torchmetrics.text import WordErrorRate
import huggingface_hub
from datasets import Dataset, Audio

def add_embeddings(item: dict) -> dict:
    """
    Adds embeddings to each dataset item using a pre-trained model.

    Args:
        item (dict): Contains the 'audio' key with subkey 'array'.

    Returns:
        dict: The item with a new 'embeddings' key holding the computed embeddings as a numpy array.
    """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./models/spkrec-ecapa-voxceleb")
    waveform = torch.tensor(item['audio']['array']).squeeze()
    embeddings = classifier.encode_batch(waveform.unsqueeze(0)).squeeze()  # Process embedding
    item['embeddings'] = embeddings.numpy()
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

def compute_metrics_and_similarity(dataset: Dataset) -> tuple[Dataset, float, float]:
    """
    Computes pairwise embeddings similarity and evaluates EER and WER for the dataset.

    Args:
        dataset (Dataset): Dataset including columns ['speaker_id', 'embeddings', 'transcript', 'asr_transcription'].

    Returns:
        tuple: A tuple containing:
            - Dataset with columns ['speaker_id1', 'speaker_id2', 'embedding1', 'embedding2', 'similarity_score'].
            - EER (Equal Error Rate) score for the dataset.
            - WER (Word Error Rate) score computed from ASR transcriptions.
    """
    print(f'computing metrics and similarity similarity for {dataset}')
    
    num_rows = len(dataset)
    speaker_ids1, speaker_ids2 = [], []
    embeddings1_list, embeddings2_list = [], []
    similarity_scores = []
    positive_scores, negative_scores = [], []  # for EER
    transcripts, predictions = [], [] # for WER

    for i in tqdm(range(num_rows)):
        transcripts.append(dataset[i]['transcript'])
        predictions.append(dataset[i]['asr_transcription']['text'])
        for j in range(i+1, num_rows):
            speaker_id1, speaker_id2 = dataset[i]['speaker_id'], dataset[j]['speaker_id']
            speaker_ids1.append(speaker_id1)
            speaker_ids2.append(speaker_id2)
            embedding1, embedding2 = torch.tensor(dataset[i]['embeddings']), torch.tensor(dataset[j]['embeddings'])
            embeddings1_list.append(embedding1.tolist())
            embeddings2_list.append(embedding2.tolist())
            similarity_score = compute_similarity_score(embedding1, embedding2)
            similarity_scores.append(similarity_score)
            (positive_scores if speaker_id1 == speaker_id2 else negative_scores).append(similarity_score)

    eer_score, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    wer = WordErrorRate()
    wer_score = wer(predictions, transcripts).item()

    pairs_dataset = Dataset.from_dict({
        'speaker_id1': speaker_ids1,
        'speaker_id2': speaker_ids2,
        'embedding1': embeddings1_list,
        'embedding2': embeddings2_list,
        'similarity_score': similarity_scores,
    })

    print('done computing metrics and pairwise similarity')
    return pairs_dataset, eer_score, wer_score

# def calculate_eer(dataset: Dataset) -> float:
#     """
#     Computes EER based on the similarity scores and labels in the dataset.
    
#     Args:
#         dataset (Dataset): Dataset including columns ['speaker_id1', 'speaker_id2', 'embedding1', 'embedding2', 'similarity_score'].
    
#     Returns:
#         float: The calculated EER and its threshold.
#     """
#     positive_scores = []
#     negative_scores = []
    
#     for item in tqdm(dataset):
#         speaker_id1 = item['speaker_id1']
#         speaker_id2 = item['speaker_id2']
#         embedding1 = torch.tensor(item['embedding1'])
#         embedding2 = torch.tensor(item['embedding2'])
#         similarity_score = compute_similarity_score(embedding1, embedding2)
        
#         if speaker_id1 == speaker_id2:
#             positive_scores.append(similarity_score)
#         else:
#             negative_scores.append(similarity_score)

#     eer, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
#     return eer, threshold


# def calculate_wer(dataset: Dataset, predictions: dict[str, str]) -> float:
#     """
#     Computes Word Error Rate (WER) based on the predictions and transcripts in the dataset.
    
#     Args:
#         dataset (Dataset): Dataset including columns ['audio', 'speaker_id', 'transcript', 'embeddings'].
#         predictions (Dict[str, str]): Dictionary mapping audio paths to predicted transcripts.

#     Returns:
#         float: The calculated WER.
#     """
#     assert len(dataset) == len(predictions), "Number of predictions should match the number of samples in the dataset"
    
#     # Sort dataset and predictions based on audio path
#     sorted_dataset = sorted(dataset, key=lambda x: x['audio']['path'])
#     sorted_predictions = [predictions[item['audio']['path']] for item in sorted_dataset]
    
#     transcripts = [item['transcript'] for item in sorted_dataset]
    
#     wer = WordErrorRate()
#     wer_score = wer(sorted_predictions, transcripts)
#     return wer_score
def speaker_verification(dataset, split):
    audio_dataset = load_dataset(dataset, split=split, cache_dir=f"{CODEBASE_DIR}/tmp")
    embeddings_dataset = process_data_to_embeddings(audio_dataset)
    similarities, eer_score, wer_score = compute_metrics_and_similarity(embeddings_dataset)
    embeddings_dataset.push_to_hub(dataset, commit_message='uploading embeddings')
    # structure of similarities is vastly different, so upload as a separate dataset
    similarities.push_to_hub(f"{dataset}-similarities", commit_message=f'uploading similarities. EER: {eer_score:.4f}, WER: {wer_score:.4f}')

def main():    

    ###############################################################################
    # TODO: Set the params here before running the script
    dataset_name = f"azain/LibriTTS-dev-clean-16khz-mono-loudnorm-100-random-samples-2024-04-18-17-34-39"
    split='dev'
    ###############################################################################
    audio_dataset = load_dataset(dataset_name, split=split, cache_dir=f"{CODEBASE_DIR}/tmp")
    print(audio_dataset[0])
    embeddings_dataset = process_data_to_embeddings(audio_dataset)
    similarities, eer_score, wer_score = compute_metrics_and_similarity(embeddings_dataset)
    metadata_description = f"EER: {eer_score:.4f}, WER: {wer_score:.4f}"
    similarities.info.description = metadata_description
    embeddings_dataset.push_to_hub(dataset_name, commit_message='uploading embeddings')
    # structure of similarities is vastly different, so upload as a separate dataset
    similarities.push_to_hub(f"{dataset_name}-similarities", commit_message=f'uploading similarities EER: {eer_score:.4f}, WER: {wer_score:.4f}')

if __name__ == "__main__":
    main()