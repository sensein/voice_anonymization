from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.metric_stats import EER
from tqdm import tqdm
from torchmetrics.text import WordErrorRate

def process_data_to_embeddings(audio_dataset: Dataset) -> Dataset:
    """
    Processes audio data from the given dataset to generate and save embeddings for each entry.

    Args:
        audio_dataset (Dataset): The dataset containing audio files and their corresponding metadata.

    Returns:
        Dataset: A new Dataset object containing the original audio, computed embeddings and existing metadata.
    
    References:
        https://github.com/sensein/fab/blob/main/tutorials/voice_anonymization/voice_anonymization.ipynb
    """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{CODEBASE_DIR}/models/spkrec-ecapa-voxceleb") # todo fix this codebase_dir nonsense
    processed_data = {
        'audio': [],
        'transcript': [],
        'speaker_id': [],
        'embeddings': []
    }

    print(f'computing embeddings for {audio_dataset}')
    for item in tqdm(audio_dataset):
        assert isinstance(item, dict), "error getting item from audio_dataset. items in audio_dataset should be dictionaries"
        
        assert 'audio' in item, "item should have an audio key"
        audio : dict = item['audio']
        
        assert classifier is not None, "error computing embeddings. Classifier should not be None" # todo eventually move assertions when creating tests
        waveform : torch.Tensor = torch.tensor(audio['array']).squeeze()
        embeddings : torch.Tensor = classifier.encode_batch(waveform.unsqueeze(0)).squeeze() # unsqueeze to turn [time] -> [batch, time], then squeeze after computing embedding
        
        processed_data['audio'].append(audio) # todo do I keep this? embeddings already calculated
        processed_data['transcript'].append(item.get('transcript', ''))
        processed_data['speaker_id'].append(item.get('speaker_id', ''))
        processed_data['embeddings'].append(embeddings.numpy()) # todo can think about storing directly as tensor

    print('done computing embeddings')
    return Dataset.from_dict(processed_data)

def compute_similarity_score(embedding1, embedding2): # todo could try ECAPA too
    """
    Computes the similarity score between two embeddings using cosine similarity.

    References:
        https://github.com/sensein/fab/blob/main/tutorials/voice_anonymization/voice_anonymization.ipynb
    """
    cos = torch.nn.CosineSimilarity(dim=-1)
    return cos(embedding1, embedding2).item()

def compute_pairwise_similarity(dataset: Dataset) -> Dataset: # todo compare between concated waveforms 
    """
    Computes cosine similarity across all pairs of embeddings in the dataset.

    Args:
        dataset (Dataset): Dataset including columns ['speaker_id', 'embeddings'].

    Returns:
        Dataset: A new Dataset object with columns ['speaker_id1', 'speaker_id2', 'embedding1', 'embedding2', 'similarity_score'].
    """
    print(f'computing pairwise similarity for {dataset}')
    num_rows = len(dataset)
    speaker_ids1 = []
    speaker_ids2 = []
    similarity_scores = []
    embeddings1_list = []
    embeddings2_list = []

    # For EER and WER
    same_speakers = []
    predictions = []
    references = []

    for i in tqdm(range(num_rows)):
        for j in range(i+1, num_rows):
            speaker_id1 = dataset[i]['speaker_id']
            speaker_id2 = dataset[j]['speaker_id']
            embedding1 = torch.tensor(dataset[i]['embeddings'])
            embedding2 = torch.tensor(dataset[j]['embeddings'])

            similarity_score = compute_similarity_score(embedding1, embedding2)
            speaker_ids1.append(speaker_id1)
            speaker_ids2.append(speaker_id2)
            similarity_scores.append(similarity_score)
            embeddings1_list.append(embedding1.tolist())
            embeddings2_list.append(embedding2.tolist())
            
            same_speakers.append(int(speaker_id1 == speaker_id2))

            # Collect transcripts for WER calculation
            predictions.append(dataset[j]['transcript'])
            references.append(dataset[i]['transcript'])

    eer_scores = EER(torch.tensor(similarity_scores), torch.tensor(same_speakers))
    wer = WordErrorRate()
    wer_scores = wer(predictions, references)

    pairs_dataset = Dataset.from_dict({
        'speaker_id1': speaker_ids1,
        'speaker_id2': speaker_ids2,
        'embedding1': embeddings1_list,
        'embedding2': embeddings2_list,
        'similarity_score': similarity_scores
    })
    print('done computing pairwise similarity')
    return pairs_dataset

def calculate_eer(dataset: Dataset) -> float:
    """
    Computes EER based on the similarity scores and labels in the dataset.
    
    Args:
        dataset (Dataset): Dataset including columns ['speaker_id1', 'speaker_id2', 'embedding1', 'embedding2', 'similarity_score'].
    
    Returns:
        float: The calculated EER and its threshold.
    """
    positive_scores = []
    negative_scores = []
    
    for item in tqdm(dataset):
        speaker_id1 = item['speaker_id1']
        speaker_id2 = item['speaker_id2']
        embedding1 = torch.tensor(item['embedding1'])
        embedding2 = torch.tensor(item['embedding2'])
        similarity_score = compute_similarity_score(embedding1, embedding2)
        
        if speaker_id1 == speaker_id2:
            positive_scores.append(similarity_score)
        else:
            negative_scores.append(similarity_score)

    eer, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    return eer, threshold


def calculate_wer(dataset: Dataset, predictions: dict[str, str]) -> float:
    """
    Computes Word Error Rate (WER) based on the predictions and transcripts in the dataset.
    
    Args:
        dataset (Dataset): Dataset including columns ['audio', 'speaker_id', 'transcript', 'embeddings'].
        predictions (Dict[str, str]): Dictionary mapping audio paths to predicted transcripts.

    Returns:
        float: The calculated WER.
    """
    assert len(dataset) == len(predictions), "Number of predictions should match the number of samples in the dataset"
    
    # Sort dataset and predictions based on audio path
    sorted_dataset = sorted(dataset, key=lambda x: x['audio']['path'])
    sorted_predictions = [predictions[item['audio']['path']] for item in sorted_dataset]
    
    references = [item['transcript'] for item in sorted_dataset]
    
    wer = WordErrorRate()
    wer_score = wer(sorted_predictions, references)
    return wer_score

def main():
    # hf_dataset_name = "azain/LibriTTS-processed"
    # audio_dataset = load_dataset(hf_dataset_name)
    
    ### # TODO temporarily do this since I cant push datasets to HuggingFace
    processed_data_path = f"{CODEBASE_DIR}/tmp/LibriTTS-processed"
    transcript_path_pattern = "{base_name}.original.txt"
    from preprocessing import create_audio_dataset
    audio_dataset = create_audio_dataset(processed_data_path, transcript_path_pattern)
    audio_dataset = audio_dataset.shuffle(seed=42) # shuffle to get random samples
    audio_dataset = audio_dataset.select(range(100)) # small dataset for testing
    print(f"{audio_dataset=}")
    ###

    ###############################################################################
    # TODO: Set the params here before running the script
    embeddings_dataset_path = f"{CODEBASE_DIR}/tmp/LibriTTS-processed-with-embeddings-small-shuffled"
    similarities_dataset_path = f"{CODEBASE_DIR}/tmp/LibriTTS-processed-with-embeddings-small-shuffled-similarities"
    ###############################################################################
    embeddings_dataset = process_data_to_embeddings(audio_dataset)
    embeddings_dataset.save_to_disk(embeddings_dataset_path)
    similarities = compute_pairwise_similarity(embeddings_dataset)
    similarities.save_to_disk(similarities_dataset_path)

if __name__ == "__main__":
    main()