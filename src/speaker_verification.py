from src.utils import CODEBASE_DIR
from datasets import Dataset, load_dataset
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

def process_data_to_embeddings(audio_dataset: Dataset) -> Dataset:
    """
    Processes audio data from the given dataset to generate and save embeddings for each entry.

    Args:
        audio_dataset (Dataset): The dataset containing audio files and their corresponding metadata.

    Returns:
        Dataset: A new Dataset object containing the original audio, transcript (if applicable), and computed embeddings. # todo once speaker id added in preprocessing, handle it here
    
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

    Args:
        embedding1 (tensor): The first embedding.
        embedding2 (tensor): The second embedding.

    Returns:
        float: The similarity score between the two embeddings.

    References:
        https://github.com/sensein/fab/blob/main/tutorials/voice_anonymization/voice_anonymization.ipynb
    """
    cos = torch.nn.CosineSimilarity(dim=-1)
    return cos(embedding1, embedding2).item()

def compute_pairwise_similarity(dataset: Dataset) -> Dataset:
    """
    Computes cosine similarity across all pairs of embeddings in the dataset.

    Args:
        dataset (Dataset): The dataset containing 'embeddings' for each entry.

    Returns:
        Dataset: A new Dataset object with columns ['speaker_id1', 'speaker_id2', 'similarity_score'].
    """
    num_rows = len(dataset)
    pair_ids1 = []
    pair_ids2 = []
    similarity_scores = []

    # Iterate over all unique pairs
    for i in tqdm(range(num_rows)):
        for j in range(i+1, num_rows):
            embedding1 = torch.tensor(dataset[i]['embeddings'])
            embedding2 = torch.tensor(dataset[j]['embeddings'])

            # Compute similarity score
            similarity_score = compute_similarity_score(embedding1, embedding2)

            # Store results
            pair_ids1.append(i)
            pair_ids2.append(j)
            similarity_scores.append(similarity_score)

    # Create a new dataset from the results
    pairs_dataset = Dataset.from_dict({
        'speaker_id1': pair_ids1,
        'speaker_id2': pair_ids2,
        'similarity_score': similarity_scores
    })

    return pairs_dataset

def main():
    # Example of loading a dataset (this should be replaced with actual dataset loading)
    # hf_dataset_name = "azain/LibriTTS-processed"
    # audio_dataset = load_dataset(hf_dataset_name)
    
    ### # TODO temporarily do this since I cant push datasets to HuggingFace
    processed_data_path = f"{CODEBASE_DIR}/tmp/LibriTTS-processed"
    transcript_path_pattern = "{base_name}.original.txt"
    from preprocessing import create_audio_dataset
    audio_dataset = create_audio_dataset(processed_data_path, transcript_path_pattern)
    audio_dataset = audio_dataset.select(range(100)) # small dataset for testing
    print(f"{audio_dataset=}")
    ###
    embeddings_dataset = process_data_to_embeddings(audio_dataset)
    embeddings_dataset.save_to_disk(f"{CODEBASE_DIR}/tmp/LibriTTS-processed-with-embeddings-small")
    similarities = compute_pairwise_similarity(embeddings_dataset)
    similarities.save_to_disk(f"{CODEBASE_DIR}/tmp/LibriTTS-processed-with-embeddings-small-similarities")
if __name__ == "__main__":
    main()