codebase_dir = '/om2/user/azain/code/voice_anonymization'

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
        Dataset: A new Dataset object containing the original path, transcript (if applicable), and computed embeddings. # todo once speaker id added in preprocessing, handle it here
    
    References:
        https://github.com/sensein/fab/blob/main/tutorials/voice_anonymization/voice_anonymization.ipynb
    """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{codebase_dir}/models/spkrec-ecapa-voxceleb") # todo fix this codebase_dir nonsense
    processed_data = {
        'path': [],
        'transcript': [],
        'embeddings': []
    }

    print(f'computing embeddings for {audio_dataset}')
    for item in tqdm(audio_dataset):
        assert isinstance(item, dict), "error getting item from audio_dataset. items in audio_dataset should be dictionaries"
        
        assert 'audio' in item, "item should have an audio key"
        audio : dict = item['audio']
        
        assert classifier is not None, "error computing embeddings. Classifier should not be None"
        waveform : torch.Tensor = torch.tensor(audio['array']).squeeze()
        embeddings : torch.Tensor = classifier.encode_batch(waveform.unsqueeze(0)).squeeze() # unsqueeze to turn [time] -> [batch, time], then squeeze after computing embedding
        
        processed_data['path'].append(audio['path'])
        processed_data['transcript'].append(item.get('transcript', ''))
        processed_data['embeddings'].append(embeddings.numpy())

    print('done computing embeddings')
    return Dataset.from_dict(processed_data)

def compute_similarity_score(embedding1, embedding2): # todo could try ECAPA too
    """
    Computes the similarity score between two embeddings using cosine similarity.

    Args:
        embedding1 (list): The first embedding.
        embedding2 (list): The second embedding.

    Returns:
        float: The similarity score between the two embeddings.

    References:
        https://github.com/sensein/fab/blob/main/tutorials/voice_anonymization/voice_anonymization.ipynb
    """
    cos = torch.nn.CosineSimilarity(dim=-1)
    similarity_score = cos(torch.tensor(embedding1), torch.tensor(embedding2))
    return similarity_score.item()

def main():
    # Example of loading a dataset (this should be replaced with actual dataset loading)
    # hf_dataset_name = "azain/LibriTTS-processed"
    # audio_dataset = load_dataset(hf_dataset_name)
    
    ### TEMP
    processed_data_path = f"{codebase_dir}/tmp/LibriTTS-processed"
    transcript_path_pattern = "{base_name}.original.txt"
    from preprocessing import load_audio_dataset
    audio_dataset = load_audio_dataset(processed_data_path, transcript_path_pattern)
    ###
    embeddings_dataset = process_data_to_embeddings(audio_dataset)
    embeddings_dataset.save_to_disk(f"{codebase_dir}/tmp/LibriTTS-processed-with-embeddings")

if __name__ == "__main__":
    main()