import torch
from datasets import load_dataset, load_from_disk
import opensmile
from tqdm.auto import tqdm
from cca_cka_utilities import cka, gram_linear, feature_space_linear_cka, cca
import numpy as np

def load_opensmile_model(feature_set, feature_level):
    """
    Load an openSMILE configuration to extract audio features.

    This function initializes the `opensmile.Smile` object which is used for extracting
    audio features based on a predefined set of features and the computational level.

    Parameters
    ----------
    feature_set : str
        The set of features to use, defined in `opensmile.FeatureSet`. Examples include 'ComParE_2016'
        and 'eGeMAPSv01a'.
    feature_level : str
        The level of feature extraction to perform, defined in `opensmile.FeatureLevel`. Examples
        include 'Functional' and 'LowLevelDescriptors'.

    Returns
    -------
    smile : opensmile.Smile
        A configured `opensmile.Smile` object ready to process audio data to extract features.

    Examples
    --------
    >>> smile_model = load_opensmile_model('ComParE_2016', 'Functional')
    >>> print(smile_model)
    <opensmile.Smile object at 0x7f88604c4cd0>
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet[feature_set],
        feature_level=opensmile.FeatureLevel[feature_level],
    )
    return smile

def extract_feats_from_batch(batch, smile):
    """
    Extract features from a batch of audio samples using the specified openSMILE model.

    This function processes each audio sample in a given batch using a provided `opensmile.Smile`
    object to compute the audio features.

    Parameters
    ----------
    batch : dict
        A dictionary containing the batch of audio data. This dictionary should have a key 'audio',
        which is associated with a list of dictionaries, each containing 'array' and 'sampling_rate'
        keys.
    smile : opensmile.Smile
        The openSMILE configuration loaded via `load_opensmile_model` that will be used to extract
        features.

    Returns
    -------
    dict
        A dictionary containing the extracted features under the key 'features'. Each entry in 'features'
        is a DataFrame containing the extracted features for one audio file.

    Examples
    --------
    >>> batch = {'audio': [{'array': np.array([0, 1, 2]), 'sampling_rate': 16000}]}
    >>> smile = load_opensmile_model('ComParE_2016', 'Functional')
    >>> features = extract_feats_from_batch(batch, smile)
    >>> print(features)
    {'features': [<pandas.DataFrame>]}
    """
    # Extracting audio data
    audio_arrays = [audio['array'] for audio in batch['audio']]
    sampling_rates = [audio['sampling_rate'] for audio in batch['audio']]

    # Processing each audio sample in the batch to compute features
    batch_features = [smile.process_signal(array, rate) for array, rate in zip(audio_arrays, sampling_rates)]

    # Returning a dictionary with only 'features' field
    return {'features': batch_features}

def extract_feats_from_dataset(dataset, feature_set, feature_level, batch_size=16):
    """
    Apply feature extraction across a dataset of audio files in batches.

    This function initializes an openSMILE model and processes an entire dataset to extract
    audio features in batches. The function assumes that the dataset is a `datasets.Dataset`
    object from the `datasets` library.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset containing the audio files to process. Each sample in the dataset should have
        at least an 'audio' field containing 'array' and 'sampling_rate'.
    feature_set : str
        The set of features to extract, corresponds to the configurations available in `opensmile.FeatureSet`.
    feature_level : str
        The computational complexity of features to extract, corresponds to the levels defined in `opensmile.FeatureLevel`.
    batch_size : int, optional
        The number of samples to process in each batch. Defaults to 16.

    Returns
    -------
    updated_dataset : datasets.Dataset
        The dataset with an added column 'features', where each entry is a DataFrame containing the extracted features.

    Examples
    --------
    >>> from datasets import load_dataset
    >>> dataset = load_dataset('path/to/dataset', split='train')
    >>> updated_dataset = extract_feats_from_dataset(dataset, 'ComParE_2016', 'Functional')
    >>> print(updated_dataset)
    Dataset({
        features: ['audio', 'features'],
        num_rows: 100
    })
    """
    smile = load_opensmile_model(feature_set, feature_level)
    updated_dataset = dataset.map(
        extract_feats_from_batch,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"smile": smile}
    )
    return updated_dataset


def transform_features(example):
    # This assumes 'features' is a dictionary and we take only the values, converting them into a list
    example['features_list'] = list(example['features'].values())
    return example

def main():
    ###############################################################################
    # TODO: Set the params here before running the script
    feature_set = "eGeMAPSv02"
    feature_level = "Functionals"
    dataset_name = "blabble-io/libritts"
    dataset_split = "dev" # we may want to switch to 'clean' or 'all' later, but pls keep 'dev' for now
    dataset_subset = "dev.clean"
    ###############################################################################
            
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_split)[dataset_subset].select(range(100))
    
    # Process dataset
    updated_dataset = extract_feats_from_dataset(dataset, feature_set, feature_level)
    print(updated_dataset)

    # Apply the function to every example in the dataset (assuming the dataset split is 'train')
    updated_dataset = updated_dataset.map(transform_features)
    features_list = np.array(updated_dataset['features_list']).squeeze()

    cka_from_examples = cka(gram_linear(features_list), gram_linear(features_list))
    cka_from_features = feature_space_linear_cka(features_list, features_list)
    cca_from_features = cca(features_list, features_list)

    print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
    print('Mean Squared CCA Correlation: {:.5f}'.format(cca_from_features))

if __name__ == "__main__":
    main()