import os
from tqdm import tqdm

CODEBASE_DIR = '/om2/user/azain/code/voice_anonymization' # TODO. find a better way to handle this

def yield_filtered_files(directory_path, file_filter_func):
    """
    Traverses a directory tree to yield the root directory and names of audio files that meet a specified filter condition.

    Args:
        directory_path (str): Root directory from which to search for audio files.
        file_filter_func (callable): Function to determine if a filename meets the criteria for yielding.

    Yields:
        tuple: Each yielded tuple contains the root directory and the name of an audio file that passed the filter.

    Progress is tracked and displayed using a tqdm progress bar.
    """
    file_count = sum(len(files) for _, _, files in os.walk(directory_path)) # used to track progress
    # file_count = sum(1 for root, dirs, files in os.walk(directory_path) for file in files if file_filter_func(file))
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(directory_path):
            for name in files:
                pbar.update(1)
                if file_filter_func(name):
                    yield root, name