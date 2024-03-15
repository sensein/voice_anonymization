import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm

def setup_device_and_dtype():
    """
    Determines the device and data type for PyTorch operations based on CUDA availability.
    
    Returns:
        device (str): The device to use ('cuda:0' or 'cpu').
        torch_dtype (torch.dtype): The data type for tensors.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype

def load_model_and_processor(model_id, device, torch_dtype):
    """
    Loads the model and processor for speech-to-text conversion.
    
    Args:
        model_id (str): Identifier for the model.
        device (str): Device to run the model on.
        torch_dtype (torch.dtype): Data type for model tensors.
    
    Returns:
        model: The loaded model.
        processor: The loaded processor.
    """
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def create_asr_pipeline(model, processor, device, torch_dtype):
    """
    Creates an automatic speech recognition (ASR) pipeline.
    
    Args:
        model: The speech-to-text model.
        processor: The processor for the model.
        device (str): The device to use for the pipeline.
        torch_dtype (torch.dtype): The data type for pipeline operations.
    
    Returns:
        pipeline: The ASR pipeline.
    """
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

"""
def transcribe_dataset(dataset, pipe):
    transcriptions = []
    for batch in tqdm(dataset.batch(batch_size=16), total=len(dataset)//16):
        try:
            batch_transcriptions = pipe(batch["audio"])
            transcriptions.extend([t["text"] for t in batch_transcriptions])
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Handle or log the error appropriately
            transcriptions.extend([None] * len(batch))  # Placeholder for failed transcriptions
    dataset = dataset.add_column("asr_transcription", transcriptions)
    return dataset

def transcribe_dataset(dataset, pipe):
    Transcribes all audio samples in the dataset and adds the transcriptions as a new column.
    transcriptions = [pipe(sample["audio"])["text"] for sample in dataset]
    return dataset.add_column("asr_transcription", transcriptions)
"""

def transcribe_audio(batch, pipe):
    """
    Transcribes a batch of audio samples and retains the original batch data.
    
    Args:
        batch: A dictionary with the key "audio" containing a list of audio file data.
        pipe: The ASR pipeline.
    
    Returns:
        The updated batch dictionary with a new key "asr_transcription" containing the transcriptions of the audio files.
    """
    # Perform transcription
    my_batch = batch.copy()
    transcriptions = pipe(batch["audio"])
    return {**my_batch, "asr_transcription": transcriptions}

def transcribe_dataset(dataset, pipe):
    """
    Transcribes all audio samples in the dataset and adds the transcriptions as a new column.
    
    Args:
        dataset: The dataset to transcribe.
        pipe: The ASR pipeline for transcription.
    
    Returns:
        The dataset with a new column "asr_transcription" containing the audio transcriptions.
    """
    # Using batched map function to transcribe audio in batches for efficiency
    updated_dataset = dataset.map(transcribe_audio, batched=True, batch_size=16, fn_kwargs={"pipe": pipe})
    return updated_dataset

def main():
    ###############################################################################
    # TODO: Set the params here before running the script
    asr_model_id = "openai/whisper-tiny.en" # "openai/whisper-large-v3"
    dataset_name = "blabble-io/libritts"
    dataset_split = "dev" # we may want to switch to 'clean' or 'all' later, but pls keep 'dev' for now
    path_to_updated_dataset = "../data/whisper_asr" # path to save the updated dataset
    ###############################################################################

    # Initialize settings and load model and processor
    device, torch_dtype = setup_device_and_dtype()

    model, processor = load_model_and_processor(asr_model_id, device, torch_dtype)
    
    # Create ASR pipeline
    pipe = create_asr_pipeline(model, processor, device, torch_dtype)
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_split)
    
    # Process dataset
    updated_dataset = transcribe_dataset(dataset, pipe)
    
    # Save updated dataset
    updated_dataset.save_to_disk(path_to_updated_dataset)
    print("Dataset updated and saved with transcriptions.")
    
    # updated_dataset = load_from_disk(path_to_updated_dataset)
    # print(updated_dataset[0])

if __name__ == "__main__":
    main()