from src.preprocessing import preprocess
from src.voice_conversion import voice_convert
from src.automatic_speech_recognition import asr
from src.speaker_verification import process_data_to_embeddings, compute_metrics_and_similarity
from src.utils import CODEBASE_DIR

def main():
    pass

if __name__ == "__main__":
    ### Parameters
    raw_data_path = f'{CODEBASE_DIR}/data/raw/LibriTTS/dev-clean'
    processed_data_path = f'{CODEBASE_DIR}/data/processed/LibriTTS-dev-clean-16khz-mono-loudnorm'
    transcript_path_pattern = "{base_name}.original.txt"
    speaker_id_pattern = r"(\d+)_"
    target_sample_rate = 16000
    num_samples = 100 # todo measure how long it takes
    dataset_name = f"azain/LibriTTS-dev-clean-16khz-mono-loudnorm-100-random-samples-2024-04-18-17-34-39"
    split = 'dev'
    file_extensions = ['.wav']
    target_index = 0 #vary (could try from outside dataset)
    asr_model_id = "openai/whisper-tiny.en" # wav2vec_children_asr
    ###
    dataset = preprocess(
        raw_data_path, processed_data_path, transcript_path_pattern, speaker_id_pattern, 
        target_sample_rate, num_samples, file_extensions
        )
    print(f"{dataset=}")
    converted_dataset = voice_convert(dataset, target_index)
    print(f"{converted_dataset=}")
    dataset_after_asr = asr(asr_model_id, dataset, split)
    print(f"{dataset_after_asr=}")
    converted_dataset_after_asr = asr(asr_model_id, converted_dataset, split)
    print(f"{converted_dataset_after_asr=}")
    dataset_with_asr_and_embeddings = process_data_to_embeddings(dataset_after_asr)
    print(f"{dataset_with_asr_and_embeddings=}")
    converted_dataset_with_asr_and_embeddings = process_data_to_embeddings(converted_dataset_after_asr)
    print(f"{converted_dataset_with_asr_and_embeddings=}")
    similarities, eer_score, wer_score = compute_metrics_and_similarity(dataset_with_asr_and_embeddings)
    print(f"{eer_score=}, {wer_score=}, {similarities=}")
    converted_similarties, converted_eer_score, converted_wer_score = compute_metrics_and_similarity(converted_dataset_with_asr_and_embeddings)
    print(f"{converted_eer_score=}, {converted_wer_score=}, {converted_similarties=}")
    dataset_with_asr_and_embeddings.push_to_hub(dataset_name, commit_message='uploading embeddings')
    converted_dataset_with_asr_and_embeddings.push_to_hub(f"{dataset_name}-converted", commit_message='uploading embeddings for converted dataset')
    similarities.push_to_hub(f"{dataset_name}-similarities", commit_message=f'uploading similarities. EER: {eer_score:.4f}, WER: {wer_score:.4f}')
    converted_similarties.push_to_hub(f"{dataset_name}-converted-similarities", commit_message=f'uploading similarities for converted dataset. EER: {converted_eer_score:.4f}, WER: {converted_wer_score:.4f}')
# todo what params could change
# target_voice (why to pick one voice over another)
#   males->same male, females->same female
# todo results
# plot of similarity, eer (speaker1=speaker2, speaker1!=speaker2)
# plot of wer (avg, stdv before/after conversion)
# cosine similarity between pairs of embeddings:
    # original vs original
    # original vs anonymized
    # anonymized vs anonymized