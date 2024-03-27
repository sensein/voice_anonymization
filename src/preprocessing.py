import torch
import torchaudio
import pyloudnorm as pyln

def resample_audio(audio_path, target_sample_rate=16000):
    """
    Resamples the audio file at the given audio_path to the target_sample_rate.

    Args:
        audio_path (str): The path to the audio file.
        target_sample_rate (int, optional): The desired sample rate for the resampled audio. Defaults to 16000.

    Returns:
        resampled_waveform (torch.Tensor): The resampled audio waveform.
        target_sample_rate (int): The sample rate of the resampled audio.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    resampled_waveform = resampler(waveform)
    return resampled_waveform, target_sample_rate

def stereo_to_mono(waveform):
    """
    Convert a stereo audio waveform to mono.

    Args:
        waveform (torch.Tensor): The input stereo audio waveform.

    Returns:
        torch.Tensor: The converted mono audio waveform.

    References:
        - https://github.com/pytorch/audio/issues/363#issuecomment-637131351
    """
    if waveform.shape[0] == 2:  # Check if the audio is stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def normalize_loudness(waveform, sample_rate):
    """
    Normalize the loudness of an audio waveform.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        sample_rate (int): The sample rate of the audio waveform.

    Returns:
        torch.Tensor: The loudness-normalized audio waveform.

    References:
        - https://github.com/csteinmetz1/pyloudnorm?tab=readme-ov-file#loudness-normalize-and-peak-normalize-audio-files
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(waveform.numpy())
    loudness_normalized_audio = pyln.normalize.loudness(waveform.numpy(), loudness, -23.0) # -23 is default target loudness
    return torch.tensor(loudness_normalized_audio)

def main():
    pass

if __name__ == "__main__":
    main()