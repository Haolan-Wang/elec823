import torchaudio
import os
import torch


def is_mono(audio_waveform):
    return audio_waveform.size(0) == 1


def stereo_to_mono(audio_waveform, sample_rate):
    waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    # Pad or trim the audio to 6 seconds
    desired_length = sample_rate * 6  # keep 6 seconds
    if waveform.size(-1) < desired_length:
        padding = desired_length - waveform.size(-1)
        waveform = torch.nn.functional.pad(waveform, (0, padding), "constant")
    elif waveform.size(-1) > desired_length:
        waveform = waveform[..., :desired_length]
    return waveform


def process_and_save_audio_files(file_path):
    # Generate the target output file path with a "_mono" suffix
    file_base, file_ext = os.path.splitext(file_path)
    output_path = f"{file_base}_mono{file_ext}"

    # Check if the target file already exists
    # if os.path.exists(output_path):
    #     print(f"Target file already exists: {output_path}")
    #     return
    waveform, sample_rate = torchaudio.load(file_path)
    original_mono = is_mono(waveform)

    if not original_mono:
        print(f"Converting stereo to mono for file: {file_path}")
        waveform = stereo_to_mono(waveform, sample_rate)

    # Do any additional processing with the waveform (mono) here

    # Save the processed audio with a "_mono" suffix
    torchaudio.save(output_path, waveform, sample_rate)

    if original_mono:
        print(f"Original file was already mono, saved as: {output_path}")
    else:
        print(f"Converted and saved mono file as: {output_path}")


if __name__ == "__main__":
    # Get the list of audio files to process
    root_dir='/home/ubuntu/elec823/hurricane'
    for mod_folder in os.listdir(root_dir):
        if mod_folder.startswith("."):
            continue
        ssn_path = os.path.join(root_dir, mod_folder, 'cs')
        if not os.path.isdir(ssn_path):
            continue
        for snr in os.listdir(ssn_path):
            if snr.startswith("."):
                continue
            snr_path = os.path.join(ssn_path, snr)
            for audio_file in os.listdir(snr_path):
                audio_path = os.path.join(snr_path, audio_file)
                process_and_save_audio_files(audio_path)
