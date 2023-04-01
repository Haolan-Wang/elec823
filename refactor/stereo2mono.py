from models import *
from utils import *
from my_loss import *


def is_mono(audio_waveform):
    return audio_waveform.size(0) == 1

def stereo_to_mono(audio_waveform):
    return audio_waveform.mean(dim=0, keepdim=True)

def process_and_save_audio_files(file_paths):
    for file_path in file_paths:
        # Generate the target output file path with a "_mono" suffix
        file_base, file_ext = os.path.splitext(file_path)
        output_path = f"{file_base}_mono{file_ext}"

        # Check if the target file already exists
        if os.path.exists(output_path):
            print(f"Target file already exists: {output_path}")
            continue

        waveform, sample_rate = torchaudio.load(file_path)
        original_mono = is_mono(waveform)

        if not original_mono:
            print(f"Converting stereo to mono for file: {file_path}")
            waveform = stereo_to_mono(waveform)

        # Do any additional processing with the waveform (mono) here

        # Save the processed audio with a "_mono" suffix
        torchaudio.save(output_path, waveform, sample_rate)

        if original_mono:
            print(f"Original file was already mono, saved as: {output_path}")
        else:
            print(f"Converted and saved mono file as: {output_path}")
            
if __name__ == "__main__":
    # Get the list of audio files to process
    CONSTANTS = InitializationTrain(
    verbose=True,
    train_data_path='/home/ubuntu/elec823/clarity_CPC1_data/clarity_data/HA_outputs/train/',
    train_info_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/CPC1.train.json',
    train_audiogram_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/listeners.CPC1_train.json',
    train_listener_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/listener_data.CPC1_train.xlsx',
    log_path='/home/ubuntu/elec823/log/',
    save_path='/home/ubuntu/elec823/checkpoints/',
    orig_freq=32000,
    new_freq=16_000,
    seed=3407,
    device=None
    )

    full_path = [CONSTANTS.DATA_PATH + path + ".wav" for path in CONSTANTS.metadata['path']]
    process_and_save_audio_files(full_path)