import torchaudio
import os
from utils import InitializationTrain
import tqdm

def is_mono(audio_waveform):
    return audio_waveform.size(0) == 1


def stereo_to_mono(audio_waveform):
    return audio_waveform[0,:], audio_waveform[1,:]


def process_and_save_audio_files(file_paths):
    for file_path in tqdm.tqdm(file_paths):
        # Generate the target output file path with a "_mono" suffix
        file_base, file_ext = os.path.splitext(file_path)
        output_path_1 = f"{file_base}_mono_1{file_ext}"
        output_path_2 = f"{file_base}_mono_2{file_ext}"

        # Check if the target file already exists
        if os.path.exists(output_path_1):
            # print(f"Target file already exists: {output_path_1}")
            continue

        waveform, sample_rate = torchaudio.load(file_path)
        original_mono = is_mono(waveform)

        if not original_mono:
            # print(f"Converting stereo to mono for file: {file_path}")
            waveform_1, waveform_2 = stereo_to_mono(waveform)

        # Do any additional processing with the waveform (mono) here

        # Save the processed audio with a "_mono" suffix
        torchaudio.save(output_path_1, waveform_1.reshape(1, -1), sample_rate)
        torchaudio.save(output_path_2, waveform_2.reshape(1, -1), sample_rate)

        # if original_mono:
        #     print(f"Original file was already mono, saved as: {output_path_1}")
        # else:
        #     print(f"Converted and saved mono file as: {output_path_1}")


if __name__ == "__main__":
    # Get the list of audio files to process
    CONSTANTS = InitializationTrain(
    )
    full_path = [CONSTANTS.DATA_PATH + path + ".wav" for path in CONSTANTS.metadata['path']]
    process_and_save_audio_files(full_path)
