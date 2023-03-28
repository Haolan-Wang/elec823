from models import *
from utils import *
from my_loss import *

import nemo.collections.asr as nemo_asr
# model = nemo_asr.models.EncDecRNNTModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")
model = nemo_asr.models.EncDecRNNTModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")

MODEL = '01_mel_mono_freeze'

CONSTANTS = InitializationTrain(
    model_name=MODEL, 
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
dataset = CPCdata(metadata=CONSTANTS.metadata)

full_path = ['/home/ubuntu/elec823/clarity_CPC1_data/clarity_data/HA_outputs/train/S09844_L0231_E013.wav',
 '/home/ubuntu/elec823/clarity_CPC1_data/clarity_data/HA_outputs/train/S09928_L0219_E021.wav',
 '/home/ubuntu/elec823/clarity_CPC1_data/clarity_data/HA_outputs/train/S09463_L0209_E013.wav']

out = model.transcribe(paths2audio_files = full_path, return_hypotheses = True)