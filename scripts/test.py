from models import *
from utils import *
from my_loss import *

MODEL = '01_mel_mono_loss_without_v2_2'
model = MelMonoModelWithout().to(device)
model = model.load_state_dict("/home/ubuntu/elec823/checkpoints/01_mel_mono_loss_without_v2_2/last.pt")
dataset = CPCdata(metadata=CONSTANTS.metadata)

CONSTANTS = InitializationTrain(
    model_name=MODEL, 
    verbose=True,
    log_path='/home/ubuntu/elec823/log/',
    save_path='/home/ubuntu/elec823/checkpoints/',
    orig_freq=32000,
    new_freq=16_000,
    seed=3407,
    device=None,
    mode='test'
    )
