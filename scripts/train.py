from models import *
from utils import *
from my_loss import *



MODEL = '01_mel_mono_mse_lr=1e-2'
model = MelMonoModel().to(device)
# for param in model.conformer_encoder.parameters():
#     param.requires_grad = False

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

train_my_model(
    model=model,
    dataset=dataset,
    CONSTANTS=CONSTANTS,
    batch_size=32,
    lr=1e-2,
    gamma=0.2,
    step_size=5,
    num_folds=5,
    num_epochs=100,
    num_workers=8,
    save_every=10,
    patience = 20,
    torlance=0.005,
    criterion=MyLoss_v0_1()
    )


