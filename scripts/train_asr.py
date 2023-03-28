import os
import torch
import datetime
import time

import torch

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
import os
from utils import *
from models import *
from my_loss import *

# REMEMBER TO CHANGE LINE 63,87 IF YOU NEED [RAW AUDIO]
# REMEMVER TO CHANGE MODEL NAME BEFORE RUNNING
def train_asr_model(model, dataset, CONSTANTS,
                   batch_size=32,
                   lr=1e-3,
                   gamma=0.2,
                   step_size=5,
                   num_folds=5,
                   num_epochs=100,
                   num_workers=8,
                   save_every=10,
                   patience=10,
                   torlance=0.005,
                   criterion=nn.MSELoss()):
    device = CONSTANTS.device
    train_writer = SummaryWriter("/home/ubuntu/elec823/log/" + CONSTANTS.MODEL_NAME + "/train")
    val_writer = SummaryWriter("/home/ubuntu/elec823/log/" + CONSTANTS.MODEL_NAME + "/val")

    time_start = time.time()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print("Start Time: ", current_time)

    kf = KFold(n_splits=num_folds, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_loader = DataLoader(dataset=[
            dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_workers)
        val_loader = DataLoader(dataset=[
            dataset[i] for i in val_idx], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

        # criterion = MyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        early_stop = EarlyStop(model, CONSTANTS.SAVE_PATH, save_every=save_every, patience=patience, torlance=torlance)
        
        for epoch in range(num_epochs):
            print(f"\nFold {fold + 1}/{num_folds}: Epoch {epoch + 1}/{num_epochs}")
            model.train()
            train_loss = 0
            train_scores = torch.zeros(0).to(device)
            train_preds = torch.zeros(0).to(device)
            for speech_input, info_dict in tqdm(train_loader, desc="Training:"):
                # speech_input = speech_input.to(device)
                score = info_dict["score"].to(torch.float32).to(device)
                mono_path = [CONSTANTS.DATA_PATH + path + "_mono.wav" for path in info_dict['path']]
                pred = model(speech_input, info_dict, mono_path)
                pred = torch.squeeze(pred)
                loss = criterion(pred, score)

                train_scores = torch.cat((train_scores, score))
                train_preds = torch.cat((train_preds, pred))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() / len(train_loader)

            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_scores = torch.zeros(0).to(device)
                val_preds = torch.zeros(0).to(device)
                for speech_input, info_dict in tqdm(val_loader, desc="Validation:"):
                    # speech_input = speech_input.to(device)
                    score = info_dict["score"].to(torch.float32).to(device)
                    mono_path = [CONSTANTS.DATA_PATH + path + "_mono.wav" for path in info_dict['path']]
                    pred = model(speech_input, info_dict, mono_path)
                    pred = torch.squeeze(pred)
                    loss = criterion(pred, score)

                    val_scores = torch.cat((val_scores, score))
                    val_preds = torch.cat((val_preds, pred))
                    val_loss += loss.item() / len(val_loader)

            train_error = ErrorCal(train_scores, train_preds)
            print(f"\t Train Loss: {train_loss:.4f}")
            print(f"\t\tMSE: {train_error.mse_loss:.4f}")
            print(f"\t\tMAE: {train_error.mae_loss:.4f}")
            print(f"\t\tPearson Correlation: {train_error.pearson_coef:.4f}")
            print(f"\t\tSpearman Correlation: {train_error.spearman_coef:.4f}")
            
            val_error = ErrorCal(val_scores, val_preds)
            print(f"\t Validation Loss: {val_loss:.4f}")
            print(f"\t\tMSE: {val_error.mse_loss:.4f}")
            print(f"\t\tMAE: {val_error.mae_loss:.4f}")
            print(f"\t\tPearson Correlation: {val_error.pearson_coef:.4f}")
            print(f"\t\tSpearman Correlation: {val_error.spearman_coef:.4f}")

            train_writer.add_scalar("loss", train_loss, fold * num_epochs + epoch)
            train_writer.add_scalar("MSE", train_error.mse_loss, fold * num_epochs + epoch)
            train_writer.add_scalar("MAE", train_error.mae_loss, fold * num_epochs + epoch)
            train_writer.add_scalar("Pearson", train_error.pearson_coef, fold * num_epochs + epoch)
            train_writer.add_scalar("Spearman", train_error.spearman_coef, fold * num_epochs + epoch)
            val_writer.add_scalar("loss", val_loss, fold * num_epochs + epoch)
            val_writer.add_scalar("MSE", val_error.mse_loss, fold * num_epochs + epoch)
            val_writer.add_scalar("MAE", val_error.mae_loss, fold * num_epochs + epoch)
            val_writer.add_scalar("Pearson", val_error.pearson_coef, fold * num_epochs + epoch)
            val_writer.add_scalar("Spearman", val_error.spearman_coef, fold * num_epochs + epoch)

            if early_stop(val_loss, fold, epoch):
                try:
                    with open(CONSTANTS.last_output+f'{CONSTANTS.MODEL_NAME}_val.txt', mode='w') as f:
                        f.write("Score, Predict\n")
                        for i in range(len(val_scores)):
                            f.write(f"{val_scores[i].item():.2f}, {val_preds[i].item():.2f}\n")
                            
                    with open(CONSTANTS.last_output+f'{CONSTANTS.MODEL_NAME}_train.txt', mode='w') as f:
                        f.write("Score, Predict\n")
                        for i in range(len(train_scores)):
                            f.write(f"{train_scores[i].item():.2f}, {train_preds[i].item():.2f}\n")
                except:
                    print("Writing file failed")
                break
        break  # Only execute the first fold so far. Remove this line to run all folds
    try:
        torch.save(model, CONSTANTS.SAVE_PATH + "/final.pt")
    except:
        print("Exit training normally. But full model save failed.")
    else:
        print("Full model saved successfully.")

    print("==========================END==========================")
    time_end = time.time()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y/%m/%d %H:%M:%S")
    hms_time = str(datetime.timedelta(seconds=int(time_end - time_start)))
    print("End Time: ", current_time)
    print(f"Total Time: {hms_time}")

    # KILL the process, release GPU memory
    try:
        pid = os.getpid()
        subprocess.call(['sudo', 'kill', str(pid), '-9'])
        print("Process killed")
    except:
        print("Process kill failed")

    print("==========================DONE==========================")
    
if __name__ == "__main__":
    MODEL = '02_asr_model_1_2_not_freeze_mse_lr=1e-3'
    model = WordConfidence().to(device)
    
    # # Freeze ASR model
    # for param in model.asr_model.parameters():
    #     param.requires_grad = False

    CONSTANTS = InitializationTrain(
        model_name=MODEL, 
        verbose=False,
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
    
    train_asr_model(
        model=model,
        dataset=dataset,
        CONSTANTS=CONSTANTS,
        batch_size=32,
        lr=1e-3,
        gamma=0.2,
        step_size=5,
        num_folds=5,
        num_epochs=100,
        num_workers=8,
        save_every=10,
        patience = 10,
        torlance=0.005,
        criterion=MyLoss_v0_1()
    )