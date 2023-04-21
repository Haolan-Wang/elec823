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
from data_process import *


###############################################
# REMEMBER TO CHANGE MODEL NAME BEFORE RUNNING#
###############################################
def train_binaural(model, dataset, CONSTANTS,
               batch_size=16,
               lr=1e-3,
               gamma=0.2,
               step_size=5,
               num_folds=5,
               num_epochs=100,
               num_workers=8,
               save_every=10,
               patience=5,
               tolerance=0.005,
               criterion=nn.MSELoss()):
    device = CONSTANTS.device
    writer = SummaryWriter("/home/ubuntu/elec823/log/" + CONSTANTS.MODEL_NAME )
    writer.add_text("Info/Model Name", CONSTANTS.MODEL_NAME)
    writer.add_text("Info/Batch size", str(batch_size))
    writer.add_text("Info/Initial learning rate", str(lr))
    writer.add_text("Info/Gamma", str(gamma))
    writer.add_text("Info/Step size", str(step_size))
    writer.add_text("Info/Number of folds", str(num_folds))
    writer.add_text("Info/Number of epochs", str(num_epochs))
    writer.add_text("Info/Number of workers", str(num_workers))
    writer.add_text("Info/save every", str(save_every))
    writer.add_text("Info/Patience", str(patience))
    writer.add_text("Info/Tolerance", str(tolerance))
    writer.add_text("Info/Criterion", str(criterion))
    
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
        early_stop = EarlyStop(model, CONSTANTS.SAVE_PATH, save_every=save_every, patience=patience, torlance=tolerance)

        for epoch in range(num_epochs):
            print(f"\nFold {fold + 1}/{num_folds}: Epoch {epoch + 1}/{num_epochs}")
            model.train()
            train_loss = 0
            train_scores = torch.zeros(0)  # .to(device)
            train_preds = torch.zeros(0)  # .to(device)
            for speech_input_l, speech_input_r, info_dict in tqdm(train_loader, desc="Training:"):
                writer.add_scalar("Info/Learning rate", lr)
                optimizer.zero_grad()
                speech_input_l = speech_input_l.to(device)
                speech_input_r = speech_input_r.to(device)
                score = info_dict["score"].to(torch.float32).to(device)
                pred = model(speech_input_l, speech_input_r, info_dict)
                pred = torch.squeeze(pred)
                if score.shape != pred.shape:
                    pred = pred.reshape_as(score)
                loss = criterion(pred, score)
                train_scores = torch.cat((train_scores, score.to('cpu')))
                train_preds = torch.cat((train_preds, pred.to('cpu')))

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() / len(train_loader)

                # del speech_input, score, pred, loss
                # torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_scores = torch.zeros(0)  # .to(device)
                val_preds = torch.zeros(0)  # .to(device)
                for speech_input_l, info_dict in tqdm(val_loader, desc="Validation:"):
                    speech_input_l = speech_input_l.to(device)
                    score = info_dict["score"].to(torch.float32).to(device)
                    pred = model(speech_input_l, info_dict)
                    pred = torch.squeeze(pred)
                    if score.shape != pred.shape:
                        pred = pred.reshape_as(score)
                    loss = criterion(pred, score)

                    val_scores = torch.cat((val_scores, score.to('cpu')))
                    val_preds = torch.cat((val_preds, pred.to('cpu')))
                    val_loss += loss.item() / len(val_loader)

                    # del speech_input, score, pred, loss
                    # torch.cuda.empty_cache()

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

            writer.add_scalar("Train/loss", train_loss, fold * num_epochs + epoch)
            writer.add_scalar("Train/MSE", train_error.mse_loss, fold * num_epochs + epoch)
            writer.add_scalar("Train/MAE", train_error.mae_loss, fold * num_epochs + epoch)
            writer.add_scalar("Train/Pearson", train_error.pearson_coef, fold * num_epochs + epoch)
            writer.add_scalar("Train/Spearman", train_error.spearman_coef, fold * num_epochs + epoch)
            writer.add_scalar("Validation/loss", val_loss, fold * num_epochs + epoch)
            writer.add_scalar("Validation/MSE", val_error.mse_loss, fold * num_epochs + epoch)
            writer.add_scalar("Validation/MAE", val_error.mae_loss, fold * num_epochs + epoch)
            writer.add_scalar("Validation/Pearson", val_error.pearson_coef, fold * num_epochs + epoch)
            writer.add_scalar("Validation/Spearman", val_error.spearman_coef, fold * num_epochs + epoch)

            # del train_loss, train_error, val_error
            # torch.cuda.empty_cache()
            with open(CONSTANTS.last_output + f'{CONSTANTS.MODEL_NAME}_val.txt', mode='w') as f:
                f.write("Score, Predict\n")
                for i in range(len(val_scores)):
                    f.write(f"{val_scores[i].item():.2f}, {val_preds[i].item():.2f}\n")

            with open(CONSTANTS.last_output + f'{CONSTANTS.MODEL_NAME}_train.txt', mode='w') as f:
                f.write("Score, Predict\n")
                for i in range(len(train_scores)):
                    f.write(f"{train_scores[i].item():.2f}, {train_preds[i].item():.2f}\n")
            if early_stop(val_loss, fold, epoch):

                # del val_loss, train_scores, train_preds, val_scores, val_preds
                # torch.cuda.empty_cache()
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
    print("==========================DONE==========================")
    print("Process killed")
    pid = os.getpid()
    subprocess.call(['sudo', 'kill', str(pid), '-9'])


if __name__ == "__main__":
    
    # WordConfidence============================
    # MODEL = 'WordConfidence_05'
    # model = WordConfidence().to(device)
    # 01: MSEPearson
    # 01.1: MSE Rerun
    # 02: MSE
    # 03: MSEConcordanceLoss
    # 04: sigmoid + MSEPearsonLoss
    # 05: MSEPearsonConcordanceLoss lr = 1e-4
    
    # EncoderPredictor==========================
    MODEL = 'EncoderPredictor_01'
    model = EncoderPredictor().to(device)
    # 01: MSE lr = 1e-4
    # 05: MSEConcordanceLoss lr = 1e-4
    # 06: MSEPearsonConcordanceLoss lr = 1e-3
    
    CONSTANTS = InitializationTrain(
        model_name=MODEL,
        verbose=True
    )
    dataset = CPCdataBinaural(metadata=CONSTANTS.metadata)
    
    train_binaural(
        model=model,
        dataset=dataset,
        CONSTANTS=CONSTANTS,
        batch_size=16,
        lr=1e-4,
        criterion=MyMSELoss()
    )
