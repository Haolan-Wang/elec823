import gc
import json
import os
import random

import numpy as np
from collections import OrderedDict
import torch
import datetime
import time

import torch
import torchaudio

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchmetrics import SpearmanCorrCoef
import subprocess
import os
import torch.functional as F



class InitializationTrain:
    def __init__(
        self, 
        model_name=None, 
        verbose=True,
        train_data_path='/home/ubuntu/elec823/clarity_CPC1_data/clarity_data/HA_outputs/train/',
        train_info_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/CPC1.train.json',
        train_audiogram_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/listeners.CPC1_train.json',
        train_listener_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/listener_data.CPC1_train.xlsx',
        test_data_path='/home/ubuntu/elec823/clarity_CPC1_data/clarity_data/HA_outputs/test/',
        test_info_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/CPC1.test.json',
        test_audiogram_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/listeners.CPC1_all.json',
        test_listener_path='/home/ubuntu/elec823/clarity_CPC1_data/metadata/listener_data.CPC1_all.xlsx',
        log_path='/home/ubuntu/elec823/log/',
        save_path='/home/ubuntu/elec823/checkpoints/',
        orig_freq=32000,
        new_freq=16_000,
        seed=3407,
        device=None,
        mode='train'
        ):
        if mode == 'train':
            self.mode = 'train'
            self.DATA_PATH = train_data_path
            self.INFO_PATH = train_info_path
            self.AUDIOGRAM_PATH = train_audiogram_path
            self.LISTENER_PATH = train_listener_path
        else:
            self.mode = 'testing'
            self.DATA_PATH = test_data_path,
            self.INFO_PATH = test_info_path
            self.AUDIOGRAM_PATH = test_audiogram_path
            self.LISTENER_PATH = test_listener_path
            
        self.model_name = model_name
        self.verbose = verbose
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.seed = seed
        self.metadata = self.load_metadata()
        self.device = self.device_check(device)

        print("Model Name: ", model_name)
        self.path_check(model_name, log_path, save_path)

        self.seed_everything(self.seed)

    def path_check(self, model_name, log_path, save_path):
        if self.verbose:
            print("Checking paths...")
        if model_name is None:
            print(
                "Warning: Model name is not specified. It's EXPECTED if you are initializing a 'models.py'. Otherwise Your model will be saved in 'checkpoints/00_TMP_MODEL' folder.")
            self.MODEL_NAME = '00_TMP_MODEL'
        else:
            self.MODEL_NAME = model_name

        self.LOG_PATH = log_path + self.MODEL_NAME
        self.SAVE_PATH = save_path + self.MODEL_NAME   
        self.last_output = log_path + "last_output/"
        self.folder_names = [self.LOG_PATH, self.SAVE_PATH, self.last_output]
        for folder_name in self.folder_names:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                if self.verbose:
                    print("Folder created:", folder_name)
            else:
                if self.verbose:
                    print("Folder already exists:", folder_name)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        if self.verbose:
            print("Seed set to:", seed)

    def device_check(self, device):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                if self.verbose:
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                device = "cpu"
                if self.verbose:
                    print("Warning: Using CPU")
            return device
        else:
            return device

    def load_metadata(self):
        with open(self.INFO_PATH, 'r') as file:
            data = json.load(file)
        paths = [item['signal'] for item in data]
        if self.mode == 'train':
            scores = [item['correctness'] / 100 for item in data]
        else:
            scores = [None for item in data]
        listeners = [item['listener'] for item in data]
        systems = [item['system'] for item in data]
        scenes = [item['scene'] for item in data]
        volumes = [item['volume'] / 100 for item in data]
        prompts = [item['prompt'] for item in data]

        # make dict
        metadata = {'path': paths, 'score': scores, 'listener': listeners,
                    'system': systems, 'scene': scenes, 'volume': volumes, 'prompt': prompts}

        return OrderedDict(metadata)


class ErrorCal():
    """Input y_true and y_pred when initilizing, return pearson, spearman, mse loss"""
    def __init__(self, y_true, y_pred):
        self.mse_loss = torch.nn.functional.mse_loss(y_true, y_pred)
        self.mae_loss = torch.nn.functional.l1_loss(y_true, y_pred)
        self.pearson_coef = torch.corrcoef(torch.stack((y_true, y_pred)))[0, 1] 
        self.spearman = SpearmanCorrCoef()
        self.spearman_coef = self.spearman(y_true, y_pred)


class EarlyStop:
    """Usage: 
        Initialize: early_stop = EarlyStop(model, path, patience=5, save_every=5, torlance=0)
        Evaluation: early_stop(val_loss, fold, epoch)
        Return: True if early stopping, False if not
    """

    def __init__(self, model, path, patience=10, save_every=20, torlance=0):
        self.model = model
        self.path = path
        self.save_every = save_every
        self.patience = patience
        self.counter = 0
        self.torlance = torlance
        self.min_loss = None

    def __call__(self, val_loss, fold, epoch):
        if self.min_loss is None:
            # First epoch
            self.min_loss = val_loss
            self.save_checkpoint(self.model, fold, epoch, state="best" ,current_loss = val_loss)
            return False

        elif val_loss > self.min_loss - self.torlance:
            # Loss is not decreasing
            self.counter += 1
            print(f'\tEarly stopping patience: [{self.counter}/{self.patience}]')
            # still save every 'save_every' epochs
            if epoch % self.save_every == self.save_every - 1:
                self.save_checkpoint(self.model, fold, epoch, current_loss = val_loss)

            if self.counter >= self.patience:
                print("Out of patience. Early stopping.")
                self.save_checkpoint(self.model, fold, epoch, state="last", current_loss = val_loss)
                return True
        elif val_loss > self.min_loss:
            # Loss is decreasing but not enough
            print("Loss is decreasing but not exceed the threshold.")
            self.min_loss = val_loss
            self.save_checkpoint(self.model, fold, epoch, state="best", current_loss=val_loss)
            self.counter += 1
            return False
        else:
            print("Loss is decreasing and exceed the threshold.")
            self.min_loss = val_loss
            self.save_checkpoint(self.model, fold, epoch, state="best", current_loss=val_loss)
            self.counter = 0
            return False

    def save_checkpoint(self, model, fold, epoch, state=None, current_loss=None):
        save_path = self.path
        try:
            if state == "best":
                torch.save(model.state_dict(), save_path+"/best.pth")
                print(f"\tSaving best model at {save_path}, with loss: {current_loss:.4f}")
            elif state == 'last':
                torch.save(model.state_dict(), save_path+"/last.pth")
                print(f"\tSaving last model at {save_path}, with loss: {current_loss:.4f}")
            else:
                torch.save(model.state_dict(), save_path+f"/{fold + 1}_{epoch + 1}.pth")
                print(f"\tRegular saving at {save_path}, with loss: {current_loss:.4f}")
        except:
            print("[Error] Saving failed.")


class MyLoss(nn.MSELoss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        pearson_loss = 1 / error.pearson_coef - 1 
        spearman_loss = 1 / error.spearman_coef - 1
        corr_loss = pearson_loss + spearman_loss
        mse_loss = error.mse_loss
        
        loss = mse_loss + corr_loss
        
        return loss
    
def train_my_model(model, dataset, CONSTANTS,
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
            for speech_input, info_dict in tqdm(train_loader):
                speech_input = speech_input.to(device)
                score = info_dict["score"].to(torch.float32).to(device)
                pred = model(speech_input, info_dict)
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
                for speech_input, info_dict in tqdm(val_loader):
                    speech_input = speech_input.to(device)
                    score = info_dict["score"].to(torch.float32).to(device)
                    pred = model(speech_input, info_dict)
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
    
    
