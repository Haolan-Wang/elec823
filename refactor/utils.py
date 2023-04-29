import gc
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torchmetrics import SpearmanCorrCoef


class InitializationTrain:
    def __init__(
            self,
            model_name=None,
            verbose=False,
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

        if self.verbose:
            print("Model Name: ", model_name)
        self.path_check(model_name, log_path, save_path)

        self.seed_everything(self.seed)

    def path_check(self, model_name, log_path, save_path):
        if self.verbose:
            print("Checking paths...")
        if model_name is None:
            if self.verbose:
                print(
                    "Warning: Model name is not specified. It's EXPECTED if you are initializing a 'models.py'. Otherwise Your model will be saved in 'checkpoints/00_TMP_MODEL' folder.")
            self.MODEL_NAME = '00_TMP_MODEL'
        else:
            self.MODEL_NAME = model_name

        self.LOG_PATH = log_path + self.MODEL_NAME
        self.SAVE_PATH = save_path + self.MODEL_NAME
        self.last_output = "/home/ubuntu/elec823/last_output/"
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
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
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


class ErrorCal:
    """Input y_true and y_pred when initializing, return pearson, spearman, mse loss"""

    def __init__(self, y_true, y_pred):
        self.mse_loss = torch.nn.functional.mse_loss(y_true, y_pred)
        self.mae_loss = torch.nn.functional.l1_loss(y_true, y_pred)
        self.pearson_coef = torch.corrcoef(torch.stack((y_true, y_pred)))[0, 1]
        self.spearman = SpearmanCorrCoef()
        self.spearman_coef = self.spearman(y_true, y_pred)
        self.ccc = self.concordance_coef(y_true, y_pred)
    
    def concordance_coef(self, y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        
        y_true_var = torch.var(y_true, unbiased=False)
        y_pred_var = torch.var(y_pred, unbiased=False)
        
        covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
        
        ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean)**2)
        return ccc
        


class EarlyStop:
    """Usage: 
        Initialize: early_stop = EarlyStop(model, path, patience=5, save_every=5, tolerance=0)
        Evaluation: early_stop(val_loss, fold, epoch)
        Return: True if early stopping, False if not
    """

    def __init__(self, model, path, patience=10, save_every=20, torlance=0):
        self.model = model
        self.path = path
        self.save_every = save_every
        self.patience = patience
        self.counter = 0
        self.tolerance = torlance
        self.min_loss = None

    def __call__(self, val_loss, fold, epoch):
        if self.min_loss is None:
            # First epoch
            self.min_loss = val_loss
            self.save_checkpoint(self.model, fold, epoch, state="best", current_loss=val_loss)
            return False

        elif val_loss > self.min_loss - self.tolerance:
            # Loss is not decreasing
            self.counter += 1
            print(f'\tEarly stopping patience: [{self.counter}/{self.patience}]')
            # still save every 'save_every' epochs
            if epoch % self.save_every == self.save_every - 1:
                self.save_checkpoint(self.model, fold, epoch, current_loss=val_loss)

            if self.counter >= self.patience:
                print("Out of patience. Early stopping.")
                self.save_checkpoint(self.model, fold, epoch, state="last", current_loss=val_loss)
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
                torch.save(model.state_dict(), save_path + "/best.pt")
                print(f"\tSaving best model at {save_path}, with loss: {current_loss:.4f}")
            elif state == 'last':
                torch.save(model.state_dict(), save_path + "/last.pt")
                print(f"\tSaving last model at {save_path}, with loss: {current_loss:.4f}")
            else:
                torch.save(model.state_dict(), save_path + f"/{fold + 1}_{epoch + 1}.pt")
                print(f"\tRegular saving at {save_path}, with loss: {current_loss:.4f}")
        except:
            print("[Error] Saving failed.")

def truncate_and_pad(tensor, desired_length):
    if type(tensor) is not torch.Tensor:
        tensor = torch.tensor(tensor)
    current_length = tensor.size(0)

    # If the tensor length is greater than the target length, truncate it
    if current_length > desired_length:
        tensor = tensor[:desired_length]
    # If the tensor length is less than the target length, pad it with zeros
    elif current_length < desired_length:
        pad_size = desired_length - current_length
        padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return tensor