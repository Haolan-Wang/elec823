import nemo.collections.asr as nemo_asr
from torch import nn
from data_process import ListenerInfo
from utils import *
from interpolate import get_interpolated_audiogram
CONSTANTS = InitializationTrain(verbose=False)
device = CONSTANTS.device


class MappingLayer(nn.Module):
    def __init__(self):
        super(MappingLayer, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return 1 / (1 + torch.exp(self.a * x + self.b))

class BetterEar(nn.Module):
    def __init__(self):
        super(BetterEar, self).__init__()
        self.a = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x, y):
        return self.a * x + self.b * y

import torch
import torch.nn as nn

class HearingImpairment(nn.Module):
    """Input: (batch_size, 160, 608)
        Output: (batch_size, 512, 151)
    """
    def __init__(self):
        super(HearingImpairment, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(160, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 37, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x.unsqueeze(-1).repeat(1, 1, 151)

class WordConfidence(nn.Module):
    """Word confidence model: Conformer + Linear predictor + exp mapping
        input(_, _, mono_path)
    """

    def __init__(self):
        super(WordConfidence, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )    
        # Freeze ASR model
        for param in self.asr_model.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(in_features=10, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        self.mapping = MappingLayer()
        self.better_ear = BetterEar()

    def forward(self, speech_l, speech_r, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence_l, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[0], return_hypotheses=True, batch_size=32
        )
        confidence_r, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[1], return_hypotheses=True, batch_size=32
        )
        confidence_l = [confidence_l[i].word_confidence for i in range(len(confidence_l))]
        confidence_r = [confidence_r[i].word_confidence for i in range(len(confidence_r))]
        # padding and truncating to 10: output shape [B, 10]
        confidence_l = torch.stack(
            list(map(self.truncate_and_pad, confidence_l)), dim=0
        ).to(device)
        confidence_r = torch.stack(
            list(map(self.truncate_and_pad, confidence_r)), dim=0
        ).to(device)
        pred_l = self.mapping(self.predictor(confidence_l))
        pred_r = self.mapping(self.predictor(confidence_r))
        
        # Better ear and mapping
        avg_pred = (pred_l + pred_r) / 2
        pred = self.mapping(avg_pred)
        # pred = self.mapping(self.better_ear(pred_l, pred_r))

        return pred

    def truncate_and_pad(self, tensor):
        FIXED_LENGTH = 10
        if type(tensor) is not torch.Tensor:
            tensor = torch.tensor(tensor)
        current_length = tensor.size(0)

        # If the tensor length is greater than the target length, truncate it
        if current_length > FIXED_LENGTH:
            tensor = tensor[:FIXED_LENGTH]
        # If the tensor length is less than the target length, pad it with zeros
        elif current_length < FIXED_LENGTH:
            pad_size = FIXED_LENGTH - current_length
            padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat((tensor, padding), dim=0)
        return tensor


class EncoderPredictor(nn.Module):
    def __init__(self):
        super(EncoderPredictor, self).__init__()
        pretrained_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        )
        self.logmel = pretrained_model.preprocessor
        self.conformer_encoder = pretrained_model.encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()
        self.better_ear = BetterEar()

    def forward(self, speech_l, speech_r, meta_data):
        # left and right [batch_size, 96000]
        # mel_out: [3, 80, 608]
        # speech_l = speech_input[0]
        # speech_r = speech_input[1]
        
        mel_feature_l, mel_feature_length = self.logmel(
            input_signal=speech_l.to(device),
            length=torch.full((speech_l.shape[0],), speech_l.shape[1]).to(device),
        )
        mel_feature_r, mel_feature_length = self.logmel(
            input_signal=speech_r.to(device),
            length=torch.full((speech_r.shape[0],), speech_r.shape[1]).to(device),
        )
        encoded_l, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature_l.to(device),
            length=torch.full((mel_feature_l.shape[0],), mel_feature_l.shape[2]).to(device),
        )
        encoded_r, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature_r.to(device),
            length=torch.full((mel_feature_r.shape[0],), mel_feature_r.shape[2]).to(device),
        )
        
        # encoder out : [32, 512, 151]
        pred_l = self.predictor(encoded_l.contiguous().view(-1, 512 * 151))
        pred_r = self.predictor(encoded_r.contiguous().view(-1, 512 * 151))
        
        # Better ear and mapping self.better_ear(pred_l, pred_r)
        avg_pred = (pred_l + pred_r) / 2
        pred = self.mapping(avg_pred)

        return pred
    

class EncoderPredictorHI(nn.Module):
    def __init__(self):
        super(EncoderPredictorHI, self).__init__()
        pretrained_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        )
        self.logmel = pretrained_model.preprocessor
        self.hearing_impairment = nn.Sequential(
            nn.Conv1d(160, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 80, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conformer_encoder = pretrained_model.encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()
        self.better_ear = BetterEar()

    def forward(self, speech_l, speech_r, meta_data):
        # left and right [batch_size, 96000]
        listener_info = ListenerInfo(meta_data['listener'])
        audiogram_l = [listener_info.info[i]['audiogram_l'] for i in range(len(listener_info.info))]
        audiogram_r = [listener_info.info[i]['audiogram_r'] for i in range(len(listener_info.info))]
        audiogram_cfs = [listener_info.info[i]['audiogram_cfs'] for i in range(len(listener_info.info))]
        
        interpolated_audiogram_l = get_interpolated_audiogram(audiogram_l, audiogram_cfs)
        interpolated_audiogram_r = get_interpolated_audiogram(audiogram_r, audiogram_cfs)
        interpolated_audiogram_l = torch.tensor(interpolated_audiogram_l, dtype=torch.float).to(device)
        interpolated_audiogram_r = torch.tensor(interpolated_audiogram_r, dtype=torch.float).to(device)
        # interpolated_audiogram: [B, 80] 
        # Repeat to [B, 80, 608]
        interpolated_audiogram_l = interpolated_audiogram_l.unsqueeze(-1).repeat(1, 1, 601)
        interpolated_audiogram_r = interpolated_audiogram_r.unsqueeze(-1).repeat(1, 1, 601)
        
        # mel_out: [B, 80, 608]
        mel_feature_l, mel_feature_length = self.logmel(
            input_signal=speech_l.to(device),
            length=torch.full((speech_l.shape[0],), speech_l.shape[1]).to(device),
        )
        mel_feature_r, mel_feature_length = self.logmel(
            input_signal=speech_r.to(device),
            length=torch.full((speech_r.shape[0],), speech_r.shape[1]).to(device),
        )
        concat_mel_feature_l = torch.cat((mel_feature_l, interpolated_audiogram_l), dim=1) # [B, 160, 608]
        concat_mel_feature_r = torch.cat((mel_feature_r, interpolated_audiogram_r), dim=1)
        
        # impaired_feature: [B, 80, 608]
        impaired_feature_l = self.hearing_impairment(concat_mel_feature_l)
        impaired_feature_r = self.hearing_impairment(concat_mel_feature_r)
        
        encoded_l, encoder_length = self.conformer_encoder(
            audio_signal=impaired_feature_l.to(device),
            length=torch.full((impaired_feature_l.shape[0],), impaired_feature_l.shape[2]).to(device),
        )
        encoded_r, encoder_length = self.conformer_encoder(
            audio_signal=impaired_feature_r.to(device),
            length=torch.full((impaired_feature_r.shape[0],), impaired_feature_r.shape[2]).to(device),
        )
        
        # encoder out : [32, 512, 151]
        pred_l = self.predictor(encoded_l.contiguous().view(-1, 512 * 151))
        pred_r = self.predictor(encoded_r.contiguous().view(-1, 512 * 151))
        
        # Better ear and mapping self.better_ear(pred_l, pred_r)
        avg_pred = (pred_l + pred_r) / 2
        # pred = self.mapping(avg_pred)

        return avg_pred
    
    
class EncoderPredictorHI_v2(nn.Module):
    """
        The interference of hearing loss acts on encoded features.
    """
    def __init__(self):
        super(EncoderPredictorHI_v2, self).__init__()
        pretrained_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        )
        self.logmel = pretrained_model.preprocessor
        self.hearing_impairment = nn.Sequential(
            nn.Conv1d(160, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 80, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conformer_encoder = pretrained_model.encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=1184 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()
        self.better_ear = BetterEar()

    def forward(self, speech_l, speech_r, meta_data):
        # left and right [batch_size, 96000]
        listener_info = ListenerInfo(meta_data['listener'])
        audiogram_l = [listener_info.info[i]['audiogram_l'] for i in range(len(listener_info.info))]
        audiogram_r = [listener_info.info[i]['audiogram_r'] for i in range(len(listener_info.info))]
        audiogram_cfs = [listener_info.info[i]['audiogram_cfs'] for i in range(len(listener_info.info))]
        
        interpolated_audiogram_l = get_interpolated_audiogram(audiogram_l, audiogram_cfs)
        interpolated_audiogram_r = get_interpolated_audiogram(audiogram_r, audiogram_cfs)
        interpolated_audiogram_l = torch.tensor(interpolated_audiogram_l, dtype=torch.float).to(device)
        interpolated_audiogram_r = torch.tensor(interpolated_audiogram_r, dtype=torch.float).to(device)
        # interpolated_audiogram: [B, 80] 
        # Repeat to [B, 80, 151]
        interpolated_audiogram_l = interpolated_audiogram_l.unsqueeze(-1).repeat(1, 1, 151)
        interpolated_audiogram_r = interpolated_audiogram_r.unsqueeze(-1).repeat(1, 1, 151)
        
        # mel_out: [B, 80, 601]
        mel_feature_l, mel_feature_length = self.logmel(
            input_signal=speech_l.to(device),
            length=torch.full((speech_l.shape[0],), speech_l.shape[1]).to(device),
        )
        mel_feature_r, mel_feature_length = self.logmel(
            input_signal=speech_r.to(device),
            length=torch.full((speech_r.shape[0],), speech_r.shape[1]).to(device),
        )
        
        encoded_l, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature_l.to(device),
            length=torch.full((mel_feature_l.shape[0],), mel_feature_l.shape[2]).to(device),
        )
        encoded_r, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature_r.to(device),
            length=torch.full((mel_feature_r.shape[0],), mel_feature_r.shape[2]).to(device),
        )
        # encoder out : [B, 512, 151]
        concatenated_l = torch.cat((encoded_l, interpolated_audiogram_l), dim=1)
        concatenated_r = torch.cat((encoded_r, interpolated_audiogram_r), dim=1)
        # [B, 592, 151] -> [B, 1184, 151]
        con_features = torch.cat((concatenated_l, concatenated_r), dim=1)
        pred = self.predictor(con_features.contiguous().view(-1, 1184 * 151))
        # pred_l = self.predictor(concatenated_l.contiguous().view(-1, 592 * 151))
        # pred_r = self.predictor(concatenated_r.contiguous().view(-1, 592 * 151))
        
        # Better ear and mapping self.better_ear(pred_l, pred_r)
        # avg_pred = (pred_l + pred_r) / 2
        pred = self.mapping(pred)

        return pred
    
    
class EncoderPredictorHI_v3(nn.Module):
    def __init__(self):
        super(EncoderPredictorHI_v3, self).__init__()
        pretrained_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        )
        self.logmel = pretrained_model.preprocessor
        self.hearing_impairment = HearingImpairment()
        self.conformer_encoder = pretrained_model.encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=2048 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()
        self.better_ear = BetterEar()

    def forward(self, speech_l, speech_r, meta_data):
        # left and right [batch_size, 96000]
        listener_info = ListenerInfo(meta_data['listener'])
        audiogram_l = [listener_info.info[i]['audiogram_l'] for i in range(len(listener_info.info))]
        audiogram_r = [listener_info.info[i]['audiogram_r'] for i in range(len(listener_info.info))]
        audiogram_cfs = [listener_info.info[i]['audiogram_cfs'] for i in range(len(listener_info.info))]
        
        interpolated_audiogram_l = get_interpolated_audiogram(audiogram_l, audiogram_cfs)
        interpolated_audiogram_r = get_interpolated_audiogram(audiogram_r, audiogram_cfs)
        interpolated_audiogram_l = torch.tensor(interpolated_audiogram_l, dtype=torch.float).to(device)
        interpolated_audiogram_r = torch.tensor(interpolated_audiogram_r, dtype=torch.float).to(device)
        # interpolated_audiogram: [B, 80] 
        # Repeat to [B, 80, 608]
        interpolated_audiogram_l = interpolated_audiogram_l.unsqueeze(-1).repeat(1, 1, 601)
        interpolated_audiogram_r = interpolated_audiogram_r.unsqueeze(-1).repeat(1, 1, 601)
        
        # mel_out: [B, 80, 608]
        mel_feature_l, mel_feature_length = self.logmel(
            input_signal=speech_l.to(device),
            length=torch.full((speech_l.shape[0],), speech_l.shape[1]).to(device),
        )
        mel_feature_r, mel_feature_length = self.logmel(
            input_signal=speech_r.to(device),
            length=torch.full((speech_r.shape[0],), speech_r.shape[1]).to(device),
        )
        # encoder out : [B, 512, 151]
        encoded_l, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature_l.to(device),
            length=torch.full((mel_feature_l.shape[0],), mel_feature_l.shape[2]).to(device),
        )
        encoded_r, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature_r.to(device),
            length=torch.full((mel_feature_r.shape[0],), mel_feature_r.shape[2]).to(device),
        )
        
        concat_mel_feature_l = torch.cat((mel_feature_l, interpolated_audiogram_l), dim=1) # [B, 160, 608]
        concat_mel_feature_r = torch.cat((mel_feature_r, interpolated_audiogram_r), dim=1)
        # impaired_feature: [B, 512, 151]
        impaired_feature_l = self.hearing_impairment(concat_mel_feature_l)
        impaired_feature_r = self.hearing_impairment(concat_mel_feature_r)
        
        # [encoded_l, impaired_feature_l, encoded_r, impaired_feature_r] -> [B, 2048, 151
        concat_feature = torch.cat((encoded_l, impaired_feature_l, encoded_r, impaired_feature_r), dim=1) # [B, 2048, 151]
        pred = self.predictor(concat_feature.contiguous().view(-1, 2048 * 151))

        pred = self.mapping(pred)

        return pred