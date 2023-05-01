import nemo.collections.asr as nemo_asr
import torch
from torch import nn
import torch.nn as nn
from data_process import ListenerInfo
from utils import *
from interpolate import get_interpolated_audiogram
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
CONSTANTS = InitializationTrain(verbose=False)
device = CONSTANTS.device

class MappingLayer(nn.Module):
    def __init__(self):
        super(MappingLayer, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return 1 / (1 + torch.exp(self.a * x + self.b))



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x, lengths):
        # Forward propagate LSTM
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = torch.stack([out[i, lengths[i] - 1] for i in range(out.size(0))])
        out = self.fc(out)
        return out


class JointPredictor(nn.Module):
    def __init__(self, time_step, encoded_feature_size, wc_feature_size):
        super(JointPredictor, self).__init__()
        self.time_step = time_step
        self.encoded_feature_size = encoded_feature_size
        self.wc_feature_size = wc_feature_size



class EncoderPredictor_BE(nn.Module):
    # This BE: max
    
    def __init__(self):
        super(EncoderPredictor_BE, self).__init__()
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_large"
        )
        self.logmel = asr_model.preprocessor
        self.conformer_encoder = asr_model.encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()

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
        
        # encoder out : [B, 512, 151]
        pred_l = self.predictor(encoded_l.contiguous().view(-1, 512 * 151))
        pred_r = self.predictor(encoded_r.contiguous().view(-1, 512 * 151))
        

        stacked_pred = torch.stack([pred_l, pred_r], dim=1).squeeze(2)
        better_ear, _ = torch.max(stacked_pred, dim=1)
        pred = self.mapping(better_ear)

        return pred
    
 
    
class EncoderPredictor_Fusion(nn.Module):
    # Fusing: predector use concat of left and right
    
    def __init__(self):
        super(EncoderPredictor_Fusion, self).__init__()
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_large"
        )
        self.logmel = asr_model.preprocessor
        self.conformer_encoder = asr_model.encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=1024 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()

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
        
        # encoder out : [B, 512, 151]
        concat_feature = torch.cat([encoded_l, encoded_r], dim=1)
        pred = self.predictor(concat_feature.contiguous().view(-1, 1024 * 151))
        pred = self.mapping(pred)

        return pred
    
    

class WordConfidence_BE(nn.Module):
    """Word confidence model: Conformer + Linear predictor + exp mapping
        input(_, _, mono_path)
    """

    def __init__(self):
        super(WordConfidence_BE, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )    
        # Freeze ASR model
        for param in self.asr_model.parameters():
            param.requires_grad = False

        self.lstm_model = LSTMModel(input_size=1, hidden_size=64, output_size=64, num_layers=1)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
            )
        self.mapping = MappingLayer()
        self.truncate_and_pad = truncate_and_pad

    def forward(self, speech_l, speech_r, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence_l, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[0], return_hypotheses=True, batch_size=32
        )
        confidence_r, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[1], return_hypotheses=True, batch_size=32
        )
        # list of tensor
        confidence_l = [torch.tensor([0.]) if torch.tensor(confidence_l[i].word_confidence).nelement() == 0 else torch.tensor(confidence_l[i].word_confidence) for i in range(len(confidence_l))]
        confidence_r = [torch.tensor([0.]) if torch.tensor(confidence_r[i].word_confidence).nelement() == 0 else torch.tensor(confidence_r[i].word_confidence) for i in range(len(confidence_r))]
        len_l = [confidence_l[i].shape[0] for i in range(len(confidence_l))]
        len_r = [confidence_r[i].shape[0] for i in range(len(confidence_r))]
        padded_confidence_l = pad_sequence(confidence_l, batch_first=True)
        padded_confidence_r = pad_sequence(confidence_r, batch_first=True)

        # BE
        pred_l = self.predictor(self.lstm_model(padded_confidence_l.to(device), len_l))
        pred_r = self.predictor(self.lstm_model(padded_confidence_r.to(device), len_r))
        stacked_pred = torch.stack([pred_l, pred_r], dim=1)
        better_ear, _ = torch.max(stacked_pred, dim=1)
        pred = self.mapping(better_ear)

        return pred
    


class WordConfidence_Fusion(nn.Module):
    def __init__(self):
        super(WordConfidence_Fusion, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )    
        # Freeze ASR model
        for param in self.asr_model.parameters():
            param.requires_grad = False

        self.lstm_model = LSTMModel(input_size=1, hidden_size=128, output_size=64, num_layers=2)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=128, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
            )
        self.mapping = MappingLayer()
        self.truncate_and_pad = truncate_and_pad

    def forward(self, speech_l, speech_r, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence_l, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[0], return_hypotheses=True, batch_size=32
        )
        confidence_r, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[1], return_hypotheses=True, batch_size=32
        )
        # list of tensor
        confidence_l = [torch.tensor([0.]) if torch.tensor(confidence_l[i].word_confidence).nelement() == 0 else torch.tensor(confidence_l[i].word_confidence) for i in range(len(confidence_l))]
        confidence_r = [torch.tensor([0.]) if torch.tensor(confidence_r[i].word_confidence).nelement() == 0 else torch.tensor(confidence_r[i].word_confidence) for i in range(len(confidence_r))]
        len_l = [confidence_l[i].shape[0] for i in range(len(confidence_l))]
        len_r = [confidence_r[i].shape[0] for i in range(len(confidence_r))]
        padded_confidence_l = pad_sequence(confidence_l, batch_first=True)
        padded_confidence_r = pad_sequence(confidence_r, batch_first=True)

        # Fusion
        lstm_feature_l = self.lstm_model(padded_confidence_l.to(device), len_l)
        lstm_feature_r = self.lstm_model(padded_confidence_r.to(device), len_r)
        pred = self.predictor(torch.cat([lstm_feature_l, lstm_feature_r], dim=1).view(-1, 128))
        pred = self.mapping(pred)

        return pred



class JointModel(nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )
        self.logmel = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large").preprocessor
        self.conformer_encoder = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large").encoder
        # Freeze ASR model
        for param in self.asr_model.parameters():
            param.requires_grad = False

        self.lstm_model = LSTMModel(input_size=1, hidden_size=128, output_size=64, num_layers=2)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=1152 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.mapping = MappingLayer()

    def forward(self, speech_l, speech_r, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence_l, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[0], return_hypotheses=True, batch_size=16
        )
        confidence_r, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[1], return_hypotheses=True, batch_size=16
        )
        # list of tensor
        confidence_l = [torch.tensor([0.]) if torch.tensor(confidence_l[i].word_confidence).nelement() == 0 else torch.tensor(confidence_l[i].word_confidence) for i in range(len(confidence_l))]
        confidence_r = [torch.tensor([0.]) if torch.tensor(confidence_r[i].word_confidence).nelement() == 0 else torch.tensor(confidence_r[i].word_confidence) for i in range(len(confidence_r))]
        len_l = [confidence_l[i].shape[0] for i in range(len(confidence_l))]
        len_r = [confidence_r[i].shape[0] for i in range(len(confidence_r))]
        padded_confidence_l = pad_sequence(confidence_l, batch_first=True)
        padded_confidence_r = pad_sequence(confidence_r, batch_first=True)

        # WC out [B,64] -> [B,64,151]
        lstm_feature_l = self.lstm_model(padded_confidence_l.to(device), len_l).unsqueeze(-1).repeat(1, 1, 151)
        lstm_feature_r = self.lstm_model(padded_confidence_r.to(device), len_r).unsqueeze(-1).repeat(1, 1, 151)
        
        # =============Encoder================
        
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
        # concat feature : [B, (512+64)*2=1152, 151]
        concat_feature = torch.cat([lstm_feature_l, encoded_l, lstm_feature_r, encoded_r], dim=1)
        pred = self.predictor(concat_feature.view(-1, 1152 * 151))
        pred = self.mapping(pred)

        return pred




class CompareWordConfidence(nn.Module):
    def __init__(self):
        super(CompareWordConfidence, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, word_confidence, valid_len):
        """
        Args:
            word_confidence (torch.Tensor): [Batch, 10]
            valid_len (torch.Tensor): [Batch, 1]
        """
        greater_than_thr = (word_confidence > self.threshold).sum(dim=1, keepdim=True).float().requires_grad_()
        output = torch.div(greater_than_thr, valid_len.view(-1, 1))
        return output
    


class WordConfidence_CMP(nn.Module):
    def __init__(self):
        super(WordConfidence_CMP, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        ) 
        # Freeze ASR model
        for param in self.asr_model.parameters():
            param.requires_grad = False
        self.mapping = MappingLayer()
        self.truncate_and_pad = truncate_and_pad
        self.cmp_wc = CompareWordConfidence()
    def forward(self, speech_l, speech_r, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence_l, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[0], return_hypotheses=True, batch_size=32
        )
        confidence_r, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path[1], return_hypotheses=True, batch_size=32
        )
        # list of tensor
        confidence_l = [torch.tensor([0.]) if torch.tensor(confidence_l[i].word_confidence).nelement() == 0 else torch.tensor(confidence_l[i].word_confidence) for i in range(len(confidence_l))]
        confidence_r = [torch.tensor([0.]) if torch.tensor(confidence_r[i].word_confidence).nelement() == 0 else torch.tensor(confidence_r[i].word_confidence) for i in range(len(confidence_r))]
        len_l = torch.tensor([confidence_l[i].shape[0] for i in range(len(confidence_l))]).to(device)
        len_r = torch.tensor([confidence_r[i].shape[0] for i in range(len(confidence_r))]).to(device)
        padded_confidence_l = pad_sequence(confidence_l, batch_first=True).to(device)
        padded_confidence_r = pad_sequence(confidence_r, batch_first=True).to(device)
        pred_l = self.cmp_wc(padded_confidence_l, len_l)
        pred_r = self.cmp_wc(padded_confidence_r, len_r)

        # Better Ear
        stacked_pred = torch.stack([pred_l, pred_r], dim=1)
        better_ear, _ = torch.max(stacked_pred, dim=1)
        # pred = self.mapping(better_ear)

        return pred_l

