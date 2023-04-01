
from copy import deepcopy

import nemo.collections.asr as nemo_asr

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from nemo.core.neural_types import *
import sys
from nemo.collections.asr.metrics.rnnt_wer import RNNTWER, RNNTDecoding, RNNTDecodingConfig
from utils import *
from data_process import ListenerInfo

CONSTANTS = InitializationTrain(verbose=True)
device = CONSTANTS.device

class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioFeatureExtractor, self).__init__()
        self.mel_bank = nemo_asr.modules.AudioToMFCCPreprocessor()
        self.wav2vec_feature_extractor = Wav2Vec2Model.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        ).feature_extractor
        self.cnn_mel = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.cnn_wav2vec = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        # [B, D=512, T =299] -> [B, D=512, T=601]
        self.transpose_w2v = nn.ConvTranspose1d(
            in_channels=512, out_channels=512, kernel_size=5, stride=2
        )

    def forward(self, speech_input: torch.Tensor) -> torch.Tensor:
        """return wav2vec feature and mel feature in the shape [B, D, T]

        Args:
        speech_input: mono only with size [B, T=96000]. If stereo, make it mono first or pass a signal channel 

        Returns:
        concatenated_feature: [B, D=1024, T=601]
        """
        mel_feature, mel_feature_length = self.mel_bank(
            input_signal=speech_input.to(device),
            length=torch.full((speech_input.shape[0],), speech_input.shape[1]).to(
                device
            ),
        )  # mel: [3, 80, 608], mfcc: [3, 64, 601]
        # print("mel_feature", mel_feature.shape)
        mel_feature = self.cnn_mel(mel_feature).to(device)  # [B, D=512, T=601]
        # print("mel_feature-cnn", mel_feature.shape)

        wav2vec_feature = self.wav2vec_feature_extractor(speech_input).to(device)
        # print("wav2vec_feature", wav2vec_feature.shape)
        wav2vec_feature = self.cnn_wav2vec(wav2vec_feature).to(
            device
        )  # [B, D=512, T=299]
        # print("wav2vec_feature", wav2vec_feature.shape)
        wav2vec_feature = self.transpose_w2v(wav2vec_feature).to(
            device
        )  # [B, D=512, T=601]
        # print("wav2vec_feature", wav2vec_feature.shape)

        concatenated_feature = torch.cat((mel_feature, wav2vec_feature), dim=1).to(
            device
        )  # [B, D=1024, T=601]

        return concatenated_feature  # [B, D=1024, T=601]
class ListenerFeatureExtractor(nn.Module):
    def __init__(self):
        super(ListenerFeatureExtractor, self).__init__()
        self.fc_audiogram = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            # [B, 32]
        )

    def forward(self, listener):
        info = ListenerInfo(listener).info
        audiogram = torch.tensor(
            [info[i]["audiogram"] for i in range(len(info))], dtype=torch.float32
        ).to(
            device
        )  # [B, 16]
        audiogram_feature = (
            self.fc_audiogram(audiogram).to(device).unsqueeze(2).repeat(1, 1, 601)
        )
        # in this case we only use audiogram feature
        listener_feature = audiogram_feature

        return listener_feature  # [B, D=32, T=601]


class NTT_01(nn.Module):
    """NTT_01: Only use audiogram as listener feature
    """

    def __init__(self):
        super(NTT_01, self).__init__()
        self.listener_feature_extractor = ListenerFeatureExtractor()  # .to(device)
        self.audio_feature_extractor = AudioFeatureExtractor()  # .to(device)
        self.audio_merger = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2),
        )
        self.cnn = nn.Conv1d(160, 80, kernel_size=5, stride=1, padding=2)
        self.conformer_encoder = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        ).encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, speech_input, meta_data):
        # audio feature size = [B, D=1024, T=601]
        # print("speech_input: ", speech_input.shape,
        #       speech_input[:, 0, :].shape)
        audio_feature_l = self.audio_feature_extractor(speech_input[:, 0, :]).to(device)
        audio_feature_r = self.audio_feature_extractor(speech_input[:, 1, :]).to(device)
        # print("audio features: ", audio_feature_l.shape, audio_feature_r.shape)

        # merged audio feature size = [B, D=128, T=601]
        audio_feature = self.audio_merger(
            torch.cat((audio_feature_l, audio_feature_r), dim=1)
        ).to(device)
        # print("audio feature: ", audio_feature.shape)

        # listener feature size = [B, D=32, T=601]
        listener_feature = self.listener_feature_extractor(meta_data["listener"]).to(
            device
        )
        # print("listener_feature:", listener_feature.shape)

        # concatenated feature size = [B, D=32+128=160, T=601]
        concatenated_feature = torch.cat((listener_feature, audio_feature), dim=1).to(
            device
        )
        # print("concatenated_feature:", concatenated_feature.shape)

        # conformer_encoder
        # use cnn to reduce the dimension of concatenated_feature to 160-80 first
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=self.cnn(concatenated_feature).to(device),
            length=torch.full(
                (concatenated_feature.shape[0],), concatenated_feature.shape[2]
            ).to(device),
        )

        # transform encoder_out [B, D=512, T=151] -> [B, D(512)*T(151)]
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 151))

        return pred


class AudioOnlyModel(nn.Module):
    def __init__(self):
        super(AudioOnlyModel, self).__init__()
        self.audio_feature_extractor = AudioFeatureExtractor()  # .to(device)
        self.audio_merger = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2),
        )
        self.cnn = nn.Conv1d(128, 80, kernel_size=5, stride=1, padding=2)
        self.conformer_encoder = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        ).encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, speech_input, meta_data):
        # merged audio feature size = [B, D=1024+1024->128, T=601]
        audio_feature_l = self.audio_feature_extractor(speech_input[:, 0, :]).to(device)
        audio_feature_r = self.audio_feature_extractor(speech_input[:, 1, :]).to(device)
        audio_feature = self.audio_merger(
            torch.cat((audio_feature_l, audio_feature_r), dim=1)
        ).to(device)

        # Then audio feature need to reduce the dimension to 128-80 by a CNN layer
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=self.cnn(audio_feature).to(device),
            length=torch.full((audio_feature.shape[0],), audio_feature.shape[2]).to(
                device
            ),
        )

        # transform encoder_out [B, D=512, T=151] -> [B, D(512)*T(151)]
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 151))

        return pred


class MFCCOnlyModel(nn.Module):
    def __init__(self):
        super(MFCCOnlyModel, self).__init__()
        self.mfcc = nemo_asr.modules.AudioToMFCCPreprocessor()
        # self.audio_feature_extractor = AudioFeatureExtractor()  # .to(device)
        self.audio_merger = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2),
        )
        self.cnn = nn.Conv1d(128, 80, kernel_size=5, stride=1, padding=2)
        self.conformer_encoder = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        ).encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, speech_input, meta_data):
        speech_input_l = speech_input[:, 0, :]
        speech_input_r = speech_input[:, 1, :]
        # mfcc_out: [3, 64, 601]
        mfcc_feature_l, mfcc_feature_length_l = self.mfcc(
            input_signal=speech_input_l.to(device),
            length=torch.full((speech_input_l.shape[0],), speech_input_l.shape[1]).to(
                device
            ),
        )
        mfcc_feature_r, mfcc_feature_length_r = self.mfcc(
            input_signal=speech_input_r.to(device),
            length=torch.full((speech_input_r.shape[0],), speech_input_r.shape[1]).to(
                device
            ),
        )

        # audio_feature
        audio_feature = self.audio_merger(
            torch.cat((mfcc_feature_l, mfcc_feature_r), dim=1)
        ).to(device)

        # Then audio feature need to reduce the dimension to 128-80 by a CNN layer
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=self.cnn(audio_feature).to(device),
            length=torch.full((audio_feature.shape[0],), audio_feature.shape[2]).to(
                device
            ),
        )

        # transform encoder_out [B, D=512, T=151] -> [B, D(512)*T(151)]
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 151))

        return pred


class MelStereoModel(nn.Module):
    def __init__(self):
        super(MelStereoModel, self).__init__()
        self.logmel = nemo_asr.modules.AudioToMelSpectrogramPreprocessor()
        self.audio_merger = nn.Sequential(
            nn.Conv1d(160, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2),
        )
        # self.cnn = nn.Conv1d(128, 80, kernel_size=5, stride=1, padding=2)
        self.conformer_encoder = nemo_asr.modules.ConformerEncoder(
            feat_in=128,
            feat_out=-1,
            n_layers=18,
            d_model=512,
            subsampling_factor=4,
            subsampling_conv_channels=512,
            ff_expansion_factor=4,
            self_attention_model="rel_pos",
            n_heads=8,
            att_context_size=[-1, -1],
        )
        # self.conformer_encoder = nemo_asr.models.EncDecCTCModel.from_pretrained(
        #     "nvidia/stt_en_conformer_ctc_large").encoder
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 152, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, speech_input, meta_data):
        speech_input_l = speech_input[:, 0, :]
        speech_input_r = speech_input[:, 1, :]
        # mel_out: [3, 64, 608]
        mel_feature_l, mel_feature_length_l = self.logmel(
            input_signal=speech_input_l.to(device),
            length=torch.full((speech_input_l.shape[0],), speech_input_l.shape[1]).to(
                device
            ),
        )
        mel_feature_r, mel_feature_length_r = self.logmel(
            input_signal=speech_input_r.to(device),
            length=torch.full((speech_input_r.shape[0],), speech_input_r.shape[1]).to(
                device
            ),
        )
        # audio_feature [B, D=128, T=608]
        audio_feature = torch.cat((mel_feature_l, mel_feature_r), dim=1).to(device)

        # VERY SIMPLE MERGE
        # Conformer encoder parameter:
        # feat_in=128, n_layers=18, d_model=512

        # encoder_out [B, D=512, T=152](MEL-stereo)
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=audio_feature.to(device),
            length=torch.full((audio_feature.shape[0],), audio_feature.shape[2]).to(
                device
            ),
        )
        # print(encoder_out.shape)

        # pred = self.predictor(encoder_out.contiguous().view(-1, 512*151))
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 152))

        return pred


class MelMonoModel(nn.Module):
    def __init__(self):
        super(MelMonoModel, self).__init__()
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
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, speech_input, meta_data):
        speech_input_l = speech_input[:, 0, :]
        speech_input_r = speech_input[:, 1, :]
        mixed_speech = (speech_input_l + speech_input_r) / 2
        # mel_out: [3, 80, 608]
        mel_feature, mel_feature_length = self.logmel(
            input_signal=mixed_speech.to(device),
            length=torch.full((mixed_speech.shape[0],), mixed_speech.shape[1]).to(
                device
            ),
        )
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature.to(device),
            length=torch.full((mel_feature.shape[0],), mel_feature.shape[2]).to(device),
        )
        # encoder out : [32, 512, 151]
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 151))

        return pred
    

class MelMonoModelWithout(nn.Module):
    """Without Sigmoid layer"""
    def __init__(self):
        super(MelMonoModelWithout, self).__init__()
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

    def forward(self, speech_input, meta_data):
        speech_input_l = speech_input[:, 0, :]
        speech_input_r = speech_input[:, 1, :]
        mixed_speech = (speech_input_l + speech_input_r) / 2
        # mel_out: [3, 80, 608]
        mel_feature, mel_feature_length = self.logmel(
            input_signal=mixed_speech.to(device),
            length=torch.full((mixed_speech.shape[0],), mixed_speech.shape[1]).to(
                device
            ),
        )
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature.to(device),
            length=torch.full((mel_feature.shape[0],), mel_feature.shape[2]).to(device),
        )
        # encoder out : [32, 512, 151]
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 151))

        return pred

class MelMonoScratch(nn.Module):
    def __init__(self):
        super(MelMonoScratch, self).__init__()
        pretrained_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            "nvidia/stt_en_conformer_ctc_large"
        )
        self.logmel = pretrained_model.preprocessor
        self.conformer_encoder = nemo_asr.modules.ConformerEncoder(
            feat_in = 80,
            feat_out = -1,
            n_layers = 18,
            d_model = 512,
            subsampling = 'striding',
            subsampling_factor = 4,
            subsampling_conv_channels = 512,
            ff_expansion_factor = 4,
            self_attention_model = 'rel_pos',
            n_heads = 8,
            att_context_size = [-1, -1],
            xscaling = True,
            untie_biases = True,
            pos_emb_max_len = 5000,
            conv_kernel_size = 31,
            dropout = 0.1,
            dropout_emb = 0.0,
            dropout_att = 0.1,
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=512 * 151, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )
    

    def forward(self, speech_input, meta_data):
        speech_input_l = speech_input[:, 0, :]
        speech_input_r = speech_input[:, 1, :]
        mixed_speech = (speech_input_l + speech_input_r) / 2
        # mel_out: [3, 80, 608]
        mel_feature, mel_feature_length = self.logmel(
            input_signal=mixed_speech.to(device),
            length=torch.full((mixed_speech.shape[0],), mixed_speech.shape[1]).to(
                device
            ),
        )
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature.to(device),
            length=torch.full((mel_feature.shape[0],), mel_feature.shape[2]).to(device),
        )
        # encoder out : [32, 512, 151]
        pred = self.predictor(encoder_out.contiguous().view(-1, 512 * 151))

        return pred

class WordConfidence(nn.Module):
    """Word confidence model: Conformer + LSTM + Sigmoid
        input(_, _, mono_path)
    """
    def __init__(self):
        super(WordConfidence, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")
        self.bidirectional_lstm=nn.LSTM(
            input_size=10,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True
            )
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, speech_input, meta_data, mono_path):
        # output of asr model is [B, word_len]
        confidence, _ = self.asr_model.transcribe(paths2audio_files = mono_path, return_hypotheses = True, batch_size = 32)
        confidence = [confidence[i].word_confidence for i in range(len(confidence))]
        # padding and truncating to 10: output shape [B, 10]
        confidence = torch.stack(list(map(self.truncate_and_pad, confidence)), dim=0).to(device)
        lstm_out, (h_n, c_n) = self.bidirectional_lstm(confidence)
        pred = self.predictor(lstm_out)
        
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
    
class WordConfidenceLinear(WordConfidence):
    """Word confidence model: Conformer + Linear + Sigmoid
        input(_, _, mono_path)
    """
    def __init__(self):
        super(WordConfidenceLinear, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=10, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, speech_input, meta_data, mono_path):
        # output of asr model is [B, word_len]
        confidence, _ = self.asr_model.transcribe(paths2audio_files = mono_path, return_hypotheses = True, batch_size = 32)
        confidence = [confidence[i].word_confidence for i in range(len(confidence))]
        # padding and truncating to 10: output shape [B, 10]
        confidence = torch.stack(list(map(self.truncate_and_pad, confidence)), dim=0).to(device)
        pred = self.predictor(confidence)
        
        return pred


class Wav2vecPredictor(nn.Module):
    def __init__(self):
        super(Wav2vecPredictor, self).__init__()
        self.MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(self.MODEL_ID) 
        self.predictor = nn.Sequential(
            nn.Linear(in_features=299 * 1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )
        
    def forward(self, speech_input, meta_data):
        # Use mono audio
        # print(speech_input.shape)
        last_layer_features = self.wav2vec_model(speech_input.to(device)).last_hidden_state
        # out of wav2vec: [B, L, D] ([3, 299, 1024])
        pred = self.predictor(last_layer_features.contiguous().view(-1, 299 * 1024))
        
        return pred