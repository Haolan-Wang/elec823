import nemo.collections.asr as nemo_asr
from torch import nn

from utils import *

CONSTANTS = InitializationTrain(verbose=False)
device = CONSTANTS.device


class MappingLayer(nn.Module):
    def __init__(self):
        super(MappingLayer, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return 1 / (1 + torch.exp(self.a * x + self.b))
        # a_expanded = self.a.expand_as(x)
        # b_expanded = self.b.expand_as(x)
        # return 1 / (1 + torch.exp(a_expanded * x + b_expanded))


class WordConfidence(nn.Module):
    """Word confidence model: Conformer + Linear predictor + exp mapping
        input(_, _, mono_path)
    """

    def __init__(self):
        super(WordConfidence, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )    # Freeze ASR model
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

    def forward(self, speech_input, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path, return_hypotheses=True, batch_size=32
        )
        confidence = [confidence[i].word_confidence for i in range(len(confidence))]
        # padding and truncating to 10: output shape [B, 10]
        confidence = torch.stack(
            list(map(self.truncate_and_pad, confidence)), dim=0
        ).to(device)
        pred = self.mapping(self.predictor(confidence))

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

    def forward(self, speech_input, meta_data):
        # mel_out: [3, 80, 608]
        mel_feature, mel_feature_length = self.logmel(
            input_signal=speech_input.to(device),
            length=torch.full((speech_input.shape[0],), speech_input.shape[1]).to(
                device
            ),
        )
        encoder_out, encoder_length = self.conformer_encoder(
            audio_signal=mel_feature.to(device),
            length=torch.full((mel_feature.shape[0],), mel_feature.shape[2]).to(device),
        )
        # encoder out : [32, 512, 151]
        pred = self.mapping(self.predictor(encoder_out.contiguous().view(-1, 512 * 151)))

        return pred
    
class WordConfidenceSigmoid(nn.Module):
    """Word confidence model: Conformer + Linear predictor + exp mapping
        input(_, _, mono_path)
    """

    def __init__(self):
        super(WordConfidenceSigmoid, self).__init__()
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )    # Freeze ASR model
        for param in self.asr_model.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(in_features=10, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, speech_input, meta_data):
        # output of asr model is [B, word_len]
        mono_path = meta_data["path"]
        confidence, _ = self.asr_model.transcribe(
            paths2audio_files=mono_path, return_hypotheses=True, batch_size=32
        )
        confidence = [confidence[i].word_confidence for i in range(len(confidence))]
        # padding and truncating to 10: output shape [B, 10]
        confidence = torch.stack(
            list(map(self.truncate_and_pad, confidence)), dim=0
        ).to(device)
        pred = self.predictor(confidence)

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