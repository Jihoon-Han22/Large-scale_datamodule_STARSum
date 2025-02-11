import torch.nn as nn

from src.models.components.uboco.cnn_1d import Conv1dBlock

class LongEnc4(nn.Module):
    def __init__(self, input=1024, nhead=16, dim_feedforward=1024, nlayers=2, output=512):
        super().__init__()

        encoder_layers = nn.TransformerEncoderLayer(d_model=input, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.relu = nn.ReLU()
        self.conv1d = Conv1dBlock(input=input, output=output)
        self.layer_norm1 = nn.LayerNorm(input)
        self.layer_norm2 = nn.LayerNorm(output)

    def forward(self, x): # 변수로 돌리기
        x = self.transformer_encoder(x)
        x = self.relu(x)
        x = self.layer_norm1(x)
        x = self.transformer_encoder(x)
        x = self.relu(x)
        x = self.layer_norm1(x)
        x = x.permute(0,2,1)
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        x = self.layer_norm2(x)
        x = self.relu(x)
        return x