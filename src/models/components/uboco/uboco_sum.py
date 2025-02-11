import torch
import torch.nn as nn

from src.models.components.uboco.cnn_1d import CNN1D
from src.models.components.uboco.long_enc4 import LongEnc4

class UBoCoEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        enc_hidden: int,
        enc1_layers: int,
        enc2_layers: int,
        enc3_layers: int,
        # enc4_nhead: int,
        # enc4_dim_feedforward: int,
        # enc4_layers: int
    ):
        super().__init__()

        self.enc1 = CNN1D(enc_hidden=enc_hidden, n_layers=enc1_layers)
        self.enc2 = CNN1D(enc_hidden=enc_hidden, n_layers=enc2_layers)
        self.enc3 = CNN1D(enc_hidden=enc_hidden, n_layers=enc3_layers)
        # self.enc4 = LongEnc4(input=enc_hidden, nhead=enc4_nhead, dim_feedforward=enc4_dim_feedforward, nlayers=enc4_layers, output=enc_hidden)
        
        # REMOVE THESE TWO
        self.feature_reduction = nn.Sequential(
                    nn.Conv1d(feature_dim, enc_hidden, 1),
                    nn.PReLU()
                )
        self.encoder_adder = nn.Linear(4, 1)

    '''
    minus l2 distance TSM
    input : torch Tensor (B, nframes, feature_dims)
    output : torch Tensor (B, nframes, nframes)
    '''
    def makeTSM_l2_dist(self, x):
        temp1 = x.unsqueeze(1)
        temp2 = x.unsqueeze(2)
        l2_dist = torch.sqrt(torch.sum((temp1 - temp2)**2, dim=-1) + 1e-8)
        l2_dist = l2_dist.unsqueeze(1)
        l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
        l2_dist = torch.squeeze(l2_dist, dim=1)
        return -l2_dist

    def forward(self, x):
        feat1 = self.enc1(x)
        feat2 = self.enc2(x)
        feat3 = self.enc3(x)
        # feat4 = self.enc4(x)

        # TSM path
        tsm1 = self.makeTSM_l2_dist(feat1)
        tsm2 = self.makeTSM_l2_dist(feat2)
        tsm3 = self.makeTSM_l2_dist(feat3)
        raw = self.makeTSM_l2_dist(x)
        # tsm4 = self.makeTSM_l2_dist(feat4)

        tsm1_temp = torch.unsqueeze(tsm1, -1)
        tsm2_temp = torch.unsqueeze(tsm2, -1)
        tsm3_temp = torch.unsqueeze(tsm3, -1)
        # tsm4_temp = torch.unsqueeze(tsm4, -1)
        raw_temp = torch.unsqueeze(raw, -1)

        concat_tsm = torch.mean(torch.cat([tsm1_temp, tsm2_temp, tsm3_temp, raw_temp], dim=-1), dim=-1)

        return [feat1, feat2, feat3, x], [tsm1, tsm2, tsm3, raw], concat_tsm