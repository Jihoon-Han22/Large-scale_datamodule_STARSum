import torch
import torch.nn as nn

class ScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_predictor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, stage):
        # if (stage < 2):
        #     B = len(x)
        #     x_list = []
        #     for b in range(B):
        #         x[b] = self.score_predictor(x[b])
        #         x[b] = x[b] / (torch.max(x[b]) + 1e-6)
        #         x_list.append(x[b])
        #     x = torch.stack(x_list)
        #     return x
        # else:
        x = self.score_predictor(x)
        x = x / (torch.max(x) + 1e-6)
        return x