import torch.nn as nn

class RegressorHead(nn.Module):
    def __init__(
        self,
        input_dim=1024, 
        output_dim=1
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2, bias=True)
        self.leaky = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.model = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_dim // 2, eps=1e-6),
            nn.Linear(input_dim // 2, output_dim, bias=True),
            nn.Sigmoid()
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # nn.init.constant_(m.bias, 0)
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            # torch.nn.init.xavier_uniform_(m.weight)
            # nn.init.constant_(m.weight, 1.0)
            if isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # nn.init.constant_(m.bias, 0)
                # m.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky(x)
        x = self.dropout(x)
        x = self.model(x)
        return x