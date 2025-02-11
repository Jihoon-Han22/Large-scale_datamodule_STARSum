import torch.nn as nn

class Conv1dBlock(nn.Module):
    def __init__(self, input=512, output=512):
        super(Conv1dBlock, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(input, output, kernel_size=1),
            nn.PReLU()
        )
    
    def forward(self, x):
        residual = 0.5 * x + 0.5 * self.enc(x)
        return residual, self.enc(x)

class CNN1D(nn.Module):
    def __init__(self, input_dim=1024, enc_hidden=512, n_layers=3):
        super(CNN1D, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv1d(input_dim, enc_hidden, kernel_size=5, padding=2),
            nn.PReLU()
        )
        layer_list = []
        for i in range(n_layers):
            layer_list.append(Conv1dBlock(enc_hidden, enc_hidden))

        self.Conv1D_layers = nn.ModuleList(layer_list)
        self.layer_norm = nn.LayerNorm(enc_hidden)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.reduce(x)
        for i, layer in enumerate(self.Conv1D_layers):
            if i == 1:
                x = nn.Dropout(0.5)(x)
            x, _ = layer(x)
        x = x.permute(0,2,1)
        x = self.layer_norm(x)
        x = self.prelu(x)
        return x