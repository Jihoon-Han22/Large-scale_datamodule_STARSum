import torch
import torch.nn as nn
from functools import partial

from src.models.components.block import Block
from torch.nn.utils.rnn import pad_sequence

class StageEnocoder(nn.Module):
    def __init__(
        self,
        depth,
        input_dim,
        num_heads,
        mlp_ratio
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(dim=input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for i in range(depth)])
    
    def forward(self, x, stage, mask=None): # scene_mask=None, shot_mask=None
        #print("forward x")
        #print(x)
        # jihoon code
        # max_len = -1
        # if (torch.is_tensor(x)):
        #     print(x.shape)
        # if (stage < 2):
        #     B = len(x)
        #     for b in range(B):
        #         n_segment = len(x[b])
        #         print("new block")
        #         for i, blk in enumerate(self.blocks):
        #             if (not torch.is_tensor(x[b])):
        #                 x[b] = torch.stack(x[b])
        #                 x[b] = torch.squeeze(x[b], 1)
        #                 x[b] = torch.unsqueeze(x[b], 0)
        #             if (x[b].dim() == 2):
        #                 x[b] = torch.unsqueeze(x[b], 0)
        #             x[b] = blk(x[b])
        #         if max_len < ((x[b].shape)[1]):
        #             max_len = ((x[b].shape)[1])
        #     # x = torch.stack(x)
        #     # x = torch.squeeze(x, 1)
        #     if (B != 1):
        #         x = pad_sequence(x, batch_first=True)
        #     x = torch.stack(x)
        #     x = torch.squeeze(x, 1)
        #     print("inside x:")
        #     print(x)
        #     print(x.shape)
        #     return x
        # else:
        #     for i, blk in enumerate(self.blocks):
        #         x = blk(x, mask=mask)
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=mask)   
        return x
        # list of list of Tensors -> Tensor
        # if (not torch.is_tensor(x[0])):
        #     B = len(x)
        #     for b in range(B):
        #         if (not torch.is_tensor(x[b])):
        #             x[b] = torch.stack(x[b])
        #             x[b] = torch.squeeze(x[b], 1)
        #             x[b] = torch.unsqueeze(x[b], 0)
        #         if (x[b].dim() == 2):
        #             x[b] = torch.unsqueeze(x[b], 0)
        #     x = torch.stack(x)
        #     x = torch.squeeze(x, 1)
        # if (stage == 0):
        #     for i, blk in enumerate(self.blocks):
        #         print("scene_mask: ")
        #         print(scene_mask) 
        #         x = blk(x, scene_mask=scene_mask)
        #     return x
        # if (stage == 1):
        #     for i, blk in enumerate(self.blocks):
        #         print("shot_mask: ")
        #         print(shot_mask) 
        #         x = blk(x, shot_mask=shot_mask)
        #     return x