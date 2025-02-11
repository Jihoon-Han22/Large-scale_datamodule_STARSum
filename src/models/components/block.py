import torch.nn as nn
from timm.models.layers import DropPath

from src.models.components.mlp import Mlp
# from src.models.components.attention import Attention
from src.utils.attention import SelfAttention

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim) ### feature dimension 
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn = SelfAttention(input_size=dim, output_size=dim, freq=10000, pos_enc=None, heads=1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask):
        # print("x")
        # print(x)
        # print(x.shape)
        # print("self.norm1(x)")
        # print(self.norm1(x))
        #print(self.norm1(x).shape)
        # print("self.attn(norm1(x))")
        # print(self.attn(self.norm1(x), mask))
        # print(self.attn(self.norm1(x), mask).shape)
        # print("mask")
        # print(mask)
        # print(mask.shape)

        # elif scene_mask:
        #     print(scene_mask)
        #     print(scene_mask.shape)
        
        # elif shot_mask:

        x = x + self.drop_path(self.attn(self.norm1(x), mask))

        # print("self.norm2(x)")
        # print(self.norm2(x))
        #print(self.norm2(x).shape)

        # print("self.mlp(self.norm2(x))")
        # print(self.mlp(self.norm2(x)))
        #print(self.mlp(self.norm2(x)).shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x