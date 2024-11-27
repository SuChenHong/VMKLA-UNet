from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block as VisionTransformerBlock


class VitDecoder(nn.Module):
    
    def __init__(self, dim, num_heads, depth, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm):
        
        super().__init__()
        # self.decoder_embed = nn.Linear()
        num_patches: int = 4
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim), requires_grad=False)  # fixed sin-cos embedding
            