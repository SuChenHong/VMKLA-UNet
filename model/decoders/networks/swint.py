import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ...SwinTransformer import SwinTransformerBlock

class SwinDecoder(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0):
        
        super().__init__()
        self.dim = dim 
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)
        ])

        # patch expand layer
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None
    
    def forward(self, x):

        if self.upsample is not None:
            x = self.upsample(x)        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
    def flops(self):

        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops

    def _init_respostnorm(self):

        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

def make_swint_decoder(**kwargs):
    swint_d = SwinDecoder(**kwargs)
    return swint_d