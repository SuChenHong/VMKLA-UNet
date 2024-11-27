import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from ...vmamba import MLP, HiLo, EfficientAdditiveAttnetion


class ConvLocalRepresentation(nn.Module):
    """ 提取局部表征 """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class ConvBlock(nn.Module):
    """ conv decoder """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5, num_heads=8, window_size=2, qkv_bias=False):
        
        super().__init__()

        self.local_representation = ConvLocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0., use_layer_scale=True
        )
        self.attn = HiLo(
            dim=dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias
        )
        self.linear = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )
        
    
    def forward(self, x):
        x = self.local_representation(x.permute(0, 3, 1, 2))
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.linear(x))
        x = x.reshape(B, H, W, C)
        return x
    

class ConvDecoder(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4., drop=0., use_layer_scale=True, drop_path=0.,
                 layer_scale_init_value=1e-5, num_heads=8, window_size=2, qkv_bias=False,
                 upsample=None, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.dim = dim
        self.depth=depth

        self.blocks = nn.ModuleList([
            ConvBlock(
                dim=dim, mlp_ratio=mlp_ratio, drop=drop,
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_layer_scale=use_layer_scale, 
                layer_scale_init_value=layer_scale_init_value,
                num_heads=num_heads, window_size=window_size,
                qkv_bias=qkv_bias
            )
            for i in range(depth)
        ])
        
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None
    
    def forward(self, x):
        
        if self.upsample:
            x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)

        return x
    

def make_conv_decoder(**kwargs):
    conv_d = ConvDecoder(**kwargs)
    return conv_d