import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """ 通道注意力 """

    def __init__(self, in_planes, ratio=16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.ratio = ratio

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    

class SpatialAttention(nn.Module):
    """ 空间注意力 """

    def __init__(self, kernel_size=7) -> None:
        super().__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class HiLo(nn.Module):
    
    def __init__(self, dim, num_heads: int = 8, qkv_bias: bool = False, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}'
        head_dim = int(dim / num_heads)
        self.dim = dim

        # self-attn heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # self-attn heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim

        # local window size
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim
        
        self.scale = qk_scale or head_dim ** -0.5

        # Lo-Fi
        if self.l_heads > 0:
            if self.ws != 1:
                # 这里的window_size是池化倍数
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
                # self.sr = PatchMerging2D(self.l_dim)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # Hi-Fi
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)


    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2] #B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x


    def forward(self, x, H=None, W=None):

        if self.h_heads == 0:
            x = self.lofi(x)
            # return x.reshape(B, N, C)
            return x

        if self.l_heads == 0:
            x = self.hifi(x)
            # return x.reshape(B, N, C)
            return x
        
        # local attention
        hifi_out = self.hifi(x)
        # global attention
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        # x = x.reshape(B, N, C)

        return x

    def flops(self, H, W):
        # pad the feature map when the height and width cannot be divided by window size
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # For Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # q @ k and attn @ v
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # projection
        hifi_flops += Np * self.h_dim * self.h_dim

        # for Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.l_dim
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        # k, v
        lofi_flops += kv_len * self.dim * self.l_dim * 2
        # q @ k and attn @ v
        lofi_flops += Np * self.l_dim * kv_len * 2
        # projection
        lofi_flops += Np * self.l_dim * self.l_dim

        return hifi_flops + lofi_flops
    

class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out