import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint
# from functools import partial
from einops import rearrange
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
try:
    from .attention.Attentions import ChannelAttention, SpatialAttention
except:
    from attention.Attentions import ChannelAttention, SpatialAttention

class LinearAttention(nn.Module):
    r""" Linear Attention with LePE & RoPE 
    
    Args:
        dim (int): 输入通道数
        num_heads (int): 注意力头
        qkv_bias (bool, optional): 是否添加偏置项
    """
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        
        super().__init__()
        self.dim = dim
        if isinstance(input_resolution, int):
            input_resolution = to_2tuple(input_resolution)
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()
    
    def forward(self, x):
        """
        Args:
            x: B N C
        """
        b, n, c = x.shape
        h = w = int(n ** .5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x

        q = self.elu(q) + 1.
        k = self.elu(k) + 1.
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** .5)
            x = x.reshape(B, H, W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x).reshape(B, L, C)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x       


class PatchExpand(nn.Module):

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)
    
    def forward(self, x):
        dim = x.dim()
        if dim == 3:
            B, L, C = x.shape
            H = W = int(L ** .5)
            x = x.reshape(B, H, W, C)
        else:
            B, _, _, C = x.shape
        x = self.expand(x)

        x = rearrange(
            x, 'b h w (p1 p2 c)->b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale,
            c=C // self.dim_scale
        )
        x = self.norm(x)
        if dim == 3:
            _, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        return x


class FinalExpand(nn.Module):

    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)
    
    def forward(self, x):
        _, _, _, C = x.shape

        x = self.expand(x)

        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale,
            c=C // self.dim_scale
        )
        x = self.norm(x)

        return x


class RoPE(nn.Module):
    """ Rotry Positional Embedding """
    
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        *channel_dims, feature_dim = x.shape[1:-1][0], x.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, "Feature dimension must be divisible by 2 * k_max"

        # 生成角度
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        angles = torch.cat([t.unsqueeze(-1).to(x.device) * theta_ks for t in torch.meshgrid([torch.arange(d).to(x.device) for d in channel_dims], indexing='ij')], dim=-1).to(x.device)

        # 计算旋转矩阵的实部和虚部
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1).to(x.device)

        # self.register_buffer('rotations', rotations)
        return rotations

    def forward(self, x):
        # if x.dtype != torch.float32:
        #     x = x.to(torch.float32)
        # 生成旋转矩阵
        rotations = self.generate_rotations(x)
        # 将 x 转换为复数形式
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        # 应用旋转矩阵
        pe_x = torch.view_as_complex(rotations) * x
        # 将结果转换回实数形式并展平最后两个维度
        return torch.view_as_real(pe_x).flatten(-2)


class KANLinear(nn.Module):
    r""" From Effective KAN source code """
    
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.,
        scale_spline=1.,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1]  
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer('grid', grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.sclae_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """  
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / grid[:, k:-1] - grid[:, : -(k + 1)]
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order
        )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.deive).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1., regularize_entropy=1.):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularize_loss_activation = l1_fake.sum()
        p = l1_fake / regularize_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularize_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )   


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1]
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range
                )
            )
    
    def forward(self, x: torch.Tensor, update_grid=False):

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1., regularize_entropy=1.):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class KanAttention(nn.Module):

    def __init__(self, dim, input_resolution=None, num_heads=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.kan = KAN([dim, 64, dim * 2])
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x = x.reshape((x.size(0), x.size(2) * x.size(3), x.size(1)))
        b, n, c = x.shape
        h = w = int(n ** .5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        x1 = x.reshape(-1, x.shape[-1])
        qk = self.kan(x1).reshape(b, n, 2 * c)
        qk = qk.reshape(b, n, 2, c).permute(2, 0, 1, 3)

        q, k, v = qk[0], qk[1], x
        q = self.elu(q) + 1.
        k = self.elu(k) + 1.

        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.transpose(2, 1).reshape(b, c, h, w)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class MKLABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)  # 左分支
        self.act_proj = nn.Linear(dim, dim) # 右分支
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = KanAttention(dim=dim, num_heads=num_heads)
        self.out_norm = norm_layer(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.in_norm(x)
        act_res = self.act(self.act_proj(x))  # 右分支
        # act_res = act_res.reshape(B, H * W, C)
        # x = self.in_proj(x.reshape(B, H, W, C)).view(B, H, W, C)
        x = self.in_proj(x).view(B, H, W, C)

        x = self.act(self.dwconv(x.permute(0, 3, 1, 2).contiguous()))
        
        # KAN attention
        x = self.attn(x).reshape(B, H * W, C)
        # print(shortcut.shape, x.shape, act_res.shape)
        # exit(0)
        x = self.out_proj(self.out_norm(x) * act_res)

        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm(x)))
        x = x.reshape(B, H, W, C)

        return x


class MKLABlockWithAttention(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)  # 左分支
        # self.act_proj = nn.Linear(dim, dim) # 右分支
        # 右分支
        self.CA = ChannelAttention(in_planes=dim)
        self.SA = SpatialAttention()

        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = KanAttention(dim=dim, num_heads=num_heads)
        self.out_norm = norm_layer(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       act_layer=act_layer, drop=drop)
        
        
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.in_norm(x)
        # 次分支
        x = x.reshape(B, C, H, W)
        ca_res = self.CA(x) * x
        sa_res = self.SA(ca_res) * ca_res
        sub_branch = sa_res.view(B, H * W, C).contiguous()

        # 主分支
        x = self.in_proj(x.view(B, H * W, C)).view(B, H, W, C)
        x = self.act(self.dwconv(x.permute(0, 3, 1, 2).contiguous()))
        
        # KAN attention
        x = self.attn(x).reshape(B, H * W, C)
        # print(shortcut.shape, x.shape, act_res.shape)
        # exit(0)
        x = self.out_proj(self.out_norm(x) * sub_branch)

        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm(x)))
        x = x.reshape(B, H, W, C)        

        return x


class MKLALayer(nn.Module):
    """ A basic MKLA layer(enc) for one stage """

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MKLABlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = downsample
    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MKLALayer_up(nn.Module):

    """ A basic MKLA layer(dec) for one stage """

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MKLABlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        if upsample is not None:
            self.upsample = upsample(dim=dim)
        else:
            self.upsample = upsample
    
    def forward(self, x):

        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class MKLALayerWithAttention(nn.Module):

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MKLABlockWithAttention(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = downsample
    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MKLALayerUpWithAttention(nn.Module):

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., drop_path=0., 
                    norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
            
            super().__init__()
            self.dim = dim
            self.depth = depth
            self.use_checkpoint = use_checkpoint

            self.blocks = nn.ModuleList([
                MKLABlockWithAttention(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    drop=drop, norm_layer=norm_layer,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                )
                for i in range(depth)
            ])
            
            if upsample is not None:
                self.upsample = upsample(dim=dim)
            else:
                self.upsample = upsample
        
    def forward(self, x):

        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x   


class MKLA(nn.Module):

    def __init__(self, embed_dim=96, in_chans=3, num_classes=1000, depths=[2, 2, 2, 2], decoder_depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24], drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 norm_pix_loss=False, use_checkpoint=False, drop_rate=0., patch_size=4, mlp_ratio=4.):
        
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(embed_dim, int):
            self.dims = [int(embed_dim * 2 ** i_layer) for i_layer in range(self.num_layers)]
            self.dec_dims = [int(self.dims[-1] // (2 ** i_layer)) for i_layer in range(self.num_layers)]
        self.embed_dim = embed_dim
        self.num_features = self.dims[-1]
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        ) 
        self.ape = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))][::-1]

        # encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MKLALayer(
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        # decoder
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MKLALayer_up(
                dim=self.dec_dims[i_layer],
                depth=decoder_depths[i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                mlp_ratio=mlp_ratio,
                drop_path=dpr_decoder[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers_up.append(layer)
        
        self.final_up = FinalExpand(dim=self.dec_dims[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(self.dec_dims[-1] // 4, num_classes, 1)
        self.final_ln = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_encoder(self, x):
        skip_list = []
        x = self.patch_embed(x)
        x, _ = self.window_mask(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        
        return x, skip_list
        

    def forward_decoder(self, x, skip_list: list):
        for idx, layer in enumerate(self.layers_up):
            if idx == 0:
                x = layer(x)
            else:
                x = layer(x + skip_list[-idx])
        
        return x

    def forward(self, x):
        x, skip_list = self.forward_encoder(x)
        x = self.forward_decoder(x, skip_list)
        x = self.final_up(x)
        x = self.final_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x, None

# MKCSA
class MKLAWithAttention(nn.Module):

    def __init__(self, embed_dim=96, in_chans=3, num_classes=1000, depths=[2, 2, 2, 2], decoder_depths=[2, 2, 2, 2],
                    num_heads=[3, 6, 12, 24], drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                    norm_pix_loss=False, use_checkpoint=False, drop_rate=0., patch_size=4, mlp_ratio=4.):
            
            super().__init__()

            self.num_classes = num_classes
            self.num_layers = len(depths)
            if isinstance(embed_dim, int):
                self.dims = [int(embed_dim * 2 ** i_layer) for i_layer in range(self.num_layers)]
                self.dec_dims = [int(self.dims[-1] // (2 ** i_layer)) for i_layer in range(self.num_layers)]
            self.embed_dim = embed_dim
            self.num_features = self.dims[-1]
            self.patch_size = patch_size
            self.norm_pix_loss = norm_pix_loss
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if patch_norm else None
            ) 

            self.ape = False
            self.pos_drop = nn.Dropout(p=drop_rate)

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))][::-1]

            # encoder
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = MKLALayerWithAttention(
                    dim=self.dims[i_layer],
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint
                )
                self.layers.append(layer)

            # decoder
            self.layers_up = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = MKLALayerUpWithAttention(
                    dim=self.dec_dims[i_layer],
                    depth=decoder_depths[i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr_decoder[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer != 0) else None,
                    use_checkpoint=use_checkpoint
                )
                self.layers_up.append(layer)
            
            self.final_up = FinalExpand(dim=self.dec_dims[-1], dim_scale=4, norm_layer=norm_layer)
            self.final_conv = nn.Conv2d(self.dec_dims[-1] // 4, num_classes, 1)
            self.final_ln = norm_layer(embed_dim)

            self.apply(self._init_weights)


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x):
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        
        return x, skip_list
        

    def forward_decoder(self, x, skip_list: list):
        for idx, layer in enumerate(self.layers_up):
            if idx == 0:
                x = layer(x)
            else:
                x = layer(x + skip_list[-idx])
        
        return x


    def forward(self, x):
        x, skip_list = self.forward_encoder(x)
        x = self.forward_decoder(x, skip_list)
        x = self.final_up(x)
        x = self.final_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x, None    




    

