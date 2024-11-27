try:
    from .MKLA import MKLALayerUpWithAttention, PatchExpand

    from .encoders.build_encoder import Encoders
    from .decoders.build_decoder import Decoders

except:
    from MKLA import MKLALayerUpWithAttention, PatchExpand

    from encoders.build_encoder import Encoders
    from decoders.build_decoder import Decoders

from typing import Literal
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import torch
import torch.nn as nn
import math

MODEL_TYPE = Literal['swin_vssm', 'swin_swin', 'swin_conv', 
                     'vssm_vssm', 'vssm_swin', 'vssm_conv', 
                     'local_global', 'mkla', 'vssm_mkla',
                     'mkcsa', 'vssm_mkla_fuse']
base_encoder = Encoders()
base_decoder = Decoders()

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
  
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape

        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        # print(x.shape)
        # exit(0)
        x = self.norm(x)

        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

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


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = 'vssm_vssm'

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)


class VssmMkla(BaseModel):
    def __init__(self, dims=96, img_size=256, patch_size=4, input_channels=3, depths=[2, 2, 2, 2], decoder_depths=[2, 2, 2, 2],
                 num_classes=1000, num_heads=[3, 6, 12, 24], drop_rate=0., drop_path_rate=.1, patch_norm=True,
                 d_state=16, attn_drop_rate=0., mlp_ratio=4., qkv_bias=True, ape=False, norm_layer=nn.LayerNorm, 
                 use_checkpoint=False):
        super().__init__()
        self.model_name = 'vssm_mkla'
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims: list = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim   :int  = dims[0]
        self.num_features:int  = dims[-1]
        self.dims        :list = dims
        self.ape         :bool = ape
        self.patch_size  :int  = patch_size
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=input_channels, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.ab_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.ab_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))][::-1]


        # encoder 
        self.layers = nn.ModuleList()
        _VssmEncoder = base_encoder('vssm')

        for i_layer in range(self.num_layers):
            layer = _VssmEncoder(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)     

        final_dim = self.num_features
        
        
        # decoder 
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MKLALayerUpWithAttention(
                dim=int(final_dim // (2 ** i_layer)),
                depth=decoder_depths[i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                mlp_ratio=mlp_ratio,
                drop_path=dpr_decoder[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=self.embed_dim, dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(self.embed_dim // 4, num_classes, 1)
        
        self.apply(self._init_weights)
    
    def __model__(self):
        return self.model_name

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'ab_pos_embed'} 
    
    def forward_encoder(self, x):
        skip_list: list = []
        x = self.patch_embed(x)
        B, L, C = x.shape
        H = W = int(L ** .5)
        x = x.reshape(B, H, W, C)   
        if self.ape:
            x = x + self.ab_pos_embed
        x = self.pos_drop(x)


        for idx, layer in enumerate(self.layers):       
            skip_list.append(x)
            x = layer(x)
        
        return x, skip_list
    
    def forward_decoder(self, x, skip_list: list):
        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-idx])
        return x


    def forward(self, x):

        x, skip_list = self.forward_encoder(x)
        x = self.forward_decoder(x, skip_list)
        x = self.final_up(x).permute(0, 3, 1, 2)
        x = self.final_conv(x) 

        return x

      
class build_model(nn.Module):

    def __init__(self, model_name: MODEL_TYPE, input_channels=3, num_classes=1, depths=[2, 2, 2, 2],
                 depths_decoder=[2, 2, 2, 2], drop_path_rate=.1, use_checkpoint=False, with_mae=False,
                 load_ckpt_path=None, with_fusion=False, encoder_ckpt_path=None, decoder_ckpt_path=None):
        
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.encoder_ckpt_path = encoder_ckpt_path
        self.decoder_ckpt_path = decoder_ckpt_path
        self.num_classes = num_classes
        self.model_name = model_name
        
        if model_name == 'vssm_mkla':
            self.model = VssmMkla(
                input_channels=input_channels,
                num_classes=num_classes,
                depths=depths,
                decoder_depths=depths_decoder,
                img_size=256,
            )
        else:
            raise ValueError(f'no this model @{model_name}')
        
    def forward_vssm_mkla(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.model(x)
        if self.num_classes == 1:
            if isinstance(logits, tuple):
                return torch.sigmoid(logits[0]), logits[1]
            else:
                return torch.sigmoid(logits)
        else:
            return logits
        

    def forward(self, x):
        if self.model_name == 'vssm_mkla':
            return self.forward_vssm_mkla(x)
        
    def load_from(self):

        if self.encoder_ckpt_path is not None:
            model_dict = self.model.state_dict()
            modelCheckpoint = torch.load(self.encoder_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
          
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
           
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.model.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
 
            print("encoder loaded finished!")
        
        if self.decoder_ckpt_path is not None:
            model_dict = self.model.state_dict()
            modelCheckpoint = torch.load(self.decoder_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
           
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.model.load_state_dict(model_dict)
            
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            
            print("decoder loaded finished!")

