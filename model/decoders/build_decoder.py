import torch.nn as nn
from typing import Literal

_DecoderType = Literal['swint', 'vit', 'vssm', 'conv']


class Decoders(nn.Module):
    
    def __init__(self):
        super().__init__()
        self._type = 'swint'  # default
    

    def _call_swint(self):
        """ swint作为decoder """

        # print(f'[decoder is {self._type}]')
        from .networks.swint import make_swint_decoder
        return make_swint_decoder

    def _call_vssm(self):
        """ vision mamba作为decoder """

        # print(f'[decoder is {self._type}]')
        from .networks.vssm import make_vssm_decoder
        return make_vssm_decoder
        
    def _call_vit(self):
        """ vit作为decoder """

        print(f'[decoder is {self._type}]')

    def _call_conv(self):
        """ 卷积作为decoder """

        # print(f'[decoder is {self._type}]')
        from .networks.conv import make_conv_decoder
        return make_conv_decoder

    def __call__(self, _type: _DecoderType):
        
        self._type = _type 
        if not isinstance(self._type, str):
            raise TypeError('type must be Type@_DecoderType or Type@str')
        
        if self._type.lower() == 'swint':
            return self._call_swint()
        elif self._type.lower() == 'vit':
            return self._call_vit()
        elif self._type.lower() == 'vssm':
            return self._call_vssm()
        elif self._type.lower() == 'conv':
            return self._call_conv()
        else:
            raise TypeError(f'no this model [{self._type}]')