import torch.nn as nn

from typing import Literal

_EncoderType = Literal['swint', 'vssm']

class Encoders(nn.Module):
    
    def __init__(self):
        super().__init__()
        self._type = 'swint'  # default
    
    def _call_swint(self):
        
        from .networks.swint import make_swint_encoder
        return make_swint_encoder
    
    def _call_vssm(self):

        from.networks.vssm import make_vssm_encoder
        return make_vssm_encoder
    
    def __call__(self, _type: _EncoderType):
        
        self._type = _type
        if not isinstance(self._type, str):
            raise TypeError('type must be Type@_DecoderType or Type@str')
        
        if self._type.lower() == 'swint':
            return self._call_swint()
        elif self._type.lower() == 'vssm':
            return self._call_vssm()
        else:
            raise TypeError(f'no this model [{self._type}]')
        