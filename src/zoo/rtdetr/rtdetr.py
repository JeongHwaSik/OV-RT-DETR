"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import time
import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', 'OVRTDETR']


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register
class OVRTDETR(nn.Module):
    __inject__ = ['backbone', 'text_backbone', 'encoder', 'decoder', ]

    def __init__(self, text_backbone: nn.Module, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.text_backbone = text_backbone
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

        self.text_backbone_time = 0
        self.img_backbone_time = 0
        self.encoder_time = 0
        self.decoder_time = 0

    def forward(self, x, targets):

        raw_text = [t['text'] for t in targets]

        start_time = time.perf_counter()
        txt_feat = self.text_backbone(raw_text) # (B, L, txt_dim)
        self.text_backbone_time = time.perf_counter() - start_time

        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        start_time = time.perf_counter()
        x = self.backbone(x)
        self.img_backbone_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        x = self.encoder(x)
        self.encoder_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        x = self.decoder(x, txt_feat, targets)
        self.decoder_time = time.perf_counter() - start_time

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
