# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add HRDAEncoderDecoder

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .bcaln_encoder_decoder import BCEncoderDecoder,BCALNEncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoder',
    'BCEncoderDecoder',
    'BCALNEncoderDecoder'
]
