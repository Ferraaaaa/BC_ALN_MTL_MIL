from ..builder import BACKBONES
from .vision_transformer_tct_ngc import VisionTransformer_TCT_NGC

@BACKBONES.register_module()
class VisionTransformer_BCALN(VisionTransformer_TCT_NGC):
    '''
        Same implementation as VisionTransformer_TCT_NGC
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)