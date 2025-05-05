import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead

@HEADS.register_module()
class MLPHead(ClsHead):

    def __init__(self,             
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )
                ):
        super(MLPHead, self).__init__(loss=loss, topk=topk)

        
    def forward_train(self, x, gt_label, **kwargs):
        if isinstance(x, tuple):
            x = x[-1]
        losses = self.loss(x, gt_label, **kwargs)
        return losses