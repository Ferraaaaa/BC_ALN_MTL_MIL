
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from ..builder import HEADS,build_loss
from mmcv.cnn import ConvModule

@HEADS.register_module()
class BCALNClsHead(nn.Module):

    def __init__(self,
                 num_classes=1,
                 in_channels=None,
                 need_cat=True,
                 index=2,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 ):
        super(BCALNClsHead, self).__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels if isinstance(in_channels,list) else [in_channels]
        self.need_cat=need_cat
        self.index=index
        self.loss_decode = build_loss(loss_decode)
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        in_channel=sum(self.in_channels)

        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=self.num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, inputs):
        inputs = inputs if not isinstance(inputs,(list,tuple)) else inputs[0]
        cls_logits = self.classifier(inputs).squeeze()
        if cls_logits.dim()==0: # bs=1
            cls_logits = cls_logits.unsqueeze(0)
        return cls_logits

    def forward_train(self,inputs,img_metas,gt_cls):
        cls_logits = self.forward(inputs)
        loss = dict()
        loss['loss_cls'] = self.loss_decode(cls_logits, gt_cls)
        return loss

    def forward_test(self, inputs, img_metas):
        return self.forward(inputs)
