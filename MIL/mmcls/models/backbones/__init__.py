# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .lenet import LeNet5
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .regnet import RegNet
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin_transformer import SwinTransformer
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tnt import TNT
from .vgg import VGG
from .vision_transformer import VisionTransformer
from .conv1x1_mlp_tct import MLP_TCT
from .vision_transformer_tct_ngc import VisionTransformer_TCT_NGC
from .mlp_mixer_tct_ngc import MlpMixer_TCT_NGC

from .vision_transformer_bcaln import VisionTransformer_BCALN

__all__ = [
    'LeNet5', 'AlexNet', 'VGG', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNeSt', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetV3', 'VisionTransformer',
    'SwinTransformer', 'TNT', 'TIMMBackbone', 'T2T_ViT', 'Res2Net', 'RepVGG',
    'MlpMixer', 'MLP_TCT', 'VisionTransformer_TCT_NGC','MlpMixer_TCT_NGC',
    'VisionTransformer_BCALN'
]
