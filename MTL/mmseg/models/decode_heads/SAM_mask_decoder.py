# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from ..sam_adapter.common import LayerNorm2d
from mmcv.cnn import constant_init, trunc_normal_init
from ...utils import get_root_logger
from ..builder import HEADS
from ..builder import build_attention
from .decode_head import BaseDecodeHead
from ..sam_adapter.prompt_encoder import PositionEmbeddingRandom
from ..losses import accuracy
from mmseg.ops import resize

@HEADS.register_module()
class SAMMaskDecoder(BaseDecodeHead):
    def __init__(
        self,
        *,
        img_size=1024,
        patch_size=16,
        transformer_dim: int =256,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        repeat_times= 1,
        num_classes= 19,
        activation: Type[nn.Module] = nn.GELU,
        pretrained=None,
        freeze=True,
        **kwargs,
    ) -> None:
        super(SAMMaskDecoder,self).__init__(
            in_channels=transformer_dim,
            channels=transformer_dim,
            num_classes=num_classes,
            **kwargs)
        self.conv_seg = nn.Sequential() 
        self.transformer_dim = transformer_dim
        self.transformer = build_attention(transformer)

        self.repeat_times=repeat_times
        self.num_classes = num_classes
        self.num_tokens = self.num_classes * self.repeat_times

        self.mask_tokens = nn.Embedding(
            num_embeddings= self.num_tokens, 
            embedding_dim= transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels= transformer_dim, 
                out_channels= transformer_dim // 4, 
                kernel_size=2, 
                stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                in_channels=transformer_dim // 4, 
                out_channels=transformer_dim // 8, 
                kernel_size=2, 
                stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(
                    input_dim= transformer_dim, 
                    hidden_dim= transformer_dim, 
                    output_dim= transformer_dim // 8, 
                    num_layers= 3)
                for i in range(self.num_tokens)
            ]
        )

        self.pe_layer = PositionEmbeddingRandom(self.transformer_dim//2)
        self.patch_size = patch_size
        self.img_size = img_size if isinstance(img_size,(tuple,list))\
                                 else (img_size,img_size)
        self.img_embedding_size = (self.img_size[0] // self.patch_size,
                                   self.img_size[1] // self.patch_size)
        self.no_mask_embed = nn.Embedding(1, transformer_dim)

        if pretrained is None:
            self.pretrained=None
        else:
            self.pretrained=pretrained
        
        self.freeze=freeze
        if self.pretrained is not None and self.freeze:
            self.freeze_param()
        
    def freeze_param(self):
        # transformer，output_upscaling，output_hypernetworks_mlps，iou_prediction_head
        for name,param in self.named_parameters():
            if  'transformer' in name or\
                'output_upscaling' in name:
                param.requires_grad=False

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

        if self.pretrained is None:
            self.apply(_init_weights)
        elif isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            state_dict = torch.load(self.pretrained)
            mask_decoder_state_dict = {}
            for key,value in state_dict.items():
                if 'mask_decoder' in key:
                    new_key = key.replace('mask_decoder.','')
                    mask_decoder_state_dict[new_key] = value

            msg = self.load_state_dict(mask_decoder_state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        else:
            raise TypeError('pretrained must be a str or None')

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.img_embedding_size).unsqueeze(0)

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
            ) -> torch.Tensor:
        
        masks = F.interpolate(
            masks,
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )

        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, 
            original_size, 
            mode="bilinear", 
            align_corners=False)

        return masks
    
    def fuse_masks(self,masks):
        B,C,H,W=masks.shape
        assert C==self.repeat_times*self.num_classes
        results=masks.\
            reshape(B,self.repeat_times,self.num_classes,H,W).\
            mean(dim=1)
        return results

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logit=False,
                      return_feat = False):
        bs = 1 
        sparse_embeddings = torch.empty(
            (bs, 0, self.transformer_dim),device=inputs.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)#\
            #.expand(bs,-1,self.img_embedding_size[0],self.img_embedding_size[1])

        out = self.forward(
            image_embeddings=inputs,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            return_feat=return_feat,
        )
        if return_feat:
            low_res_masks = out[0]
            feat = out[1]

        low_res_masks = self.fuse_masks(low_res_masks)
        masks = self.postprocess_masks(
            masks=low_res_masks,
            input_size=self.img_size,
            original_size=self.img_size,
        )
        losses = self.losses(masks, gt_semantic_seg, seg_weight)

        if return_logit:
            losses['logit'] = masks
        if return_feat:
            losses['feat'] = feat
        
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, return_feat = False):
        bs = 1 
        sparse_embeddings = torch.empty(
            (bs, 0, self.transformer_dim),device=inputs.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)#\
            #.expand(bs,-1,self.img_embedding_size[0],self.img_embedding_size[1])

        out = self.forward(
            image_embeddings=inputs,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            return_feat=return_feat,
        )
        if return_feat:
            low_res_masks = out[0]
            feat = out[1]

        low_res_masks=self.fuse_masks(low_res_masks)

        masks = self.postprocess_masks(
            masks=low_res_masks,
            input_size=self.img_size,
            original_size=self.img_size,
        )

        if return_feat:
            return masks, feat
        else:
            return masks

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        return_feat = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            return_feat=return_feat,
        )

        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        return_feat = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](hs[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        if return_feat:
            return masks, upscaled_embedding
        else:
            return masks
    
    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        if self.loss_decode.use_sigmoid==False:
            return super().losses(seg_logit, seg_label, seg_weight)
        elif self.loss_decode.use_sigmoid==True:
            loss = dict()
            seg_label = seg_label.squeeze(1)
            seg_logit = seg_logit.squeeze(1)
            loss['loss_seg'] = self.loss_decode(
                seg_logit,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)
            return loss


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
