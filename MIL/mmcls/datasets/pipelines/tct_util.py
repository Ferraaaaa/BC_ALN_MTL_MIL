import os.path as osp

import numpy as np
import torch

from .loading import LoadImageFromFile
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadPtFromFile(LoadImageFromFile):

    def __init__(self,
                 self_normalize=True,
                 **kwargs):
        self.self_normalize = self_normalize
        super().__init__(**kwargs)

    def __call__(self, results):

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        img = torch.load(filename)
        if self.self_normalize:
            mean = torch.mean(img, dim=(1, 2)).reshape(-1, 1, 1)
            std = torch.std(img, dim=(1, 2), unbiased=False).reshape(-1, 1, 1)
            img = (img - mean) / std

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
