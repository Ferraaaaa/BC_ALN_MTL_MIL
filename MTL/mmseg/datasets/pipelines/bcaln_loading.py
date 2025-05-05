# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
from typing import Any

import mmcv
import numpy as np

import cv2
from ..builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os


@PIPELINES.register_module()
class BCALNLoadAnnotations(object):

    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id

        gt_semantic_seg[gt_semantic_seg==255]=1
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class BCALNCrop(object):
    def __init__(self,
                 blur_size=5,
                 dilate_size=5,
                 otsu_thresh=0.5,
                 margin=[0.05,0.15,0.05,0.05],
                 ):
        self.blur_size=blur_size
        self.dilate_size=dilate_size
        self.otsu_thresh=otsu_thresh
        self.margin=margin

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def getMarginBBox(self,bbox,margin,shape):
        y1,y2,x1,x2=bbox
        y1 = max(0,y1-int(margin[0]*shape[0]))
        y2 = min(shape[0],y2+int(margin[1]*shape[0]))
        x1 = max(0,x1-int(margin[2]*shape[1]))
        x2 = min(shape[1],x2+int(margin[3]*shape[1]))
        bbox= [y1,y2,x1,x2]
        return bbox

    def __call__(self, results):
        img = results['img']
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        gray = cv2.dilate(gray,kernel=np.ones((5,5),np.uint8),iterations=2)
        th, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, binary = cv2.threshold(gray,int(0.5*th),255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        x,y,w,h = cv2.boundingRect(contours[np.argmax(areas)])
        margin_bbox = self.getMarginBBox([y,y+h,x,x+w],self.margin,img.shape)

        img = self.crop(img,margin_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['BCALN_bbox'] = margin_bbox

        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], margin_bbox)
        return results         


@PIPELINES.register_module()
class BCALNCollect(object):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg','BCALN_bbox')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
