# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from functools import reduce

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS, build_head
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class BCEncoderDecoder(EncoderDecoder):

    def __init__(self,cls_head=None,**kwargs):
        super(BCEncoderDecoder, self).__init__(**kwargs)
        assert cls_head is not None
        self.cls_head = build_head(cls_head)
        self.temperature = 2
    
    def encode_decode(self, img, img_metas,**kwargs):
        if self.with_neck:
            x,ori_x = self.extract_feat(img)
        else:
            x=ori_x = self.extract_feat(img)
        cls_out = self._cls_head_forward_test(ori_x,img_metas,**kwargs)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return cls_out,out
    
    def _cls_head_forward_train(self,x,img_metas,gt_cls):
        losses = dict()
        loss_cls = self.cls_head.forward_train(
            inputs=x,
            img_metas=img_metas,
            gt_cls=gt_cls
        )

        losses.update(add_prefix(loss_cls, 'cls'))
        return losses
    
    def _cls_head_forward_test(self,x,img_metas):
        return self.cls_head.forward_test(x,img_metas)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      cls,
                      seg_weight=None,
                      return_feat=False):
        if self.with_neck:
            x,ori_x = self.extract_feat(img)
        else:
            x=ori_x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x
        
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        loss_cls = self._cls_head_forward_train(ori_x,img_metas,cls)

        losses.update(loss_cls)
        losses.update(loss_decode)

        return losses

    def get_feats_and_logits(self, img, img_metas, **kwargs):
        img = img[0].cuda()
        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas,self.test_cfg,return_feat=True)
        mask, feat = out[0], out[1]
        cls_feat = x * torch.sigmoid(
            resize(mask.detach(), x.shape[2:], mode='bilinear') / self.temperature
        )
        cls_logit = self.cls_head.classifier(cls_feat)
        return cls_feat, cls_logit
    
    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta)
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        cls_logit,seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                if img_meta[0].get('BC_bbox',None) is not None:
                    y1,y2,x1,x2 = img_meta[0]['BC_bbox']
                    size = (y2-y1,x2-x1)
                else:
                    size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return cls_logit,seg_logit

    def inference(self, img, img_meta, rescale):

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            cls_logit,seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            cls_logit,seg_logit = self.whole_inference(img, img_meta, rescale)

        cls_logit = F.sigmoid(cls_logit)
        output = F.sigmoid(seg_logit)

        return cls_logit,output

    def simple_test(self, img, img_meta, rescale=True,**kwargs):
        """Simple test with single image."""
        cls_logit,seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = (seg_logit > 0.5).long()
        if img_meta[0].get('BC_bbox',None) is not None:
            y1,y2,x1,x2 = img_meta[0]['BC_bbox']
            y,x = img_meta[0]['ori_shape'][:2]
            pad_size = (x1,x-x2,y1,y-y2)
            seg_pred = F.pad(input=seg_pred,
                            pad=pad_size,
                            mode='constant',
                            value=0)
        cls_pred = (cls_logit>0.5).long()
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        cls_pred = cls_pred.cpu().numpy()

        # unravel batch dim
        seg_pred = list(seg_pred)
        cls_logit = cls_logit.cpu().numpy()[0,cls_pred]
        cls_logit = cls_logit if isinstance(cls_logit,list) else [cls_logit]
        cls_pred = cls_pred if isinstance(cls_pred,list) else [cls_pred]
        return cls_pred,seg_pred,cls_logit

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        # to save memory, we get augmented seg logit inplace
        cls_logit,seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_cls_logit,cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            cls_logit += cur_cls_logit
            seg_logit += cur_seg_logit
        cls_logit /= len(imgs)
        seg_logit /= len(imgs)
        cls_pred = cls_logit>0.5
        seg_pred = seg_logit>0.5
        cls_pred = cls_pred.cpu().numpy()
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        cls_pred = cls_pred if isinstance(cls_pred,list) else [cls_pred]
        seg_pred = list(seg_pred)
        return cls_pred,seg_pred

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        img = mmcv.imread(img)
        img = img.copy()
        opacity = 0.6
        cls = result[0][0]
        seg = result[1][0].squeeze()
        cls_logit = round(100*result[2][0][0],2)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in seg]))
        else:
            num_classes = len(self.CLASSES)
            
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(
                    0, 255, size=(num_classes, 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == num_classes
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if label==0:continue 
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if cls==1:
            cls_text='positive'+':'+str(cls_logit)+'%'
        else:
            cls_text='negative'+':'+str(cls_logit)+'%'
        cv2.putText(img,cls_text,(40,40),cv2.FONT_HERSHEY_COMPLEX,0.9,(255,255,255),1,4)

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img



@SEGMENTORS.register_module()
class BCALNEncoderDecoder(BCEncoderDecoder):

    def __init__(self,mask_cls_head,**kwargs):
        super(BCALNEncoderDecoder, self).__init__(**kwargs)
        self.mask_cls_head = build_head(mask_cls_head)
    
    def _mask_cls_head_forward_train(self,x,img_metas,gt_cls):
        losses = dict()
        loss_cls = self.mask_cls_head.forward_train(
            inputs=x,
            img_metas=img_metas,
            gt_cls=gt_cls
        )

        losses.update(add_prefix(loss_cls, 'mask_cls'))
        return losses

    def _mask_cls_head_forward_test(self,x,img_metas):
        return self.mask_cls_head.forward_test(x,img_metas)

    def get_mask_feats_and_logits(self, img, img_metas, **kwargs):
        img = img[0].cuda()
        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas,self.test_cfg,return_feat=True)
        mask, feat = out[0], out[1]
        mask_cls_feat = feat * torch.sigmoid(
            resize(mask.detach(), feat.shape[2:], mode='bilinear') / self.temperature
        )
        mask_cls_logit = self.mask_cls_head.classifier(mask_cls_feat)
        return mask_cls_feat, mask_cls_logit
    
    def get_patient_tensor(self, img, img_metas, **kwargs):
        img = img[0].cuda()
        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas,self.test_cfg,return_feat=True)
        mask, feat = out[0], out[1]
        cls_feat = x * torch.sigmoid(
            resize(mask.detach(), x.shape[2:], mode='bilinear') / self.temperature
        )
        pool_feat = torch._adaptive_avg_pool2d(cls_feat, (1,1)).squeeze()
        return pool_feat
    
    def encode_decode(self, img, img_metas,**kwargs):
        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas, self.test_cfg, return_feat=True)
        logit, feat = out[0], out[1]
        cls_feat = x * torch.sigmoid(
            resize(logit.detach(), x.shape[2:], mode='bilinear') / self.temperature
        )
        mask_cls_feat = feat * torch.sigmoid(
            resize(logit.detach(), feat.shape[2:], mode='bilinear') / self.temperature
        )
        cls_out = self._cls_head_forward_test(cls_feat,img_metas,**kwargs)
        mask_cls_out = self._mask_cls_head_forward_test(mask_cls_feat,img_metas,**kwargs)
        seg_out = resize(
            input=logit,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return cls_out,seg_out,mask_cls_out

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      cls,
                      mask_cls,
                      seg_weight=None,
                      return_feat=False):
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x
        
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight,
                                                     return_logit = True,
                                                     return_feat=True)
        logit = loss_decode.pop('logit')
        feat = loss_decode.pop('feat')
        cls_feat = x * torch.sigmoid(
            resize(logit.detach(), x.shape[2:], mode='bilinear') / self.temperature
        )
        mask_cls_feat = feat * torch.sigmoid(
            resize(logit.detach(), feat.shape[2:], mode='bilinear') / self.temperature
        )
        loss_cls = self._cls_head_forward_train(cls_feat,img_metas,cls)
        loss_mask_cls = self._mask_cls_head_forward_train(mask_cls_feat,img_metas,mask_cls)

        losses.update(loss_cls)
        losses.update(loss_decode)
        losses.update(loss_mask_cls)

        return losses

    def whole_inference(self, img, img_meta, rescale):
        cls_logit,seg_logit,mask_cls_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                if img_meta[0].get('BC_bbox',None) is not None:
                    y1,y2,x1,x2 = img_meta[0]['BC_bbox']
                    size = (y2-y1,x2-x1)
                else:
                    size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return cls_logit,seg_logit,mask_cls_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            cls_logit,seg_logit,mask_cls_logit = self.slide_inference(img, img_meta, rescale)
        else:
            cls_logit,seg_logit,mask_cls_logit = self.whole_inference(img, img_meta, rescale)

        cls_pred = F.sigmoid(cls_logit)
        output = F.sigmoid(seg_logit)
        mask_cls_logit = F.sigmoid(mask_cls_logit)

        return cls_pred,output,mask_cls_logit

    def simple_test(self, img, img_meta, rescale=True,**kwargs):
        """Simple test with single image."""
        cls_logit,seg_logit,mask_cls_logit = self.inference(img, img_meta, rescale)
        seg_pred = (seg_logit>0.5).long()
        if img_meta[0].get('BC_bbox',None) is not None:
            y1,y2,x1,x2 = img_meta[0]['BC_bbox']
            y,x = img_meta[0]['ori_shape'][:2]
            pad_size = (x1,x-x2,y1,y-y2)
            seg_pred = F.pad(input=seg_pred,
                            pad=pad_size,
                            mode='constant',
                            value=0)
        cls_pred = (cls_logit>0.5).long()
        mask_cls_pred = (mask_cls_logit>0.5).long()
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        cls_pred = cls_pred.cpu().numpy()
        mask_cls_pred = mask_cls_pred.cpu().numpy()
        
        # unravel batch dim
        seg_pred = list(seg_pred)
        cls_logit = cls_logit.cpu().numpy()
        cls_logit = cls_logit if isinstance(cls_logit,list) else [cls_logit]
        cls_pred = cls_pred if isinstance(cls_pred,list) else [cls_pred]
        mask_cls_pred = mask_cls_pred if isinstance(mask_cls_pred,list) else [mask_cls_pred]
        return cls_pred,seg_pred,mask_cls_pred,cls_logit

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        img = mmcv.imread(img)
        img = img.copy()
        opacity = 0.6
        cls = result[0][0][0]
        seg = result[1][0].squeeze()
        mask_cls = result[2][0][0]
        cls_logit = round(100*result[3][0][0],2)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in seg]))
        else:
            num_classes = len(self.CLASSES)
            
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(
                    0, 255, size=(num_classes, 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == num_classes
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        
        seg[seg==1] = mask_cls + 1 
        for label, color in enumerate(palette):
            if label==0:continue 
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img


@SEGMENTORS.register_module()
class BCALNClassifier(EncoderDecoder):
    def __init__(self, cls_head, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = build_head(cls_head)
        self.decode_head = nn.Sequential()
    
    def _cls_head_forward_train(self,x,img_metas,gt_cls):
        losses = dict()
        loss_cls = self.cls_head.forward_train(
            inputs=x,
            img_metas=img_metas,
            gt_cls=gt_cls
        )

        losses.update(add_prefix(loss_cls, 'cls'))
        return losses
    
    def _cls_head_forward_test(self,x,img_metas):
        return self.cls_head.forward_test(x,img_metas)

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        cls_logit = self._cls_head_forward_test(x ,img_metas)
        return cls_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      cls,
                      mask_cls,
                      seg_weight=None,
                      return_feat=False):
        x = self.extract_feat(img)

        losses = dict()
        loss_cls = self._cls_head_forward_train(x,img_metas,cls)
        losses.update(loss_cls)

        return losses
    
    def whole_inference(self, img, img_meta, rescale):
        cls_logit = self.encode_decode(img, img_meta)
        return cls_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            cls_logit,seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            cls_logit = self.whole_inference(img, img_meta, rescale)
            cls_logit = F.softmax(cls_logit,dim=1)
        return cls_logit

    def simple_test(self, img, img_meta, rescale=True,**kwargs):
        """Simple test with single image."""
        cls_logit = self.inference(img, img_meta, rescale)
        cls_pred = cls_logit.argmax(dim=1)
        cls_pred = cls_pred.cpu().numpy()
        cls_logit = cls_logit.cpu().numpy()[0,cls_pred]
        cls_logit = cls_logit if isinstance(cls_logit,list) else [cls_logit]
        cls_pred = cls_pred if isinstance(cls_pred,list) else [cls_pred]
        return cls_pred,None,cls_logit