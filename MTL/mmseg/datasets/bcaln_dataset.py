import math
import mmcv
import numpy as np
import os
import os.path as osp
import pandas as pd
import time
import torch
from collections import OrderedDict, defaultdict
from functools import reduce
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from ..core.evaluation import metrics
from .builder import DATASETS
from .builder import build_dataset
from .bc_dataset import BCDataset


@DATASETS.register_module()
class BCALNDataset(BCDataset):

    def __init__(self, **kwargs):
        super(BCDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,split):
        img_infos=super().load_annotations(
            img_dir, img_suffix, ann_dir, seg_map_suffix,split
        )
        for img_info in img_infos:
            class_name = img_info['filename'].split(sep='/')[-2]
            if class_name=='T':
                img_info['mask_cls'] = 0
            elif class_name=='L':
                img_info['mask_cls'] = 1
            else:
                raise NotImplementedError
            
        return img_infos
    
    def prepare_train_img(self, idx):
        results=super().prepare_train_img(idx)
        results['mask_cls']=self.img_infos[idx]['mask_cls']
        return results

    def get_mask_cls_gt(self):
        return np.array(list(map(lambda x:x['mask_cls'],self.img_infos)))

    def get_patient_wise_cls_gt(self):
        patient_wise_cls_gt = dict()
        for img_info in self.img_infos:
            each_cls = img_info['cls']
            filename = img_info['filename'].split(sep='/')[1]
            patient_wise_cls_gt[filename] = \
                patient_wise_cls_gt.get(filename,False) or each_cls
        return patient_wise_cls_gt

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU', 'mIoU2']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        
        assert len(self)==len(results)
        eval_results = {}
        cls_results = [results[i][0][0] for i in range(len(self))]
        seg_results = [results[i][1][0] for i in range(len(self))]
        mask_cls_results = [results[i][2][0] for i in range(len(self))]
        cls_results= np.array(cls_results).squeeze()
        mask_cls_results = np.array(mask_cls_results).squeeze()
        gt_cls = self.get_cls_gt()
        gt_seg_maps = self.get_gt_seg_maps()
        gt_mask_cls = self.get_mask_cls_gt()
        patient_wise_cls_gt_dict = self.get_patient_wise_cls_gt()
        patient_names,patient_wise_cls_gt = \
            list(patient_wise_cls_gt_dict.keys()),np.array(list(patient_wise_cls_gt_dict.values()))

        # patient level classification
        all_names = list(map(lambda x:x['filename'].split(sep='/')[1],self.img_infos))
        patient_results_dict = dict()
        for i in range(len(cls_results)):
            patient_results_dict[all_names[i]] = \
                patient_results_dict.get(all_names[i],False) or cls_results[i]
        patient_results = np.array(list(patient_results_dict.values()))
        per_patient_acc = 100*np.mean(patient_results==patient_wise_cls_gt)

        # each lesion
        lesion_acc = []
        lesion_acc.append(np.mean(gt_mask_cls==mask_cls_results))
        for i in range(len(reduce(np.union1d, [np.unique(_) for _ in gt_mask_cls]))):
            per_lesion_acc = (gt_mask_cls==i) & (mask_cls_results==i)
            lesion_acc.append(np.sum(per_lesion_acc)/np.sum(gt_mask_cls==i))
        
        # convert gt and pred
        for gt,gt_i,pred,pred_i in zip(gt_seg_maps,gt_mask_cls,seg_results,mask_cls_results):
            gt[gt==255] = gt_i + 1 
            pred[pred==1] = pred_i+1

        # save results to csv file
        save_csv = kwargs.get('save_csv',False)
        if save_csv:
            csv_dict = {}
            csv_dict['patient'] = patient_names
            csv_dict['gt'] = patient_wise_cls_gt
            csv_dict['pred'] = patient_results
            pd_csv = pd.DataFrame(csv_dict)
            save_dir = kwargs.get('save_dir',None)
            save_path = os.path.join(save_dir,
                    time.strftime('%Y%m%d_%H%M%S', time.localtime())+\
                        '-'+'patient_pred.csv')
            pd_csv.to_csv(save_path)
            print(f'\nSuccessfully saved to {save_path}')

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        iou, f1, ppv, s = metrics(
            seg_results, gt_seg_maps, num_classes,
            use_sigmoid=False,
            ignore_index=self.ignore_index)  # evaluate
        
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        summary_str = ''
        summary_str += '\n\nper patient acc:{:.2f}\n'.format(per_patient_acc)
        summary_str += 'results of acc:{:.2f}\n'.format(100*np.mean(cls_results==gt_cls))
        summary_str += 'mask classification result:\n'
        summary_str += 'Overall:{:<10}'.format(round(lesion_acc[0]*100,2))
        for i in range(1,num_classes):
            summary_str += '{}:{:<10}'.format(class_names[i],round(lesion_acc[i]*100,2))
        summary_str += '\n'
        summary_str += 'per class results:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'F1', 'PPV', 'S')
        for i in range(num_classes):
            ppv_str = '{:.2f}'.format(ppv[i] * 100)
            s_str = '{:.2f}'.format(s[i] * 100)
            f1_str = '{:.2f}'.format(f1[i] * 100)
            iou_str = '{:.2f}'.format(iou[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, f1_str, ppv_str, s_str)

        mIoU = np.nanmean(np.nan_to_num(iou[-4:], nan=0))
        mF1 = np.nanmean(np.nan_to_num(f1[-4:], nan=0))
        mPPV = np.nanmean(np.nan_to_num(ppv[-4:], nan=0))
        mS = np.nanmean(np.nan_to_num(s[-4:], nan=0))

        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mF1', 'mPPV', 'mS')

        iou_str = '{:.2f}'.format(mIoU * 100)
        f1_str = '{:.2f}'.format(mF1 * 100)
        ppv_str = '{:.2f}'.format(mPPV * 100)
        s_str = '{:.2f}'.format(mS * 100)
        summary_str += line_format.format('global', iou_str, f1_str, ppv_str, s_str)

        eval_results['mIoU'] = mIoU
        eval_results['mF1'] = mF1
        eval_results['mPPV'] = mPPV
        eval_results['mS'] = mS

        # NEW: for two classes metric
        if metric == 'mIoU2':
            summary_str += '\n'

        print_log(summary_str, logger)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return eval_results
