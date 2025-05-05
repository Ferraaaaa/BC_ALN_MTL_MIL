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
from .custom import CustomDataset
from .pipelines import Compose
from .builder import build_dataset

@DATASETS.register_module()
class BCDataset(CustomDataset):

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
            if img_info['filename'].startswith('Y'):
                img_info['cls'] = 1
            elif img_info['filename'].startswith('N'):
                img_info['cls'] = 0
            else:
                raise NotImplementedError
            
        return img_infos
    
    def get_cls_gt(self):
        return np.array(list(map(lambda x:x['cls'],self.img_infos)))

    def get_patient_wise_cls_gt(self):
        patient_wise_cls_gt = dict()
        for img_info in self.img_infos:
            each_cls = img_info['cls']
            filename = img_info['filename'].split(sep='/')[0]
            patient_wise_cls_gt[filename] = \
                patient_wise_cls_gt.get(filename,False) or each_cls
        return patient_wise_cls_gt
    
    def prepare_train_img(self, idx):
        results=super().prepare_train_img(idx)
        results['cls']=self.img_infos[idx]['cls']
        return results

    # def prepare_test_img(self, idx):
    #     results=super().prepare_test_img(idx)
    #     results['cls']=self.img_infos[idx]['cls']
    #     return results

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU', 'mIoU2']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        cls_results,seg_results=[results[i][0][0] for i in range(len(self))],\
                                [results[i][1][0] for i in range(len(self))]
        gt_cls = self.get_cls_gt()
        cls_results= np.array(cls_results).squeeze()

        # patient-level gt
        patient_wise_cls_gt_dict = self.get_patient_wise_cls_gt()
        patient_names,patient_wise_cls_gt = \
            list(patient_wise_cls_gt_dict.keys()),np.array(list(patient_wise_cls_gt_dict.values()))
        
        all_names = list(map(lambda x:x['filename'].split(sep='/')[0],self.img_infos))
        patient_results_dict = dict()
        for i in range(len(cls_results)):
            patient_results_dict[all_names[i]] = \
                patient_results_dict.get(all_names[i],False) or cls_results[i]
        patient_results = np.array(list(patient_results_dict.values()))
        per_patient_acc = 100*np.mean(patient_results==patient_wise_cls_gt)

        # save to csv file
        save_csv = kwargs.get('save_csv',False)
        if save_csv:
            csv_dict = {}
            csv_dict['patient'] = patient_names
            csv_dict['gt'] = patient_wise_cls_gt
            csv_dict['pred'] = patient_results
            pd_csv = pd.DataFrame(csv_dict)
            save_dir = kwargs.get('save_dir',None)
            assert save_dir is not None,f'if save_csv is True, save_dir should not be None'
            save_path = os.path.join(save_dir,
                    time.strftime('%Y%m%d_%H%M%S', time.localtime())+\
                        '-'+'patient_pred.csv')
            pd_csv.to_csv(save_path)
            print(f'\nSuccessfully saved to {save_path}')

        gt_seg_maps = self.get_gt_seg_maps()
        for each_map in gt_seg_maps:
            each_map[each_map==255]=1

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        iou, f1, ppv, s = metrics(
            seg_results, gt_seg_maps, num_classes,
            use_sigmoid=False,
            ignore_index=self.ignore_index)  # evaluate
        
        summary_str = ''
        summary_str += '\n\nper patient acc:{:.2f}\n'.format(per_patient_acc)
        summary_str += 'results of acc:{:.2f}\n'.format(100*np.mean(cls_results==gt_cls))
        summary_str += 'per class results:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'F1', 'PPV', 'S')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
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

# https://mmclassification.readthedocs.io/zh_CN/latest/_modules/mmcls/datasets/dataset_wrappers.html
# #ClassBalancedDataset
@DATASETS.register_module()
class BCBalancedDataset(object):

    def __init__(self, dataset, oversample_thr=0.2):
        self.dataset = build_dataset(dataset) 
        self.oversample_thr = oversample_thr

        repeat_factors = self._get_repeat_factors(self.dataset,oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices 

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        category_freq = defaultdict(int)

        num_images = len(dataset)
        cls_gts = dataset.get_cls_gt()

        for cls_gt in cls_gts:
            category_freq[cls_gt] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }
        # 
        repeat_factors = []
        for cls_gt in cls_gts:
            repeat_factor = category_repeat[cls_gt]
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        return self.dataset(results,metric,logger, **kwargs)

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (
            f'\n{self.__class__.__name__} ({self.dataset.__class__.__name__}) '
            f'{dataset_type} dataset with total number of samples {len(self)}.'
        )
        return result

