# copyright see SAM
# some details of implementation or idea see SAM/amg.py

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
from scipy import ndimage
from torchvision.ops.boxes import batched_nms
from mmseg.ops import resize
import cv2

def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    '''see sam/amg.py for more details'''
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions

def get_dist_prob(img_size,gt_semantic_seg,alpha=0.2):
    alpha=np.clip(alpha,0.01,0.99)
    pos_matrix=np.ones(gt_semantic_seg.shape,dtype=np.bool)
    pos_matrix[:,img_size[0]//2,img_size[1]//2]=False
    dist_matrix=ndimage.distance_transform_edt(pos_matrix)
    sigma=dist_matrix.max()/np.log(alpha)
    # 标准正态分布是e**(-u**2/(2*sigma**2)))
    dist_prob=np.exp(dist_matrix/sigma)
    return dist_prob

def get_pixel_prob(gt_semantic_seg):
    h,w=gt_semantic_seg.shape[-2:]
    pos_prob=np.zeros(gt_semantic_seg.shape)
    classes,counts=np.unique(gt_semantic_seg,return_counts=True)
    class_wieght=np.log((h*w)/counts)
    for i in range(len(classes)):
        if classes[i]==255:
            continue
        pos_prob[gt_semantic_seg==classes[i]]=class_wieght[i]
    return pos_prob

def get_prompt(img_size,
               gt_semantic_seg=None,
               batch_size=2,
               mode='random-100',
               dev=None,
               alpha=0.2):
    mode,num= mode.split(sep='-')
    num=eval(num)

    if mode=='random':
        points=torch.as_tensor(np.array([[[each_x,each_y] for each_x,each_y in zip(
            np.random.choice(range(img_size[0]),num,replace=True),
            np.random.choice(range(img_size[1]),num,replace=True)
        )] for i in range(batch_size)]),device=dev)
    elif mode=='grid':
        grid_w,grid_h= img_size[0]//num,img_size[1]//num
        points=torch.as_tensor(np.array(
            [[[xi*grid_w+grid_w//2,yi*grid_h+grid_h//2]
                for xi in range(num) for yi in range(num)]
                    for i in range(batch_size)]),device=dev)
    elif mode=='guassrandom':
        assert gt_semantic_seg is not None,f'if mode is \'guassrandom\', gt should not be None'
        points=list()
        for i in range(batch_size):
            dist_prob=get_dist_prob(img_size,gt_semantic_seg[i],alpha).flatten()
            index=np.random.choice(a=np.arange(img_size[0]*img_size[1]),
                                   size=num,
                                   replace=False,
                                   p=dist_prob/dist_prob.sum())
            points.append(np.array([
                [each_x,each_y] for each_x,each_y in zip(index//img_size[0],index%img_size[0])
            ]))
        points=torch.as_tensor(points,device=dev)
    elif mode=='select':
        assert gt_semantic_seg is not None,f'if mode is \'select\', gt should not be None'
        points=list()
        for i in range(batch_size):
            dist_prob=get_dist_prob(img_size,gt_semantic_seg[i],alpha).flatten()
            pixel_prob=get_pixel_prob(gt_semantic_seg[i].cpu().numpy()).flatten()
            select_prob=dist_prob*pixel_prob
            index=np.random.choice(a=np.arange(img_size[0]*img_size[1]),
                                   size=num,
                                   replace=False,
                                   p=select_prob/select_prob.sum())
            points.append(np.array([
                [each_x,each_y] for each_x,each_y in zip(index//img_size[0],index%img_size[0])
            ]))
        points=torch.as_tensor(points,device=dev)
    else:
        raise NotImplementedError

    points_label=torch.ones(points.shape[:-1],device=dev)
    if gt_semantic_seg is not None:
        gt_cls= torch.cat([
            gt_semantic_seg[i,:,points[i].T[1],points[i].T[0]] 
            for i in range(batch_size)
        ])
    else:
        gt_cls= None
    
    return points,points_label,gt_cls

def get_gt_from_masks(low_res_masks,gt_semantic_seg):
    # [bs,B,H/4,W/4],[bs,1,H,W]
    bs,B,h,w=low_res_masks.shape
    gt_semantic_seg=resize(
        input=gt_semantic_seg.type(torch.float32),
        size=[h,w],
        mode='nearest'
    ).reshape(bs*1,h*w).long()
    low_res_masks=low_res_masks.reshape(bs,B,h*w)
    mask_gt=list()
    for i in range(bs):
        for j in range(B):
            classes,counts=torch.unique(gt_semantic_seg[i][low_res_masks[i,j]],
                                        return_counts=True)
            ids=torch.argmax(counts)
            if classes[ids]==255:
                counts[ids]=0
                ids=torch.argmax(counts)
            mask_gt.append(classes[ids])
    mask_gt=torch.as_tensor(mask_gt,device=gt_semantic_seg.device)
    mask_gt=mask_gt.reshape(bs,B)
    return mask_gt
    

def make_segmentation(low_res_masks,
                      cls_preds,
                      iou_preds,
                      stability_scores,
                      area,
                      img_size,
                      ignore_index,
                      threshold,
                      cls_gt,):
    cls_gt=cls_gt.type(torch.int32)

    bs=cls_preds.shape[0]
    dev=low_res_masks.device
    crop_mask=torch.ones((bs,img_size[0],img_size[1]),
                         device=dev,
                         dtype=torch.int32)*ignore_index
    final_score=(iou_preds+stability_scores)/torch.log10(area.type(torch.float32))
    for i in range(bs):
        _,cls_label=torch.max(cls_preds[i],dim=0)
        index=torch.argsort(final_score[i],descending=False)

        cls_label=cls_label.type(torch.int32)
        for each_index in index:
            each_mask=low_res_masks[i][each_index]
            crop_mask[i,each_mask]=cls_label[each_index]

    return crop_mask

def get_best_masks(masks,iou_preds):
    # mask:[bs,B,4,H,W]
    # iou:[bs,B,4]
    # index:[bs,B,4]--one_hot
    bs,B,n,H,W=masks.shape
    index=torch.argmax(iou_preds,dim=2)
    index=F.one_hot(index,num_classes=n)
    index=torch.where(index)
    
    masks = masks[index[0],index[1],index[2],:,:].view(bs,B,H,W)
    iou_preds = iou_preds[index[0],index[1],index[2]].view(bs,B)

    return masks,iou_preds

def get_fused_masks(masks,iou_preds):
    # mask:[bs,B,4,H,W]
    # iou:[bs,B,4]
    bs,B,n,H,W=masks.shape

    fused_masks=masks.flatten(0,2)*(iou_preds.flatten(0,2)).reshape(-1,1,1)
    fused_masks=fused_masks.unflatten(0,(bs,B,n)).mean(dim=2)
    fused_iou_preds=iou_preds.mean(dim=2)

    return fused_masks,fused_iou_preds

def data_filter(inputs,keep_index):
    keep_index=keep_index.flatten()
    for key in inputs.keys():
        inputs[key]=inputs[key][keep_index]
    return inputs

def mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    '''more details please refer to SAM/amg.py'''
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out

def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def postprocess_small_regions(mask_data, img_size, min_area, nms_thresh,mask_thresh):
    # Filter small disconnected regions and holes
    dev=mask_data['masks'].device
    mask_data['masks']=resize(
        input=mask_data['masks'].unsqueeze(0),
        size=img_size,
        mode='bilinear',
        align_corners=False,
    ).squeeze().cpu().numpy()
    mask_data['masks']=mask_data['masks']>mask_thresh
    new_masks = []
    scores = []
    for i in range(mask_data['masks'].shape[0]):    # [bsxB]
        mask = mask_data['masks'][i]
        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    # Recalculate boxes and remove any new duplicates
    masks = torch.cat(new_masks, dim=0)
    boxes = mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros(len(boxes)),  # categories
        iou_threshold=nms_thresh,
    )
    # Only recalculate RLEs for masks that have changed
    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_data['masks'][i_mask] = masks[i_mask]

    mask_data=data_filter(mask_data,keep_index=keep_by_nms)
    mask_data['masks']=torch.as_tensor(mask_data['masks'],device=dev)
    return mask_data

def calculate_area(masks):
    b,h,w=masks.shape
    area_list=list()
    for i in range(b):
        area_list.append(masks[i].sum())
    area=torch.as_tensor(area_list,device=masks.device)
    return area

def postprocess_masks(masks,
                      iou_preds,
                      cls_preds,
                      stability_scores,
                      cls_gt,
                      img_size,
                      mask_threshold=0,
                      iou_threshold=0,
                      score_threshold=0,
                      nms_threshold=0.7,
                      min_area_threshold=30,):
    bs=masks.shape[0]

    data=dict(masks=masks.flatten(0,1),
              iou_preds=iou_preds.flatten(),
              cls_preds=cls_preds.permute(0,2,1).flatten(0,1),
              stability_scores=stability_scores.flatten(),
              cls_gt=cls_gt.flatten())

    if iou_threshold>0:
        data=data_filter(data,data['iou_preds']>iou_threshold)

    if score_threshold>0:
        data=data_filter(data,data['stability_scores']>score_threshold)

    data['boxes']=mask_to_box(data['masks']>mask_threshold)
    keep_by_nms = batched_nms(
        data['boxes'].float(),
        data['iou_preds'],
        torch.zeros(len(data["boxes"])),  # categories
        iou_threshold=nms_threshold,)
    data=data_filter(data,keep_index=keep_by_nms)

    if min_area_threshold > 0:
        data = postprocess_small_regions(
            mask_data=data,
            img_size=img_size,
            min_area=min_area_threshold,
            nms_thresh=nms_threshold,
            mask_thresh=mask_threshold,
        )
    
    data['boxes']=mask_to_box(data['masks'])
    data['area']=calculate_area(data['masks'])

    final_num_masks=data['masks'].shape[0]
    unflatten_shape=(bs,final_num_masks//bs)
    masks,iou_preds,cls_preds,stability_scores,area,cls_gt=\
        (data['masks'].unflatten(0,unflatten_shape),
         data['iou_preds'].unflatten(0,unflatten_shape),
         data['cls_preds'].unflatten(0,unflatten_shape).permute(0,2,1),
         data['stability_scores'].unflatten(0,unflatten_shape),
         data['area'].unflatten(0,unflatten_shape),
         data['cls_gt'].unflatten(0,unflatten_shape),
         )
    
    return masks,iou_preds,cls_preds,stability_scores,area,cls_gt
