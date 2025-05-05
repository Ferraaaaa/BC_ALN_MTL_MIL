# https://blog.csdn.net/sinat_37532065/article/details/103362517

import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
 

def draw_CAM(model, data, img_path, save_path, visual_heatmap=False):
    # feature/score
    model.eval()
    model.module = model.module.cuda()
    features, output = model.module.get_feats_and_logits(**data)

    # get gradient
    def extract(g):
        global features_grad
        features_grad = g
 
    # get pred
    pred = torch.sigmoid(output)
    features.register_hook(extract)
    pred.backward()
 
    grads = features_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # default bs=1
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]
 
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap) 
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) 
    cam_img = heatmap * 0.5 + img * 0.5
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, cam_img)
