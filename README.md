# Applying a Multi-task and Multi-instance Framework to Predict Axillary Lymph Node Metastases in Breast Cancer

### Environment
Our code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMClassification](https://github.com/open-mmlab/mmpretrain).

In this work, we used:
- python=3.8.5
- pytorch=2.0.1
- mmcv=1.7.1
- mmsegmentation=0.16.0
- mmclassification=0.18.0

The environment can be installed with:
```
conda create -n bcaln python=3.8.5
conda activate bcaln

pip install -r requirements.txt
chmod u+x scripts/*
```

### Dataset
Please organize the structure of dataset like this:
```
Dataset
|——images
|  |——train
|  |  |——N (Negative)
|  |  |  |——AAA (name)
|  |  |  |  |——T (TIs)
|  |  |  |  |  |——N_AAA_T_1.jpg
|  |  |  |  |  |——...
|  |  |  |  |——L (Lymph)
|  |  |  |  |  |——N_AAA_L_1.jpg
|  |  |  |  |  |——...
|  |  |  |——...
|  |  |——Y (Positive)
|  |  |  |——BBB (name)
|  |  |  |  |——T (TIs)
|  |  |  |  |  |——Y_BBB_T_1.jpg
|  |  |  |  |  |——...
|  |  |  |  |——L (Lymph)
|  |  |  |  |  |——Y_BBB_L_1.jpg
|  |  |  |  |  |——...
|  |  |  |——...
|  |——val
|  |——test
|——labels
|  |——train
|  |  |——N (Negative)
|  |  |  |——AAA (name)
|  |  |  |  |——T (TIs)
|  |  |  |  |  |——N_AAA_T_1.png
|  |  |  |  |  |——...
|  |  |  |  |——L (Lymph)
|  |  |  |  |  |——N_AAA_L_1.png
|  |  |  |  |  |——...
|  |  |  |——...
|  |  |——Y (Positive)
|  |  |  |——BBB (name)
|  |  |  |  |——T (TIs)
|  |  |  |  |  |——Y_BBB_T_1.jpg
|  |  |  |  |  |——...
|  |  |  |  |——L (Lymph)
|  |  |  |  |  |——Y_BBB_L_1.jpg
|  |  |  |  |  |——...
|  |  |  |——...
|  |——val
|  |——test
```

Then the structure of the project could be like this:
```
BCALN
|——MIL
|——MTL
|——scripts
|——data
|  |——Dataset
```
Or you can change the `data_root` in the MTL config files to the dir path to your dataset.

### Training & Testing
To train or test a model, you can use the `.sh` files in `./scripts` dir. 

The trained models of MTL stage will be stored to `./mtl_work_dirs`. The trained models of MIL stage will be stored to `./mil_work_dirs`. The inference results will be stored to `./results`.

To train the full stage, you can use the following commands:
```
# train
CUDA_VISIBLE_DEVICES=GPU_IDs PORT=12345 scripts/train_all.sh MTL_config MIL_config GPU_num 
# example
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/train_all.sh MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py MIL/configs/bcaln/bcaln_vit_segformer.py 4
```

Or you can train the models separately, you can use commands like this:
```
# train MTL
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/train_mtl.sh MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py 4

# inference MTL
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/inference_mtl.sh MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py

# train MIL
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/train_mil.sh MIL/configs/bcaln/bcaln_vit_segformer.py 4

# inference MIL
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/inference_mil.sh MIL/configs/bcaln/bcaln_vit_segformer.py
```

To test a MTL model or a MIL model, you can use commands like this:
```
# test MTL
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/test_mtl.sh MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py mtl_work_dirs/bcaln_segformer+mask_decoder_512/latest.pth 4

# test MIL
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/test_mil.sh MIL/configs/bcaln/bcaln_vit_segformer.py mil_work_dirs/bcaln_vit_segformer/latest.pth 4
```

For inference, you can use the following commands:
```
# save to pt file
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/inference_mtl.sh MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py

# save predictions to csv file
CUDA_VISIBLE_DEVICES=0 PORT=12345 MTL/tools/test_with_visualization.py MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py mtl_work_dirs/bcaln_segformer+mask_decoder_512/latest.pth --save_csv

# save cam visualization
CUDA_VISIBLE_DEVICES=0 PORT=12345 MTL/tools/test_with_visualization.py MTL/configs/bcaln/bcaln_segformer+mask_decoder_512.py mtl_work_dirs/bcaln_segformer+mask_decoder_512/latest.pth --cam

# save patient preds
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 scripts/inference_mil.sh MIL/configs/bcaln/bcaln_vit_segformer.py
```
These results will be saved to `results/saved_pt`, `results/mtl_preds`, `results/visualization` and `results/patient_preds`, respectively.

### Acknowledgements
Last, we thank these authors for sharing their source code:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMClassification](https://github.com/open-mmlab/mmpretrain)
- [tb-wsi-classifier](https://github.com/CVIU-CSU/icassp2023_tb-wsi-classifier) (mainly for MIL training stage)
