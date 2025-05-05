
# dataset settings
dataset_type = 'BCALNDataset'
data_root = 'data/bcaln'

img_norm_cfg = dict(
    mean=[39.02,39.34,41.44], std=[54.45,53.53,53.74], to_rgb=True)
CLASSES = ['bg' , 'TIS' , 'Lymph']
PALETTE = [[0,0,0],
           [255,255,0],
           [0,255,255]]
img_scale = crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BCALNLoadAnnotations'),
    dict(type='Resize', img_scale=img_scale,ratio_range=(0.8,1.25)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='labels/train',
        classes = CLASSES,
        palette = PALETTE,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        classes = CLASSES,
        palette = PALETTE,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='labels/test',
        classes = CLASSES,
        palette = PALETTE,
        pipeline=test_pipeline),
    inference = [
        dict(type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='labels/train',
            classes = CLASSES,
            palette = PALETTE,
            pipeline=test_pipeline),
        dict(type=dataset_type,
            data_root=data_root,
            img_dir='images/val',
            ann_dir='labels/val',
            classes = CLASSES,
            palette = PALETTE,
            pipeline=test_pipeline),
        dict(type=dataset_type,
            data_root=data_root,
            img_dir='images/test',
            ann_dir='labels/test',
            classes = CLASSES,
            palette = PALETTE,
            pipeline=test_pipeline),
    ]
)
