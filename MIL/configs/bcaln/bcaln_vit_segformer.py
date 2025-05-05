_base_ = [
    '../_base_/models/bcaln_vit-base-p16.py',
    '../_base_/schedules/adamw.py',
    '../_base_/default_runtime.py'
]
classes = [
    'neg',
    'pos'
]
model = dict(backbone=dict(patch_num=200, arch='ts_6'))

dataset_type = 'BCALNClsDataset'
data_root = 'results/saved_pt/bcaln_segformer+mask_decoder_512/'

train_pipeline = [
    dict(type='LoadPtFromFile_vit', self_normalize=True),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadPtFromFile_vit', self_normalize=True),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    #samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val = [
        dict(type=dataset_type,
            classes=classes,
            data_prefix=data_root + 'val',
            pipeline=test_pipeline),
        dict(type=dataset_type,
            classes=classes,
            data_prefix=data_root + 'test',
            pipeline=test_pipeline)
    ],
    test=dict(type=dataset_type,
            classes=classes,
            data_prefix=data_root + 'test',
            pipeline=test_pipeline),
)

evaluation = dict(
    interval=20,
    metric=['accuracy', 'precision', 'recall', 'f1_score'],
    metric_options=dict(
        topk=(1,),  # topk=(1, 5),
        choose_classes=[1]
    )
)
# seed
seed=0
#find_unused_paramters=True
work_dir = 'mil_work_dirs/'
runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(interval=200)
checkpoint_config = dict(interval=100)