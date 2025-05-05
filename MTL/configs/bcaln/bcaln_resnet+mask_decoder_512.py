_base_ =[
    # datasets
    '../_base_/datasets/bcaln_512.py',
    # network backbone and classifier
    '../_base_/models/res50+mask_decoder.py',
    # default log config and training workflow config
    '../_base_/default_runtime.py',
    # Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0

size=512
model=dict(
    decode_head=dict(
        img_size=size,
        patch_size=8,
        freeze=False,
    ),
    test_cfg=dict(
        mode='whole',
        batched_slide=True,
        stride=[size//2,size//2],
        crop_size=[size,size],
    )
)
data=dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
# Optimizer Hyperparameters
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
work_dir = 'mtl_work_dirs/'
runner = dict(type='IterBasedRunner', max_iters=10000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
evaluation = dict(interval=2000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')
