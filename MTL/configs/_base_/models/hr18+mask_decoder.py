norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim=256
model = dict(
    type='BCALNEncoderDecoder',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    cls_head=dict(
        type='BCALNClsHead',
        num_classes=1,
        in_channels=embed_dim,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    mask_cls_head=dict( 
        type='BCALNClsHead',
        num_classes=1,
        in_channels=embed_dim//8,
        norm_cfg=norm_cfg,
        loss_decode=dict( 
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    neck=dict(
        type='SegFormerNeck',
        in_channels=[18, 36, 72, 144],
        out_channel=embed_dim,
        index=1,
    ),
    decode_head=dict(
        type='SAMMaskDecoder',
        transformer_dim=embed_dim,
        patch_size=8,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        repeat_times=1,
        num_classes=1,
        loss_decode=dict( 
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
train_cfg = dict(),
test_cfg = dict(mode='whole'))
