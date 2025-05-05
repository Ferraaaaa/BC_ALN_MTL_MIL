norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim=256
model = dict(
    type='BCALNEncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    cls_head=dict(
        type='BCALNClsHead',
        num_classes=1,
        in_channels=embed_dim,
        need_cat=False,
        index=-1,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    mask_cls_head=dict(
        type='BCALNClsHead',
        num_classes=1,
        in_channels=embed_dim//8,
        need_cat=False,
        index=-1,
        norm_cfg=norm_cfg,
        loss_decode=dict( 
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    neck=dict(
        type='SegFormerNeck',
        in_channels=[256,512,1024,2048],
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
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
