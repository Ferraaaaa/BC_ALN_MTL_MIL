norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim=256
model = dict(
    type='BCALNEncoderDecoder',
    pretrained='pretrain/MiT/mit_b2.pth',
    backbone=dict(type='mit_b2', style='pytorch'),
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
        in_channels=[64,128,320,512],
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
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))