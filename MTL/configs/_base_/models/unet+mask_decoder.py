# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim=256
model = dict(
    type='BCALNEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
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
        in_channels=[64,128,256,512,1024],
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
    test_cfg=dict(mode='slide', crop_size=256, stride=170))