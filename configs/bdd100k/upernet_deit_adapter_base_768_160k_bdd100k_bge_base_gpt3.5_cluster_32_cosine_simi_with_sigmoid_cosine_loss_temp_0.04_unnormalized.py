norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_cluster_embed',
    pretrained='pretrained/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(
        type='ViTAdapter',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            False, False, False, False, False, False, False, False, False,
            False, False, False
        ],
        window_size=[
            None, None, None, None, None, None, None, None, None, None, None,
            None
        ]),
    decode_head=dict(
        type='UPerHead_cluster_embed',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=768,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CosineSimilarityLoss', use_sigmoid=False, loss_weight=1.0),
        ignore_index=None,
        desc_model_name=
        'bge-base-en-v1.5_gpt3.5_cluster_32_embeddings_and_labels',
        desc_weights_dict_path=
        'embeddings/cityscapes_bdd_desc_bge-base-en-v1.5_gpt3.5_cluster_32_embedding_bank.pth',
        get_logit_mode='cosine_similarity_with_sigmoid',
        sigmoid_temperature=0.04,
        image_embedding_normalize=False),
    auxiliary_head=dict(
        type='FCNHead_cluster_embed',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=768,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CosineSimilarityLoss', use_sigmoid=False, loss_weight=0.4),
        ignore_index=None,
        desc_model_name=
        'bge-base-en-v1.5_gpt3.5_cluster_32_embeddings_and_labels',
        desc_weights_dict_path=
        'embeddings/cityscapes_bdd_desc_bge-base-en-v1.5_gpt3.5_cluster_32_embedding_bank.pth',
        get_logit_mode='cosine_similarity_with_sigmoid',
        sigmoid_temperature=0.04,
        image_embedding_normalize=False),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        crop_size=(768, 768),
        stride=(512, 512),
        test_dataset_name='cityscapes'),
    desc_model_name='bge-base-en-v1.5_gpt3.5_cluster_32_embeddings_and_labels',
    desc_weights_dict_path=
    'embeddings/cityscapes_bdd_desc_bge-base-en-v1.5_gpt3.5_cluster_32_embedding_bank.pth',
    get_logit_mode='cosine_similarity_with_sigmoid',
    sigmoid_temperature=0.04,
    image_embedding_normalize=False)
dataset_type = 'BDD100K_Dataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(768, 768), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(768, 768), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='BDD100K_Dataset',
        data_root='data/bdd100k/',
        img_dir='images/10k/train',
        ann_dir='labels/sem_seg/masks/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(768, 768), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(768, 768), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='BDD100K_Dataset',
        data_root='data/bdd100k/',
        img_dir='images/10k/val',
        ann_dir='labels/sem_seg/masks/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='BDD100K_Dataset',
        data_root='data/bdd100k/',
        img_dir='images/10k/val',
        ann_dir='labels/sem_seg/masks/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
nchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(
    interval=16000, metric='mIoU', pre_eval=True, save_best='mIoU')
pretrained = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
work_dir = './work_dirs/bdd100k/upernet_deit_adapter_base_768_160k_bdd100k_bge_base_gpt3.5_cluster_32_cosine_simi_with_sigmoid_cosine_loss_temp_0.04_unnormalized/'
gpu_ids = range(0, 8)
auto_resume = False
