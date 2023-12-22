# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_beit.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth'
pretrained = 'pretrained/beit_large_patch16_224_pt22k_ft22k.pth'

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder_cluster_embed',
    pretrained=pretrained,
    backbone=dict(
        type='BEiTAdapter',
        img_size=640,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    ),
    decode_head=dict(
        type='UPerHead_cluster_embed',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=768,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        ignore_index=None,
        loss_decode=dict(
            type='CosineSimilarityLoss', loss_weight=1.0),
        desc_model_name='bge-base-en-v1.5_gpt3.5_reformated_cluster256_embeddings_and_labels',
        desc_weights_dict_path='ade_desc_all-MiniLM-L6-v2_gpt3.5_cluster_embedding_bank.pth',
        get_logit_mode='cosine_similarity_with_sigmoid',
        sigmoid_temperature=0.01,
        image_embedding_normalize=True,
        ),
    auxiliary_head=dict(
        type='FCNHead_cluster_embed',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=768,
        ignore_index=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CosineSimilarityLoss', loss_weight=0.4),
        desc_model_name='bge-base-en-v1.5_gpt3.5_reformated_cluster256_embeddings_and_labels',
        desc_weights_dict_path='ade_desc_all-MiniLM-L6-v2_gpt3.5_cluster_embedding_bank.pth',
        get_logit_mode='cosine_similarity_with_sigmoid',
        sigmoid_temperature=0.01,
        image_embedding_normalize=True,
        ),
    desc_model_name='bge-base-en-v1.5_gpt3.5_reformated_cluster256_embeddings_and_labels',
    desc_weights_dict_path='ade_desc_all-MiniLM-L6-v2_gpt3.5_cluster_embedding_bank.pth',
    get_logit_mode='cosine_similarity_with_sigmoid',
    sigmoid_temperature=0.01,
    image_embedding_normalize=True,
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426))
)



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2,
        train=dict(pipeline=train_pipeline),
        val=dict(pipeline=test_pipeline),
        test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
