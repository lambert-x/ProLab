# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)

# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
pretrained = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
model = dict(
    type='EncoderDecoder_cluster_embed',
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
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
        window_attn=[False] * 12,
        window_size=[None] * 12),
    decode_head=dict(
        type='UPerHead_cluster_embed',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=768,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=None,
        loss_decode=dict(
            type='CosineSimilarityLoss', loss_weight=1.0),
        desc_model_name='bge-base-en-v1.5_gpt3.5_reformated_cluster256_embeddings_and_labels',
        desc_weights_dict_path='ade_desc_all-MiniLM-L6-v2_gpt3.5_cluster_embedding_bank.pth',
        get_logit_mode='cosine_similarity_with_sigmoid',
        sigmoid_temperature=0.04,
        image_embedding_normalize=False,
        ),
    auxiliary_head=dict(
        type='FCNHead_cluster_embed',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=768,
        ignore_index=None,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CosineSimilarityLoss', loss_weight=0.4),
        desc_model_name='bge-base-en-v1.5_gpt3.5_reformated_cluster256_embeddings_and_labels',
        desc_weights_dict_path='ade_desc_all-MiniLM-L6-v2_gpt3.5_cluster_embedding_bank.pth',
        get_logit_mode='cosine_similarity_with_sigmoid',
        sigmoid_temperature=0.04,
        image_embedding_normalize=False,
        ),
    desc_model_name='bge-base-en-v1.5_gpt3.5_reformated_cluster256_embeddings_and_labels',
    desc_weights_dict_path='ade_desc_all-MiniLM-L6-v2_gpt3.5_cluster_embedding_bank.pth',
    get_logit_mode='cosine_similarity_with_sigmoid',
    sigmoid_temperature=0.04,
    image_embedding_normalize=False,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
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
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                constructor='LayerDecayOptimizerConstructor',
                paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
lr_config = dict(_delete_=True, policy='poly',
                warmup='linear',
                warmup_iters=1500,
                warmup_ratio=1e-6,
                power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2,
        val=dict(pipeline=test_pipeline),
        test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
