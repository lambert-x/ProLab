# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.models.losses import accuracy
# from ..utils import clip_utils
import clip
import mmseg.models.decode_heads.uper_head

@HEADS.register_module()
class UPerHead_CLIP_T5(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), 
                 clip_model_name = 'ViT-B/32',
                 clip_weights_dict_path = 'ade_zeroshot_weights_dict.pth',
                 t5_model_name='t5-v1_1-base_decoder_llama2-7b_concat_desc_paper_template',
                 t5_weight_dict_path='embeddings/ade_desc_embeddings_llama2-7b_bank.pth',
                 clip_loss_weight=0.5,
                 t5_loss_weight=0.5,
                 dataset_names = None,
                 background_index_value=255,
                 split_embed=False,
                 **kwargs):
        super(UPerHead_CLIP_T5, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.clip_loss_weight = clip_loss_weight
        self.t5_loss_weight = t5_loss_weight
        
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


        clip_weights_paths_dict = {
            'ADE20K': 'ade_zeroshot_weights_dict.pth',
            'COCOStuff_164K': 'coco_zeroshot_weights_dict.pth',
            'PascalContext': 'pascalcontext59_zeroshot_weights_dict.pth',
            'cityscapes': 'cityscapes_zeroshot_weights_dict.pth',
            'mapillary_v1': 'mapillary_zeroshot_weights_dict.pth'
        }
        
        if clip_weights_dict_path is not None:
            assert dataset_names is None
            clip_weights_dict = torch.load(clip_weights_dict_path)
            clip_embedding_bank = clip_weights_dict[clip_model_name].float().t()
        else:
            assert dataset_names is not None
            # if len(dataset_names) > 1:
            #     assert background_index_value > 255 
            clip_weights_dict = dict()
            clip_embedding_bank = list()
            for dataset_name in dataset_names:
                assert dataset_name in clip_weights_paths_dict.keys()
                clip_weights_dict_path = clip_weights_paths_dict[dataset_name]
                clip_weights_dict = torch.load(clip_weights_dict_path)
                clip_embedding_bank.append(clip_weights_dict[clip_model_name].float().t())
            clip_embedding_bank = torch.cat(clip_embedding_bank, dim=0)
        clip_embedding_bank = clip_embedding_bank.cpu()
        self.num_classes = clip_embedding_bank.shape[0]
        feature_len = clip_embedding_bank.shape[1]
        self.clip_embedding_bank = torch.cat([clip_embedding_bank, torch.zeros(1, feature_len).to(clip_embedding_bank.device)]).cuda()
        
        t5_weight_dict = torch.load(t5_weight_dict_path)
        t5_embedding_bank = t5_weight_dict[t5_model_name].float().t()
        t5_embedding_bank = t5_embedding_bank.cpu()
        t5_feature_len = t5_embedding_bank.shape[1]
        self.t5_embedding_bank = torch.cat([t5_embedding_bank, torch.zeros(1, t5_feature_len).to(t5_embedding_bank.device)]).cuda()
        self.background_index_value = background_index_value
        self.split_embed = split_embed
        # assert self.dataset_name in ['ADE20K']
        # assert self.clip_model_name in clip.available_models()
        #
        #
        # self.clip_embedding_bank = torch.load(f'{self.dataset_name}_{self.clip_model_name}.pt').float().t()

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output

    @force_fp32(apply_to=('seg_embedding', ))
    def losses(self, seg_embedding, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_embedding = resize(
            input=seg_embedding,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_embedding, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        #seg_embedding Shape B x 512 x H X W
        #seg_label Shape B x H x W
        #clip_embedding_bank 150 x 512
        # cosine similarity loss
        #seg_label_clip_embedding = clip_embedding_bank[seg_label, :]  , shape B x 512 x H x W

        # cosine_simi_loss(seg_embedding, seg_label_clip_embedding)

        seg_label_clip = seg_label.clone()
        if self.ignore_index is not None:
            self.valid_mask = ((seg_label_clip >= 0) & (seg_label_clip != self.ignore_index)).float().clone()
        else:
            self.valid_mask = None
            
        seg_label_clip[seg_label_clip==self.background_index_value] = self.num_classes
        seg_label_t5_embedding = self.t5_embedding_bank[seg_label_clip, :].permute(0,3,1,2)

        # print('seg_label_clip shape: ', seg_label_clip.shape)
        # print('clip_embedding_bank shape: ', self.clip_embedding_bank.shape)
        # print('seg_label_clip unique values: ', torch.unique(seg_label_clip))
        
        seg_label_clip_embedding = self.clip_embedding_bank[seg_label_clip, :].permute(0,3,1,2)

        # print(self.valid_mask.shape)
        # print(seg_embedding.shape)
        # print(seg_label_clip_embedding.shape)
        if self.split_embed:
            seg_embedding_clip, seg_logit_t5 = torch.chunk(seg_embedding, 2, dim=1)
            seg_logit_clip = torch.einsum('bchw,nc->bnhw',seg_embedding_clip, self.clip_embedding_bank[:self.num_classes,:])      
            seg_logit_t5 = torch.einsum('bchw,nc->bnhw',seg_logit_t5, self.t5_embedding_bank[:self.num_classes,:])
        # print('logit_shape:', seg_logit.shape, 'seg_label_shape:', seg_label_clip_embedding.shape)
        for loss_decode in losses_decode:
            if 'cos_simi' in loss_decode.loss_name:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_embedding,
                        seg_label_clip_embedding,
                        seg_label_t5_embedding,
                        self.clip_loss_weight,
                        self.t5_loss_weight,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_embedding,
                        seg_label_clip_embedding,
                        seg_label_t5_embedding,
                        self.clip_loss_weight,
                        self.t5_loss_weight,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
            else:
                raise NotImplementedError
                # if loss_decode.loss_name not in loss:
                #     loss[loss_decode.loss_name] = loss_decode(
                #         seg_logit_clip,
                #         seg_label,
                #         weight=seg_weight,
                #         ignore_index=self.ignore_index)
                # else:
                #     loss[loss_decode.loss_name] += loss_decode(
                #         seg_logit_clip,
                #         seg_label,
                #         weight=seg_weight,
                #         ignore_index=self.ignore_index)
        
        # print(seg_logits.shape)
        # print(seg_label.shape)

        loss['acc_seg_clip'] = accuracy(seg_logit_clip, seg_label)
        loss['acc_seg_t5'] = accuracy(seg_logit_t5, seg_label)
        return loss

