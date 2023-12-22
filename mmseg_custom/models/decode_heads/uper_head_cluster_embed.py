# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.models.losses import accuracy
# from ..utils import desc_utils
import clip
import mmseg.models.decode_heads.uper_head
import numpy as np

@HEADS.register_module()
class UPerHead_cluster_embed(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), 
                 desc_model_name = 'ViT-B/32',
                 desc_weights_dict_path = 'ade_zeroshot_weights_dict.pth',
                 dataset_names = None,
                 background_index_value=255,
                 get_logit_mode='cosine_similarity',
                 sigmoid_temperature=1.0,
                 image_embedding_normalize=True,
                 **kwargs):
        super(UPerHead_cluster_embed, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module

        
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


        desc_weights_paths_dict = {
            'ADE20K': 'ade_zeroshot_weights_dict.pth',
            'COCOStuff_164K': 'coco_zeroshot_weights_dict.pth',
            'PascalContext': 'pascalcontext59_zeroshot_weights_dict.pth',
            'cityscapes': 'cityscapes_zeroshot_weights_dict.pth',
            'mapillary_v1': 'mapillary_zeroshot_weights_dict.pth'
        }
        
        if desc_weights_dict_path is not None:
            assert dataset_names is None
            desc_weights_dict = torch.load(desc_weights_dict_path)
            # desc_clustered_embedding_bank 384x256
            # desc_cluster2label 256x150, multi-hot encoding
            desc_clustered_embedding_bank, desc_cluster2label = desc_weights_dict[desc_model_name]
            if type(desc_clustered_embedding_bank) == np.ndarray:
                desc_clustered_embedding_bank = torch.from_numpy(desc_clustered_embedding_bank)
            if type(desc_cluster2label) == np.ndarray:
                desc_cluster2label = torch.from_numpy(desc_cluster2label)
            desc_clustered_embedding_bank = desc_clustered_embedding_bank.float().t()
            desc_cluster2label = desc_cluster2label.float().t()
        else:
            raise NotImplementedError
            # assert dataset_names is not None
            # # if len(dataset_names) > 1:
            # #     assert background_index_value > 255 
            # desc_weights_dict = dict()
            # desc_clustered_embedding_bank = list()
            # for dataset_name in dataset_names:
            #     assert dataset_name in desc_weights_paths_dict.keys()
            #     desc_weights_dict_path = desc_weights_paths_dict[dataset_name]
            #     desc_weights_dict = torch.load(desc_weights_dict_path)
            #     desc_clustered_embedding_bank.append(desc_weights_dict[desc_model_name].float().t())
            # desc_clustered_embedding_bank = torch.cat(desc_clustered_embedding_bank, dim=0)
        # print(desc_clustered_embedding_bank.shape)
        desc_clustered_embedding_bank = desc_clustered_embedding_bank.cpu()
        
        feature_len = desc_clustered_embedding_bank.shape[1]
        cluster_num = desc_cluster2label.shape[1]
        self.num_classes = desc_cluster2label.shape[0]
        assert cluster_num == desc_clustered_embedding_bank.shape[0]
        # desc_cluster2label 150x256, multi-hot encoding, append a zero vector to the end for background
        self.desc_cluster2label = torch.cat([desc_cluster2label, torch.zeros(1, cluster_num).to(desc_clustered_embedding_bank.device)]).cuda()
        self.desc_clustered_embedding_bank = desc_clustered_embedding_bank.cuda()
        self.desc_clustered_embedding_bank = F.normalize(self.desc_clustered_embedding_bank, p=2, dim=1)
        self.background_index_value = background_index_value
        self.get_logit_mode = get_logit_mode
        self.sigmoid_temperature = sigmoid_temperature
        self.image_embedding_normalize = image_embedding_normalize
        
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
        #desc_embedding_bank 150 x 512
        # cosine similarity loss
        #seg_label_cluster_embed_embedding = desc_embedding_bank[seg_label, :]  , shape B x 512 x H x W

        # cosine_simi_loss(seg_embedding, seg_label_cluster_embed_embedding)

        seg_label_desc = seg_label.clone()
        if self.ignore_index is not None:
            self.valid_mask = ((seg_label_desc >= 0) & (seg_label_desc != self.ignore_index)).float().clone()
        else:
            self.valid_mask = None
        # print(torch.unique(seg_label_desc))
        
        seg_label_desc[seg_label_desc==self.background_index_value] = self.num_classes
        # print('seg_label_desc shape: ', seg_label_desc.shape)
        # print('desc_clustered_embedding_bank shape: ', self.desc_clustered_embedding_bank.shape)
        # print('seg_label_desc unique values: ', torch.unique(seg_label_desc))
        seg_label_desc = self.desc_cluster2label[seg_label_desc, :].permute(0,3,1,2)
        if self.image_embedding_normalize:
            seg_embedding = F.normalize(seg_embedding, p=2, dim=1)
        seg_logit_cluster = torch.einsum('bchw,kc->bkhw', seg_embedding, self.desc_clustered_embedding_bank)
        #  bx256xhw-> bx150xhw
        #options: L1 / L2 / Cosine / Dot
        #  desc_clustered_embedding_bank 384x256
        #  desc_cluster2label 151x256, multi-hot encoding
        if self.get_logit_mode == 'cosine_similarity':
            desc_cluster2label_normalized = F.normalize(self.desc_cluster2label, p=2, dim=1)
            seg_logit_cluster_normalized = F.normalize(seg_logit_cluster, p=2, dim=1)
            # out = torch.einsum('bkhw,nk->bnhw', out, (self.desc_cluster2label / self.desc_cluster2label.sum(dim=-1, keepdim=True)))
            seg_logit = torch.einsum('bkhw,nk->bnhw', seg_logit_cluster_normalized, desc_cluster2label_normalized)
        elif self.get_logit_mode == 'cosine_similarity_with_sigmoid':
            seg_logit_cluster = torch.sigmoid(seg_logit_cluster / self.sigmoid_temperature)
            desc_cluster2label_normalized = F.normalize(self.desc_cluster2label, p=2, dim=1)
            seg_logit_cluster_normalized = F.normalize(seg_logit_cluster, p=2, dim=1)
            # out = torch.einsum('bkhw,nk->bnhw', out, (self.desc_cluster2label / self.desc_cluster2label.sum(dim=-1, keepdim=True)))
            seg_logit = torch.einsum('bkhw,nk->bnhw', seg_logit_cluster_normalized, desc_cluster2label_normalized)
        elif self.get_logit_mode == 'L2_with_sigmoid':
            B, K, H, W = seg_logit_cluster.shape
            N = self.desc_cluster2label.shape[0]
            seg_logit_cluster = torch.sigmoid(seg_logit_cluster / self.sigmoid_temperature)
            seg_logit = torch.einsum('bkhw->bhwk', seg_logit_cluster)
            cluster_num = seg_logit.shape[-1]
            seg_logit = seg_logit.reshape(-1, cluster_num)
            diff = seg_logit.unsqueeze(1) - self.desc_cluster2label.unsqueeze(0)
            l2_dist = torch.norm(diff, dim=2) / torch.sqrt(torch.tensor(cluster_num).float())
            seg_logit = (1 - l2_dist).mean(dim=2)
            seg_logit = seg_logit.view(B, N, H, W)
        else:
            raise NotImplementedError
        
        for loss_decode in losses_decode:
            if 'cos_simi' in loss_decode.loss_name:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit_cluster,
                        seg_label_desc,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit_cluster,
                        seg_label_desc,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
            elif 'MSE' in loss_decode.loss_name or 'BCE' in loss_decode.loss_name:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit_cluster,
                        seg_label_desc,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit_cluster,
                        seg_label_desc,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
            else:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
        
        # print(seg_logits.shape)
        # print(seg_label.shape)

        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

