# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmseg.models.losses import accuracy
from mmseg.models.decode_heads.aspp_head import ASPPHead, ASPPModule


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHead_cluster_embed(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, 
                 desc_model_name = 'ViT-B/32',
                 desc_weights_dict_path = 'ade_zeroshot_weights_dict.pth',
                 background_index_value=255,
                 dataset_names=None,
                 get_logit_mode='cosine_similarity',
                 sigmoid_temperature=1.0,
                 image_embedding_normalize=True,
                 **kwargs):
        super(DepthwiseSeparableASPPHead_cluster_embed, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        
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
            desc_clustered_embedding_bank = desc_clustered_embedding_bank.float().t()
            desc_cluster2label = desc_cluster2label.float().t()
        else:
            raise NotImplementedError
        

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
        

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output
    
    @force_fp32(apply_to=('seg_embedding',))
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

        # seg_embedding Shape B x H X W x 512
        # seg_label Shape B x H x W
        # desc_clustered_embedding_bank 150 x 512
        # cosine similarity loss
        # seg_label_desc = desc_clustered_embedding_bank[seg_label, :]  , shape B x H x W x 512
        # cosine_simi_loss(seg_embedding, seg_label_desc)
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
        #  clip_clustered_embedding_bank 384x256
        #  clip_cluster2label 151x256, multi-hot encoding
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