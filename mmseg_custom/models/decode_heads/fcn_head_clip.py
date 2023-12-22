# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.ops import resize


@HEADS.register_module()
class FCNHead_CLIP(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 clip_model_name = 'ViT-B/32',
                 clip_weights_dict_path = 'ade_zeroshot_weights_dict.pth',
                 background_index_value=255,
                 dataset_names=None,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead_CLIP, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels
        
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
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
        # print(clip_embedding_bank.shape)
        clip_embedding_bank = clip_embedding_bank.cpu()
        self.num_classes = clip_embedding_bank.shape[0]
        feature_len = clip_embedding_bank.shape[1]
        self.clip_embedding_bank = torch.cat([clip_embedding_bank, torch.zeros(1, feature_len).to(clip_embedding_bank.device)]).cuda()
        self.background_index_value = background_index_value

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
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
        # clip_embedding_bank 150 x 512
        # cosine similarity loss
        # seg_label_clip = clip_embedding_bank[seg_label, :]  , shape B x H x W x 512
        # cosine_simi_loss(seg_embedding, seg_label_clip)
        seg_label_clip = seg_label.clone()
        if self.ignore_index is not None:
            self.valid_mask = ((seg_label_clip >= 0) & (seg_label_clip != self.ignore_index)).float().clone()
        else:
            self.valid_mask = None
        # print(torch.unique(seg_label_clip))
        
        seg_label_clip[seg_label_clip==self.background_index_value] = self.num_classes
        # print('seg_label_clip shape: ', seg_label_clip.shape)
        # print('clip_embedding_bank shape: ', self.clip_embedding_bank.shape)
        # print('seg_label_clip unique values: ', torch.unique(seg_label_clip))
        seg_label_clip_embedding = self.clip_embedding_bank[seg_label_clip, :].permute(0,3,1,2)

        # print(self.valid_mask.shape)
        # print(seg_embedding.shape)
        # print(seg_label_clip_embedding.shape)
        seg_logit = torch.einsum('bchw,nc->bnhw',seg_embedding, self.clip_embedding_bank[:self.num_classes,:])      
        # print(seg_label_clip_embedding.shape, seg_logit.shape)
        for loss_decode in losses_decode:
            if 'cos_simi' in loss_decode.loss_name:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_embedding,
                        seg_label_clip_embedding,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                        valid_mask=self.valid_mask
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_embedding,
                        seg_label_clip_embedding,
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


