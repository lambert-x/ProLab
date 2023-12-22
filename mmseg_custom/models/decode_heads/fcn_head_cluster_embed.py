# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.ops import resize
import numpy as np

@HEADS.register_module()
class FCNHead_cluster_embed(BaseDecodeHead):
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
                 desc_model_name = 'ViT-B/32',
                 desc_weights_dict_path = 'ade_zeroshot_weights_dict.pth',
                 background_index_value=255,
                 dataset_names=None,
                 get_logit_mode='cosine_similarity',
                 sigmoid_temperature=1.0,
                 image_embedding_normalize=True,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead_cluster_embed, self).__init__(**kwargs)
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
