# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss




@LOSSES.register_module(force=True)
class CosineSimilarityLoss2(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_cos_simi_2',
                 split_embed=False,
                 avg_non_ignore=False,
                 dim=1):
        super(CosineSimilarityLoss2, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.split_embed = split_embed

        self.criterion = torch.nn.CosineSimilarity(dim=dim)
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                embedding_pred,
                embedding_target_1,
                embedding_target_2,
                target_weight_1=0.5,
                target_weight_2=0.5,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                valid_mask=None,
                **kwargs):
        """Forward function."""

        # v1 = nn.functional.normalize(embedding_pred, dim=1)
        # v2 = nn.functional.normalize(embedding _target, dim=1)
        if self.split_embed:
            v1_1, v1_2 = torch.chunk(embedding_pred, 2, dim=1)
            v2_1 = embedding_target_1
            v2_2 = embedding_target_2
            if valid_mask is not None:
                avg_factor = valid_mask.sum().item()
                loss_1 = ((1 - self.criterion(v1_1, v2_1)) * valid_mask).sum() / avg_factor
                loss_2 = ((1 - self.criterion(v1_2, v2_2)) * valid_mask).sum() / avg_factor
                loss = target_weight_1 * loss_1 + target_weight_2 * loss_2
            else:
                loss_1 = (1 - self.criterion(v1_1, v2_1)).mean()
                loss_2 = (1 - self.criterion(v1_2, v2_2)).mean()
                loss = target_weight_1 * loss_1 + target_weight_2 * loss_2
            return loss
        else:
            v1 = embedding_pred
            v2_1 = embedding_target_1
            v2_2 = embedding_target_2
            if valid_mask is not None:
                avg_factor = valid_mask.sum().item()
                loss_1 = ((1 - self.criterion(v1, v2_1)) * valid_mask).sum() / avg_factor
                loss_2 = ((1 - self.criterion(v1, v2_2)) * valid_mask).sum() / avg_factor
                loss = target_weight_1 * loss_1 + target_weight_2 * loss_2
            else:
                loss_1 = (1 - self.criterion(v1, v2_1)).mean()
                loss_2 = (1 - self.criterion(v1, v2_2)).mean()
                loss = target_weight_1 * loss_1 + target_weight_2 * loss_2
            return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
