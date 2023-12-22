# Copyright (c) OpenMMLab. All rights reserved.
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .uper_head_cluster_embed import UPerHead_cluster_embed
from .fcn_head_cluster_embed import FCNHead_cluster_embed
from .sep_aspp_head_cluster_embed import DepthwiseSeparableASPPHead_cluster_embed
from .segformer_head_cluster_embed import SegformerHead_cluster_embed
__all__ = [
    'MaskFormerHead',
    'Mask2FormerHead',
    'UPerHead_cluster_embed',
    'FCNHead_cluster_embed',
    'DepthwiseSeparableASPPHead_cluster_embed',
    'SegformerHead_cluster_embed'
]
