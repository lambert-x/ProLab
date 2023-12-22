# Copyright (c) OpenMMLab. All rights reserved.
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .fcn_head_clip import FCNHead_CLIP
from .uper_head_clip import UPerHead_CLIP
from .fcn_head_clip_t5 import FCNHead_CLIP_T5
from .uper_head_clip_t5 import UPerHead_CLIP_T5
from .uper_head_cluster_embed import UPerHead_cluster_embed
from .fcn_head_cluster_embed import FCNHead_cluster_embed
from .sep_aspp_head_cluster_embed import DepthwiseSeparableASPPHead_cluster_embed
from .segformer_head_cluster_embed import SegformerHead_cluster_embed
__all__ = [
    'MaskFormerHead',
    'Mask2FormerHead',
    'FCNHead_CLIP',
    'UPerHead_CLIP',
    'FCNHead_CLIP_T5',
    'UPerHead_CLIP_T5',
    'UPerHead_cluster_embed',
    'FCNHead_cluster_embed',
    'DepthwiseSeparableASPPHead_cluster_embed',
    'SegformerHead_cluster_embed'
]
