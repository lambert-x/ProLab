# Copyright (c) OpenMMLab. All rights reserved.
from .mapillary import MapillaryDataset  # noqa: F401,F403
from .potsdam import PotsdamDataset  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
from .rvc_seg import RVC_SEG_Dataset
from .mapillary_official import MapillaryDataset_v1
from .bdd100k import BDD100K_Dataset
from .coco_panoptic_133classes import COCO_PANO_133_Dataset
from .voc_customized import PascalVOCDataset_NoBackground
from .ade20k_847 import ADE20K_847_Dataset
from .pascal_459 import Pascal_459_Dataset