# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
from mmcv.utils import print_log
from .pipelines.loading import LoadAnnotations_INT16


PASCAL_CTX_459_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "accordion"},
    {"color": [180, 120, 120], "id": 1, "isthing": 0, "name": "aeroplane"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "air conditioner"},
    {"color": [80, 50, 50], "id": 3, "isthing": 0, "name": "antenna"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "artillery"},
    {"color": [120, 120, 80], "id": 5, "isthing": 0, "name": "ashtray"},
    {"color": [140, 140, 140], "id": 6, "isthing": 0, "name": "atrium"},
    {"color": [204, 5, 255], "id": 7, "isthing": 0, "name": "baby carriage"},
    {"color": [230, 230, 230], "id": 8, "isthing": 0, "name": "bag"},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "ball"},
    {"color": [224, 5, 255], "id": 10, "isthing": 0, "name": "balloon"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "bamboo weaving"},
    {"color": [150, 5, 61], "id": 12, "isthing": 0, "name": "barrel"},
    {"color": [120, 120, 70], "id": 13, "isthing": 0, "name": "baseball bat"},
    {"color": [8, 255, 51], "id": 14, "isthing": 0, "name": "basket"},
    {"color": [255, 6, 82], "id": 15, "isthing": 0, "name": "basketball backboard"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "bathtub"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "bed"},
    {"color": [255, 51, 7], "id": 18, "isthing": 0, "name": "bedclothes"},
    {"color": [204, 70, 3], "id": 19, "isthing": 0, "name": "beer"},
    {"color": [0, 102, 200], "id": 20, "isthing": 0, "name": "bell"},
    {"color": [61, 230, 250], "id": 21, "isthing": 0, "name": "bench"},
    {"color": [255, 6, 51], "id": 22, "isthing": 0, "name": "bicycle"},
    {"color": [11, 102, 255], "id": 23, "isthing": 0, "name": "binoculars"},
    {"color": [255, 7, 71], "id": 24, "isthing": 0, "name": "bird"},
    {"color": [255, 9, 224], "id": 25, "isthing": 0, "name": "bird cage"},
    {"color": [9, 7, 230], "id": 26, "isthing": 0, "name": "bird feeder"},
    {"color": [220, 220, 220], "id": 27, "isthing": 0, "name": "bird nest"},
    {"color": [255, 9, 92], "id": 28, "isthing": 0, "name": "blackboard"},
    {"color": [112, 9, 255], "id": 29, "isthing": 0, "name": "board"},
    {"color": [8, 255, 214], "id": 30, "isthing": 0, "name": "boat"},
    {"color": [7, 255, 224], "id": 31, "isthing": 0, "name": "bone"},
    {"color": [255, 184, 6], "id": 32, "isthing": 0, "name": "book"},
    {"color": [10, 255, 71], "id": 33, "isthing": 0, "name": "bottle"},
    {"color": [255, 41, 10], "id": 34, "isthing": 0, "name": "bottle opener"},
    {"color": [7, 255, 255], "id": 35, "isthing": 0, "name": "bowl"},
    {"color": [224, 255, 8], "id": 36, "isthing": 0, "name": "box"},
    {"color": [102, 8, 255], "id": 37, "isthing": 0, "name": "bracelet"},
    {"color": [255, 61, 6], "id": 38, "isthing": 0, "name": "brick"},
    {"color": [255, 194, 7], "id": 39, "isthing": 0, "name": "bridge"},
    {"color": [255, 122, 8], "id": 40, "isthing": 0, "name": "broom"},
    {"color": [0, 255, 20], "id": 41, "isthing": 0, "name": "brush"},
    {"color": [255, 8, 41], "id": 42, "isthing": 0, "name": "bucket"},
    {"color": [255, 5, 153], "id": 43, "isthing": 0, "name": "building"},
    {"color": [6, 51, 255], "id": 44, "isthing": 0, "name": "bus"},
    {"color": [235, 12, 255], "id": 45, "isthing": 0, "name": "cabinet"},
    {"color": [160, 150, 20], "id": 46, "isthing": 0, "name": "cabinet door"},
    {"color": [0, 163, 255], "id": 47, "isthing": 0, "name": "cage"},
    {"color": [140, 140, 140], "id": 48, "isthing": 0, "name": "cake"},
    {"color": [250, 10, 15], "id": 49, "isthing": 0, "name": "calculator"},
    {"color": [20, 255, 0], "id": 50, "isthing": 0, "name": "calendar"},
    {"color": [31, 255, 0], "id": 51, "isthing": 0, "name": "camel"},
    {"color": [255, 31, 0], "id": 52, "isthing": 0, "name": "camera"},
    {"color": [255, 224, 0], "id": 53, "isthing": 0, "name": "camera lens"},
    {"color": [153, 255, 0], "id": 54, "isthing": 0, "name": "can"},
    {"color": [0, 0, 255], "id": 55, "isthing": 0, "name": "candle"},
    {"color": [255, 71, 0], "id": 56, "isthing": 0, "name": "candle holder"},
    {"color": [0, 235, 255], "id": 57, "isthing": 0, "name": "cap"},
    {"color": [0, 173, 255], "id": 58, "isthing": 0, "name": "car"},
    {"color": [31, 0, 255], "id": 59, "isthing": 0, "name": "card"},
    {"color": [120, 120, 120], "id": 60, "isthing": 0, "name": "cart"},
    {"color": [180, 120, 120], "id": 61, "isthing": 0, "name": "case"},
    {"color": [6, 230, 230], "id": 62, "isthing": 0, "name": "casette recorder"},
    {"color": [80, 50, 50], "id": 63, "isthing": 0, "name": "cash register"},
    {"color": [4, 200, 3], "id": 64, "isthing": 0, "name": "cat"},
    {"color": [120, 120, 80], "id": 65, "isthing": 0, "name": "cd"},
    {"color": [140, 140, 140], "id": 66, "isthing": 0, "name": "cd player"},
    {"color": [204, 5, 255], "id": 67, "isthing": 0, "name": "ceiling"},
    {"color": [230, 230, 230], "id": 68, "isthing": 0, "name": "cell phone"},
    {"color": [4, 250, 7], "id": 69, "isthing": 0, "name": "cello"},
    {"color": [224, 5, 255], "id": 70, "isthing": 0, "name": "chain"},
    {"color": [235, 255, 7], "id": 71, "isthing": 0, "name": "chair"},
    {"color": [150, 5, 61], "id": 72, "isthing": 0, "name": "chessboard"},
    {"color": [120, 120, 70], "id": 73, "isthing": 0, "name": "chicken"},
    {"color": [8, 255, 51], "id": 74, "isthing": 0, "name": "chopstick"},
    {"color": [255, 6, 82], "id": 75, "isthing": 0, "name": "clip"},
    {"color": [143, 255, 140], "id": 76, "isthing": 0, "name": "clippers"},
    {"color": [204, 255, 4], "id": 77, "isthing": 0, "name": "clock"},
    {"color": [255, 51, 7], "id": 78, "isthing": 0, "name": "closet"},
    {"color": [204, 70, 3], "id": 79, "isthing": 0, "name": "cloth"},
    {"color": [0, 102, 200], "id": 80, "isthing": 0, "name": "clothes tree"},
    {"color": [61, 230, 250], "id": 81, "isthing": 0, "name": "coffee"},
    {"color": [255, 6, 51], "id": 82, "isthing": 0, "name": "coffee machine"},
    {"color": [11, 102, 255], "id": 83, "isthing": 0, "name": "comb"},
    {"color": [255, 7, 71], "id": 84, "isthing": 0, "name": "computer"},
    {"color": [255, 9, 224], "id": 85, "isthing": 0, "name": "concrete"},
    {"color": [9, 7, 230], "id": 86, "isthing": 0, "name": "cone"},
    {"color": [220, 220, 220], "id": 87, "isthing": 0, "name": "container"},
    {"color": [255, 9, 92], "id": 88, "isthing": 0, "name": "control booth"},
    {"color": [112, 9, 255], "id": 89, "isthing": 0, "name": "controller"},
    {"color": [8, 255, 214], "id": 90, "isthing": 0, "name": "cooker"},
    {"color": [7, 255, 224], "id": 91, "isthing": 0, "name": "copying machine"},
    {"color": [255, 184, 6], "id": 92, "isthing": 0, "name": "coral"},
    {"color": [10, 255, 71], "id": 93, "isthing": 0, "name": "cork"},
    {"color": [255, 41, 10], "id": 94, "isthing": 0, "name": "corkscrew"},
    {"color": [7, 255, 255], "id": 95, "isthing": 0, "name": "counter"},
    {"color": [224, 255, 8], "id": 96, "isthing": 0, "name": "court"},
    {"color": [102, 8, 255], "id": 97, "isthing": 0, "name": "cow"},
    {"color": [255, 61, 6], "id": 98, "isthing": 0, "name": "crabstick"},
    {"color": [255, 194, 7], "id": 99, "isthing": 0, "name": "crane"},
    {"color": [255, 122, 8], "id": 100, "isthing": 0, "name": "crate"},
    {"color": [0, 255, 20], "id": 101, "isthing": 0, "name": "cross"},
    {"color": [255, 8, 41], "id": 102, "isthing": 0, "name": "crutch"},
    {"color": [255, 5, 153], "id": 103, "isthing": 0, "name": "cup"},
    {"color": [6, 51, 255], "id": 104, "isthing": 0, "name": "curtain"},
    {"color": [235, 12, 255], "id": 105, "isthing": 0, "name": "cushion"},
    {"color": [160, 150, 20], "id": 106, "isthing": 0, "name": "cutting board"},
    {"color": [0, 163, 255], "id": 107, "isthing": 0, "name": "dais"},
    {"color": [140, 140, 140], "id": 108, "isthing": 0, "name": "disc"},
    {"color": [250, 10, 15], "id": 109, "isthing": 0, "name": "disc case"},
    {"color": [20, 255, 0], "id": 110, "isthing": 0, "name": "dishwasher"},
    {"color": [31, 255, 0], "id": 111, "isthing": 0, "name": "dock"},
    {"color": [255, 31, 0], "id": 112, "isthing": 0, "name": "dog"},
    {"color": [255, 224, 0], "id": 113, "isthing": 0, "name": "dolphin"},
    {"color": [153, 255, 0], "id": 114, "isthing": 0, "name": "door"},
    {"color": [0, 0, 255], "id": 115, "isthing": 0, "name": "drainer"},
    {"color": [255, 71, 0], "id": 116, "isthing": 0, "name": "dray"},
    {"color": [0, 235, 255], "id": 117, "isthing": 0, "name": "drink dispenser"},
    {"color": [0, 173, 255], "id": 118, "isthing": 0, "name": "drinking machine"},
    {"color": [31, 0, 255], "id": 119, "isthing": 0, "name": "drop"},
    {"color": [120, 120, 120], "id": 120, "isthing": 0, "name": "drug"},
    {"color": [180, 120, 120], "id": 121, "isthing": 0, "name": "drum"},
    {"color": [6, 230, 230], "id": 122, "isthing": 0, "name": "drum kit"},
    {"color": [80, 50, 50], "id": 123, "isthing": 0, "name": "duck"},
    {"color": [4, 200, 3], "id": 124, "isthing": 0, "name": "dumbbell"},
    {"color": [120, 120, 80], "id": 125, "isthing": 0, "name": "earphone"},
    {"color": [140, 140, 140], "id": 126, "isthing": 0, "name": "earrings"},
    {"color": [204, 5, 255], "id": 127, "isthing": 0, "name": "egg"},
    {"color": [230, 230, 230], "id": 128, "isthing": 0, "name": "electric fan"},
    {"color": [4, 250, 7], "id": 129, "isthing": 0, "name": "electric iron"},
    {"color": [224, 5, 255], "id": 130, "isthing": 0, "name": "electric pot"},
    {"color": [235, 255, 7], "id": 131, "isthing": 0, "name": "electric saw"},
    {"color": [150, 5, 61], "id": 132, "isthing": 0, "name": "electronic keyboard"},
    {"color": [120, 120, 70], "id": 133, "isthing": 0, "name": "engine"},
    {"color": [8, 255, 51], "id": 134, "isthing": 0, "name": "envelope"},
    {"color": [255, 6, 82], "id": 135, "isthing": 0, "name": "equipment"},
    {"color": [143, 255, 140], "id": 136, "isthing": 0, "name": "escalator"},
    {"color": [204, 255, 4], "id": 137, "isthing": 0, "name": "exhibition booth"},
    {"color": [255, 51, 7], "id": 138, "isthing": 0, "name": "extinguisher"},
    {"color": [204, 70, 3], "id": 139, "isthing": 0, "name": "eyeglass"},
    {"color": [0, 102, 200], "id": 140, "isthing": 0, "name": "fan"},
    {"color": [61, 230, 250], "id": 141, "isthing": 0, "name": "faucet"},
    {"color": [255, 6, 51], "id": 142, "isthing": 0, "name": "fax machine"},
    {"color": [11, 102, 255], "id": 143, "isthing": 0, "name": "fence"},
    {"color": [255, 7, 71], "id": 144, "isthing": 0, "name": "ferris wheel"},
    {"color": [255, 9, 224], "id": 145, "isthing": 0, "name": "fire extinguisher"},
    {"color": [9, 7, 230], "id": 146, "isthing": 0, "name": "fire hydrant"},
    {"color": [220, 220, 220], "id": 147, "isthing": 0, "name": "fire place"},
    {"color": [255, 9, 92], "id": 148, "isthing": 0, "name": "fish"},
    {"color": [112, 9, 255], "id": 149, "isthing": 0, "name": "fish tank"},
    {"color": [8, 255, 214], "id": 150, "isthing": 0, "name": "fishbowl"},
    {"color": [7, 255, 224], "id": 151, "isthing": 0, "name": "fishing net"},
    {"color": [255, 184, 6], "id": 152, "isthing": 0, "name": "fishing pole"},
    {"color": [10, 255, 71], "id": 153, "isthing": 0, "name": "flag"},
    {"color": [255, 41, 10], "id": 154, "isthing": 0, "name": "flagstaff"},
    {"color": [7, 255, 255], "id": 155, "isthing": 0, "name": "flame"},
    {"color": [224, 255, 8], "id": 156, "isthing": 0, "name": "flashlight"},
    {"color": [102, 8, 255], "id": 157, "isthing": 0, "name": "floor"},
    {"color": [255, 61, 6], "id": 158, "isthing": 0, "name": "flower"},
    {"color": [255, 194, 7], "id": 159, "isthing": 0, "name": "fly"},
    {"color": [255, 122, 8], "id": 160, "isthing": 0, "name": "foam"},
    {"color": [0, 255, 20], "id": 161, "isthing": 0, "name": "food"},
    {"color": [255, 8, 41], "id": 162, "isthing": 0, "name": "footbridge"},
    {"color": [255, 5, 153], "id": 163, "isthing": 0, "name": "forceps"},
    {"color": [6, 51, 255], "id": 164, "isthing": 0, "name": "fork"},
    {"color": [235, 12, 255], "id": 165, "isthing": 0, "name": "forklift"},
    {"color": [160, 150, 20], "id": 166, "isthing": 0, "name": "fountain"},
    {"color": [0, 163, 255], "id": 167, "isthing": 0, "name": "fox"},
    {"color": [140, 140, 140], "id": 168, "isthing": 0, "name": "frame"},
    {"color": [250, 10, 15], "id": 169, "isthing": 0, "name": "fridge"},
    {"color": [20, 255, 0], "id": 170, "isthing": 0, "name": "frog"},
    {"color": [31, 255, 0], "id": 171, "isthing": 0, "name": "fruit"},
    {"color": [255, 31, 0], "id": 172, "isthing": 0, "name": "funnel"},
    {"color": [255, 224, 0], "id": 173, "isthing": 0, "name": "furnace"},
    {"color": [153, 255, 0], "id": 174, "isthing": 0, "name": "game controller"},
    {"color": [0, 0, 255], "id": 175, "isthing": 0, "name": "game machine"},
    {"color": [255, 71, 0], "id": 176, "isthing": 0, "name": "gas cylinder"},
    {"color": [0, 235, 255], "id": 177, "isthing": 0, "name": "gas hood"},
    {"color": [0, 173, 255], "id": 178, "isthing": 0, "name": "gas stove"},
    {"color": [31, 0, 255], "id": 179, "isthing": 0, "name": "gift box"},
    {"color": [120, 120, 120], "id": 180, "isthing": 0, "name": "glass"},
    {"color": [180, 120, 120], "id": 181, "isthing": 0, "name": "glass marble"},
    {"color": [6, 230, 230], "id": 182, "isthing": 0, "name": "globe"},
    {"color": [80, 50, 50], "id": 183, "isthing": 0, "name": "glove"},
    {"color": [4, 200, 3], "id": 184, "isthing": 0, "name": "goal"},
    {"color": [120, 120, 80], "id": 185, "isthing": 0, "name": "grandstand"},
    {"color": [140, 140, 140], "id": 186, "isthing": 0, "name": "grass"},
    {"color": [204, 5, 255], "id": 187, "isthing": 0, "name": "gravestone"},
    {"color": [230, 230, 230], "id": 188, "isthing": 0, "name": "ground"},
    {"color": [4, 250, 7], "id": 189, "isthing": 0, "name": "guardrail"},
    {"color": [224, 5, 255], "id": 190, "isthing": 0, "name": "guitar"},
    {"color": [235, 255, 7], "id": 191, "isthing": 0, "name": "gun"},
    {"color": [150, 5, 61], "id": 192, "isthing": 0, "name": "hammer"},
    {"color": [120, 120, 70], "id": 193, "isthing": 0, "name": "hand cart"},
    {"color": [8, 255, 51], "id": 194, "isthing": 0, "name": "handle"},
    {"color": [255, 6, 82], "id": 195, "isthing": 0, "name": "handrail"},
    {"color": [143, 255, 140], "id": 196, "isthing": 0, "name": "hanger"},
    {"color": [204, 255, 4], "id": 197, "isthing": 0, "name": "hard disk drive"},
    {"color": [255, 51, 7], "id": 198, "isthing": 0, "name": "hat"},
    {"color": [204, 70, 3], "id": 199, "isthing": 0, "name": "hay"},
    {"color": [0, 102, 200], "id": 200, "isthing": 0, "name": "headphone"},
    {"color": [61, 230, 250], "id": 201, "isthing": 0, "name": "heater"},
    {"color": [255, 6, 51], "id": 202, "isthing": 0, "name": "helicopter"},
    {"color": [11, 102, 255], "id": 203, "isthing": 0, "name": "helmet"},
    {"color": [255, 7, 71], "id": 204, "isthing": 0, "name": "holder"},
    {"color": [255, 9, 224], "id": 205, "isthing": 0, "name": "hook"},
    {"color": [9, 7, 230], "id": 206, "isthing": 0, "name": "horse"},
    {"color": [220, 220, 220], "id": 207, "isthing": 0, "name": "horse-drawn carriage"},
    {"color": [255, 9, 92], "id": 208, "isthing": 0, "name": "hot-air balloon"},
    {"color": [112, 9, 255], "id": 209, "isthing": 0, "name": "hydrovalve"},
    {"color": [8, 255, 214], "id": 210, "isthing": 0, "name": "ice"},
    {"color": [7, 255, 224], "id": 211, "isthing": 0, "name": "inflator pump"},
    {"color": [255, 184, 6], "id": 212, "isthing": 0, "name": "ipod"},
    {"color": [10, 255, 71], "id": 213, "isthing": 0, "name": "iron"},
    {"color": [255, 41, 10], "id": 214, "isthing": 0, "name": "ironing board"},
    {"color": [7, 255, 255], "id": 215, "isthing": 0, "name": "jar"},
    {"color": [224, 255, 8], "id": 216, "isthing": 0, "name": "kart"},
    {"color": [102, 8, 255], "id": 217, "isthing": 0, "name": "kettle"},
    {"color": [255, 61, 6], "id": 218, "isthing": 0, "name": "key"},
    {"color": [255, 194, 7], "id": 219, "isthing": 0, "name": "keyboard"},
    {"color": [255, 122, 8], "id": 220, "isthing": 0, "name": "kitchen range"},
    {"color": [0, 255, 20], "id": 221, "isthing": 0, "name": "kite"},
    {"color": [255, 8, 41], "id": 222, "isthing": 0, "name": "knife"},
    {"color": [255, 5, 153], "id": 223, "isthing": 0, "name": "knife block"},
    {"color": [6, 51, 255], "id": 224, "isthing": 0, "name": "ladder"},
    {"color": [235, 12, 255], "id": 225, "isthing": 0, "name": "ladder truck"},
    {"color": [160, 150, 20], "id": 226, "isthing": 0, "name": "ladle"},
    {"color": [0, 163, 255], "id": 227, "isthing": 0, "name": "laptop"},
    {"color": [140, 140, 140], "id": 228, "isthing": 0, "name": "leaves"},
    {"color": [250, 10, 15], "id": 229, "isthing": 0, "name": "lid"},
    {"color": [20, 255, 0], "id": 230, "isthing": 0, "name": "life buoy"},
    {"color": [31, 255, 0], "id": 231, "isthing": 0, "name": "light"},
    {"color": [255, 31, 0], "id": 232, "isthing": 0, "name": "light bulb"},
    {"color": [255, 224, 0], "id": 233, "isthing": 0, "name": "lighter"},
    {"color": [153, 255, 0], "id": 234, "isthing": 0, "name": "line"},
    {"color": [0, 0, 255], "id": 235, "isthing": 0, "name": "lion"},
    {"color": [255, 71, 0], "id": 236, "isthing": 0, "name": "lobster"},
    {"color": [0, 235, 255], "id": 237, "isthing": 0, "name": "lock"},
    {"color": [0, 173, 255], "id": 238, "isthing": 0, "name": "machine"},
    {"color": [31, 0, 255], "id": 239, "isthing": 0, "name": "mailbox"},
    {"color": [120, 120, 120], "id": 240, "isthing": 0, "name": "mannequin"},
    {"color": [180, 120, 120], "id": 241, "isthing": 0, "name": "map"},
    {"color": [6, 230, 230], "id": 242, "isthing": 0, "name": "mask"},
    {"color": [80, 50, 50], "id": 243, "isthing": 0, "name": "mat"},
    {"color": [4, 200, 3], "id": 244, "isthing": 0, "name": "match book"},
    {"color": [120, 120, 80], "id": 245, "isthing": 0, "name": "mattress"},
    {"color": [140, 140, 140], "id": 246, "isthing": 0, "name": "menu"},
    {"color": [204, 5, 255], "id": 247, "isthing": 0, "name": "metal"},
    {"color": [230, 230, 230], "id": 248, "isthing": 0, "name": "meter box"},
    {"color": [4, 250, 7], "id": 249, "isthing": 0, "name": "microphone"},
    {"color": [224, 5, 255], "id": 250, "isthing": 0, "name": "microwave"},
    {"color": [235, 255, 7], "id": 251, "isthing": 0, "name": "mirror"},
    {"color": [150, 5, 61], "id": 252, "isthing": 0, "name": "missile"},
    {"color": [120, 120, 70], "id": 253, "isthing": 0, "name": "model"},
    {"color": [8, 255, 51], "id": 254, "isthing": 0, "name": "money"},
    {"color": [255, 6, 82], "id": 255, "isthing": 0, "name": "monkey"},
    {"color": [143, 255, 140], "id": 256, "isthing": 0, "name": "mop"},
    {"color": [204, 255, 4], "id": 257, "isthing": 0, "name": "motorbike"},
    {"color": [255, 51, 7], "id": 258, "isthing": 0, "name": "mountain"},
    {"color": [204, 70, 3], "id": 259, "isthing": 0, "name": "mouse"},
    {"color": [0, 102, 200], "id": 260, "isthing": 0, "name": "mouse pad"},
    {"color": [61, 230, 250], "id": 261, "isthing": 0, "name": "musical instrument"},
    {"color": [255, 6, 51], "id": 262, "isthing": 0, "name": "napkin"},
    {"color": [11, 102, 255], "id": 263, "isthing": 0, "name": "net"},
    {"color": [255, 7, 71], "id": 264, "isthing": 0, "name": "newspaper"},
    {"color": [255, 9, 224], "id": 265, "isthing": 0, "name": "oar"},
    {"color": [9, 7, 230], "id": 266, "isthing": 0, "name": "ornament"},
    {"color": [220, 220, 220], "id": 267, "isthing": 0, "name": "outlet"},
    {"color": [255, 9, 92], "id": 268, "isthing": 0, "name": "oven"},
    {"color": [112, 9, 255], "id": 269, "isthing": 0, "name": "oxygen bottle"},
    {"color": [8, 255, 214], "id": 270, "isthing": 0, "name": "pack"},
    {"color": [7, 255, 224], "id": 271, "isthing": 0, "name": "pan"},
    {"color": [255, 184, 6], "id": 272, "isthing": 0, "name": "paper"},
    {"color": [10, 255, 71], "id": 273, "isthing": 0, "name": "paper box"},
    {"color": [255, 41, 10], "id": 274, "isthing": 0, "name": "paper cutter"},
    {"color": [7, 255, 255], "id": 275, "isthing": 0, "name": "parachute"},
    {"color": [224, 255, 8], "id": 276, "isthing": 0, "name": "parasol"},
    {"color": [102, 8, 255], "id": 277, "isthing": 0, "name": "parterre"},
    {"color": [255, 61, 6], "id": 278, "isthing": 0, "name": "patio"},
    {"color": [255, 194, 7], "id": 279, "isthing": 0, "name": "pelage"},
    {"color": [255, 122, 8], "id": 280, "isthing": 0, "name": "pen"},
    {"color": [0, 255, 20], "id": 281, "isthing": 0, "name": "pen container"},
    {"color": [255, 8, 41], "id": 282, "isthing": 0, "name": "pencil"},
    {"color": [255, 5, 153], "id": 283, "isthing": 0, "name": "person"},
    {"color": [6, 51, 255], "id": 284, "isthing": 0, "name": "photo"},
    {"color": [235, 12, 255], "id": 285, "isthing": 0, "name": "piano"},
    {"color": [160, 150, 20], "id": 286, "isthing": 0, "name": "picture"},
    {"color": [0, 163, 255], "id": 287, "isthing": 0, "name": "pig"},
    {"color": [140, 140, 140], "id": 288, "isthing": 0, "name": "pillar"},
    {"color": [250, 10, 15], "id": 289, "isthing": 0, "name": "pillow"},
    {"color": [20, 255, 0], "id": 290, "isthing": 0, "name": "pipe"},
    {"color": [31, 255, 0], "id": 291, "isthing": 0, "name": "pitcher"},
    {"color": [255, 31, 0], "id": 292, "isthing": 0, "name": "plant"},
    {"color": [255, 224, 0], "id": 293, "isthing": 0, "name": "plastic"},
    {"color": [153, 255, 0], "id": 294, "isthing": 0, "name": "plate"},
    {"color": [0, 0, 255], "id": 295, "isthing": 0, "name": "platform"},
    {"color": [255, 71, 0], "id": 296, "isthing": 0, "name": "player"},
    {"color": [0, 235, 255], "id": 297, "isthing": 0, "name": "playground"},
    {"color": [0, 173, 255], "id": 298, "isthing": 0, "name": "pliers"},
    {"color": [31, 0, 255], "id": 299, "isthing": 0, "name": "plume"},
    {"color": [120, 120, 120], "id": 300, "isthing": 0, "name": "poker"},
    {"color": [180, 120, 120], "id": 301, "isthing": 0, "name": "poker chip"},
    {"color": [6, 230, 230], "id": 302, "isthing": 0, "name": "pole"},
    {"color": [80, 50, 50], "id": 303, "isthing": 0, "name": "pool table"},
    {"color": [4, 200, 3], "id": 304, "isthing": 0, "name": "postcard"},
    {"color": [120, 120, 80], "id": 305, "isthing": 0, "name": "poster"},
    {"color": [140, 140, 140], "id": 306, "isthing": 0, "name": "pot"},
    {"color": [204, 5, 255], "id": 307, "isthing": 0, "name": "pottedplant"},
    {"color": [230, 230, 230], "id": 308, "isthing": 0, "name": "printer"},
    {"color": [4, 250, 7], "id": 309, "isthing": 0, "name": "projector"},
    {"color": [224, 5, 255], "id": 310, "isthing": 0, "name": "pumpkin"},
    {"color": [235, 255, 7], "id": 311, "isthing": 0, "name": "rabbit"},
    {"color": [150, 5, 61], "id": 312, "isthing": 0, "name": "racket"},
    {"color": [120, 120, 70], "id": 313, "isthing": 0, "name": "radiator"},
    {"color": [8, 255, 51], "id": 314, "isthing": 0, "name": "radio"},
    {"color": [255, 6, 82], "id": 315, "isthing": 0, "name": "rail"},
    {"color": [143, 255, 140], "id": 316, "isthing": 0, "name": "rake"},
    {"color": [204, 255, 4], "id": 317, "isthing": 0, "name": "ramp"},
    {"color": [255, 51, 7], "id": 318, "isthing": 0, "name": "range hood"},
    {"color": [204, 70, 3], "id": 319, "isthing": 0, "name": "receiver"},
    {"color": [0, 102, 200], "id": 320, "isthing": 0, "name": "recorder"},
    {"color": [61, 230, 250], "id": 321, "isthing": 0, "name": "recreational machines"},
    {"color": [255, 6, 51], "id": 322, "isthing": 0, "name": "remote control"},
    {"color": [11, 102, 255], "id": 323, "isthing": 0, "name": "road"},
    {"color": [255, 7, 71], "id": 324, "isthing": 0, "name": "robot"},
    {"color": [255, 9, 224], "id": 325, "isthing": 0, "name": "rock"},
    {"color": [9, 7, 230], "id": 326, "isthing": 0, "name": "rocket"},
    {"color": [220, 220, 220], "id": 327, "isthing": 0, "name": "rocking horse"},
    {"color": [255, 9, 92], "id": 328, "isthing": 0, "name": "rope"},
    {"color": [112, 9, 255], "id": 329, "isthing": 0, "name": "rug"},
    {"color": [8, 255, 214], "id": 330, "isthing": 0, "name": "ruler"},
    {"color": [7, 255, 224], "id": 331, "isthing": 0, "name": "runway"},
    {"color": [255, 184, 6], "id": 332, "isthing": 0, "name": "saddle"},
    {"color": [10, 255, 71], "id": 333, "isthing": 0, "name": "sand"},
    {"color": [255, 41, 10], "id": 334, "isthing": 0, "name": "saw"},
    {"color": [7, 255, 255], "id": 335, "isthing": 0, "name": "scale"},
    {"color": [224, 255, 8], "id": 336, "isthing": 0, "name": "scanner"},
    {"color": [102, 8, 255], "id": 337, "isthing": 0, "name": "scissors"},
    {"color": [255, 61, 6], "id": 338, "isthing": 0, "name": "scoop"},
    {"color": [255, 194, 7], "id": 339, "isthing": 0, "name": "screen"},
    {"color": [255, 122, 8], "id": 340, "isthing": 0, "name": "screwdriver"},
    {"color": [0, 255, 20], "id": 341, "isthing": 0, "name": "sculpture"},
    {"color": [255, 8, 41], "id": 342, "isthing": 0, "name": "scythe"},
    {"color": [255, 5, 153], "id": 343, "isthing": 0, "name": "sewer"},
    {"color": [6, 51, 255], "id": 344, "isthing": 0, "name": "sewing machine"},
    {"color": [235, 12, 255], "id": 345, "isthing": 0, "name": "shed"},
    {"color": [160, 150, 20], "id": 346, "isthing": 0, "name": "sheep"},
    {"color": [0, 163, 255], "id": 347, "isthing": 0, "name": "shell"},
    {"color": [140, 140, 140], "id": 348, "isthing": 0, "name": "shelves"},
    {"color": [250, 10, 15], "id": 349, "isthing": 0, "name": "shoe"},
    {"color": [20, 255, 0], "id": 350, "isthing": 0, "name": "shopping cart"},
    {"color": [31, 255, 0], "id": 351, "isthing": 0, "name": "shovel"},
    {"color": [255, 31, 0], "id": 352, "isthing": 0, "name": "sidecar"},
    {"color": [255, 224, 0], "id": 353, "isthing": 0, "name": "sidewalk"},
    {"color": [153, 255, 0], "id": 354, "isthing": 0, "name": "sign"},
    {"color": [0, 0, 255], "id": 355, "isthing": 0, "name": "signal light"},
    {"color": [255, 71, 0], "id": 356, "isthing": 0, "name": "sink"},
    {"color": [0, 235, 255], "id": 357, "isthing": 0, "name": "skateboard"},
    {"color": [0, 173, 255], "id": 358, "isthing": 0, "name": "ski"},
    {"color": [31, 0, 255], "id": 359, "isthing": 0, "name": "sky"},
    {"color": [120, 120, 120], "id": 360, "isthing": 0, "name": "sled"},
    {"color": [180, 120, 120], "id": 361, "isthing": 0, "name": "slippers"},
    {"color": [6, 230, 230], "id": 362, "isthing": 0, "name": "smoke"},
    {"color": [80, 50, 50], "id": 363, "isthing": 0, "name": "snail"},
    {"color": [4, 200, 3], "id": 364, "isthing": 0, "name": "snake"},
    {"color": [120, 120, 80], "id": 365, "isthing": 0, "name": "snow"},
    {"color": [140, 140, 140], "id": 366, "isthing": 0, "name": "snowmobiles"},
    {"color": [204, 5, 255], "id": 367, "isthing": 0, "name": "sofa"},
    {"color": [230, 230, 230], "id": 368, "isthing": 0, "name": "spanner"},
    {"color": [4, 250, 7], "id": 369, "isthing": 0, "name": "spatula"},
    {"color": [224, 5, 255], "id": 370, "isthing": 0, "name": "speaker"},
    {"color": [235, 255, 7], "id": 371, "isthing": 0, "name": "speed bump"},
    {"color": [150, 5, 61], "id": 372, "isthing": 0, "name": "spice container"},
    {"color": [120, 120, 70], "id": 373, "isthing": 0, "name": "spoon"},
    {"color": [8, 255, 51], "id": 374, "isthing": 0, "name": "sprayer"},
    {"color": [255, 6, 82], "id": 375, "isthing": 0, "name": "squirrel"},
    {"color": [143, 255, 140], "id": 376, "isthing": 0, "name": "stage"},
    {"color": [204, 255, 4], "id": 377, "isthing": 0, "name": "stair"},
    {"color": [255, 51, 7], "id": 378, "isthing": 0, "name": "stapler"},
    {"color": [204, 70, 3], "id": 379, "isthing": 0, "name": "stick"},
    {"color": [0, 102, 200], "id": 380, "isthing": 0, "name": "sticky note"},
    {"color": [61, 230, 250], "id": 381, "isthing": 0, "name": "stone"},
    {"color": [255, 6, 51], "id": 382, "isthing": 0, "name": "stool"},
    {"color": [11, 102, 255], "id": 383, "isthing": 0, "name": "stove"},
    {"color": [255, 7, 71], "id": 384, "isthing": 0, "name": "straw"},
    {"color": [255, 9, 224], "id": 385, "isthing": 0, "name": "stretcher"},
    {"color": [9, 7, 230], "id": 386, "isthing": 0, "name": "sun"},
    {"color": [220, 220, 220], "id": 387, "isthing": 0, "name": "sunglass"},
    {"color": [255, 9, 92], "id": 388, "isthing": 0, "name": "sunshade"},
    {"color": [112, 9, 255], "id": 389, "isthing": 0, "name": "surveillance camera"},
    {"color": [8, 255, 214], "id": 390, "isthing": 0, "name": "swan"},
    {"color": [7, 255, 224], "id": 391, "isthing": 0, "name": "sweeper"},
    {"color": [255, 184, 6], "id": 392, "isthing": 0, "name": "swim ring"},
    {"color": [10, 255, 71], "id": 393, "isthing": 0, "name": "swimming pool"},
    {"color": [255, 41, 10], "id": 394, "isthing": 0, "name": "swing"},
    {"color": [7, 255, 255], "id": 395, "isthing": 0, "name": "switch"},
    {"color": [224, 255, 8], "id": 396, "isthing": 0, "name": "table"},
    {"color": [102, 8, 255], "id": 397, "isthing": 0, "name": "tableware"},
    {"color": [255, 61, 6], "id": 398, "isthing": 0, "name": "tank"},
    {"color": [255, 194, 7], "id": 399, "isthing": 0, "name": "tap"},
    {"color": [255, 122, 8], "id": 400, "isthing": 0, "name": "tape"},
    {"color": [0, 255, 20], "id": 401, "isthing": 0, "name": "tarp"},
    {"color": [255, 8, 41], "id": 402, "isthing": 0, "name": "telephone"},
    {"color": [255, 5, 153], "id": 403, "isthing": 0, "name": "telephone booth"},
    {"color": [6, 51, 255], "id": 404, "isthing": 0, "name": "tent"},
    {"color": [235, 12, 255], "id": 405, "isthing": 0, "name": "tire"},
    {"color": [160, 150, 20], "id": 406, "isthing": 0, "name": "toaster"},
    {"color": [0, 163, 255], "id": 407, "isthing": 0, "name": "toilet"},
    {"color": [140, 140, 140], "id": 408, "isthing": 0, "name": "tong"},
    {"color": [250, 10, 15], "id": 409, "isthing": 0, "name": "tool"},
    {"color": [20, 255, 0], "id": 410, "isthing": 0, "name": "toothbrush"},
    {"color": [31, 255, 0], "id": 411, "isthing": 0, "name": "towel"},
    {"color": [255, 31, 0], "id": 412, "isthing": 0, "name": "toy"},
    {"color": [255, 224, 0], "id": 413, "isthing": 0, "name": "toy car"},
    {"color": [153, 255, 0], "id": 414, "isthing": 0, "name": "track"},
    {"color": [0, 0, 255], "id": 415, "isthing": 0, "name": "train"},
    {"color": [255, 71, 0], "id": 416, "isthing": 0, "name": "trampoline"},
    {"color": [0, 235, 255], "id": 417, "isthing": 0, "name": "trash bin"},
    {"color": [0, 173, 255], "id": 418, "isthing": 0, "name": "tray"},
    {"color": [31, 0, 255], "id": 419, "isthing": 0, "name": "tree"},
    {"color": [120, 120, 120], "id": 420, "isthing": 0, "name": "tricycle"},
    {"color": [180, 120, 120], "id": 421, "isthing": 0, "name": "tripod"},
    {"color": [6, 230, 230], "id": 422, "isthing": 0, "name": "trophy"},
    {"color": [80, 50, 50], "id": 423, "isthing": 0, "name": "truck"},
    {"color": [4, 200, 3], "id": 424, "isthing": 0, "name": "tube"},
    {"color": [120, 120, 80], "id": 425, "isthing": 0, "name": "turtle"},
    {"color": [140, 140, 140], "id": 426, "isthing": 0, "name": "tvmonitor"},
    {"color": [204, 5, 255], "id": 427, "isthing": 0, "name": "tweezers"},
    {"color": [230, 230, 230], "id": 428, "isthing": 0, "name": "typewriter"},
    {"color": [4, 250, 7], "id": 429, "isthing": 0, "name": "umbrella"},
    {"color": [224, 5, 255], "id": 430, "isthing": 0, "name": "unknown"},
    {"color": [235, 255, 7], "id": 431, "isthing": 0, "name": "vacuum cleaner"},
    {"color": [150, 5, 61], "id": 432, "isthing": 0, "name": "vending machine"},
    {"color": [120, 120, 70], "id": 433, "isthing": 0, "name": "video camera"},
    {"color": [8, 255, 51], "id": 434, "isthing": 0, "name": "video game console"},
    {"color": [255, 6, 82], "id": 435, "isthing": 0, "name": "video player"},
    {"color": [143, 255, 140], "id": 436, "isthing": 0, "name": "video tape"},
    {"color": [204, 255, 4], "id": 437, "isthing": 0, "name": "violin"},
    {"color": [255, 51, 7], "id": 438, "isthing": 0, "name": "wakeboard"},
    {"color": [204, 70, 3], "id": 439, "isthing": 0, "name": "wall"},
    {"color": [0, 102, 200], "id": 440, "isthing": 0, "name": "wallet"},
    {"color": [61, 230, 250], "id": 441, "isthing": 0, "name": "wardrobe"},
    {"color": [255, 6, 51], "id": 442, "isthing": 0, "name": "washing machine"},
    {"color": [11, 102, 255], "id": 443, "isthing": 0, "name": "watch"},
    {"color": [255, 7, 71], "id": 444, "isthing": 0, "name": "water"},
    {"color": [255, 9, 224], "id": 445, "isthing": 0, "name": "water dispenser"},
    {"color": [9, 7, 230], "id": 446, "isthing": 0, "name": "water pipe"},
    {"color": [220, 220, 220], "id": 447, "isthing": 0, "name": "water skate board"},
    {"color": [255, 9, 92], "id": 448, "isthing": 0, "name": "watermelon"},
    {"color": [112, 9, 255], "id": 449, "isthing": 0, "name": "whale"},
    {"color": [8, 255, 214], "id": 450, "isthing": 0, "name": "wharf"},
    {"color": [7, 255, 224], "id": 451, "isthing": 0, "name": "wheel"},
    {"color": [255, 184, 6], "id": 452, "isthing": 0, "name": "wheelchair"},
    {"color": [10, 255, 71], "id": 453, "isthing": 0, "name": "window"},
    {"color": [255, 41, 10], "id": 454, "isthing": 0, "name": "window blinds"},
    {"color": [7, 255, 255], "id": 455, "isthing": 0, "name": "wineglass"},
    {"color": [224, 255, 8], "id": 456, "isthing": 0, "name": "wire"},
    {"color": [102, 8, 255], "id": 457, "isthing": 0, "name": "wood"},
    {"color": [255, 61, 6], "id": 458, "isthing": 0, "name": "wool"},
]


@DATASETS.register_module()
class Pascal_459_Dataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = [info['name'] for info in PASCAL_CTX_459_CATEGORIES]

    PALETTE = None

    def __init__(self, gt_seg_map_loader_cfg=None, **kwargs):
        super(Pascal_459_Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            ignore_index=32767,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_INT16() if gt_seg_map_loader_cfg is None else LoadAnnotations_INT16(
            **gt_seg_map_loader_cfg)
        self.ignore_index=32767

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.tif')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint16))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
    
    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']