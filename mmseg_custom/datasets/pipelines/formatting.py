# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module(force=True)
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(to_tensor(
                results['gt_semantic_seg'][None, ...].astype(np.int64)),
                                            stack=True)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(to_tensor(results['gt_masks']))
        if 'gt_labels' in results:
            results['gt_labels'] = DC(to_tensor(results['gt_labels']))

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ToMask(object):
    """Transfer gt_semantic_seg to binary mask and generate gt_labels."""
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, results):
        gt_semantic_seg = results['gt_semantic_seg']
        gt_labels = np.unique(gt_semantic_seg)
        # remove ignored region
        gt_labels = gt_labels[gt_labels != self.ignore_index]

        gt_masks = []
        for class_id in gt_labels:
            gt_masks.append(gt_semantic_seg == class_id)

        if len(gt_masks) == 0:
            # Some image does not have annotation (all ignored)
            gt_masks = np.empty((0, ) + results['pad_shape'][:-1], dtype=np.int64)
            gt_labels = np.empty((0, ),  dtype=np.int64)
        else:
            gt_masks = np.asarray(gt_masks, dtype=np.int64)
            gt_labels = np.asarray(gt_labels, dtype=np.int64)

        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(ignore_index={self.ignore_index})'



@PIPELINES.register_module()
class Remapping_SegGT(object):

    def __init__(self,
                 ignore_index=255,
                 ignore_index_target_value=255,
                 label_shift_value=0,
                 remapping_dict_path=None
                 ):
        self.ignore_index = ignore_index
        self.ignore_index_target_value = ignore_index_target_value
        self.label_shift_value = label_shift_value
        self.remapping_dict_path = remapping_dict_path
        if self.remapping_dict_path is not None:
            self.remapping_dict = np.load(self.remapping_dict_path, allow_pickle=True).item()
        else:
            self.remapping_dict = None
    def __call__(self, results):
         
        assert 'gt_semantic_seg' in results.keys(), 'gt_semantic_seg is not in results'
        # convert to long
        results['gt_semantic_seg'] = results['gt_semantic_seg'].astype(np.int64)
        
        results['gt_semantic_seg'] = results['gt_semantic_seg'] + self.label_shift_value
        ignore_index_shifted = self.ignore_index + self.label_shift_value
        results['gt_semantic_seg'][results['gt_semantic_seg'] == ignore_index_shifted] \
            = self.ignore_index_target_value
        if self.remapping_dict is not None:
            available_labels = np.unique(results['gt_semantic_seg'])
            for original_label, target_label in self.remapping_dict.items():
                original_label = original_label + self.label_shift_value
                if original_label not in available_labels:
                    continue
                results['gt_semantic_seg'][results['gt_semantic_seg'] == original_label] = target_label
                
        return results

    def __repr__(self):
        return self.__class__.__name__

