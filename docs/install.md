# Prepare Environment

Our segmentation code is developed on top of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/) and [ViT-Adapter](https://github.com/czczup/ViT-Adapter).

## Install

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
ln -s ../detection/ops ./
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

#### **ADE20K/Cityscapes/COCO Stuff/Pascal Context**

Please follow the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation to download **ADE20K, Cityscapes, COCO Stuff and Pascal Context**.

#### BDD

Please visit the [official website](https://bdd-data.berkeley.edu/) to download the **BDD** dataset.

| Name   | Year | Type       | Data         | Repo                                                                      | Paper                                     |
| ------ | ---- | ---------- | ------------ | ------------------------------------------------------------------------- | ----------------------------------------- |
| DeiT   | 2021 | Supervised | ImageNet-1K  | [repo](https://github.com/facebookresearch/deit/blob/main/README_deit.md) | [paper](https://arxiv.org/abs/2012.12877) |
| BEiT   | 2021 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit)               | [paper](https://arxiv.org/abs/2106.08254) |
| BEiTv2 | 2022 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2)              | [paper](https://arxiv.org/abs/2208.06366) |
