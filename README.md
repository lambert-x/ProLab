# ProLab: Property-level Label Space

## News

## Method

## Contents

## Getting Started

Our segmentation code is developed on top of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/) and [ViT-Adapter](https://github.com/czczup/ViT-Adapter).

## Install

We have two tested environments based on **torch 1.9+cuda 11.1+MMSegmentation 0.20.2** and **torch 1.13.1+torch11.7+MMSegmentation**

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
conda create -n prolab python=3.8
conda activate prolab
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
pip install -r requirements.txt
cd ops
sh make.sh # compile deformable attention
```

### Data Preparation

##### **ADE20K/Cityscapes/COCO Stuff/Pascal Context**

Please follow the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation to download **ADE20K, Cityscapes, COCO Stuff and Pascal Context**.

##### BDD

Please visit the [official website](https://bdd-data.berkeley.edu/) to download the **BDD** dataset.

### Training

### Testing

## Model Zoo

**ADE20K**

| Method  | Backbone      | Pretrain                                                                                                                   | Lr schd | Crop Size | mIoU | Config | Download |
|:-------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------:|:-------:|:---------:|:----:|:------:|:--------:|
| UperNet | ViT-Adapter-B | [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                           | 320k    | 512       | 49.0 |        |          |
| UperNet | ViT-Adapter-L | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)      | 160k    | 640       | 58.2 |        |          |
| UperNet | ViT-Adapter-L | [BEiTv2-L](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth) | 80K     | 896       | 58.7 |        |          |

**COCO-Stuff-164K**

**Pascal Context**

**Cityscapes**

**BDD**

## Catalog

## Citations

## Acknowledgement

For retrieving knowledf 

Our segmentation code is based on the great codebases [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/) and [ViT-Adapter](https://github.com/czczup/ViT-Adapter). 


