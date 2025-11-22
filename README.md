

# Cross-Model Nested Fusion Network for Salient Object Detection in Optical Remote Sensing Images

Mingzhu Xu, Sen Wang, Yupeng Hu, Haoyu Tang, Runmin Cong, Liqiang Nie, IEEE Transactions on Cybernetics 2025.

## Structure

![image-20251122164015033](./Fig/CMNFNet.png)

## Introduction

This repository is the official implementation of our TGRS 2025 paper: [Cross-Model Nested Fusion Network for Salient Object Detection in Optical Remote Sensing Images](https://ieeexplore.ieee.org/document/11163514)

In this paper, we propose a novel Cross-Model Nested Fusion Network (CMNFNet), which leverages heterogeneous features to increase the performance of salient object detection in optical remote sensing images (ORSI-SOD). Specifically, CMNFNet comprises two heterogeneous encoders: a conventional CNN-based encoder that can model local pattern features, and a specially designed graph convolutional network (GCN)-based encoder with local and global receptive fields that can model local and global features simultaneously. To effectively differentiate between multiple salient objects of different sizes or complex topological structures within an image, we project the image into two different graphs with different receptive fields and conduct message passing through two parallel graph convolutions. Finally, the heterogeneous features extracted from the two encoders are fused in the well-designed Attention Enhanced Cross Model Nested Fusion Module (AECMNFM). This module is meticulously crafted to integrate features progressively, allowing the model to adaptively eliminate background interference while simultaneously refining the feature representations. Comprehensive experimental analyses on benchmark datasets (ORSSD, EORSSD, and ORSI-4199) demonstrate the superiority of our CMNFNet over 16 state-of-the-art (SOTA) models. The contributions of this paper are as follows:

- We innovatively devise a CMNFNet for ORSI-SOD. Unlike existing models that rely on a single encoder or direct fusion, it employs two heterogeneous encoders (CNN and a custom-designed GCN) and progressively fuses their features through a novel nested fusion strategy, effectively complementing heterogeneous representations.
- We propose a novel graph-based convolution subnetwork (Encoder-GCN) as an auxiliary encoder. It employs dual parallel graph convolutions in distinct semantic spaces with varying receptive fields to jointly model local and global context, facilitating the multiscale perception of complex salient objects.
- We propose a novel Attention Enhanced Cross Model Nested Fusion Module (AECMNFM). Unlike traditional fusion methods that cause mutual interference, this approach progressively integrates heterogeneous features in a complementary manner, enhanced by a hybrid attention mechanism that selectively emphasizes salient information while effectively suppressing background noise.
- We conduct thorough experiments across three challenging datasets to evaluate the overall performance and the component effectiveness, demonstrating that our method outperforms 16 advanced models in ORSI-SOD.

## Setting Up

### Preliminaries

The code has been verified to work with PyTorch v2.0.0 + CUDA 11.8 and Python 3.8.

### Package Dependencies

```
# 1.Create a new Conda environment with Python 3.8 then activate it
conda create -n cmnfnet python=3.8
conda activate cmnfnet

# 2.Install PyTorch v2.0.0 with a CUDA 11.8 version
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-sparse torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 3.Install the packages in requirements.txt
pip install -r requirements.txt
```

## Training

```
python mainNest.py
```

## Saliency maps

We provide saliency maps of our ACCoNet (VGG_backbone and ResNet_backbone on ORSSD, EORSSD, and additional ORSI-4199 datasets.

## Citation

**Please kindly cite the papers if this code is useful and helpful for your research.**

```
@ARTICLE{11163514,
  author={Xu, Mingzhu and Wang, Sen and Hu, Yupeng and Tang, Haoyu and Cong, Runmin and Nie, Liqiang},
  journal={IEEE Transactions on Cybernetics}, 
  title={Cross-Model Nested Fusion Network for Salient Object Detection in Optical Remote Sensing Images}, 
  year={2025},
  volume={55},
  number={11},
  pages={5332-5345},
  keywords={Feature extraction;Transformers;Adaptation models;Object detection;Remote sensing;Interference;Semantics;Optical sensors;Optical imaging;Image edge detection;Cross-model nested fusion;graph convolution network;optical remote sensing images (ORSIs);salient object detection (SOD)},
  doi={10.1109/TCYB.2025.3571913}}
```

