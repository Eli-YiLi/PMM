# Pseudo-mask Matters in Weakly-supervised Semantic Segmentation

By Yi Li, Zhanghui Kuang, Liyang Liu, Yimin Chen, Wayne Zhang

SenseTime, Tsinghua University

### Table of Contents
0. [Introduction](#Introduction)
0. [Classification](#Classification)
0. [Segmentation](#Segmentation)
0. [License](#License)

### Introduction

This is a PyTorch implementation of [Pseudo-mask Matters in Weakly-supervised Semantic Segmentation](https://arxiv.org/pdf/2108.12995.pdf).(ICCV2021).

In this paper, we propose Coefficient of Variation Smoothing and Proportional Pseudo-mask Generation to generate high quality pseudo-mask in classification part.
In segmentation part, we propose Pretended Under-Fitting strategy and Cyclic Pseudo-mask for better utilization of pseudo-mask.

### Classification

#### Data Preparation
1. Download VOC12 [OneDrive](https://1drv.ms/f/s!Agn5nXKXMkK5aigB0g238YxuTxs), [BaiduYun](https://pan.baidu.com/s/1GL3zXZuapuXmH9E7Xy8-Fg)
2. Download COCO14 [BaiduYun](https://pan.baidu.com/s/1GL3zXZuapuXmH9E7Xy8-Fg)
3. Download pretrained models [OneDrive](https://1drv.ms/f/s!Agn5nXKXMkK5aigB0g238YxuTxs), [BaiduYun](https://pan.baidu.com/s/1GL3zXZuapuXmH9E7Xy8-Fg)

(extract code of BaiduYun: mtci)


#### Get Started
    git clone https://github.com/Eli-YiLi/PMM
    cd PMM
    ln -s [path to model files] models
    ln -s [path to VOC12] voc12
    ln -s [path to COCO14] coco14
    pip3 install -r requirements.txt
    bash slurm_run.sh [partition name] [dataset name] / bash dist_run.sh [dataset name]

### Segmentation
Please refer to [WSSS_MMSeg](https://github.com/Eli-YiLi/WSSS_MMSeg)

### License
Please refer to: [LICENSE](LICENSE).
