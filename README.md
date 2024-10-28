# CL4EF

This repository provides the official PyTorch implementation of our paper "Short-Term Earthquake Forecasting Using Electromagnetic and Geoacoustic Observations via Contrastive Learning".

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- python 3.7.16
- cudatoolkit 11.1.1
- torch 1.8.0
- torchvision 0.9.0
- numpy 1.21.5
- scikit-learn 1.0.2

## Quick Example

- Download the preprocessed [test set](https://drive.google.com/file/d/1xAKkkElaHxJgbGqsVcpSoOIJrjWEXWnA/view?usp=sharing) and put it in the `datasets/` directory.

- Download the pre-trained [model](https://drive.google.com/file/d/16ByM4Bv8ukbfjnAVIHZ3kVX8R1pUDZ2v/view?usp=sharing) and put it in the `results/` directory.

- To do the quick test, run:

```bash
python downstream_test.py
```

## Datasets

- Download [AETA](https://platform.aeta.cn/zh-CN/competitionpage/download) dataset and [earthquake catalog](https://news.ceic.ac.cn/index.html?time=1704271080).

- All downloaded electromagnetic and geoacoustic observational data are stored in the `datasets/` directory, comprising 150 CSV files. Each file within this directory contains data from a single observational station.


## Pretext Task

- We introduce the synchronous response consistency hypothesis, assuming that different observations within the same time window should respond consistently to the same physical process. Following this hypothesis, we design a contrastive loss to perform contrastive learning in the pretext task.

- To do the pretext prediction task on a large-scale dataset composed of all samples, run:

```python
python main_pretext.py
```

## Downstream Task

- We set the classification task as a downstream task, focusing on whether major earthquakes will occur in the coming week.

- To do the downstream classification task on a small-scale yet balanced dataset built through undersampling, run:

```python
python main_downstream.py
```