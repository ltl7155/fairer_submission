# Fairer: Fairness as Decision Rationale Alignment.
This is an anonymous submission to reproduce the work.

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

This repo contains required scripts to reproduce results from paper:

In this version, we integrate our code into a network pruning process. We hope we can combine our method with pruning methods to improve fairness.

Fairer <br>
anonymous .<br>


## Installation


This code requires PyTorch 1.7+ and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit -c pytorch 
```
* Please check pytorch.org for installing instruction.

For the best reproducibility of results you will need NVIDIA GeForce RTX 3090. 


The code was tested with python3.6 the following software versions:

| Software        | version | 
| ------------- |-------------| 
| cuDNN         | 8.0.5 |
| Pytorch      | 1.9.1  |
| torchvision      | 0.10.1  |
| CUDA | v11.1    |
|cudatoolkit | 11.1.1|


# Celeb-A Experiment

* This is a sample cmd to run code, for more details, please check the paper.

## Dataset
Place the CelebA dataset (```list_attr_celeba.txt```, ```list_eval_partition.txt```, ```img_align_celeba```) under directory ```./celeba``` and run ```data_processing.py``` to process the dataset. 

# Vanilla & Vaseline
```
cat_run_van_alexnet.py
```

# FairReg(No Aug) 

```
cat run_FairReg_alexnet.py
```

# FairReg(Aug)
```
cat run_FairReg_Aug_alexnet.py
```

# DRAlign(Ours)
```
cat run_CAIGA_alexnet.py
```

