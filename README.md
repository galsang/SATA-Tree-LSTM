# SATA Tree-LSTM

This repository contains the implementation of 
**SATA** (**S**tructure-**A**ware **T**ag **A**ugmented) **Tree-LSTM**, 
which is presented in **Dynamic Compositionality in Recursive Neural Networks with Structure-aware Tag Representations** (AAAI 2019).
For a detailed illustration of the architecture, please refer to the [paper]((https://arxiv.org/pdf/1809.02286.pdf)).

When doing following work with this code, please cite our paper with the following BibTex at this time
(It will be changed as the proceeding comes out).

    @article{kim2018dynamic,
      title={Dynamic Compositionality in Recursive Neural Networks with Structure-aware Tag Representations},
      author={Kim, Taeuk and Choi, Jihun and Edmiston, Daniel and Bae, Sanghwan and Lee, Sang-goo},
      journal={arXiv preprint arXiv:1809.02286},
      year={2018}
    }


## Experimental Results (reported in the paper)

| Dataset        | SST-2      | SST-5      | MR         | SUBJ       | TREC       | 
|:--------------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Test acc. (%)  | 91.3       | 54.4       | 83.8       | 95.4       | 96.2       |


## Development Environment

- Ubuntu 16.04 LTS (64bit)
- GPU support with Titan XP or GTX 1080
- **Python** (>3.6)
- **PyTorch** (>0.4.0)


## Pre-requisites

Please install the following libraries specified in the **requirements.txt** first.

    numpy==1.15.4
    nltk==3.2.4
    torch==0.4.1
    tensorboardX==1.4
    PyYAML==3.13
    torchtext==0.3.1


## Training

> python train.py --help

	usage: train.py [-h] --dataset DATASET --random-seed RANDOM_SEED
                [--optimizer OPTIMIZER] [--use-leafLSTM USE_LEAFLSTM]
                [--gpu GPU]

    optional arguments:
      -h, --help                    show this help message and exit
      --dataset DATASET             options: SST2, SST5, MR, SUBJ, TREC
      --optimizer OPTIMIZER         options: Adadelta, AdadeltaW, Adam, AdamW
      --use-leafLSTM USE_LEAFLSTM   options: 0==FF, 1==LSTM, 2==bi-LSTM
      --gpu GPU


* More task-specific hyper-parameters can be customized by modifying **yaml** files (e.g. MR.yaml) in the **config** folder.
* We do not support SNLI in this code as our own parsed dataset is quite large to be uploaded on Github.
* \[**NOTE**\] As experimental results can be varied depending on random seeds, we provide pre-trained models in the **saved-models**.


## Test

> python test.py --help

	usage: test.py [-h] --path PATH [--gpu GPU]

    optional arguments:
      -h, --help   show this help message and exit
      --path PATH
      --gpu GPU

- _PATH_ means an indicator to the target in which saved models and arguments are located
(they are automatically created after the end of training). e.g. 
    
        --path=saved_models/SST2/yyyymmdd-HH:MM:SS 
        --path=saved_models/MR/yyyymmdd-HH:MM:SS
    
    
## Visualization
   
When you want to check how accuracy and loss move as learning goes by, 
you can visualize learning curves on Tensorboard by typing the command following:

> tensorboard --logdir=runs

Moreover, you can check training logs in the **logs** folder.

- \[**WARNING**\] Tensorboard logs are written automatically for every training.


## Supplemental materials
   
The supplemental materials for our paper can be found in **supplemental_materials**.
It includes the details about specific settings for our experiments. 