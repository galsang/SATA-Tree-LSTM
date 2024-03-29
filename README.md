# SATA Tree-LSTM

This repository contains the implementation of 
**SATA** (**S**tructure-**A**ware **T**ag **A**ugmented) **Tree-LSTM**, 
which is proposed by **Dynamic Compositionality in Recursive Neural Networks with Structure-aware Tag Representations (AAAI 2019)**.
For a detailed illustration of the architecture, refer to our [paper](https://aaai.org/ojs/index.php/AAAI/article/view/4628).

When utilizing this code for future work, please cite our paper with the following BibTex.

	@inproceedings{kim2019dynamic,
	  title={Dynamic Compositionality in Recursive Neural Networks with Structure-Aware Tag Representations},
	  author={Kim, Taeuk and Choi, Jihun and Edmiston, Daniel and Bae, Sanghwan and Lee, Sang-goo},
	  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	  volume={33},
	  pages={6594--6601},
	  year={2019}
	}


**\[NOTE\]** If you need to parse your own data, check [the example codes](https://github.com/galsang/parser) (implemented in JAVA, resorting to the Standford parser) which were utilized to build the experimental data (the files located in .data) for our paper.

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
