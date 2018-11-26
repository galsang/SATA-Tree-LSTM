import logging
import os
import pickle
import random
import yaml

from functools import reduce

import torch

from data.SST import SST
# from data.SST_elmo import SST
from data.SUBJ import SUBJ
from data.TREC import TREC
from data.MR import MR


def load_config(args):
    with open(f'config/{args.dataset}.yaml', 'r', encoding='utf-8') as f:
        saved_args = yaml.load(f.read())
        for key in saved_args:
            if not hasattr(args, key):
                setattr(args, key, saved_args[key])

    return args


def load_data(args):
    datasets = {'SST2': SST, 'SST5': SST, 'MR': MR, 'SUBJ': SUBJ, 'TREC': TREC}
    class_sizes = {'SST2': 2, 'SST5': 5, 'MR': 2, 'SUBJ': 2, 'TREC': 6}

    if args.dataset not in datasets.keys():
        raise NotImplementedError('Not available dataset!')
    data = datasets[args.dataset](args)
    setattr(args, 'class_size', class_sizes[args.dataset])
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'word_tag_vocab_size', len(data.WORD_TAG.vocab))
    setattr(args, 'cons_tag_vocab_size', len(data.CONS_TAG.vocab))
    return args, data


def set_logger(args, mode='train'):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    logger.addHandler(logging.StreamHandler())

    if mode == 'train':
        path = f'logs/{args.dataset}'
        if not os.path.exists(path):
            os.makedirs(path)
        logger.addHandler(logging.FileHandler(
            f'{path}/{args.model_time}.log'))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def save_model(args, model, dev_acc=None, test_acc=None, cv=0):
    path = f'saved_models/{args.dataset}'

    if not os.path.exists(path):
        os.makedirs(path)

    if cv:
        path += f'/{args.model_time}/{cv}'
        if not os.path.exists(path):
            os.makedirs(path)
        path += f'/{test_acc:.3f}'
    else:
        path += f'/{args.model_time}'
        if dev_acc:
            path += f'-{dev_acc:.3f}'
        if test_acc:
            path += f'-{test_acc:.3f}'

    torch.save(model.state_dict(), path + '.pt')
    with open(f'{path}.args', 'wb') as f:
        pickle.dump(args, f)


def print_params(model):
    total = 0
    logging.info('<List of parameters>')
    for name, param in model.named_parameters():
        if param.requires_grad and name != 'word_emb.weight':
            logging.info(f'{name} {param.size()} {reduce(lambda x, y: x * y, param.size())}')
            total += param.numel()
    logging.info(f'<Total # of paramters>: {total}')


def print_args(args):
    logging.info('<List of arguments>')
    for a in args.__dict__:
        logging.info(f'{a}: {args.__dict__[a]}')
