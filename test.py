import argparse
import logging
import glob
import pickle
import numpy as np

import torch
import torch.nn as nn
import torchtext

from modules.classifier import Classifier
from utils.tools import load_data, set_logger, set_seed


def test(model, data, mode):
    iterator = iter(getattr(data, f'{mode}_iter'))
    criterion = nn.CrossEntropyLoss()
    acc, loss, size = 0, 0, 0

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in iterator:
            pred = model(batch)
            batch_loss = criterion(pred, batch.label)
            loss += batch_loss.item()
            _, pred = pred.max(dim=1)
            acc += (pred == batch.label).sum().float()
            size += len(pred)

    acc /= size
    acc = acc.cpu().item()
    return loss, acc


def cross_validation_test(args, data, path):
    fields = [('text', data.WORD), ('label', data.LABEL), ('transitions', data.PARSE),
              ('word_tag', data.WORD_TAG), ('cons_tag', data.CONS_TAG)]
    results = []

    for cv in range(10):
        split_start = (len(data.examples) // 10) * cv
        split_end = (len(data.examples) // 10) * (cv + 1)

        model = Classifier(args, data.WORD.vocab.vectors).to(torch.device(args.device))
        model.load_state_dict(torch.load(glob.glob(f'{path}/{cv+1}/*.pt')[0]))

        data.train = torchtext.data.Dataset(
            examples=data.examples[:split_start] + data.examples[split_end:],
            fields=fields)
        data.test = torchtext.data.Dataset(
            examples=data.examples[split_start:split_end],
            fields=fields)
        data.train_iter, data.test_iter = torchtext.data.BucketIterator.splits(
            (data.train, data.test),
            batch_sizes=[args.batch_size] * 2,
            device=args.device,
            sort_key=lambda x: len(x.text))

        _, test_acc = test(model, data, 'test')
        logging.info(f'CV{cv+1} test acc: {test_acc:.3f}')
        results.append(test_acc)
    logging.info(f'Averaged test acc: {np.mean(results):.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()
    path = args.path
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    args_path = (glob.glob(f'{path}*.args') + glob.glob(f'{path}/1/*.args'))[0]
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
        setattr(args, 'device', device)
    set_logger(args.model_time, 'test')
    set_seed(args.random_seed)
    logging.info(f'Dataset: {args.dataset}')
    logging.info('Loading data...')
    args, data = load_data(args)

    logging.info('Test start!')
    if not hasattr(data, 'test_iter'):  # cross validation (no test data)
        cross_validation_test(args, data, path)
    else:
        model = Classifier(args, data.WORD.vocab.vectors).to(torch.device(args.device))
        model.load_state_dict(torch.load(glob.glob(f'{path}*.pt')[0]))

        _, test_acc = test(model, data, 'test')
        logging.info(f'Test acc: {test_acc:.3f}')


if __name__ == '__main__':
    main()
