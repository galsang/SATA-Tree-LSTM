import argparse
import copy
import datetime
import logging
import numpy as np

import torch
import torchtext
from torch import nn, optim
from tensorboardX import SummaryWriter

from modules.classifier import Classifier
from utils.tools import load_config, load_data
from utils.tools import set_logger, set_seed
from utils.tools import save_model, print_params, print_args
from test import test


def train(args, data, cv=None):
    model = Classifier(args, data.WORD.vocab.vectors).to(torch.device(args.device))
    print_params(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer.endswith('W'):
        optimizer = getattr(optim, args.optimizer[:-1])(parameters, lr=args.learning_rate)
    else:
        optimizer = getattr(optim, args.optimizer)(parameters,
                                                   lr=args.learning_rate,
                                                   weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    log_dir = f'runs/{args.dataset}/{args.model_time}'
    if cv: log_dir += f'/{cv}'
    writer = SummaryWriter(log_dir=log_dir)

    loss, iterations = 0, 0
    dev_loss, dev_acc = 0, 0
    max_dev_acc, max_test_acc = 0, 0
    iterator = data.train_iter

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.epoch * len(iterator),
                                                     eta_min=args.eta_min)

    for e in range(1, args.epoch + 1):
        train_acc, train_size = 0, 0
        logging.info(f'Epoch: {e}')
        for i, batch in enumerate(iterator):
            iterations += 1

            model.train()
            pred = model(batch)
            train_acc += (pred.max(dim=1)[1] == batch.label).sum().float()
            train_size += len(pred)

            optimizer.zero_grad()
            batch_loss = criterion(pred, batch.label)
            loss += batch_loss.item()
            if args.optimizer.endswith('W'):
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-args.weight_decay * args.learning_rate, param.data)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step(iterations - 1)

            if iterations % args.print_freq == 0:
                c = iterations // args.print_freq
                writer.add_scalar('loss/train', loss, c)
                if hasattr(data, 'dev_iter'):
                    dev_loss, dev_acc = test(model, data, mode='dev')
                    writer.add_scalar('loss/dev', dev_loss, c)
                    writer.add_scalar('acc/dev', dev_acc, c)
                test_loss, test_acc = test(model, data, mode='test')
                writer.add_scalar('loss/test', test_loss, c)
                writer.add_scalar('acc/test', test_acc, c)

                logging.info(f'L: {loss:.3f} / DL: {dev_loss:.3f} / TL: {test_loss:.3f} '
                             f'/ DAcc: {dev_acc:.3f} / TAcc: {test_acc:.3f}')
                loss = 0

                if hasattr(data, 'dev_iter'):
                    if dev_acc > max_dev_acc:
                        max_dev_acc = dev_acc
                        max_test_acc = test_acc
                        best_model = copy.deepcopy(model)
                else:
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        best_model = copy.deepcopy(model)

        train_acc = (train_acc / train_size).cpu().item()
        logging.info(f'Train_acc: {train_acc:.3f}')
        writer.add_scalar('acc/train', train_acc, e)

    writer.close()
    logging.info(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')
    return best_model, max_dev_acc, max_test_acc


def cross_validation(args, data):
    fields = [('text', data.WORD), ('label', data.LABEL), ('transitions', data.PARSE),
              ('word_tag', data.WORD_TAG), ('cons_tag', data.CONS_TAG)]
    results = []

    for cv in range(10):
        logging.info(f'---CV {cv+1}---')
        split_start = (len(data.examples) // 10) * cv
        split_end = (len(data.examples) // 10) * (cv + 1)

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

        best_model, _, max_test_acc = train(args, data, cv + 1)
        save_model(args, best_model, None, max_test_acc, cv=cv + 1)
        results.append(max_test_acc)
    logging.info(f'Averaged test acc: {np.mean(results):.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SST2',
                        help='options: SST2, SST5, MR, SUBJ, TREC')
    parser.add_argument('--optimizer', default='AdadeltaW',
                        help='options: Adadelta, AdadeltaW, Adam, AdamW')
    parser.add_argument('--use-leafLSTM', default=1, type=int,
                        help='options: 0==FF, 1==LSTM, 2==bi-LSTM')
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'model_time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    set_logger(args)
    logging.info(f'Start time: {args.model_time}')
    set_seed(args.random_seed)
    logging.info('Loading saved config from yaml...')
    args = load_config(args)
    logging.info(f'Dataset: {args.dataset}')
    logging.info('Loading data...')
    args, data = load_data(args)
    logging.info('Training start!')
    print_args(args)

    if not hasattr(data, 'test_iter'):  # cross validation (no test data)
        cross_validation(args, data)
    else:
        best_model, max_dev_acc, max_test_acc = train(args, data)
        save_model(args, best_model, max_dev_acc, max_test_acc)
    logging.info('Training finished!')


if __name__ == '__main__':
    main()
