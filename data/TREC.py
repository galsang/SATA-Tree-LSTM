import os
import torch

from torchtext import data
from torchtext.vocab import GloVe

from utils.data_utils import preprocess_WORD, preprocess_PARSE, \
    preprocess_WORD_TAG, preprocess_CONS_TAG_root_divided


class TREC():
    def __init__(self, args):
        path = '.data/'
        dataset_path = path + f'/{args.dataset}/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        test_examples_path = dataset_path + 'test_examples.pt'

        self.WORD = data.Field(batch_first=True,
                               tokenize=lambda s: s,
                               lower=True,
                               include_lengths=True,
                               preprocessing=preprocess_WORD)
        self.PARSE = data.Field(batch_first=True,
                                tokenize=lambda s: s,
                                include_lengths=True,
                                unk_token=None,
                                preprocessing=preprocess_PARSE)
        self.WORD_TAG = data.Field(batch_first=True,
                                   tokenize=lambda s: s,
                                   include_lengths=True,
                                   unk_token=None,
                                   preprocessing=preprocess_WORD_TAG)
        self.CONS_TAG = data.Field(batch_first=True,
                                   tokenize=lambda s: s,
                                   include_lengths=True,
                                   unk_token=None,
                                   preprocessing=preprocess_CONS_TAG_root_divided)
        self.LABEL = data.Field(sequential=False, unk_token=None, preprocessing=None)

        if os.path.exists(dataset_path):
            print("Loading splits...")

            fields = [('text', self.WORD), ('label', self.LABEL), ('transitions', self.PARSE),
                      ('word_tag', self.WORD_TAG), ('cons_tag', self.CONS_TAG)]

            train_examples = torch.load(train_examples_path)
            test_examples = torch.load(test_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=fields)
            self.test = data.Dataset(examples=test_examples, fields=fields)
        else:
            print("building splits...")

            fields = {'parse': [('text', self.WORD), ('transitions', self.PARSE),
                                ('word_tag', self.WORD_TAG), ('cons_tag', self.CONS_TAG)],
                      'label': ('label', self.LABEL)}

            self.train, self.test = data.TabularDataset.splits(path='.data/TREC/parsed/',
                                                               train='trec_train.jsonl',
                                                               test='trec_test.jsonl',
                                                               format='json',
                                                               fields=fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.test.examples, test_examples_path)

        self.WORD.build_vocab(self.train, self.test, vectors=GloVe(name='840B', dim=300))
        self.WORD_TAG.build_vocab(self.train)
        self.CONS_TAG.build_vocab(self.train)
        self.LABEL.build_vocab(self.train)
        self.PARSE.build_vocab(self.train)

        self.train_iter, self.test_iter = data.BucketIterator.splits((self.train, self.test),
                                                                     batch_sizes=[args.batch_size] * 2,
                                                                     device=args.device,
                                                                     sort_key=lambda x: len(x.text))
