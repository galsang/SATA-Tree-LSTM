import os
import torch
import random

from torchtext import data
from torchtext.vocab import GloVe

from utils.data_utils import preprocess_WORD, preprocess_WORD_TAG, \
    preprocess_PARSE, preprocess_CONS_TAG_root_merged


class MR():
    def __init__(self, args):
        path = '.data/'
        dataset_path = path + f'/{args.dataset}/torchtext/'
        examples_path = dataset_path + 'examples.pt'

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
                                   preprocessing=preprocess_CONS_TAG_root_merged)
        self.LABEL = data.Field(sequential=False, unk_token=None, preprocessing=None)

        if os.path.exists(dataset_path):
            print("Loading splits...")

            fields = [('text', self.WORD), ('label', self.LABEL), ('transitions', self.PARSE),
                      ('word_tag', self.WORD_TAG), ('cons_tag', self.CONS_TAG)]

            self.examples = torch.load(examples_path)
            self.dataset = data.Dataset(examples=self.examples, fields=fields)
        else:
            print("building splits...")

            fields = {'parse': [('text', self.WORD), ('transitions', self.PARSE),
                                ('word_tag', self.WORD_TAG), ('cons_tag', self.CONS_TAG)],
                      'label': ('label', self.LABEL)}

            self.dataset = data.TabularDataset.splits(path='.data/MR/parsed/',
                                                      train='mr.jsonl',
                                                      format='json',
                                                      fields=fields)[0]
            self.examples = self.dataset.examples
            random.shuffle(self.examples)

            os.makedirs(dataset_path)
            torch.save(self.examples, examples_path)

        self.WORD.build_vocab(self.dataset, vectors=GloVe(name='840B', dim=300))
        self.WORD_TAG.build_vocab(self.dataset)
        self.CONS_TAG.build_vocab(self.dataset)
        self.LABEL.build_vocab(self.dataset)
        self.PARSE.build_vocab(self.dataset)
