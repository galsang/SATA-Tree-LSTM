import logging
import os
import torch
import json

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import Tree
from utils.data_utils import parse, \
    preprocess_WORD, preprocess_PARSE, preprocess_WORD_TAG, \
    preprocess_CONS_TAG_root_divided, preprocess_CONS_TAG_root_merged


class SST():
    def __init__(self, args):
        path = '.data/sst'
        dataset_path = path + f'/{args.dataset}/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'
        test_examples_path = dataset_path + 'test_examples.pt'

        def preprocess_label(s):
            l = Tree.fromstring(s).label()

            pre = 'very ' if args.dataset == 'SST5' else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[l]

        if args.dataset == 'SST5':
            preprocess_cons_tag = preprocess_CONS_TAG_root_merged
        else:
            preprocess_cons_tag = preprocess_CONS_TAG_root_divided

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
                                   preprocessing=preprocess_cons_tag)
        self.LABEL = data.Field(sequential=False,
                                unk_token=None,
                                preprocessing=preprocess_label)

        fields = [('text', self.WORD), ('label', self.LABEL), ('transitions', self.PARSE),
                  ('word_tag', self.WORD_TAG), ('cons_tag', self.CONS_TAG)]

        if os.path.exists(dataset_path):
            logging.info("Loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)
            test_examples = torch.load(test_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=fields)
            self.dev = data.Dataset(examples=dev_examples, fields=fields)
            self.test = data.Dataset(examples=test_examples, fields=fields)
        else:
            logging.info("Building splits...")
            self.train, self.dev, self.test = \
                revised_torchtext_SST.splits(self.WORD, self.LABEL, self.PARSE,
                                             self.WORD_TAG, self.CONS_TAG,
                                             train_subtrees=True,
                                             fine_grained=(args.dataset == 'SST5'))

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)
            torch.save(self.test.examples, test_examples_path)

        self.WORD.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.WORD_TAG.build_vocab(self.train, self.dev, self.test)
        self.CONS_TAG.build_vocab(self.train, self.dev, self.test)
        self.LABEL.build_vocab(self.train)
        self.PARSE.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=args.device,
                                       sort_key=lambda x: len(x.text))


class revised_torchtext_SST(datasets.SST):
    def __init__(self, path, text_field, label_field, parse_field,
                 word_tag_field, cons_tag_field, subtrees=False, fine_grained=False, **kwargs):
        fields = {'tagged_parse': [('text', text_field), ('transitions', parse_field),
                                   ('word_tag', word_tag_field), ('cons_tag', cons_tag_field)],
                  'original_parse': ('label', label_field)}

        def test(t1, t2):
            t1 = parse(t1)
            t2 = parse(t2)

            for j, token1 in enumerate(t1):
                token2 = t2[j]
                if (token1 == "(" and token2 != "(") or (token1 == ")" and token2 != ")"):
                    return False

            return True

        with open(os.path.expanduser(path)) as f:
            if subtrees:
                examples = []
                for line in f:
                    json_line = json.loads(line)
                    if test(json_line['original_parse'], json_line['tagged_parse']):
                        tagged_parse_subtrees = list(Tree.fromstring(json_line['tagged_parse']).subtrees())
                        original_parse_subtrees = list(Tree.fromstring(json_line['original_parse']).subtrees())
                        for i in range(len(tagged_parse_subtrees)):
                            target = {'tagged_parse': ' '.join(str(tagged_parse_subtrees[i]).split()),
                                      'original_parse': ' '.join(str(original_parse_subtrees[i]).split())}
                            examples.append(data.Example.fromdict(target, fields))
                    else:
                        examples.append(data.Example.fromJSON(line, fields))
            else:
                examples = [data.Example.fromJSON(line, fields) for line in f]

        if not fine_grained:
            examples = [e for e in examples if e.label != 'neutral']

        fields = [('text', text_field), ('label', label_field), ('transitions', parse_field),
                  ('word_tag', word_tag_field), ('cons_tag', cons_tag_field)]
        super(datasets.SST, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, parse_field, word_tag_field, cons_tag_field,
               root='.data', train='train.jsonl', validation='dev.jsonl', test='test.jsonl',
               train_subtrees=False, **kwargs):
        path = '.data/sst/parsed/'

        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, parse_field,
            word_tag_field, cons_tag_field, subtrees=train_subtrees, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, parse_field,
            word_tag_field, cons_tag_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, parse_field,
            word_tag_field, cons_tag_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
