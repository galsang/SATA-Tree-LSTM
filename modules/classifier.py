import torch.nn as nn

from modules.encoder import Encoder
from utils.nn import Linear


class Classifier(nn.Module):
    def __init__(self, args, pretrained):
        super(Classifier, self).__init__()
        self.args = args

        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=args.freeze_word_emb)
        self.encoder = Encoder(args)

        self.fc1 = nn.Sequential(nn.BatchNorm1d(self.args.hidden_size),
                                 Linear(self.args.hidden_size, args.output_mlp_size,
                                        dropout=args.mlp_dropout),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.BatchNorm1d(args.output_mlp_size),
                                 Linear(args.output_mlp_size, args.class_size,
                                        dropout=args.mlp_dropout))
        self.dropout = nn.Dropout(args.emb_dropout)
        self.reset_params()

    def reset_params(self):
        # initialize <unk> in the word embedding with uniform dist.
        nn.init.uniform_(self.word_emb.weight[0], -0.005, 0.005)
        # initialize the last mlp with uniform dist. instead of kaiming normal
        nn.init.uniform_(self.fc2[1].linear.weight, -0.005, 0.005)

    def forward(self, batch):
        # (batch, sent_len, word_dim)
        text = self.word_emb(batch.text[0])
        text = self.dropout(text)
        text_len = batch.text[1]
        # (batch, action_len)
        action = batch.transitions[0]
        action_len = batch.transitions[1]

        word_tag = batch.word_tag[0]
        cons_tag = batch.cons_tag[0]

        # (batch, hidden_size)
        x = self.encoder(text, text_len, action, action_len, word_tag, cons_tag).contiguous()
        x = self.fc2(self.fc1(x))
        return x
