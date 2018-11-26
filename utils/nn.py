import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.orthogonal_(self.rnn.weight_hh)
        nn.init.kaiming_normal_(self.rnn.weight_ih)
        nn.init.constant_(self.rnn.bias_hh, val=0)
        nn.init.constant_(self.rnn.bias_ih, val=0)
        self.rnn.bias_hh.chunk(4)[1].fill_(1)

    def forward(self, x, hx, cx):
        return self.rnn(x, (hx, cx))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, bias=0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.bias = bias
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, self.bias)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
