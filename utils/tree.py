import torch
import torch.nn as nn


class TreeLSTM_with_cell_input(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(TreeLSTM_with_cell_input, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.weights = nn.Linear(hidden_size * 2 + input_size, hidden_size * 5)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.weights.weight)
        nn.init.constant_(self.weights.bias, 0)
        nn.init.constant_(self.weights.bias[:self.hidden_size * 2], 1)

    def forward(self, hl, cl, hr, cr, e):
        x = torch.cat([hl, hr, e], dim=-1)
        fl, fr, i, o, g = torch.chunk(self.weights(x), 5, dim=-1)
        # (batch, sent_len, hidden_size)
        c = fl.sigmoid() * cl + fr.sigmoid() * cr + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        # (batch, hidden_size * 2)
        return torch.cat([h, c], dim=-1)


class TreeLSTM(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.hidden_weights = nn.Linear(hidden_size * 2, hidden_size * 5)
        self.input_weights = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.input_weights.weight)
        nn.init.kaiming_normal_(self.hidden_weights.weight)
        nn.init.constant_(self.hidden_weights.bias, 0)
        nn.init.constant_(self.hidden_weights.bias[:self.hidden_size * 2], 1)

    def forward(self, hl, cl, hr, cr, e):
        x = torch.cat([hl, hr], dim=-1)
        fl, fr, i, o, g = torch.chunk(self.hidden_weights(x), 5, dim=-1)
        fl2, fr2, i2, o2 = torch.chunk(self.input_weights(e), 4, dim=-1)
        c = (fl + fl2).sigmoid() * cl + (fr + fr2).sigmoid() * cr + (i + i2).sigmoid() * g.tanh()
        h = (o + o2).sigmoid() * c.tanh()
        # (batch, hidden_size * 2)
        return torch.cat([h, c], dim=-1)


class NormalizedTreeLSTM(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(NormalizedTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.gate_weights = nn.Linear(hidden_size * 2 + input_size, hidden_size * 4)
        self.cell_weights = nn.Linear(hidden_size * 2, hidden_size)

        for g in ['fl', 'fr', 'i', 'o']:
            setattr(self, f'LN_{g}', nn.LayerNorm(hidden_size))

    def reset_params(self):
        nn.init.kaiming_normal_(self.gate_weights.weight)
        nn.init.constant_(self.gate_weights.bias, 0)
        nn.init.constant_(self.gate_weights.bias[:self.hidden_size * 2], 1)

        nn.init.kaiming_normal_(self.cell_weights.weight)
        nn.init.constant_(self.cell_weights.bias, 0)

    def forward(self, hl, cl, hr, cr, e):
        fl, fr, i, o = torch.chunk(self.gate_weights(torch.cat([hl, hr, e], dim=-1)), 4, dim=-1)
        g = self.cell_weights(torch.cat([hl, hr], dim=-1))
        # (batch, sent_len, hidden_size)
        c = self.LN_fl(fl).sigmoid() * cl + self.LN_fr(fr).sigmoid() * cr + self.LN_i(i).sigmoid() * g.tanh()
        h = self.LN_o(o).sigmoid() * c.tanh()
        # (batch, hidden_size * 2)
        return torch.cat([h, c], dim=-1)
