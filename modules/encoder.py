import torch
import torch.nn as nn

from utils.nn import LSTMCell, Linear
from utils.tree import TreeLSTM, TreeLSTM_with_cell_input, NormalizedTreeLSTM


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        if self.args.use_leafLSTM == 0:
            self.word_leaf = nn.Sequential(Linear(args.word_dim, args.hidden_size * 2), nn.Tanh())
        elif self.args.use_leafLSTM == 1:
            self.word_leaf = LSTMCell(args.word_dim, args.hidden_size)
        elif self.args.use_leafLSTM == 2:
            assert args.hidden_size % 2 == 0
            self.word_leaf = LSTMCell(args.word_dim, args.hidden_size // 2)
            self.word_leaf_bw = LSTMCell(args.word_dim, args.hidden_size // 2)
        else:
            raise NotImplementedError('not available option for leaf module!')

        if self.args.dataset == 'SST5':
            self.word_tree = NormalizedTreeLSTM(args.hidden_size, args.tag_emb_dim)
        else:
            self.word_tree = TreeLSTM(args.hidden_size, args.tag_emb_dim)
        self.tag_leaf = nn.Sequential(Linear(args.tag_emb_dim, args.tag_emb_dim * 2), nn.Tanh())
        self.tag_tree = TreeLSTM_with_cell_input(args.tag_emb_dim, args.tag_emb_dim)

        self.tag_emb = nn.Embedding(args.word_tag_vocab_size + args.cons_tag_vocab_size, args.tag_emb_dim)
        nn.init.uniform_(self.tag_emb.weight, -0.005, 0.005)
        # nn.init.normal_(self.tag_emb.weight, 0, 0.01)
        nn.init.constant_(self.tag_emb.weight[0], 0)
        nn.init.constant_(self.tag_emb.weight[args.word_tag_vocab_size], 0)

    def forward(self, x, x_len, a, a_len, word_tag, cons_tag):
        """
        :param x: batched sentences (batch, sent_len, word_dim)
        :param x_len: lengths of the batched sentences (batch)
        :param a: batched actions (batch, action_len)
        :param a_len: lengths of the batched actions (batch)
        :param word_tag: word-level (leaf-level) tags (batch, sent_len)
        :param cons_tag: phrase-level tags (batch, sent_len * 2 - 1)
        :return: batched sentence representation (batch, hidden_size)
        """
        device = torch.device(torch.cuda.current_device())
        B = x.size(0)
        T = a.size(1)
        sent_len = x.size(1)

        buffer_cursor = torch.zeros(B, device=device, dtype=torch.long)
        queue_cursor = torch.ones(B, device=device, dtype=torch.long)
        tag_cursor = torch.zeros(B, device=device, dtype=torch.long)

        if self.args.use_leafLSTM == 0:
            x = self.word_leaf(x)
        elif self.args.use_leafLSTM == 1:
            hx = torch.zeros(B, self.args.hidden_size).to(device)
            cx = torch.zeros(B, self.args.hidden_size).to(device)
            new_x = []
            for t in range(sent_len):
                nhx, ncx = self.word_leaf(x[:, t], hx, cx)
                done = x_len.le(t).float().unsqueeze(1)
                hx = nhx * (1 - done) + hx * done
                cx = ncx * (1 - done) + cx * done
                new_x.append(torch.cat([hx, cx], dim=-1))
            x = torch.stack(new_x, dim=1)
        elif self.args.use_leafLSTM == 2:
            hx = torch.zeros(B, self.args.hidden_size // 2).to(device)
            cx = torch.zeros(B, self.args.hidden_size // 2).to(device)
            new_x = []
            for t in range(sent_len):
                nhx, ncx = self.word_leaf(x[:, t], hx, cx)
                done = x_len.le(t).float().unsqueeze(1)
                hx = nhx * (1 - done) + hx * done
                cx = ncx * (1 - done) + cx * done
                new_x.append(torch.cat([hx, cx], dim=-1))
            new_x = torch.stack(new_x, dim=1)
            new_xh, new_xc = new_x.chunk(2, dim=-1)

            hx_bw = torch.zeros(B, self.args.hidden_size // 2).to(device)
            cx_bw = torch.zeros(B, self.args.hidden_size // 2).to(device)
            new_x_bw = []
            for t in range(sent_len - 1, -1, -1):
                nhx_bw, ncx_bw = self.word_leaf_bw(x[:, t], hx_bw, cx_bw)
                done = x_len.gt(t).float().unsqueeze(1)
                hx_bw = nhx_bw * done + hx_bw * (1 - done)
                cx_bw = ncx_bw * done + cx_bw * (1 - done)
                new_x_bw.append(torch.cat([hx_bw, cx_bw], dim=-1))
            new_x_bw.reverse()
            new_x_bw = torch.stack(new_x_bw, dim=1)
            new_xh_bw, new_xc_bw = new_x_bw.chunk(2, dim=-1)

            x = torch.cat([new_xh, new_xh_bw, new_xc, new_xc_bw], dim=-1)
        else:
            raise (NotImplementedError('not available option for leaf module!'))

        # dummy for blank buffer
        x = torch.cat([x, torch.zeros(B, 1, self.args.hidden_size * 2, device=device)], dim=1)
        # stack
        S = torch.zeros(B, T + 1, self.args.hidden_size * 2, device=device)
        # for 'only a word' cases
        S[:, 0] = x[:, 0]
        S_tag = torch.zeros(B, T + 1, self.args.tag_emb_dim * 2, device=device)
        # queue
        Q = torch.zeros(B, sent_len + 2, device=device, dtype=torch.long)

        for t in range(1, T + 1):
            # [<pad>: 0, shift: 1, reduce: 2]
            action = a.select(1, t - 1)
            shift_indices = (action == 1).nonzero()
            reduce_indices = (action == 2).nonzero()
            shift_indices = shift_indices.squeeze(len(shift_indices.size()) - 1)
            reduce_indices = reduce_indices.squeeze(len(reduce_indices.size()) - 1)

            # 0. doing nothing as default (padding)
            S[:, t] = S[:, t - 1]
            # 1. shift
            if shift_indices.nelement() > 0:
                # move buffer top to stack
                S[shift_indices, t] = x[shift_indices, buffer_cursor[shift_indices]]
                S_tag[shift_indices, t] = self.tag_leaf(
                    self.tag_emb(word_tag[shift_indices, buffer_cursor[shift_indices]]))
                # update buffer cursor
                buffer_cursor[shift_indices] += 1
                # add this time step to queue
                queue_cursor[shift_indices] += 1
                Q[shift_indices, queue_cursor[shift_indices]] = t
            # 2. reduce
            if reduce_indices.nelement() > 0:
                # (batch, sent_len, hidden_size) * 2
                l_idx = Q[reduce_indices, queue_cursor[reduce_indices] - 1]
                r_idx = Q[reduce_indices, queue_cursor[reduce_indices]]

                hl, cl = S[reduce_indices, l_idx].chunk(2, dim=-1)
                hr, cr = S[reduce_indices, r_idx].chunk(2, dim=-1)

                hl_tag, cl_tag = S_tag[reduce_indices, l_idx].chunk(2, dim=-1)
                hr_tag, cr_tag = S_tag[reduce_indices, r_idx].chunk(2, dim=-1)
                p_tag = self.tag_emb(
                    cons_tag[reduce_indices, tag_cursor[reduce_indices]] + self.args.word_tag_vocab_size)
                tag_result = self.tag_tree(hl_tag, cl_tag, hr_tag, cr_tag, p_tag)

                S_tag[reduce_indices, t] = tag_result
                tag_cursor[reduce_indices] += 1
                S[reduce_indices, t] = self.word_tree(hl, cl, hr, cr, tag_result.chunk(2, dim=-1)[0])

                queue_cursor[reduce_indices] -= 1
                # add this time step to queue
                Q[reduce_indices, queue_cursor[reduce_indices]] = t

        result = S.select(1, T).chunk(2, dim=-1)[0]
        return result
