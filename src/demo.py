import math
import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """ """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 5000*100
        print("size: {}".format(pe.size()))

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 5000*1
        print("size: {}".format(position.size()))

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model)^((2i)/d_model) -> e^(((2i)/d_model) * ln(d_model)), 50
        print("size: {}".format(div_term.size()))

        # begin from 0 with stride is 2
        pe[:, 0::2] = torch.sin(position * div_term)  # 5000*100
        pe[:, 1::2] = torch.cos(position * div_term)  # 5000*100
        pe = pe.unsqueeze(0).transpose(0, 1)  # 5000*1*100
        print("size: {}".format(pe.size()))

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    seq_q = [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
    seq_q_tensor = torch.LongTensor(seq_q)

    vec = torch.ones(500, 2, 100)
    test = vec[:10, :]
    print(test.size())

    model = PositionalEncoding(100)
    res = model.forward(vec)

    print(res.size())
