import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    positional encoding
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        constructor of positional encoding

        :param d_model: embedding_dimension
        :param max_len: max length of sentence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [5000, 100]
        # print("pe size: {}".format(pe.size()))

        # unsqueeze(idx)，在指定位置上增加维度
        # position: (0,1,...,4999)，指示每个单词的位置
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]
        # print("position size: {}".format(position.size()))

        # (10000)^((2i)/d_model) -> e^(((2i)/d_model) * ln(10000))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [50]
        # print("div_term size: {}".format(div_term.size()))

        # 进行位置编码，与具体的句子无关
        # 0::2, begin from 0 with stride is 2
        pe[:, 0::2] = torch.sin(position * div_term)  # [5000, 100]
        pe[:, 1::2] = torch.cos(position * div_term)  # [5000, 100]
        # print("pe size: {}".format(pe.size()))

        # transpose(0, 1)，修改维度的顺序
        pe = pe.unsqueeze(0).transpose(0, 1)  # [5000, 1, 100]
        # print("revised pe size: {}".format(pe.size()))

        # 该方法的作用是定义一组参数，该组参数的特别之处在于：
        # 模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），、
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    print("------------testing positional encode------------")

    vec = torch.ones(500, 2, 100)  # ve: [seq_len, batch_size, d_model]
    res = vec[:10, :]  # 切片操作
    # print("test for split: {}".format(res.size()))
    print("enc_inputs size: {}".format(vec.size()))

    model = PositionalEncoding(100)
    dst = model.forward(vec)
    print("enc_inputs decoded size: {}".format(dst.size()))
