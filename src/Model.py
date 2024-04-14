import torch
import numpy as np
import torch.nn as nn
from DataSet import *
from PositionalEncoding import *

# Transformer Parameters

d_model = (
    512  # Embedding Size, the number of expected features in the encoder/decoder inputs
)
d_ff = 2048  # FeedForward dimension, the dimension of of the feedforward network model
d_k = 64  # dimension of K(=Q)
d_v = 64  # dimension of V
n_layers = 6  # number of Encoder and Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


def get_attn_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]

    seq_len could be src_len or it could be tgt_len, and the seq_len in seq_q and seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # unsqueeze，在指定 dim 上增加张量维度
    # eq(0), elements equal to 0 were set to True
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(
        1
    )  # pad_attn_mask: [batch_size, 1, len_k]

    # expand()
    # Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
    return pad_attn_mask.expand(
        batch_size, len_q, len_k
    )  # pad_attn_mask: [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    """
    :param seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # Byte类型张量
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """

        # MatMul and Scale
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # Mask(opt.)
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        # SoftMax
        attn = nn.Softmax(dim=-1)(scores)

        # MatMul
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """

        residual, batch_size = input_Q, input_Q.size(0)

        # B: batch_size, S: seq_len, D: d_model, H: n_heads, W: d_k
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # repeat(num), 在指定维度上进行重复
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v],
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # Concat
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        # Linear
        # output: [batch_size, len_q, d_model]
        output = self.fc(context)

        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs

        # FeedForward
        # output: [batch_size, seq_len, d_model]
        output = self.fc(inputs)

        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model],
        # attn: [batch_size, n_heads, src_len, src_len]
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )

        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 词嵌入：[输入词库元素数目，嵌入维度]
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码：[嵌入维度]
        self.pos_emb = PositionalEncoding(d_model)
        # 多层encoder
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """

        # [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # enc_inputs的自注意力机制矩阵
        # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model],
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        )

        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """

        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)

        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)

        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # gt(), a boolean tensor that is True where input is greater than other and False elsewhere
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0
        )

        # 互注意力矩阵
        # dec_enc_attn_mask: [batc_size, tgt_len, src_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model];
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        """

        # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outpus: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs
        )

        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)

        return (
            dec_logits.view(-1, dec_logits.size(-1)),
            enc_self_attns,
            dec_self_attns,
            dec_enc_attns,
        )


if __name__ == "__main__":
    print("------------testing model------------")

    model = Transformer()
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    # enc_inputs: [batch_size, src_len]
    # dec_inputs: [batch_size, tgt_len]
    # dec_outputs: [batch_size, tgt_len]
    for enc_inputs, dec_inputs, dec_outputs in loader:
        print("------------epoch------------")
        print("enc_inputs size: {}".format(enc_inputs.size()))
        print("dec_inputs size: {}".format(dec_inputs.size()))
        print("dec_outputs size: {}".format(dec_outputs.size()))

        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            enc_inputs, dec_inputs
        )
        print("outputs size: {}".format(outputs.size()))
