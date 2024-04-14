import torch
import torch.utils.data as Data

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length

# enc -> encoder
# dec -> decoder
# S: Symbol that shows starting of decoding input
# E: Symbol that shows ending of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
# enc_input           dec_input         dec_output
sentences = [
    ["ich mochte ein bier P", "S i want a beer .", "i want a beer . E"],
    ["ich mochte ein cola P", "S i want a coke .", "i want a coke . E"],
]

# Padding Should be Zero
src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4, "cola": 5}

tgt_vocab = {
    "P": 0,
    "i": 1,
    "want": 2,
    "a": 3,
    "beer": 4,
    "coke": 5,
    "S": 6,
    "E": 7,
    ".": 8,
}

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)


idx2word = {i: w for i, w in enumerate(tgt_vocab)}


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [
            [src_vocab[n] for n in sentences[i][0].split()]
        ]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [
            [tgt_vocab[n] for n in sentences[i][1].split()]
        ]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [
            [tgt_vocab[n] for n in sentences[i][2].split()]
        ]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        # Extend list by appending elements from the iterable.
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    # 包含整型数据的张量
    return (
        torch.LongTensor(enc_inputs),
        torch.LongTensor(dec_inputs),
        torch.LongTensor(dec_outputs),
    )


# 将语言序列进行编码
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        # 返回数据集中语言序列对的数目（数据量）
        return self.enc_inputs.size(0)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


if __name__ == "__main__":
    # enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    # print(enc_inputs)
    # print(dec_inputs)
    # print(dec_outputs)print

    print("------------testing dataloader------------")
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 1, True)
    for enc_inputs, dec_inputs, dec_outputs in loader:
        print("------------epoch------------")
        print(enc_inputs)
        print(dec_inputs)
        print(dec_outputs)
