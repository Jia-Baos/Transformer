import torch
import torch.utils.data as Data

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc -> encoder
    # dec -> decoder
    # enc_input           dec_input         dec_output
    ["ich mochte ein bier P", "S i want a beer .", "E i want a beer ."],
    ["ich mochte ein cola P", "S i want a coke .", "E i want a coke ."],
]

# Padding Should be Zero
src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4, "cola": 5}
src_vocab_size = len(src_vocab)

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

idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length


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
        ]  # [[7, 2, 3, 4, 8, 1], [7, 2, 3, 5, 8, 1]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    # 包含整型数据的张量
    return (
        torch.LongTensor(enc_inputs),
        torch.LongTensor(dec_inputs),
        torch.LongTensor(dec_outputs),
    )


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


if __name__ == "__main__":
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    print(enc_inputs)
    print(dec_inputs)
    print(dec_outputs)
