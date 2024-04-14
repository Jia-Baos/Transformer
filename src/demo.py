import torch
import numpy as np

# subsequence_mask = np.triu(np.ones((4, 4)), k=1)  # Upper triangular matrix
# print(subsequence_mask)


vec1 = torch.Tensor([[1, 2], [3, 0]])
vec2 = torch.Tensor([[1, 1]])
mask = vec1.data.eq(0)

attn_shape = [vec1.size(0), vec1.size(1), vec1.size(1)]
subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # Byte类型张量

if __name__ == "__main__":
    print(subsequence_mask)
