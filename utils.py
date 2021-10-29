import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import ipdb


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):  # embedding_dim: 是一个size, 例如[batch_size, len, embedding_dim], 也可以是embedding_dim。。
        super(LayerNorm, self).__init__()
        # 用 parameter 封装，代表模型的参数，作为调节因子
        self.a = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        # 其实就是对最后一维做标准化
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x-mean) / (std+self.eps) + self.b


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, forward_expansion):
        super(FeedForwardLayer, self).__init__()
        self.w1 = nn.Linear(d_model, d_model*forward_expansion)
        self.w2 = nn.Linear(d_model*forward_expansion, d_model)

    def forward(self, x):
        return self.w2((F.relu(self.w1(x))))


class WordEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):  # d_model是embedding的维数，vocab是词表的大小
        super(WordEmbeddings, self).__init__()

        self.lookup = nn.Embedding(vocab, d_model)

    def forward(self, x):
        embedding = self.lookup(x)  # x是经过词表映射之后的one-hot向量
        return embedding


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000): # max_len是每个句子的最大长度
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        x = position * div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe: [max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x

if __name__ == "__main__":
    # word_embedding = WordEmbeddings(256, 1000)
    # x = torch.randint(0, 1000, size=(2, 100))
    # print(x.shape)
    # output = word_embedding(x)
    # print(output.shape)

    position_embedding = PositionEmbedding(256)
    x = torch.randint(0, 1000, size=(2, 100, 256))
    output = position_embedding(x)
    print(output.shape)

