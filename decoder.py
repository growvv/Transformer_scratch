import torch
import torch.nn as nn
from multi_head_attn import MultihHeadAttention
from utils import  LayerNorm, FeedForwardLayer
from encoder import TransformerBlock
import ipdb

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.norm = LayerNorm(embed_size)
        self.attn = MultihHeadAttention(embed_size, heads, dropout)
        self.transformer = TransformerBlock(embed_size, heads, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attn = self.attn(x, x, x, trg_mask)
        query = self.dropout(self.norm(attn+x))
        out = self.transformer(query, value, key, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
            
        )

    def forward(self, x, encoder_out, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)

        return x

    
if __name__ == "__main__":
    decoder = Decoder(embed_size=256, num_layers=8, heads=8, forward_expansion=4)
    x = torch.randn(2, 100, 256)
    encoder_out = torch.randn(2, 100, 256)
    src_mask = torch.tril(torch.ones((2, 100, 100)))
    trg_mask = torch.tril(torch.zeros((2, 1, 100)))
    output = decoder(x, encoder_out, src_mask, trg_mask)
    print(output.shape)


