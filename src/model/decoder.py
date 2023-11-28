import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from src.model.modules import MultiHeadSelfAttention, PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        embed_dim,
        feedforward_dim,
        dropout,
        vocab_size,
        max_length
    ):
        super().__init__()
        
        self.word_encoding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(
            max_len=max_length, embed_dim=embed_dim
        )

        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                max_length=max_length,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)


    def forward(self, x):
        x = self.word_encoding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.linear(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, max_length, dropout=0.1):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.max_length = max_length

        self.MultiHeadSelfAttention = MultiHeadSelfAttention(
            d_model=embed_dim,
            n_head=num_heads,
            max_len=max_length,
            d_k=embed_dim // num_heads,
            d_v=embed_dim // num_heads
        )
        self.FFN = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)
        """

        outputs, attention = self.MultiHeadSelfAttention(
            q=x, k=x, v=x, mask=mask
        )

        outputs = self.dropout_1(outputs)
        outputs = self.layer_norm_1(x + outputs)
        
        outputs = self.FFN(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.layer_norm_2(outputs)
        
        if return_attention:
            return outputs, attention
        else:
            return outputs


