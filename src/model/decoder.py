import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from src.model.modules import MultiHeadSelfAttention, PositionalEncoding, RMSNorm


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        embed_dim,
        feedforward_dim,
        attn_dropout,
        ff_dropout,
        use_flash_attention,
        use_rms_norm,
        vocab_size,
        max_length,
        dtype=torch.bfloat16
    ):
        super().__init__()
        
        self.dtype = dtype
        
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
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_flash_attention=use_flash_attention,
                use_rms_norm=use_rms_norm,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            self.layer_norm = RMSNorm(d=embed_dim, dtype=dtype)
        else:
            self.layer_norm = nn.LayerNorm(embed_dim).to(dtype=dtype)

        self.linear = nn.Linear(embed_dim, vocab_size).to(dtype=dtype)

    def forward(self, x):
        x = self.word_encoding(x).to(dtype=self.dtype)
        x = self.positional_encoding(x).to(dtype=self.dtype)

        for layer in self.layers:
            x = layer(x)
        
        x = self.layer_norm(x)
        x = self.linear(x)
        return x.float()


class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_dim,
                 max_length,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 use_flash_attention=True,
                 use_rms_norm=False,
                 dtype=torch.bfloat16):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        self.dtype = dtype

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.max_length = max_length
        self.use_flash_attention = use_flash_attention

        self.MultiHeadSelfAttention = MultiHeadSelfAttention(
            d_model=embed_dim,
            n_head=num_heads,
            max_len=max_length,
            d_k=embed_dim // num_heads,
            d_v=embed_dim // num_heads,
            use_flash_attention=use_flash_attention,
            dtype=dtype
        )
        self.FFN = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        ).to(dtype=dtype)

        if use_rms_norm:
            self.layer_norm_1 = RMSNorm(d=self.embed_dim, dtype=dtype)
            self.layer_norm_2 = RMSNorm(d=self.embed_dim, dtype=dtype)
        else:
            self.layer_norm_1 = nn.LayerNorm(self.embed_dim).to(dtype=dtype)
            self.layer_norm_2 = nn.LayerNorm(self.embed_dim).to(dtype=dtype)

        self.attn_dropout = attn_dropout
        if not self.use_flash_attention:
            self.dropout_1 = nn.Dropout(attn_dropout).to(dtype=dtype)
        self.dropout_2 = nn.Dropout(ff_dropout).to(dtype=dtype)

    def forward(self, x, mask=None):
        """
        Args:
            x: torch.Tensor (B, L, D), in specified dtype
        Returns:
            outputs: torch.Tensor (B, L, D)
        """
        x_norm = self.layer_norm_1(x)
        outputs = x + self.MultiHeadSelfAttention(
            q=x_norm, k=x_norm, v=x_norm, mask=mask,
            attn_dropout=self.attn_dropout
        )
        if not self.use_flash_attention:
            outputs = self.dropout_1(outputs)
        
        outputs = outputs + self.FFN(self.layer_norm_2(outputs))
        outputs = self.dropout_2(outputs)
        
        return outputs
