import math
import torch
import torch.nn.functional as F

from torch import nn


class ScaledDotProductSelfAttention(nn.Module):
    ''' Scaled Dot-Product Self Attention '''

    def __init__(self, max_len, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, q, k, v, mask=None):
        # q, k, v: [ (batch_size * n_heads) x seq_len x hidden_size ]

        d = math.sqrt(q.shape[-1])
        attn = F.softmax(q @ k.transpose(-2, -1) / d, dim=-1)
        # attn: [ (batch_size * n_heads) x seq_len x seq_len ]
        
        seq_len = q.shape[1]
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, 0.)

        output = attn @ v
        # output: [ (batch_size * n_heads) x seq_len x hidden_size ]
        return output


class MultiHeadSelfAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, max_len, n_head, d_model, d_k, d_v, use_flash_attention=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            self.attention = F.scaled_dot_product_attention
        else:
            self.attention = ScaledDotProductSelfAttention(
                max_len=max_len,
                temperature=d_k**0.5
            )

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        
        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=math.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=math.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=math.sqrt(2.0 / (self.d_model + self.d_v))) 
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        
        if self.use_flash_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=False, 
                enable_mem_efficient=False
            ):
                output = self.attention(q, k, v, attn_mask=mask)
        output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        
        return output
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        freq_indices = torch.arange(embed_dim, dtype=torch.float32)
        freq_indices[1::2] -= 1
        frequencies = torch.exp(-math.log(10000) * freq_indices / embed_dim).unsqueeze(0)

        pos_encodings = positions @ frequencies
        pos_encodings[:, 0::2] = torch.sin(pos_encodings[:, 0::2])
        pos_encodings[:, 1::2] = torch.cos(pos_encodings[:, 1::2])

        # we need 1 x MaxLength x EmbedDim
        self.pos_encodings = pos_encodings.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', self.pos_encodings, persistent=False)

    def forward(self, x):
        # assert x.shape[-1] == self.pos_encodings.shape[-1], "Embedding dimension is not the same"
        return x + self.pe[:, :x.shape[1], :]
