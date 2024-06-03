import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])  # scale factor for dot-product attention
        scores = torch.einsum("bthd,bshd->bhts", q, k) * softmax_scale
        causal_mask = torch.triu(torch.full((q.size(1), q.size(1)), float('-inf'), device=scores.device), 1)
        scores = scores.masked_fill(causal_mask == float('-inf'), float('-inf'))
        attention = F.softmax(scores, dim=-1)
        attention_drop = F.dropout(attention, p=self.dropout_p, training=self.training)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output

class MHC(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 1, bias: bool = True, dropout: float = 0.0, layer_idx: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        context = self.inner_attn(q, k, v)
        # Apply convolution on the flattened outputs
        context = rearrange(context, "b s h d -> b (h d) s")
        conv_output = self.conv1d(context)
        conv_output = rearrange(conv_output, "b (h d) s -> b s h d", h=self.num_heads)
        # Use conv_output as queries, reuse k and v
        context2 = self.inner_attn(conv_output, k, v)
        out = self.out_proj(rearrange(context2, "b s h d -> b s (h d)"))
        return out

    def state_size(self, batch_size: int = 1, sequence_length: int = 2048):
        return 2 * self.d_model * sequence_length

