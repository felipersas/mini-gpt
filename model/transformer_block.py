"""Transformer Block pre-norm: attention e feed-forward, cada um com LayerNorm e residual."""

import math

import torch
from torch import nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """x = x + MHA(LayerNorm(x)); depois x = x + FFN(LayerNorm(x)).

    [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    `num_layers` é o total de blocos do modelo, usado só na inicialização.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
        attention_backend: str = "manual",
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, backend=attention_backend)
        self.ln_2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # O modelo soma 2 · num_layers contribuições ao residual stream. Encolher as matrizes
        # que escrevem nele por 1/√(2 · num_layers) mantém a variância da soma estável (GPT-2).
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.attention.out_proj.weight, std=residual_std)
        nn.init.normal_(self.feed_forward.down_proj.weight, std=residual_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model], o residual stream
        x = x + self.attention(self.ln_1(x))  # comunicação entre posições
        x = x + self.feed_forward(self.ln_2(x))  # computação em cada posição
        return x  # [batch, seq_len, d_model]
