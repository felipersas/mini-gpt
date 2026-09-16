"""Feed-forward: rede de duas camadas aplicada a cada posição separadamente."""

import torch
from torch import nn


class FeedForward(nn.Module):
    """Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> dropout.

    [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # weight: [d_ff, d_model]; linha i = padrão que ativa o neurônio i
        self.up_proj = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        # weight: [d_model, d_ff]; coluna i = vetor que o neurônio i escreve na saída
        self.down_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        for proj in (self.up_proj, self.down_proj):
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]; nn.Linear atua só na última dimensão,
        # então cada posição passa pela mesma rede, sem ver as outras.
        hidden = self.activation(self.up_proj(x))  # [batch, seq_len, d_ff]
        return self.dropout(self.down_proj(hidden))  # [batch, seq_len, d_model]
