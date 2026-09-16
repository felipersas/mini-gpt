"""Language Model Head: estado final de cada posição -> logits sobre o vocabulário."""

import torch
from torch import nn


class LMHead(nn.Module):
    """[batch, seq_len, d_model] -> [batch, seq_len, vocab_size].

    logits[..., i] = h · W[i]: um score por token; a softmax transforma em probabilidades.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # weight: [vocab_size, d_model]; linha i = direção que favorece o token i
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq_len, d_model]
        return self.proj(h)  # [batch, seq_len, vocab_size]
