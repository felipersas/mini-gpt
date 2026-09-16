"""Embeddings de entrada: vetor do token somado ao vetor da sua posição."""

import torch
from torch import nn


class Embeddings(nn.Module):
    """IDs [batch, seq_len] -> vetores [batch, seq_len, d_model].

    Positional embeddings aprendidos, como no GPT-2: uma linha por posição,
    de 0 a context_length - 1.
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.context_length = context_length
        # weight: [vocab_size, d_model]
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # weight: [context_length, d_model]
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # N(0, 0.02²) como no GPT-2; o padrão do PyTorch, N(0, 1), é grande demais.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [batch, seq_len]
        seq_len = ids.shape[1]
        if seq_len > self.context_length:
            raise ValueError(f"seq_len={seq_len} excede context_length={self.context_length}")

        positions = torch.arange(seq_len, device=ids.device)  # [seq_len]
        tok = self.token_embedding(ids)  # [batch, seq_len, d_model]
        pos = self.position_embedding(positions)  # [seq_len, d_model]

        # Broadcasting: o mesmo vetor de posição é somado em todas as sequências do batch.
        return self.dropout(tok + pos)  # [batch, seq_len, d_model]
