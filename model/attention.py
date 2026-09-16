"""Scaled dot-product attention causal e multi-head attention."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Máscara [seq_len, seq_len]: True onde a posição i pode ver a posição j (j <= i)."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def attention_weights(
    q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """softmax(Q Kᵀ / √d_k), com peso zero onde `mask` é False.

    q, k: [..., seq_len, d_k]  ->  pesos: [..., seq_len, seq_len]
    A linha i diz como a posição i distribui sua atenção entre as posições j.
    """
    d_k = q.shape[-1]
    # scores[..., i, j] = q_i · k_j
    scores = q @ k.transpose(-2, -1)  # [..., seq_len, seq_len]
    # Sem a escala, a variância de q·k cresce com d_k e a softmax satura.
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        # exp(-inf) = 0: posições proibidas recebem peso zero após a softmax.
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V.

    q, k: [..., seq_len, d_k]; v: [..., seq_len, d_v]
    Retorna (saída [..., seq_len, d_v], pesos [..., seq_len, seq_len]).
    """
    weights = attention_weights(q, k, mask)
    # Cada saída é uma média ponderada dos values.
    return weights @ v, weights


class CausalSelfAttention(nn.Module):
    """Uma head de self-attention causal.

    [batch, seq_len, d_model] -> [batch, seq_len, head_dim]
    """

    def __init__(self, d_model: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        # nn.Linear guarda weight: [head_dim, d_model] e calcula x @ weightᵀ.
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        q = self.q_proj(x)  # [batch, seq_len, head_dim]
        k = self.k_proj(x)  # [batch, seq_len, head_dim]
        v = self.v_proj(x)  # [batch, seq_len, head_dim]

        mask = causal_mask(seq_len, device=x.device)  # [seq_len, seq_len], vale para todo o batch
        weights = attention_weights(q, k, mask)  # [batch, seq_len, seq_len]
        # Dropout nos pesos: corta ligações aleatórias entre posições durante o treino.
        weights = self.dropout(weights)
        return weights @ v  # [batch, seq_len, head_dim]


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim].

    A head i recebe o bloco de features [i * head_dim, (i + 1) * head_dim).
    """
    batch, seq_len, d_model = x.shape
    x = x.view(batch, seq_len, num_heads, d_model // num_heads)  # [batch, seq_len, heads, head_dim]
    # Heads antes de seq_len: a attention trata [batch, heads] como dimensões de lote.
    return x.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]: concatena as heads."""
    batch, num_heads, seq_len, head_dim = x.shape
    # transpose só troca os strides; view exige memória contígua, então copiamos antes.
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)


def _check_heads(d_model: int, num_heads: int) -> None:
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) precisa ser divisível por num_heads ({num_heads})")


class LoopMultiHeadAttention(nn.Module):
    """Multi-head attention como uma lista de heads independentes (versão didática).

    Calcula o mesmo que `MultiHeadAttention`, uma head por vez.
    [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        _check_heads(d_model, num_heads)
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            CausalSelfAttention(d_model, head_dim, dropout) for _ in range(num_heads)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]; cada head devolve [batch, seq_len, head_dim]
        heads = [head(x) for head in self.heads]
        concatenated = torch.cat(heads, dim=-1)  # [batch, seq_len, num_heads * head_dim]
        return self.dropout(self.out_proj(concatenated))  # [batch, seq_len, d_model]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention causal, com todas as heads num único tensor.

    [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float, backend: str = "manual"
    ) -> None:
        super().__init__()
        _check_heads(d_model, num_heads)
        if backend not in ("manual", "sdpa"):
            raise ValueError(f"backend inválido: {backend!r} (use 'manual' ou 'sdpa')")
        self.num_heads = num_heads
        self.backend = backend
        # Uma matriz [d_model, d_model] = num_heads matrizes [head_dim, d_model] empilhadas.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        q = split_heads(self.q_proj(x), self.num_heads)  # [batch, heads, seq_len, head_dim]
        k = split_heads(self.k_proj(x), self.num_heads)  # [batch, heads, seq_len, head_dim]
        v = split_heads(self.v_proj(x), self.num_heads)  # [batch, heads, seq_len, head_dim]

        if self.backend == "sdpa":
            # Mesma matemática de attention_weights, calculada pelo kernel do PyTorch: nunca
            # materializa a matriz [T, T] de scores, e escolhe o kernel mais rápido do device.
            drop = self.dropout.p if self.training else 0.0
            heads = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            mask = causal_mask(seq_len, device=x.device)  # [seq_len, seq_len], p/ batch e heads
            weights = attention_weights(q, k, mask)  # [batch, heads, seq_len, seq_len]
            weights = self.dropout(weights)
            heads = weights @ v  # [batch, heads, seq_len, head_dim]

        out = self.out_proj(merge_heads(heads))  # [batch, seq_len, d_model]
        # Dropout também no que será somado ao residual stream, como no feed-forward.
        return self.dropout(out)
