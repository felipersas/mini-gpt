"""LayerNorm: normaliza cada vetor usando as suas próprias features."""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """y = γ · (x − média) / √(variância + eps) + β, calculado na última dimensão.

    Equivalente a nn.LayerNorm(d_model); os parâmetros têm os mesmos nomes (weight = γ, bias = β).
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # γ: começa sem alterar a escala
        self.bias = nn.Parameter(torch.zeros(d_model))  # β: começa sem deslocar

    def extra_repr(self) -> str:
        # Aparece em print(model), como no nn.LayerNorm.
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]; as estatísticas são de cada vetor, nunca do batch ou da sequência.
        mean = x.mean(dim=-1, keepdim=True)  # [..., 1]
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)  # [..., 1], divide por d_model
        # eps evita divisão por zero quando todas as features são iguais.
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # média 0, variância ~1
        return self.weight * x_hat + self.bias  # [..., d_model]
