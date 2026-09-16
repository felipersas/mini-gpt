"""Cross-entropy: a loss de next-token prediction."""

import torch


def token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """−log p(token correto) em cada posição, sem tirar a média.

    logits:  [..., vocab_size], por exemplo [batch, seq_len, vocab_size]
    targets: [...] com os IDs corretos, por exemplo [batch, seq_len]
    Devolve um tensor com o shape de targets.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} e targets {tuple(targets.shape)} não combinam"
        )
    # log p_i = z_i − log Σ_j exp(z_j); logsumexp calcula o segundo termo sem overflow.
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # [..., vocab_size]
    # Em cada posição, fica só o log p do token correto.
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [...]


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Média de −log p(token correto) sobre todas as posições: um escalar."""
    return token_losses(logits, targets).mean()
