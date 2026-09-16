"""Geração autoregressiva: greedy, temperature, top-k e top-p."""

import torch

from device import model_device
from model.gpt import GPT


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mantém os k maiores logits; os demais viram −inf (probabilidade zero)."""
    if k >= logits.shape[-1]:
        return logits
    kth_largest = torch.topk(logits, k).values[..., -1, None]  # [..., 1]
    return logits.masked_fill(logits < kth_largest, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Mantém o menor conjunto de tokens mais prováveis cuja probabilidade somada chega a p."""
    if not 0 < p <= 1:
        raise ValueError(f"top_p deve estar em (0, 1], recebido {p}")
    if p == 1:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    # Probabilidade acumulada ANTES de cada token: se já chegou a p, o token sai.
    # O mais provável tem acumulado 0 e sempre fica.
    remove_sorted = (torch.cumsum(probs, dim=-1) - probs) >= p
    remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)  # ordem original
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Escolhe o próximo token a partir dos logits de uma posição, [vocab_size]."""
    if temperature == 0:
        return int(logits.argmax())  # greedy: sempre o mais provável
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Gera até `max_new_tokens` tokens depois do prompt, um de cada vez.

    Devolve o prompt seguido dos tokens gerados. Para antes se gerar `eos_id`.
    """
    if not prompt_ids:
        raise ValueError("o prompt precisa ter pelo menos um token")
    was_training = model.training
    model.eval()
    ids = list(prompt_ids)
    context_length = model.config.context_length
    device = model_device(model)

    for _ in range(max_new_tokens):
        # O modelo aceita no máximo context_length posições: ficam as mais recentes.
        context = torch.tensor([ids[-context_length:]], device=device)  # [1, T]
        # O sorteio acontece na CPU: um torch.Generator de CPU serve para qualquer device.
        logits = model(context)[0, -1].cpu()  # [vocab_size]: previsão para a posição seguinte
        next_id = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        ids.append(next_id)
        if next_id == eos_id:
            break

    model.train(was_training)
    return ids
