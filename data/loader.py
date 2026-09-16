"""Divisão treino/validação e criação dos DataLoaders."""

import torch
from torch.utils.data import DataLoader

from data.dataset import NextTokenDataset


def split_train_val(ids: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide a sequência em dois trechos contíguos: início para treino, final para validação."""
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction deve estar em (0, 1), recebido {val_fraction}")
    # Corte por posição, não aleatório: janelas sorteadas dos dois lados compartilhariam tokens.
    split = len(ids) - int(len(ids) * val_fraction)
    return ids[:split], ids[split:]


def create_dataloader(
    ids: torch.Tensor,
    *,
    context_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """DataLoader de batches (x, y), ambos com shape [batch_size, context_length]."""
    dataset = NextTokenDataset(ids, context_length, stride)
    # Com drop_last, um dataset menor que o batch geraria zero batches sem nenhum erro.
    if len(dataset) < batch_size:
        raise ValueError(f"{len(dataset)} janelas não completam um batch de {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # todo batch com o mesmo shape
        generator=torch.Generator().manual_seed(seed),
    )
