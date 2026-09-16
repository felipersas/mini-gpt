"""Dataset de next-token prediction sobre uma sequência longa de IDs."""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """Recorta a sequência em janelas (x, y), com y deslocado uma posição à frente.

    Para a janela i, com início s = i * stride e T = context_length:

        x = ids[s     : s + T]
        y = ids[s + 1 : s + T + 1]

    Assim, y[t] é o token que segue x[0..t].
    """

    def __init__(self, ids: Sequence[int] | torch.Tensor, context_length: int, stride: int) -> None:
        ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError(f"ids deve ser 1-D, recebido shape {tuple(ids.shape)}")
        if context_length < 1 or stride < 1:
            raise ValueError("context_length e stride devem ser >= 1")
        # Uma janela consome T + 1 tokens: o último só aparece em y.
        if len(ids) < context_length + 1:
            raise ValueError(
                f"{len(ids)} tokens não formam nenhuma janela com context_length={context_length}"
            )
        self.ids = ids
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        # Último início válido: len(ids) - (T + 1).
        return (len(self.ids) - self.context_length - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"janela {index} fora do intervalo (0..{len(self) - 1})")
        start = index * self.stride
        # window: [T + 1]; x, y: [T]
        window = self.ids[start : start + self.context_length + 1]
        return window[:-1], window[1:]
