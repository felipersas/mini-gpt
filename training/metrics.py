"""Métricas de avaliação: perplexidade, bits por token, memória e o histórico do treino."""

import json
import math
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def perplexity(loss: float) -> float:
    """e^loss: com k opções igualmente prováveis, a loss é ln k e a perplexidade é k."""
    return math.exp(loss)


def bits_per_token(loss: float) -> float:
    """A loss em bits: com logaritmo natural a unidade é o nat, e 1 nat = 1/ln 2 bits."""
    return loss / math.log(2)


def peak_memory_mb() -> float:
    """Pico de memória residente (RSS) do processo até agora, em MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # O macOS informa em bytes; o Linux, em kilobytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def save_history(path: str | Path, records: list[Any]) -> None:
    """Grava a lista de EpochRecord como JSON, um objeto por época."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")


def load_history(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
