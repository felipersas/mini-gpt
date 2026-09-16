"""Checkpoints: o modelo treinado e, se pedido, todo o estado necessário para continuar o treino."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from config import Config, ModelConfig, config_from_dict
from model.gpt import GPT
from tokenizer import CharTokenizer, Vocab
from training.trainer import EpochRecord, TrainingProgress


def save_checkpoint(
    path: str | Path,
    model: GPT,
    tokenizer: CharTokenizer,
    *,
    config: Config | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    progress: TrainingProgress | None = None,
) -> None:
    """Grava o modelo e, se fornecidos, a config, o otimizador e o progresso do treino.

    Só com o modelo, o arquivo serve para inferência. Com os três extras, também para retomar.
    """
    extras = (config, optimizer, progress)
    if any(extra is None for extra in extras) and any(extra is not None for extra in extras):
        raise ValueError("para retomar o treino, config, optimizer e progress precisam vir juntos")

    checkpoint: dict[str, Any] = {
        "model_config": asdict(model.config),  # hiperparâmetros da arquitetura
        "vocab": tokenizer.vocab.id_to_token,  # a ordem dos tokens define os IDs
        "model_state": model.state_dict(),  # todos os pesos, por nome
    }
    if progress is not None:
        checkpoint["config"] = asdict(config)  # dados, modelo e treino
        checkpoint["optimizer_state"] = optimizer.state_dict()  # m e v do AdamW, por parâmetro
        checkpoint["progress"] = asdict(progress)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Grava num arquivo temporário e só então troca o nome: uma interrupção durante a escrita
    # nunca deixa um checkpoint pela metade no lugar do anterior.
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    # weights_only=True: só aceita tensores e tipos simples, sem executar código do arquivo.
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[GPT, CharTokenizer]:
    tokenizer = CharTokenizer(Vocab(checkpoint["vocab"]))
    model = GPT(ModelConfig(**checkpoint["model_config"]), vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def load_checkpoint(path: str | Path) -> tuple[GPT, CharTokenizer]:
    """Recria o tokenizer e o modelo, carrega os pesos e deixa o modelo em modo de inferência."""
    return model_from_checkpoint(read_checkpoint(path))


@dataclass
class TrainingCheckpoint:
    config: Config
    model: GPT
    tokenizer: CharTokenizer
    optimizer_state: dict[str, Any]
    progress: TrainingProgress


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    """Tudo o que o treino precisa para continuar: config, modelo, otimizador e progresso."""
    checkpoint = read_checkpoint(path)
    if "progress" not in checkpoint:
        raise ValueError(
            f"{path} só tem os pesos: serve para inferência, não para retomar o treino"
        )
    model, tokenizer = model_from_checkpoint(checkpoint)
    progress = dict(checkpoint["progress"])
    progress["history"] = [EpochRecord(**record) for record in progress["history"]]
    return TrainingCheckpoint(
        config=config_from_dict(checkpoint["config"]),
        model=model,
        tokenizer=tokenizer,
        optimizer_state=checkpoint["optimizer_state"],
        progress=TrainingProgress(**progress),
    )
