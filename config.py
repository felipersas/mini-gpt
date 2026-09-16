"""Configuração de dados, modelo e treinamento, carregada de um arquivo YAML."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    corpus_path: str  # relativo à raiz do projeto
    val_fraction: float
    stride: int


@dataclass
class ModelConfig:
    """Hiperparâmetros da arquitetura. O vocab_size vem do tokenizer."""

    context_length: int
    d_model: int
    num_heads: int
    num_layers: int
    dropout: float
    tie_weights: bool = True  # o LM head reusa a matriz do token embedding
    # "manual" materializa a matriz [T, T] de scores; "sdpa" usa o kernel do PyTorch
    # (torch.nn.functional.scaled_dot_product_attention), mais rápido em CUDA e MPS.
    attention_backend: str = "manual"

    def __post_init__(self) -> None:
        # Cada head recebe uma fatia igual do embedding.
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) precisa ser divisível por num_heads ({self.num_heads})"
            )
        if self.attention_backend not in ("manual", "sdpa"):
            raise ValueError(
                f"attention_backend inválido: {self.attention_backend!r} (use 'manual' ou 'sdpa')"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def d_ff(self) -> int:
        # Dimensão interna do feed-forward, como no GPT: 4 × d_model.
        return 4 * self.d_model


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float  # valor máximo, atingido ao fim do warmup
    min_learning_rate: float  # valor final do decaimento cosseno
    warmup_steps: int
    epochs: int
    weight_decay: float
    gradient_clip: float
    seed: int
    log_every: int  # passos entre linhas de log
    checkpoint_every: int  # passos entre checkpoints periódicos (checkpoint_<passo>.pt); 0 desliga
    # forward e loss em bfloat16 num acelerador; pesos e gradientes continuam em float32.
    mixed_precision: bool = False


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def config_from_dict(raw: dict) -> Config:
    """Monta a Config; chaves ausentes ou desconhecidas geram TypeError."""
    return Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )


def load_config(path: str | Path) -> Config:
    """Lê o YAML e monta a Config."""
    with open(path, encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f))
