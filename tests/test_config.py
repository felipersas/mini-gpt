from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from config import ModelConfig, config_from_dict, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_tiny_config_loads():
    config = load_config(ROOT / "configs" / "tiny.yaml")

    assert config.model.d_model == 256
    assert config.model.head_dim == 64
    assert config.model.d_ff == 1024
    assert config.model.tie_weights is True
    # "3e-4" no YAML viraria string.
    assert isinstance(config.training.learning_rate, float)
    assert (ROOT / config.data.corpus_path).is_file()
    assert config.training.checkpoint_every > 0


def test_config_round_trips_through_a_dict():
    # É assim que a config vai para dentro de um checkpoint e volta.
    config = load_config(ROOT / "configs" / "tiny.yaml")
    assert config_from_dict(asdict(config)) == config


def test_d_model_must_be_divisible_by_num_heads():
    with pytest.raises(ValueError):
        ModelConfig(context_length=8, d_model=10, num_heads=3, num_layers=1, dropout=0.0)


def test_attention_backend_defaults_to_manual_and_is_validated():
    config = ModelConfig(context_length=8, d_model=8, num_heads=2, num_layers=1, dropout=0.0)
    assert config.attention_backend == "manual"
    with pytest.raises(ValueError):
        ModelConfig(
            context_length=8,
            d_model=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            attention_backend="flash",
        )


def test_mixed_precision_defaults_to_false():
    config = load_config(ROOT / "configs" / "tiny.yaml")
    assert config.training.mixed_precision is False


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "data: {corpus_path: x.txt, val_fraction: 0.1, stride: 8}\n"
        "model: {context_length: 8, d_model: 8, num_heads: 2, num_layers: 1,"
        " dropout: 0.0, typo_key: 1}\n"
        "training: {batch_size: 1, learning_rate: 0.1, epochs: 1,"
        " weight_decay: 0.0, gradient_clip: 1.0, seed: 0}\n"
    )
    with pytest.raises(TypeError):
        load_config(path)


def test_torch_runs_on_cpu():
    x = torch.ones(2, 3)
    assert (x @ x.T).shape == (2, 2)
