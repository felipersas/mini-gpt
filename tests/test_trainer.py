import math

import pytest
import torch

from config import ModelConfig
from data.loader import create_dataloader
from model import GPT
from training.loss import cross_entropy
from training.trainer import (
    configure_optimizer,
    evaluate,
    learning_rate_at,
    train,
    train_step,
)

VOCAB_SIZE = 10
CONFIG = ModelConfig(context_length=8, d_model=32, num_heads=4, num_layers=2, dropout=0.0)
# Sequência previsível: o próximo token é sempre o atual + 1 (mod 10).
IDS = torch.arange(VOCAB_SIZE).repeat(40)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(CONFIG, VOCAB_SIZE)


def make_loader(shuffle: bool = True):
    return create_dataloader(IDS, context_length=8, stride=8, batch_size=4, shuffle=shuffle)


def first_batch():
    return next(iter(make_loader(shuffle=False)))


# --- configure_optimizer ---


def test_weight_decay_only_on_matrices(model):
    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])


def test_every_parameter_is_optimized_exactly_once(model):
    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in optimized}) == len(optimized) == len(list(model.parameters()))


# --- learning_rate_at ---


def lr(step: int, warmup_steps: int = 10) -> float:
    return learning_rate_at(
        step, max_lr=1.0, min_lr=0.1, warmup_steps=warmup_steps, total_steps=110
    )


def test_warmup_grows_linearly_to_max():
    assert lr(0) == pytest.approx(0.1)
    assert lr(4) == pytest.approx(0.5)
    assert lr(9) == pytest.approx(1.0)


def test_cosine_decay_goes_from_max_to_min():
    assert lr(10) == pytest.approx(1.0)
    assert lr(60) == pytest.approx(0.55)  # metade do decaimento: média entre max e min
    assert lr(110) == pytest.approx(0.1)
    assert lr(500) == pytest.approx(0.1)


def test_decay_is_monotonic():
    values = [lr(step) for step in range(10, 111)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_without_warmup_starts_at_max():
    assert lr(0, warmup_steps=0) == pytest.approx(1.0)


# --- train_step ---


def test_train_step_changes_parameters(model):
    optimizer = configure_optimizer(model, learning_rate=1e-2, weight_decay=0.0)
    before = [p.detach().clone() for p in model.parameters()]
    train_step(model, *first_batch(), optimizer, gradient_clip=1.0)
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_gradients_do_not_accumulate_between_steps(model):
    optimizer = configure_optimizer(model, learning_rate=0.0, weight_decay=0.0)  # pesos não mudam
    x, y = first_batch()
    train_step(model, x, y, optimizer, gradient_clip=math.inf)
    first = model.blocks[0].attention.q_proj.weight.grad.clone()
    train_step(model, x, y, optimizer, gradient_clip=math.inf)
    torch.testing.assert_close(model.blocks[0].attention.q_proj.weight.grad, first)


def test_clipping_limits_the_gradient_norm(model):
    optimizer = configure_optimizer(model, learning_rate=0.0, weight_decay=0.0)
    result = train_step(model, *first_batch(), optimizer, gradient_clip=1e-3)
    norm_after = torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
    assert result.grad_norm > 1e-3
    assert norm_after.item() == pytest.approx(1e-3, rel=1e-3)


def test_repeated_steps_reduce_the_loss(model):
    optimizer = configure_optimizer(model, learning_rate=3e-3, weight_decay=0.0)
    x, y = first_batch()
    losses = [train_step(model, x, y, optimizer, gradient_clip=1.0).loss for _ in range(40)]
    assert losses[-1] < 0.5 * losses[0]


# --- evaluate ---


def test_evaluate_matches_manual_average(model):
    loader = make_loader(shuffle=False)
    model.eval()
    with torch.no_grad():
        losses = [cross_entropy(model(x), y).item() for x, y in loader]
    assert evaluate(model, loader) == pytest.approx(sum(losses) / len(losses), rel=1e-6)


def test_evaluate_does_not_touch_parameters_or_mode(model):
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    evaluate(model, make_loader(shuffle=False))
    assert model.training
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert all(p.grad is None for p in model.parameters())


# --- train ---


def test_train_runs_epochs_and_learns(model):
    optimizer = configure_optimizer(model, learning_rate=3e-3, weight_decay=0.0)
    logs: list[str] = []
    loader = make_loader()
    initial = evaluate(model, make_loader(shuffle=False))
    history = train(
        model,
        loader,
        make_loader(shuffle=False),
        optimizer,
        epochs=3,
        max_lr=3e-3,
        min_lr=3e-4,
        warmup_steps=5,
        gradient_clip=1.0,
        log_every=10,
        train_eval_loader=make_loader(shuffle=False),
        log=logs.append,
    )
    assert [r.epoch for r in history] == [1, 2, 3]
    assert history[-1].step == 3 * len(loader)
    assert history[-1].val_loss < 0.5 * initial
    assert sum(line.startswith("== época") for line in logs) == 3
    last = history[-1]
    assert last.train_eval_loss is not None and last.train_eval_loss < 0.5 * initial
    # O último passo é o de índice total − 1: o cosseno ainda não chegou exatamente a min_lr.
    expected_lr = learning_rate_at(
        last.step - 1, max_lr=3e-3, min_lr=3e-4, warmup_steps=5, total_steps=last.step
    )
    assert last.learning_rate == pytest.approx(expected_lr)
    assert last.grad_norm > 0 and last.tokens_per_second > 0 and last.peak_memory_mb > 0


def test_train_without_train_eval_loader_skips_that_measurement(model):
    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    history = train(
        model,
        make_loader(),
        make_loader(shuffle=False),
        optimizer,
        epochs=1,
        max_lr=1e-3,
        min_lr=1e-3,
        warmup_steps=0,
        gradient_clip=1.0,
        log_every=100,
        log=lambda line: None,
    )
    assert history[0].train_eval_loss is None
