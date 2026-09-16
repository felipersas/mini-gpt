import copy

import pytest
import torch

from config import ModelConfig
from data.loader import create_dataloader
from device import autocast, get_device_rng_state, model_device, select_device, set_device_rng_state
from generation.generate import generate
from model import GPT
from training.trainer import configure_optimizer, evaluate, train_step

AVAILABLE = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]
ACCELERATORS = [name for name, available in AVAILABLE if available]
needs_accelerator = pytest.mark.skipif(not ACCELERATORS, reason="nenhum acelerador neste ambiente")

VOCAB_SIZE = 10
CONFIG = ModelConfig(context_length=8, d_model=16, num_heads=2, num_layers=2, dropout=0.0)


def make_loader():
    ids = torch.arange(VOCAB_SIZE).repeat(8)  # 9 janelas: 2 batches de 4
    return create_dataloader(ids, context_length=8, stride=8, batch_size=4, shuffle=False)


def cpu_and_copy_on(name):
    torch.manual_seed(0)
    cpu_model = GPT(CONFIG, VOCAB_SIZE)
    return cpu_model, copy.deepcopy(cpu_model).to(name)


# --- escolha do device ---


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"


def test_unavailable_device_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    for name in ("cuda", "mps"):
        with pytest.raises(ValueError):
            select_device(name)


def test_cpu_is_always_available_and_has_no_separate_rng_state():
    device = select_device("cpu")
    assert device == torch.device("cpu")
    assert get_device_rng_state(device) is None


def test_model_device_follows_the_parameters():
    torch.manual_seed(0)
    assert model_device(GPT(CONFIG, VOCAB_SIZE)).type == "cpu"


# --- o mesmo trabalho num acelerador ---


@needs_accelerator
@pytest.mark.parametrize("name", ACCELERATORS)
def test_train_step_on_accelerator_matches_cpu(name):
    cpu_model, model = cpu_and_copy_on(name)
    x, y = next(iter(make_loader()))  # na CPU: train_step leva o batch ao device do modelo
    results = [
        train_step(m, x, y, configure_optimizer(m, 1e-3, 0.0), gradient_clip=1.0)
        for m in (cpu_model, model)
    ]
    assert results[1].loss == pytest.approx(results[0].loss, rel=1e-5)
    for cpu_param, param in zip(cpu_model.parameters(), model.parameters(), strict=True):
        assert param.device.type == name
        torch.testing.assert_close(param.grad.cpu(), cpu_param.grad, rtol=1e-4, atol=1e-6)


@needs_accelerator
@pytest.mark.parametrize("name", ACCELERATORS)
def test_evaluate_on_accelerator_matches_cpu(name):
    cpu_model, model = cpu_and_copy_on(name)
    assert evaluate(model, make_loader()) == pytest.approx(
        evaluate(cpu_model, make_loader()), rel=1e-5
    )


@needs_accelerator
@pytest.mark.parametrize("name", ACCELERATORS)
def test_greedy_generation_on_accelerator_matches_cpu(name):
    cpu_model, model = cpu_and_copy_on(name)
    prompt = [1, 2, 3]
    assert generate(model, prompt, 20, temperature=0) == generate(
        cpu_model, prompt, 20, temperature=0
    )


@needs_accelerator
@pytest.mark.parametrize("name", ACCELERATORS)
def test_accelerator_rng_state_reproduces_dropout(name):
    device = torch.device(name)
    torch.manual_seed(0)
    model = GPT(ModelConfig(8, 16, 2, 2, dropout=0.5), VOCAB_SIZE).to(device).train()
    x = next(iter(make_loader()))[0].to(device)
    state = get_device_rng_state(device)
    first = model(x)
    assert not torch.equal(model(x), first)  # outro sorteio de dropout
    set_device_rng_state(device, state)
    assert torch.equal(model(x), first)  # o mesmo sorteio de novo


# --- mixed precision ---


def test_autocast_disabled_keeps_float32():
    with autocast(torch.device("cpu"), False):
        out = torch.randn(2, 2) @ torch.randn(2, 2)
    assert out.dtype == torch.float32


def test_autocast_enabled_casts_matmul_to_bfloat16():
    with autocast(torch.device("cpu"), True):
        out = torch.randn(2, 2) @ torch.randn(2, 2)
    assert out.dtype == torch.bfloat16


def test_mixed_precision_train_step_keeps_weights_and_grads_in_float32():
    torch.manual_seed(0)
    model = GPT(CONFIG, VOCAB_SIZE)
    x, y = next(iter(make_loader()))
    optimizer = configure_optimizer(model, 1e-3, 0.0)
    train_step(model, x, y, optimizer, gradient_clip=1.0, mixed_precision=True)
    for p in model.parameters():
        assert p.dtype == torch.float32
        assert torch.isfinite(p.grad).all()


@needs_accelerator
@pytest.mark.parametrize("name", ACCELERATORS)
def test_mixed_precision_loss_close_to_full_precision(name):
    cpu_model, model = cpu_and_copy_on(name)
    x, y = next(iter(make_loader()))
    full = train_step(cpu_model, x, y, configure_optimizer(cpu_model, 1e-3, 0.0), gradient_clip=1.0)
    mixed = train_step(
        model, x, y, configure_optimizer(model, 1e-3, 0.0), gradient_clip=1.0, mixed_precision=True
    )
    assert mixed.loss == pytest.approx(full.loss, abs=0.05)
