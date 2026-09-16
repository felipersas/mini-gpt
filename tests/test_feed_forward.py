import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from model.feed_forward import FeedForward

D_MODEL, D_FF = 16, 64


@pytest.fixture
def ff() -> FeedForward:
    torch.manual_seed(0)
    return FeedForward(D_MODEL, D_FF, dropout=0.0)


@pytest.fixture
def ff_random_biases(ff) -> FeedForward:
    # Na inicialização os biases são zero; aqui eles precisam aparecer nas contas.
    with torch.no_grad():
        for param in ff.parameters():
            nn.init.normal_(param)
    return ff


# --- forma e fórmula ---


def test_output_shape(ff):
    assert ff(torch.randn(3, 10, D_MODEL)).shape == (3, 10, D_MODEL)


def test_hidden_layer_expands_to_d_ff(ff):
    assert ff.up_proj(torch.randn(3, 10, D_MODEL)).shape == (3, 10, D_FF)


def test_matches_manual_formula(ff_random_biases):
    ff = ff_random_biases
    x = torch.randn(2, 5, D_MODEL)
    hidden = F.gelu(x @ ff.up_proj.weight.T + ff.up_proj.bias)  # [2, 5, d_ff]
    expected = hidden @ ff.down_proj.weight.T + ff.down_proj.bias  # [2, 5, d_model]
    torch.testing.assert_close(ff(x), expected)


def test_output_is_a_sum_of_neurons(ff_random_biases):
    # ff(x) = Σ_i GELU(k_i · x + b_i) · v_i + b_2, com k_i = linha i de W1 e v_i = coluna i de W2.
    ff = ff_random_biases
    x = torch.randn(D_MODEL)
    activations = F.gelu(ff.up_proj.weight @ x + ff.up_proj.bias)  # [d_ff]
    total = ff.down_proj.bias + sum(activations[i] * ff.down_proj.weight[:, i] for i in range(D_FF))
    torch.testing.assert_close(ff(x), total, rtol=1e-4, atol=1e-4)


def test_without_activation_two_layers_collapse_into_one(ff_random_biases):
    ff = ff_random_biases
    x = torch.randn(4, D_MODEL)
    W = ff.down_proj.weight @ ff.up_proj.weight  # [d_model, d_model]
    b = ff.down_proj.weight @ ff.up_proj.bias + ff.down_proj.bias  # [d_model]
    torch.testing.assert_close(ff.down_proj(ff.up_proj(x)), x @ W.T + b, rtol=1e-4, atol=1e-4)


# --- GELU ---


def test_gelu_matches_formula():
    x = torch.linspace(-5, 5, 1001)
    phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # CDF da normal padrão
    torch.testing.assert_close(F.gelu(x), x * phi)


def test_gelu_shape():
    assert F.gelu(torch.tensor(0.0)) == 0
    torch.testing.assert_close(F.gelu(torch.tensor([8.0, -8.0])), torch.tensor([8.0, 0.0]))
    grid = torch.linspace(-3, 3, 60001)
    values = F.gelu(grid)
    assert values.min().item() == pytest.approx(-0.170, abs=1e-3)
    assert grid[values.argmin()].item() == pytest.approx(-0.752, abs=1e-2)


# --- processamento por posição ---


def test_each_position_is_processed_independently(ff):
    x = torch.randn(1, 8, D_MODEL)
    changed = x.clone()
    changed[:, 5] = torch.randn(D_MODEL)
    diff = (ff(x) - ff(changed)).abs().amax(dim=-1)[0]
    assert diff.nonzero().flatten().tolist() == [5]


def test_same_vector_gives_same_output_at_any_position(ff):
    x = torch.randn(1, 8, D_MODEL)
    x[:, 6] = x[:, 1]
    out = ff(x)
    torch.testing.assert_close(out[:, 6], out[:, 1])


def test_permuting_positions_permutes_output(ff):
    x = torch.randn(2, 7, D_MODEL)
    perm = torch.randperm(7)
    torch.testing.assert_close(ff(x[:, perm]), ff(x)[:, perm])


def test_gradient_only_reaches_the_same_position(ff):
    x = torch.randn(1, 6, D_MODEL, requires_grad=True)
    ff(x)[0, 2].sum().backward()
    grad_norms = x.grad[0].norm(dim=-1)
    assert grad_norms.nonzero().flatten().tolist() == [2]


# --- parâmetros, inicialização, treino ---


def test_parameter_count(ff):
    n_params = sum(p.numel() for p in ff.parameters())
    assert n_params == D_MODEL * D_FF + D_FF + D_FF * D_MODEL + D_MODEL


def test_initialization():
    ff = FeedForward(256, 1024, dropout=0.0)
    for proj in (ff.up_proj, ff.down_proj):
        assert proj.weight.std().item() == pytest.approx(0.02, rel=0.05)
        assert (proj.bias == 0).all()


def test_gradients_reach_all_parameters(ff):
    ff(torch.randn(2, 5, D_MODEL)).sum().backward()
    for name, param in ff.named_parameters():
        assert param.grad.abs().sum() > 0, name


def test_dropout_only_in_train_mode():
    ff = FeedForward(D_MODEL, D_FF, dropout=0.5)
    x = torch.randn(4, 8, D_MODEL)
    ff.eval()
    torch.testing.assert_close(ff(x), ff(x))
    ff.train()
    assert (ff(x) == 0).any()
