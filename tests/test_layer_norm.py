import pytest
import torch
from torch import nn

from model.layer_norm import LayerNorm

D = 16


@pytest.fixture
def ln() -> LayerNorm:
    torch.manual_seed(0)
    module = LayerNorm(D)
    with torch.no_grad():
        module.weight.normal_()
        module.bias.normal_()
    return module


def test_matches_pytorch_layer_norm(ln):
    reference = nn.LayerNorm(D, eps=1e-5)
    with torch.no_grad():
        reference.weight.copy_(ln.weight)
        reference.bias.copy_(ln.bias)
    x = torch.randn(3, 7, D) * 4 + 2
    torch.testing.assert_close(ln(x), reference(x))


def test_normalized_vectors_have_zero_mean_and_unit_variance():
    x = torch.randn(3, 7, D) * 5 + 3
    out = LayerNorm(D)(x)  # γ = 1, β = 0
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(3, 7), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.pow(2).mean(dim=-1), torch.ones(3, 7), atol=1e-4, rtol=0)


def test_invariant_to_scaling_and_shifting_the_input(ln):
    x = torch.randn(2, 5, D)
    torch.testing.assert_close(ln(3 * x + 7), ln(x), atol=1e-4, rtol=1e-4)


def test_each_position_is_normalized_independently(ln):
    x = torch.randn(1, 6, D)
    changed = x.clone()
    changed[:, 4] = torch.randn(D)
    diff = (ln(x) - ln(changed)).abs().amax(dim=-1)[0]
    assert diff.nonzero().flatten().tolist() == [4]


def test_constant_vector_does_not_produce_nan(ln):
    out = ln(torch.full((1, 3, D), 2.5))
    assert not out.isnan().any()
    # x − média = 0, então só sobra β.
    torch.testing.assert_close(out.detach(), ln.bias.detach().expand(1, 3, D))


def test_initial_parameters():
    ln = LayerNorm(D)
    assert torch.equal(ln.weight, torch.ones(D))
    assert torch.equal(ln.bias, torch.zeros(D))
    assert sum(p.numel() for p in ln.parameters()) == 2 * D
