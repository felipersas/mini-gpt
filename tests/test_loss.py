import math

import pytest
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import GPT
from training.loss import cross_entropy, token_losses

V = 10


def test_token_losses_keep_one_value_per_position_and_average_to_cross_entropy():
    logits, targets = torch.randn(4, 7, V), torch.randint(0, V, (4, 7))
    losses = token_losses(logits, targets)
    assert losses.shape == (4, 7)
    assert (losses >= 0).all()
    torch.testing.assert_close(losses.mean(), cross_entropy(logits, targets))


# --- valor da loss ---


def test_matches_pytorch_for_flat_inputs():
    logits, targets = torch.randn(20, V), torch.randint(0, V, (20,))
    torch.testing.assert_close(cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_matches_pytorch_for_sequences():
    logits, targets = torch.randn(4, 7, V), torch.randint(0, V, (4, 7))
    expected = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    torch.testing.assert_close(cross_entropy(logits, targets), expected)


def test_small_example_by_hand():
    # softmax(2, 1, 0) = (0,665; 0,245; 0,090); o alvo é o token 0.
    loss = cross_entropy(torch.tensor([[2.0, 1.0, 0.0]]), torch.tensor([0]))
    expected = -math.log(math.exp(2) / (math.exp(2) + math.exp(1) + 1))
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_uniform_logits_give_log_vocab_size():
    loss = cross_entropy(torch.zeros(5, V), torch.randint(0, V, (5,)))
    assert loss.item() == pytest.approx(math.log(V), abs=1e-6)


def test_confident_correct_prediction_gives_almost_zero_loss():
    logits = torch.zeros(1, V)
    logits[0, 3] = 50.0
    assert cross_entropy(logits, torch.tensor([3])).item() < 1e-6


def test_confident_wrong_prediction_is_heavily_penalized():
    logits = torch.zeros(1, V)
    logits[0, 3] = 50.0
    assert cross_entropy(logits, torch.tensor([4])).item() == pytest.approx(50.0, abs=1e-3)


def test_adding_a_constant_to_the_logits_does_not_change_the_loss():
    logits, targets = torch.randn(6, V), torch.randint(0, V, (6,))
    torch.testing.assert_close(cross_entropy(logits + 100, targets), cross_entropy(logits, targets))


def test_large_logits_are_numerically_stable():
    loss = cross_entropy(torch.tensor([[1000.0, 0.0, -1000.0]]), torch.tensor([2]))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(2000.0, rel=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cross_entropy(torch.randn(4, 7, V), torch.randint(0, V, (4, 6)))


# --- gradiente ---


def test_gradient_is_softmax_minus_one_hot():
    logits = torch.randn(8, V, requires_grad=True)
    targets = torch.randint(0, V, (8,))
    cross_entropy(logits, targets).backward()
    expected = (torch.softmax(logits, dim=-1) - F.one_hot(targets, V)) / 8
    torch.testing.assert_close(logits.grad, expected.detach())


def test_gradient_of_each_position_sums_to_zero():
    logits = torch.randn(8, V, requires_grad=True)
    cross_entropy(logits, torch.randint(0, V, (8,))).backward()
    torch.testing.assert_close(logits.grad.sum(dim=-1), torch.zeros(8), atol=1e-7, rtol=0)


def test_loss_reaches_every_parameter_of_the_model():
    torch.manual_seed(0)
    config = ModelConfig(context_length=16, d_model=32, num_heads=4, num_layers=2, dropout=0.0)
    model = GPT(config, vocab_size=V)
    ids, targets = torch.randint(0, V, (2, 12)), torch.randint(0, V, (2, 12))
    cross_entropy(model(ids), targets).backward()
    for name, param in model.named_parameters():
        assert param.grad.abs().sum() > 0, name
