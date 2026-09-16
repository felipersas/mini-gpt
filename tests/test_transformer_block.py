import math

import pytest
import torch
from torch import nn

from model.transformer_block import TransformerBlock

D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS = 16, 4, 64, 4


def make_block(dropout: float = 0.0) -> TransformerBlock:
    return TransformerBlock(D_MODEL, NUM_HEADS, D_FF, dropout, NUM_LAYERS)


def silence_sublayers(block: TransformerBlock) -> None:
    """Zera as matrizes que escrevem no residual stream: as subcamadas passam a somar zero."""
    with torch.no_grad():
        block.attention.out_proj.weight.zero_()
        block.feed_forward.down_proj.weight.zero_()


@pytest.fixture
def block() -> TransformerBlock:
    torch.manual_seed(0)
    return make_block()


def test_output_shape(block):
    assert block(torch.randn(3, 10, D_MODEL)).shape == (3, 10, D_MODEL)


def test_matches_manual_composition(block):
    x = torch.randn(2, 7, D_MODEL)
    after_attention = x + block.attention(block.ln_1(x))
    expected = after_attention + block.feed_forward(block.ln_2(after_attention))
    torch.testing.assert_close(block(x), expected)


def test_block_is_identity_when_sublayers_write_nothing(block):
    silence_sublayers(block)
    x = torch.randn(2, 7, D_MODEL)
    torch.testing.assert_close(block(x), x)


def test_residual_path_passes_gradient_unchanged(block):
    silence_sublayers(block)
    x = torch.randn(2, 7, D_MODEL, requires_grad=True)
    block(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_future_tokens_do_not_affect_past(block):
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = torch.randn(1, 4, D_MODEL)
    torch.testing.assert_close(block(x)[:, :6], block(changed)[:, :6])


def test_gradient_only_reaches_current_and_past_inputs(block):
    x = torch.randn(1, 6, D_MODEL, requires_grad=True)
    block(x)[0, 2].sum().backward()
    grad_norms = x.grad[0].norm(dim=-1)
    assert (grad_norms[:3] > 0).all()
    assert (grad_norms[3:] == 0).all()


def test_stacked_blocks_keep_shape_and_causality():
    torch.manual_seed(0)
    stack = nn.Sequential(*(make_block() for _ in range(3)))
    x = torch.randn(1, 8, D_MODEL)
    changed = x.clone()
    changed[:, 5:] = torch.randn(1, 3, D_MODEL)
    out = stack(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out[:, :5], stack(changed)[:, :5])


def test_residual_projections_use_depth_scaled_init():
    torch.manual_seed(0)
    block = TransformerBlock(d_model=256, num_heads=4, d_ff=1024, dropout=0.0, num_layers=8)
    scaled_std = 0.02 / math.sqrt(2 * 8)
    for weight in (block.attention.out_proj.weight, block.feed_forward.down_proj.weight):
        assert weight.std().item() == pytest.approx(scaled_std, rel=0.05)
    for weight in (block.attention.q_proj.weight, block.feed_forward.up_proj.weight):
        assert weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_parameter_count(block):
    n_params = sum(p.numel() for p in block.parameters())
    # attention 4d² + feed-forward (8d² + 5d) + 2 LayerNorms (4d)
    assert n_params == 12 * D_MODEL**2 + 9 * D_MODEL


def test_gradients_reach_all_parameters(block):
    block(torch.randn(2, 5, D_MODEL)).sum().backward()
    for name, param in block.named_parameters():
        assert param.grad.abs().sum() > 0, name


def test_dropout_only_in_train_mode():
    torch.manual_seed(0)
    block = make_block(dropout=0.5)
    x = torch.randn(2, 6, D_MODEL)
    block.eval()
    torch.testing.assert_close(block(x), block(x))
    block.train()
    assert not torch.allclose(block(x), block(x))
