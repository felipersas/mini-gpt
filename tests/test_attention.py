import math

import pytest
import torch
import torch.nn.functional as F

from model.attention import (
    CausalSelfAttention,
    attention_weights,
    causal_mask,
    scaled_dot_product_attention,
)

D_MODEL, HEAD_DIM = 16, 8


@pytest.fixture
def attn() -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(D_MODEL, HEAD_DIM, dropout=0.0)


# --- causal_mask ---


def test_causal_mask_is_lower_triangular():
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(causal_mask(4), expected)


# --- funções de attention ---


def test_matches_explicit_loops():
    torch.manual_seed(0)
    q, k, v = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 3)
    out, _ = scaled_dot_product_attention(q, k, v, causal_mask(4))

    expected = torch.zeros(4, 3)
    for i in range(4):
        # Só j <= i: a posição i não vê o futuro.
        scores = torch.tensor([torch.dot(q[i], k[j]) / math.sqrt(8) for j in range(i + 1)])
        weights = torch.softmax(scores, dim=0)
        expected[i] = sum(weights[j] * v[j] for j in range(i + 1))
    torch.testing.assert_close(out, expected)


def test_matches_pytorch_implementation():
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 6)
    out, _ = scaled_dot_product_attention(q, k, v, causal_mask(5))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_weights_rows_sum_to_one():
    weights = attention_weights(torch.randn(2, 5, 8), torch.randn(2, 5, 8), causal_mask(5))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))


def test_masked_positions_get_zero_weight():
    weights = attention_weights(torch.randn(2, 5, 8), torch.randn(2, 5, 8), causal_mask(5))
    assert (weights[:, ~causal_mask(5)] == 0).all()


def test_first_position_attends_only_to_itself():
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 3)
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(4))
    assert weights[0, 0, 0] == 1
    torch.testing.assert_close(out[0, 0], v[0, 0])


def test_equal_scores_average_visible_values():
    # q = 0 deixa todos os scores iguais: cada saída é a média simples dos values visíveis.
    v = torch.randn(1, 4, 3)
    out, _ = scaled_dot_product_attention(
        torch.zeros(1, 4, 8), torch.randn(1, 4, 8), v, causal_mask(4)
    )
    prefix_means = v.cumsum(dim=1) / torch.arange(1, 5).view(1, 4, 1)
    torch.testing.assert_close(out, prefix_means)


def test_scaling_keeps_score_variance_near_one():
    torch.manual_seed(0)
    d_k = 256
    q, k = torch.randn(4096, d_k), torch.randn(4096, d_k)
    dot = (q * k).sum(dim=-1)  # 4096 produtos escalares independentes
    assert dot.var().item() == pytest.approx(d_k, rel=0.1)
    assert (dot / math.sqrt(d_k)).var().item() == pytest.approx(1.0, rel=0.1)


def test_no_nan_with_large_scores():
    q = torch.randn(2, 16, 8) * 1e3
    weights = attention_weights(q, q, causal_mask(16))
    assert not weights.isnan().any()


# --- CausalSelfAttention ---


def test_module_output_shape(attn):
    assert attn(torch.randn(3, 10, D_MODEL)).shape == (3, 10, HEAD_DIM)


def test_module_matches_function(attn):
    x = torch.randn(2, 7, D_MODEL)
    expected, _ = scaled_dot_product_attention(
        attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), causal_mask(7)
    )
    torch.testing.assert_close(attn(x), expected)


def test_future_tokens_do_not_affect_past(attn):
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = torch.randn(1, 4, D_MODEL)
    out, out_changed = attn(x), attn(changed)
    torch.testing.assert_close(out[:, :6], out_changed[:, :6])
    assert not torch.allclose(out[:, 6:], out_changed[:, 6:])


def test_past_token_affects_every_later_position(attn):
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[:, 0] = torch.randn(D_MODEL)
    diff = (attn(x) - attn(changed)).abs().amax(dim=-1)[0]
    assert (diff > 0).all()


def test_gradient_only_reaches_current_and_past_inputs(attn):
    x = torch.randn(1, 6, D_MODEL, requires_grad=True)
    attn(x)[0, 2].sum().backward()
    grad_norms = x.grad[0].norm(dim=-1)
    assert (grad_norms[:3] > 0).all()
    assert (grad_norms[3:] == 0).all()


def test_gradients_reach_all_projections(attn):
    attn(torch.randn(2, 5, D_MODEL)).sum().backward()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        assert proj.weight.grad.abs().sum() > 0


def test_deterministic_in_eval_mode():
    x = torch.randn(2, 5, D_MODEL)
    outputs = []
    for _ in range(2):
        torch.manual_seed(0)
        module = CausalSelfAttention(D_MODEL, HEAD_DIM, dropout=0.1).eval()
        outputs.append(module(x))
    torch.testing.assert_close(outputs[0], outputs[1])


def test_parameter_count(attn):
    assert sum(p.numel() for p in attn.parameters()) == 3 * D_MODEL * HEAD_DIM
