import pytest
import torch
from torch import nn

from model.attention import (
    LoopMultiHeadAttention,
    MultiHeadAttention,
    causal_mask,
    merge_heads,
    split_heads,
)

D_MODEL, NUM_HEADS = 16, 4
HEAD_DIM = D_MODEL // NUM_HEADS


def copy_heads(loop: LoopMultiHeadAttention, fused: MultiHeadAttention) -> None:
    """Empilha as matrizes de cada head nas projeções da versão com tensor único."""
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj"):
            stacked = torch.cat([getattr(head, name).weight for head in loop.heads], dim=0)
            getattr(fused, name).weight.copy_(stacked)
        fused.out_proj.weight.copy_(loop.out_proj.weight)


@pytest.fixture
def mha() -> MultiHeadAttention:
    torch.manual_seed(0)
    return MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)


# --- split_heads / merge_heads ---


def test_split_heads_shape():
    x = torch.randn(2, 5, D_MODEL)
    assert split_heads(x, NUM_HEADS).shape == (2, NUM_HEADS, 5, HEAD_DIM)


def test_each_head_gets_a_contiguous_block_of_features():
    x = torch.randn(2, 5, D_MODEL)
    heads = split_heads(x, NUM_HEADS)
    for i in range(NUM_HEADS):
        torch.testing.assert_close(heads[:, i], x[..., i * HEAD_DIM : (i + 1) * HEAD_DIM])


def test_merge_undoes_split():
    x = torch.randn(2, 5, D_MODEL)
    torch.testing.assert_close(merge_heads(split_heads(x, NUM_HEADS)), x)


def test_merge_works_on_a_fresh_attention_output():
    # A attention devolve um tensor novo [B, h, T, d_h]; o transpose dele não é contíguo.
    heads = torch.randn(2, NUM_HEADS, 5, HEAD_DIM)
    merged = merge_heads(heads)
    assert merged.shape == (2, 5, D_MODEL)
    for i in range(NUM_HEADS):
        torch.testing.assert_close(merged[..., i * HEAD_DIM : (i + 1) * HEAD_DIM], heads[:, i])


def test_head_uses_only_its_block_of_weights(mha):
    x = torch.randn(1, 5, D_MODEL)
    before = split_heads(mha.q_proj(x), NUM_HEADS)
    with torch.no_grad():
        mha.q_proj.weight[HEAD_DIM : 2 * HEAD_DIM] += 1.0  # linhas da head 1
    after = split_heads(mha.q_proj(x), NUM_HEADS)
    changed = [not torch.allclose(before[:, i], after[:, i]) for i in range(NUM_HEADS)]
    assert changed == [False, True, False, False]


# --- MultiHeadAttention ---


def test_output_shape(mha):
    assert mha(torch.randn(3, 10, D_MODEL)).shape == (3, 10, D_MODEL)


@pytest.mark.parametrize("cls", [MultiHeadAttention, LoopMultiHeadAttention])
def test_d_model_must_be_divisible_by_num_heads(cls):
    with pytest.raises(ValueError):
        cls(d_model=10, num_heads=3, dropout=0.0)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_fused_matches_loop(num_heads):
    torch.manual_seed(0)
    loop = LoopMultiHeadAttention(D_MODEL, num_heads, dropout=0.0)
    fused = MultiHeadAttention(D_MODEL, num_heads, dropout=0.0)
    copy_heads(loop, fused)
    x = torch.randn(2, 7, D_MODEL)
    torch.testing.assert_close(fused(x), loop(x))


# --- backend "sdpa" ---


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError):
        MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0, backend="flash")


def test_sdpa_backend_matches_manual_backend():
    torch.manual_seed(0)
    manual = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0, backend="manual")
    sdpa = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0, backend="sdpa")
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, 7, D_MODEL)
    torch.testing.assert_close(sdpa(x), manual(x), atol=1e-6, rtol=1e-5)


def test_sdpa_backend_is_also_causal():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0, backend="sdpa").eval()
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = torch.randn(1, 4, D_MODEL)
    torch.testing.assert_close(mha(x)[:, :6], mha(changed)[:, :6])


def test_sdpa_backend_gradients_reach_all_projections():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0, backend="sdpa")
    mha(torch.randn(2, 5, D_MODEL)).sum().backward()
    for proj in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
        assert proj.weight.grad.abs().sum() > 0


def test_matches_pytorch_multihead_attention(mha):
    torch_mha = nn.MultiheadAttention(D_MODEL, NUM_HEADS, bias=False, batch_first=True)
    with torch.no_grad():
        qkv = torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight])
        torch_mha.in_proj_weight.copy_(qkv)
        torch_mha.out_proj.weight.copy_(mha.out_proj.weight)
    x = torch.randn(2, 7, D_MODEL)
    # No PyTorch, True em attn_mask significa "proibido": o inverso da nossa máscara.
    expected, _ = torch_mha(x, x, x, attn_mask=~causal_mask(7), need_weights=False)
    torch.testing.assert_close(mha(x), expected, atol=1e-6, rtol=1e-5)


def test_future_tokens_do_not_affect_past(mha):
    x = torch.randn(1, 10, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = torch.randn(1, 4, D_MODEL)
    torch.testing.assert_close(mha(x)[:, :6], mha(changed)[:, :6])


def test_gradient_only_reaches_current_and_past_inputs(mha):
    x = torch.randn(1, 6, D_MODEL, requires_grad=True)
    mha(x)[0, 2].sum().backward()
    grad_norms = x.grad[0].norm(dim=-1)
    assert (grad_norms[:3] > 0).all()
    assert (grad_norms[3:] == 0).all()


def test_gradients_reach_all_projections(mha):
    mha(torch.randn(2, 5, D_MODEL)).sum().backward()
    for proj in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
        assert proj.weight.grad.abs().sum() > 0


def test_both_versions_have_4_d_model_squared_parameters():
    for cls in (MultiHeadAttention, LoopMultiHeadAttention):
        module = cls(D_MODEL, NUM_HEADS, dropout=0.0)
        assert sum(p.numel() for p in module.parameters()) == 4 * D_MODEL**2


def test_deterministic_in_eval_mode():
    x = torch.randn(2, 5, D_MODEL)
    outputs = []
    for _ in range(2):
        torch.manual_seed(0)
        outputs.append(MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.1).eval()(x))
    torch.testing.assert_close(outputs[0], outputs[1])
