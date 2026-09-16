import math
from dataclasses import replace

import pytest
import torch

from config import ModelConfig
from model.gpt import GPT

VOCAB_SIZE = 20
CONFIG = ModelConfig(context_length=16, d_model=32, num_heads=4, num_layers=3, dropout=0.0)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(CONFIG, VOCAB_SIZE)


def random_ids(batch: int, seq_len: int) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


# --- forma e composição ---


@pytest.mark.parametrize("seq_len", [1, 7, CONFIG.context_length])
def test_hidden_states_shape(model, seq_len):
    assert model.hidden_states(random_ids(2, seq_len)).shape == (2, seq_len, CONFIG.d_model)


def test_sequence_longer_than_context_raises(model):
    with pytest.raises(ValueError):
        model(random_ids(1, CONFIG.context_length + 1))


def test_id_outside_vocab_raises(model):
    with pytest.raises(IndexError):
        model(torch.tensor([[VOCAB_SIZE]]))


def test_has_one_block_per_layer(model):
    assert len(model.blocks) == CONFIG.num_layers


def test_matches_manual_composition(model):
    ids = random_ids(2, 9)
    h = model.embeddings(ids)
    for block in model.blocks:
        h = block(h)
    hidden = model.ln_final(h)
    torch.testing.assert_close(model.hidden_states(ids), hidden)
    torch.testing.assert_close(model(ids), model.lm_head(hidden))


def test_final_layer_norm_standardizes_each_position(model):
    out = model.hidden_states(random_ids(2, 10))  # γ = 1 e β = 0 na inicialização
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(2, 10), atol=1e-5, rtol=0)
    # Um pouco abaixo de 1: o stream é pequeno na inicialização, e o eps pesa em var / (var + eps).
    torch.testing.assert_close(out.pow(2).mean(dim=-1), torch.ones(2, 10), atol=0.05, rtol=0)


def test_parameter_count(model):
    # Com weight tying (padrão), o LM head não acrescenta parâmetros.
    d, T, L = CONFIG.d_model, CONFIG.context_length, CONFIG.num_layers
    expected = VOCAB_SIZE * d + T * d + L * (12 * d**2 + 9 * d) + 2 * d
    assert model.num_parameters() == expected


# --- causalidade e independência ---


def test_future_tokens_do_not_affect_past(model):
    ids = random_ids(1, 12)
    changed = ids.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % VOCAB_SIZE
    out, out_changed = model(ids), model(changed)
    torch.testing.assert_close(out[:, :8], out_changed[:, :8])
    assert not torch.allclose(out[:, 8:], out_changed[:, 8:])


def test_sequences_in_a_batch_are_independent(model):
    ids = random_ids(2, 10)
    changed = ids.clone()
    changed[1] = random_ids(1, 10)[0]
    torch.testing.assert_close(model(ids)[0], model(changed)[0])


# --- gradientes e inicialização ---


def test_gradients_reach_all_parameters(model):
    out = model(random_ids(2, 10))
    (out * torch.randn_like(out)).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad.abs().sum() > 0, name


def test_sum_of_hidden_states_gives_almost_no_gradient_before_final_layer_norm(model):
    # Cada vetor normalizado tem soma 0, qualquer que seja a entrada: a soma é constante.
    model.hidden_states(random_ids(2, 10)).sum().backward()
    assert model.embeddings.token_embedding.weight.grad.abs().max() < 1e-4


def test_all_blocks_use_depth_scaled_init():
    torch.manual_seed(0)
    config = ModelConfig(context_length=16, d_model=256, num_heads=4, num_layers=8, dropout=0.0)
    model = GPT(config, VOCAB_SIZE)
    expected = 0.02 / math.sqrt(2 * config.num_layers)
    for block in model.blocks:
        assert block.attention.out_proj.weight.std().item() == pytest.approx(expected, rel=0.05)
        assert block.feed_forward.down_proj.weight.std().item() == pytest.approx(expected, rel=0.05)


def test_dropout_only_in_train_mode():
    torch.manual_seed(0)
    model = GPT(replace(CONFIG, dropout=0.5), VOCAB_SIZE)
    ids = random_ids(2, 10)
    model.eval()
    torch.testing.assert_close(model(ids), model(ids))
    model.train()
    assert not torch.allclose(model(ids), model(ids))
