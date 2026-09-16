import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from config import ModelConfig
from model.gpt import GPT
from model.lm_head import LMHead

VOCAB_SIZE, D_MODEL = 20, 32
CONFIG = ModelConfig(context_length=16, d_model=D_MODEL, num_heads=4, num_layers=2, dropout=0.0)


def make_model(tie_weights: bool = True) -> GPT:
    torch.manual_seed(0)
    return GPT(replace(CONFIG, tie_weights=tie_weights), VOCAB_SIZE)


def random_ids(batch: int, seq_len: int) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


# --- LMHead ---


def test_head_output_shape():
    head = LMHead(D_MODEL, VOCAB_SIZE)
    assert head(torch.randn(3, 7, D_MODEL)).shape == (3, 7, VOCAB_SIZE)


def test_logits_are_dot_products_with_each_row():
    head = LMHead(D_MODEL, VOCAB_SIZE)
    h = torch.randn(D_MODEL)
    expected = torch.stack([torch.dot(h, head.proj.weight[i]) for i in range(VOCAB_SIZE)])
    torch.testing.assert_close(head(h), expected)


def test_head_has_no_bias_and_small_init():
    torch.manual_seed(0)
    head = LMHead(256, 1000)
    assert head.proj.bias is None
    assert head.proj.weight.std().item() == pytest.approx(0.02, rel=0.05)


# --- GPT com LM head ---


@pytest.mark.parametrize("tie_weights", [True, False])
def test_model_returns_logits_over_the_vocabulary(tie_weights):
    assert make_model(tie_weights)(random_ids(2, 9)).shape == (2, 9, VOCAB_SIZE)


def test_softmax_gives_a_distribution_per_position():
    probs = torch.softmax(make_model()(random_ids(2, 9)), dim=-1)
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, 9))


def test_logits_are_causal():
    model = make_model()
    ids = random_ids(1, 12)
    changed = ids.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % VOCAB_SIZE
    torch.testing.assert_close(model(ids)[:, :8], model(changed)[:, :8])


def test_tied_head_shares_the_token_embedding_matrix():
    model = make_model(tie_weights=True)
    assert model.lm_head.proj.weight is model.embeddings.token_embedding.weight


def test_untied_head_has_its_own_matrix():
    model = make_model(tie_weights=False)
    assert model.lm_head.proj.weight is not model.embeddings.token_embedding.weight


def test_tied_logits_are_similarities_with_token_embeddings():
    model = make_model(tie_weights=True)
    ids = random_ids(2, 9)
    expected = model.hidden_states(ids) @ model.embeddings.token_embedding.weight.T
    torch.testing.assert_close(model(ids), expected)


def test_tying_saves_vocab_size_times_d_model_parameters():
    saved = make_model(False).num_parameters() - make_model(True).num_parameters()
    assert saved == VOCAB_SIZE * D_MODEL


@pytest.mark.parametrize("tie_weights", [True, False])
def test_initial_loss_is_close_to_a_uniform_guess(tie_weights):
    model = make_model(tie_weights)
    ids, targets = random_ids(8, 16), random_ids(8, 16)
    loss = F.cross_entropy(model(ids).view(-1, VOCAB_SIZE), targets.view(-1))
    assert loss.item() == pytest.approx(math.log(VOCAB_SIZE), abs=0.15)


@pytest.mark.parametrize("tie_weights, every_row", [(True, True), (False, False)])
def test_tying_sends_gradient_to_every_embedding_row(tie_weights, every_row):
    model = make_model(tie_weights)
    # Só os tokens 0..4 aparecem; a softmax da saída ainda envolve todos os 20.
    ids, targets = torch.randint(0, 5, (4, 16)), torch.randint(0, 5, (4, 16))
    F.cross_entropy(model(ids).view(-1, VOCAB_SIZE), targets.view(-1)).backward()
    rows_with_grad = model.embeddings.token_embedding.weight.grad.abs().sum(dim=1) > 0
    assert rows_with_grad[:5].all()
    assert rows_with_grad.all().item() is every_row
