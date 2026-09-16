import pytest
import torch
import torch.nn.functional as F

from model.embeddings import Embeddings

VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL = 20, 8, 16


@pytest.fixture
def emb() -> Embeddings:
    torch.manual_seed(0)
    return Embeddings(VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, dropout=0.0)


@pytest.mark.parametrize("seq_len", [1, 5, CONTEXT_LENGTH])
def test_output_shape(emb, seq_len):
    ids = torch.randint(0, VOCAB_SIZE, (3, seq_len))
    assert emb(ids).shape == (3, seq_len, D_MODEL)


def test_sequence_longer_than_context_raises(emb):
    with pytest.raises(ValueError):
        emb(torch.zeros(1, CONTEXT_LENGTH + 1, dtype=torch.long))


def test_id_outside_vocab_raises(emb):
    with pytest.raises(IndexError):
        emb(torch.tensor([[VOCAB_SIZE]]))


def test_lookup_equals_one_hot_times_matrix(emb):
    ids = torch.tensor([[3, 0, 3, 19]])
    one_hot = F.one_hot(ids, VOCAB_SIZE).float()  # [1, 4, vocab_size]
    torch.testing.assert_close(emb.token_embedding(ids), one_hot @ emb.token_embedding.weight)


def test_output_is_token_plus_position(emb):
    ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    expected = emb.token_embedding.weight[ids] + emb.position_embedding.weight[:5]
    torch.testing.assert_close(emb(ids), expected)


def test_same_token_differs_by_position(emb):
    out = emb(torch.tensor([[7, 7]]))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_token_and_position_match_across_batch(emb):
    out = emb(torch.tensor([[7, 1], [7, 2]]))
    torch.testing.assert_close(out[0, 0], out[1, 0])


def test_gradient_reaches_only_used_rows(emb):
    emb(torch.tensor([[2, 5, 5]])).sum().backward()
    token_rows = emb.token_embedding.weight.grad.abs().sum(dim=1).nonzero().flatten()
    position_rows = emb.position_embedding.weight.grad.abs().sum(dim=1).nonzero().flatten()
    assert token_rows.tolist() == [2, 5]
    assert position_rows.tolist() == [0, 1, 2]


def test_gradient_accumulates_over_repeated_tokens(emb):
    emb(torch.tensor([[2, 5, 5]])).sum().backward()
    grad = emb.token_embedding.weight.grad
    # Com loss = soma das saídas, cada ocorrência contribui com um gradiente de uns.
    torch.testing.assert_close(grad[5], 2 * grad[2])


def test_parameter_count(emb):
    n_params = sum(p.numel() for p in emb.parameters())
    assert n_params == VOCAB_SIZE * D_MODEL + CONTEXT_LENGTH * D_MODEL


def test_init_std_is_small():
    emb = Embeddings(vocab_size=1000, context_length=512, d_model=64, dropout=0.0)
    for table in (emb.token_embedding, emb.position_embedding):
        assert table.weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_dropout_only_in_train_mode():
    emb = Embeddings(VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, dropout=0.5)
    ids = torch.randint(0, VOCAB_SIZE, (4, CONTEXT_LENGTH))
    emb.eval()
    torch.testing.assert_close(emb(ids), emb(ids))
    emb.train()
    assert (emb(ids) == 0).any()
