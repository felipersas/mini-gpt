import pytest
import torch

from data.dataset import NextTokenDataset
from data.loader import create_dataloader, split_train_val
from tokenizer import CharTokenizer


def window_starts(n: int, context_length: int, stride: int) -> list[int]:
    return [s for s in range(0, n, stride) if s + context_length + 1 <= n]


# --- NextTokenDataset ---


def test_first_and_last_windows():
    ds = NextTokenDataset(torch.arange(10), context_length=3, stride=1)
    assert len(ds) == 7
    x, y = ds[0]
    assert (x.tolist(), y.tolist()) == ([0, 1, 2], [1, 2, 3])
    x, y = ds[len(ds) - 1]
    assert (x.tolist(), y.tolist()) == ([6, 7, 8], [7, 8, 9])


@pytest.mark.parametrize(
    "n, context_length, stride", [(10, 3, 1), (10, 3, 3), (11, 3, 3), (100, 8, 5), (9, 8, 4)]
)
def test_windows_match_brute_force(n, context_length, stride):
    ds = NextTokenDataset(torch.arange(n), context_length, stride)
    starts = window_starts(n, context_length, stride)
    assert len(ds) == len(starts)
    for i, start in enumerate(starts):
        x, y = ds[i]
        assert x.tolist() == list(range(start, start + context_length))
        assert y.tolist() == list(range(start + 1, start + context_length + 1))


def test_target_is_input_shifted_by_one():
    ds = NextTokenDataset(torch.randint(0, 50, (200,)), context_length=16, stride=7)
    for x, y in ds:
        assert torch.equal(y[:-1], x[1:])


def test_accepts_list_and_returns_long():
    x, y = NextTokenDataset([5, 6, 7, 8], context_length=2, stride=1)[0]
    assert x.dtype == y.dtype == torch.long


def test_sequence_too_short_raises():
    with pytest.raises(ValueError):
        NextTokenDataset(torch.arange(3), context_length=3, stride=1)


@pytest.mark.parametrize("context_length, stride", [(0, 1), (3, 0)])
def test_invalid_sizes_raise(context_length, stride):
    with pytest.raises(ValueError):
        NextTokenDataset(torch.arange(10), context_length, stride)


def test_rejects_2d_ids():
    with pytest.raises(ValueError):
        NextTokenDataset(torch.zeros(2, 10, dtype=torch.long), context_length=3, stride=1)


@pytest.mark.parametrize("index", [-1, 7])
def test_index_out_of_range_raises(index):
    ds = NextTokenDataset(torch.arange(10), context_length=3, stride=1)
    with pytest.raises(IndexError):
        ds[index]


# --- split_train_val ---


def test_split_is_contiguous():
    ids = torch.arange(100)
    train, val = split_train_val(ids, val_fraction=0.1)
    assert (len(train), len(val)) == (90, 10)
    assert torch.equal(torch.cat([train, val]), ids)


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1])
def test_split_rejects_invalid_fraction(val_fraction):
    with pytest.raises(ValueError):
        split_train_val(torch.arange(100), val_fraction)


# --- create_dataloader ---


def make_loader(n=1000, shuffle=True, seed=0):
    return create_dataloader(
        torch.arange(n), context_length=16, stride=16, batch_size=8, shuffle=shuffle, seed=seed
    )


def test_batches_have_consistent_shape():
    batches = list(make_loader())
    # (1000 - 17) // 16 + 1 = 62 janelas -> 7 batches completos
    assert len(batches) == 62 // 8
    for x, y in batches:
        assert x.shape == y.shape == (8, 16)


def test_without_shuffle_windows_come_in_order():
    x, _ = next(iter(make_loader(shuffle=False)))
    assert x[:, 0].tolist() == [i * 16 for i in range(8)]


def test_shuffle_is_reproducible_with_seed():
    def first_batch(seed):
        return next(iter(make_loader(seed=seed)))[0]

    assert torch.equal(first_batch(0), first_batch(0))
    assert not torch.equal(first_batch(0), first_batch(1))


def test_dataset_smaller_than_batch_raises():
    with pytest.raises(ValueError):
        create_dataloader(torch.arange(20), context_length=4, stride=4, batch_size=8, shuffle=False)


def test_pipeline_from_text():
    text = "o gato está na casa. " * 20
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text))
    loader = create_dataloader(ids, context_length=12, stride=5, batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    for xi, yi in zip(x, y, strict=True):
        assert tok.decode(yi)[:-1] == tok.decode(xi)[1:]
