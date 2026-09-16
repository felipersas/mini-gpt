import math
from dataclasses import asdict

import pytest

from training.metrics import bits_per_token, load_history, peak_memory_mb, perplexity, save_history
from training.trainer import EpochRecord


def test_perplexity_of_a_uniform_guess_is_the_number_of_options():
    assert perplexity(math.log(97)) == pytest.approx(97)


def test_zero_loss_has_perplexity_one():
    assert perplexity(0.0) == 1.0


def test_bits_per_token_converts_nats_to_bits():
    assert bits_per_token(math.log(2)) == pytest.approx(1.0)
    assert bits_per_token(math.log(97)) == pytest.approx(math.log2(97))


def test_history_round_trip(tmp_path):
    records = [
        EpochRecord(
            epoch=1,
            step=82,
            train_loss=3.2,
            train_eval_loss=2.5,
            val_loss=2.47,
            learning_rate=1e-3,
            grad_norm=0.6,
            tokens_per_second=40_000.0,
            peak_memory_mb=250.0,
            seconds=19.0,
        )
    ]
    path = tmp_path / "checkpoints" / "history.json"
    save_history(path, records)
    assert load_history(path) == [asdict(records[0])]


def test_peak_memory_is_measured():
    assert peak_memory_mb() > 1
