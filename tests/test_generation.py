import pytest
import torch

from config import ModelConfig
from generation.generate import generate, sample_next_token, top_k_filter, top_p_filter
from model import GPT

# --- filtros ---


def test_top_k_keeps_the_k_largest():
    logits = torch.tensor([1.0, 4.0, 2.0, 3.0])
    filtered = top_k_filter(logits, 2)
    assert torch.isfinite(filtered).tolist() == [False, True, False, True]
    assert filtered[1] == 4.0 and filtered[3] == 3.0


def test_top_k_at_least_vocab_size_changes_nothing():
    logits = torch.randn(5)
    torch.testing.assert_close(top_k_filter(logits, 5), logits)


def test_top_p_keeps_the_smallest_set_that_reaches_p():
    logits = torch.tensor([0.05, 0.5, 0.15, 0.3]).log()
    # Em ordem: 0,5 + 0,3 = 0,8 ≥ 0,7, então só esses dois ficam.
    assert torch.isfinite(top_p_filter(logits, 0.7)).tolist() == [False, True, False, True]


def test_top_p_always_keeps_the_most_likely_token():
    logits = torch.tensor([0.0, 10.0, 0.0])
    assert torch.isfinite(top_p_filter(logits, 0.01)).tolist() == [False, True, False]


def test_top_p_of_one_keeps_everything():
    logits = torch.tensor([20.0, 0.0, -20.0])
    assert torch.isfinite(top_p_filter(logits, 1.0)).all()


# --- escolha do próximo token ---


def test_temperature_zero_is_greedy():
    assert sample_next_token(torch.tensor([1.0, 3.0, 2.0]), temperature=0.0) == 1


def softmax_at(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def test_low_temperature_sharpens_and_high_temperature_flattens():
    logits = torch.tensor([2.0, 1.0, 0.0])
    assert softmax_at(logits, 0.2)[0] > softmax_at(logits, 1.0)[0] > softmax_at(logits, 5.0)[0]
    flat = softmax_at(logits, 100.0)
    assert flat.max() - flat.min() < 0.01


def test_sampling_follows_the_probabilities():
    generator = torch.Generator().manual_seed(0)
    logits = torch.tensor([0.5, 0.3, 0.2]).log()
    counts = torch.zeros(3)
    for _ in range(5000):
        counts[sample_next_token(logits, generator=generator)] += 1
    torch.testing.assert_close(counts / 5000, torch.tensor([0.5, 0.3, 0.2]), atol=0.03, rtol=0)


def test_top_k_of_one_is_greedy():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(20)
    samples = {sample_next_token(logits, top_k=1, generator=generator) for _ in range(20)}
    assert samples == {int(logits.argmax())}


def test_sampling_never_picks_a_filtered_token():
    generator = torch.Generator().manual_seed(0)
    logits = torch.tensor([0.0, 0.1, 0.2, 5.0, 4.0])
    samples = {sample_next_token(logits, top_k=2, generator=generator) for _ in range(200)}
    assert samples <= {3, 4}


# --- generate ---


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    config = ModelConfig(context_length=8, d_model=16, num_heads=2, num_layers=1, dropout=0.0)
    return GPT(config, vocab_size=10)


def test_generate_appends_max_new_tokens(model):
    out = generate(model, [1, 2, 3], 5, temperature=0.0)
    assert len(out) == 8
    assert out[:3] == [1, 2, 3]


def test_greedy_generation_matches_a_manual_loop(model):
    ids = [1, 2]
    for _ in range(4):
        ids.append(int(model(torch.tensor([ids[-8:]]))[0, -1].argmax()))
    assert generate(model, [1, 2], 4, temperature=0.0) == ids


def test_prompt_longer_than_context_is_cropped(model):
    out = generate(model, list(range(10)) * 2, 3, temperature=0.0)  # 20 tokens > context 8
    assert len(out) == 23


def test_generation_stops_at_eos(model):
    first = generate(model, [1, 2], 1, temperature=0.0)[-1]
    assert generate(model, [1, 2], 10, temperature=0.0, eos_id=first) == [1, 2, first]


def test_same_seed_gives_the_same_text(model):
    texts = [generate(model, [1], 10, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
    assert texts[0] == texts[1]


def test_generate_restores_training_mode(model):
    model.train()
    generate(model, [1], 2, temperature=0.0)
    assert model.training


def test_empty_prompt_raises(model):
    with pytest.raises(ValueError):
        generate(model, [], 3)
