import torch

from config import ModelConfig
from model import GPT
from tokenizer import CharTokenizer
from training.trainer import configure_optimizer, train_step

SENTENCE = "o gato está na casa"


def test_gpt_memorizes_a_short_sentence():
    # Prova de sanidade: se o modelo não decora uma frase tão curta, há um bug no pipeline.
    torch.manual_seed(0)
    tok = CharTokenizer.from_text(SENTENCE)
    ids = torch.tensor([tok.encode(SENTENCE, add_special_tokens=True)])
    x, y = ids[:, :-1], ids[:, 1:]
    config = ModelConfig(context_length=32, d_model=32, num_heads=4, num_layers=2, dropout=0.0)
    model = GPT(config, tok.vocab_size)
    optimizer = configure_optimizer(model, learning_rate=3e-3, weight_decay=0.0)

    for _ in range(200):
        result = train_step(model, x, y, optimizer, gradient_clip=1.0)

    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=-1)
    assert result.loss < 0.05
    assert torch.equal(predictions, y)
