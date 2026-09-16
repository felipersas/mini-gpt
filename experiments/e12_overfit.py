"""Experimento: decorar de propósito, para provar que o treino funciona.

Rode com:  uv run python -m experiments.e12_overfit
"""

import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch

import model.attention as attention_module
from config import Config, ModelConfig, load_config
from data.loader import create_dataloader
from model.gpt import GPT
from tokenizer import CharTokenizer
from training.loss import cross_entropy
from training.trainer import configure_optimizer, train_step

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "o gato está na casa"
OTHER = "a gata está na casa"  # mesmos caracteres, outra frase
STEPS, LOG_EVERY = 300, 25


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def pairs(tok: CharTokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """<BOS> + texto + <EOS>, recortado em entrada x e alvo y deslocado: [1, n + 1] cada."""
    ids = torch.tensor([tok.encode(text, add_special_tokens=True)])
    return ids[:, :-1], ids[:, 1:]


def name(tok: CharTokenizer, token_id: int) -> str:
    token = tok.vocab.get_token(token_id)
    return token if token.startswith("<") else repr(token)


def see_everything(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Máscara sem nenhuma restrição: toda posição vê todas, inclusive as futuras."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)


@torch.no_grad()
def measure(model: GPT, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Loss e acurácia sem dropout: a acurácia é a fração de posições com argmax == alvo."""
    was_training = model.training
    model.eval()
    logits = model(x)
    loss = cross_entropy(logits, y).item()
    accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
    model.train(was_training)
    return loss, accuracy


@torch.no_grad()
def greedy_from_bos(model: GPT, tok: CharTokenizer, max_new_tokens: int = 40) -> str:
    """Prévia da geração: a partir de <BOS>, escolhe sempre o token mais provável até <EOS>."""
    model.eval()
    ids = [tok.vocab.bos_id]
    for _ in range(max_new_tokens):
        next_id = int(model(torch.tensor([ids]))[0, -1].argmax())
        if next_id == tok.vocab.eos_id:
            break
        ids.append(next_id)
    return tok.decode(ids)


@torch.no_grad()
def greedy_continue(
    model: GPT, tok: CharTokenizer, prompt: str, new_tokens: int, window: int
) -> str:
    """Continua o prompt escolhendo sempre o mais provável, olhando os últimos `window` tokens."""
    model.eval()
    ids = tok.encode(prompt)
    for _ in range(new_tokens):
        ids.append(int(model(torch.tensor([ids[-window:]]))[0, -1].argmax()))
    return tok.decode(ids)


def the_dataset(tok: CharTokenizer) -> None:
    section("1. O dataset: uma única frase")
    x, y = pairs(tok, SENTENCE)
    print(f"frase: {SENTENCE!r} ({len(SENTENCE)} caracteres)")
    print(f"vocabulário: {tok.vocab_size} tokens ({tok.vocab_size - 4} caracteres + 4 especiais)")
    print(f"x: {tuple(x.shape)}   y: {tuple(y.shape)}   -> {y.numel()} previsões para decorar\n")
    print(f"  {'t':>2}  {'entrada x[t]':<13} {'alvo y[t]':<9}")
    for t in range(y.shape[1]):
        print(f"  {t:>2}  {name(tok, int(x[0, t])):<13} {name(tok, int(y[0, t])):<9}")


def bigram_floor(tok: CharTokenizer) -> None:
    section("2. Olhar só o caractere anterior não basta")
    x, y = pairs(tok, SENTENCE)
    successors: dict[int, Counter] = defaultdict(Counter)
    for a, b in zip(x[0].tolist(), y[0].tolist(), strict=True):
        successors[a][b] += 1
    print("caracteres seguidos por mais de um caractere diferente:")
    for a, counts in successors.items():
        if len(counts) > 1:
            options = ", ".join(f"{name(tok, b)} ({n}×)" for b, n in counts.items())
            print(f"  depois de {name(tok, a):<5} vem: {options}")
    # O melhor que um modelo de "caractere anterior" faz é usar as frequências da própria frase.
    losses = [
        -math.log(successors[a][b] / sum(successors[a].values()))
        for a, b in zip(x[0].tolist(), y[0].tolist(), strict=True)
    ]
    floor = sum(losses) / len(losses)
    print(f"\nmenor loss possível olhando só o caractere anterior: {floor:.3f} nats")
    print("para chegar a ~0, o modelo precisa de mais informação: a posição ou o contexto")


def train_on_sentence(tok: CharTokenizer, config: ModelConfig) -> GPT:
    section("3. Treinando na frase, repetidamente")
    x, y = pairs(tok, SENTENCE)
    x_other, y_other = pairs(tok, OTHER)
    torch.manual_seed(0)
    model = GPT(config, tok.vocab_size)
    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    print(f"{model.num_parameters():,} parâmetros, AdamW lr 1e-3, sem dropout e sem weight decay\n")
    other = "loss em " + repr(OTHER)
    print(f"  {'passo':>5} | {'loss (frase)':>12} | {'acurácia':>8} | {'grad':>5} | {other:>28}")
    for step in range(1, STEPS + 1):
        result = train_step(model, x, y, optimizer, gradient_clip=1.0)
        if step == 1 or step % LOG_EVERY == 0:
            loss, accuracy = measure(model, x, y)
            other_loss, _ = measure(model, x_other, y_other)
            print(
                f"  {step:>5} | {loss:>12.4f} | {accuracy:>8.0%} | {result.grad_norm:>5.2f}"
                f" | {other_loss:>28.3f}"
            )
    return model


def memorized(model: GPT, tok: CharTokenizer) -> None:
    section("4. O modelo decorou a frase")
    x, y = pairs(tok, SENTENCE)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0]  # [n, V]
    print(f"  {'t':>2}  {'contexto':<22} {'alvo':<7} {'previsto':<9} {'p(alvo)':>8}")
    for t in range(y.shape[1]):
        context = tok.decode(x[0, : t + 1])
        target, predicted = int(y[0, t]), int(probs[t].argmax())
        print(
            f"  {t:>2}  {context!r:<22} {name(tok, target):<7} {name(tok, predicted):<9}"
            f" {probs[t, target]:>8.4f}"
        )
    text = greedy_from_bos(model, tok)
    print(f"\ngeração a partir de <BOS>, sempre o mais provável: {text!r}")
    print(f"igual à frase: {text == SENTENCE}")


def position_shortcut(model: GPT, tok: CharTokenizer) -> None:
    section("5. O atalho: o modelo decorou posições, não o texto")
    _, y = pairs(tok, SENTENCE)
    x_other, y_other = pairs(tok, OTHER)
    with torch.no_grad():
        predicted = model(x_other).argmax(dim=-1)  # [1, n]
    print(f"entrada (outra frase): {tok.decode(x_other[0])!r}")
    print(f"previsões do modelo:   {tok.decode(predicted[0])!r}")
    memorized_rate = (predicted == y).float().mean()
    right_rate = (predicted == y_other).float().mean()
    print(f"previsões iguais aos alvos da frase decorada: {memorized_rate:.0%}")
    print(f"previsões iguais aos alvos certos desta frase: {right_rate:.0%}")
    print("\nposições onde as duas frases pedem caracteres diferentes (ou onde o modelo errou):")
    print(f"  {'t':>2}  {'entrada':<8} {'alvo certo':<11} {'alvo decorado':<14} {'previsto':<8}")
    for t in range(y.shape[1]):
        right, memo, pred = int(y_other[0, t]), int(y[0, t]), int(predicted[0, t])
        if right != memo or pred != right:
            print(
                f"  {t:>2}  {name(tok, int(x_other[0, t])):<8} {name(tok, right):<11}"
                f" {name(tok, memo):<14} {name(tok, pred):<8}"
            )


def without_causal_mask(config: Config, model_config: ModelConfig) -> None:
    steps, window, new_tokens = 400, 32, 80
    section("6. Um bug proposital: decorar um parágrafo sem a causal mask")
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    start = text.index("Uma noite")
    paragraph = text[start : start + 400]
    ids = torch.tensor(tok.encode(paragraph))
    print(f"parágrafo de {len(paragraph)} caracteres; janelas de {window} com stride 1")
    print(
        "(a mesma posição da janela recebe caracteres diferentes: só o contexto identifica o alvo)"
    )

    original = attention_module.causal_mask
    for label, mask_function in [("com máscara", original), ("sem máscara", see_everything)]:
        attention_module.causal_mask = mask_function
        try:
            torch.manual_seed(0)
            model = GPT(model_config, tok.vocab_size)
            optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
            loader = create_dataloader(
                ids, context_length=window, stride=1, batch_size=32, shuffle=True, seed=0
            )
            losses, step = {}, 0
            while step < steps:
                for xb, yb in loader:
                    result = train_step(model, xb, yb, optimizer, gradient_clip=1.0)
                    step += 1
                    if step in (1, 100, 200, steps):
                        losses[step] = result.loss
                    if step == steps:
                        break
            generated = greedy_continue(model, tok, paragraph[:window], new_tokens, window)
        finally:
            attention_module.causal_mask = original
        correct = sum(a == b for a, b in zip(generated[window:], paragraph[window:], strict=False))
        print(
            f"\n  {label}: loss de treino "
            + " | ".join(f"passo {s}: {v:.4f}" for s, v in losses.items())
        )
        print(f"  continuação dos {window} primeiros caracteres: {generated!r}")
        print(f"  dos {new_tokens} caracteres gerados, iguais ao original: {correct}")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    model_config = replace(config.model, dropout=0.0)
    tok = CharTokenizer.from_text(SENTENCE + OTHER)

    the_dataset(tok)
    bigram_floor(tok)
    model = train_on_sentence(tok, model_config)
    memorized(model, tok)
    position_shortcut(model, tok)
    without_causal_mask(config, model_config)


if __name__ == "__main__":
    main()
