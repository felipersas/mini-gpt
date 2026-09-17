"""Experimento: a cross-entropy, o seu gradiente e réguas para interpretar a loss.

Rode com:  uv run python -m experiments.e10_loss
"""

import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.gpt import GPT
from tokenizer import CharTokenizer
from training.loss import cross_entropy

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def fmt(vector: torch.Tensor, spec: str = "+.3f") -> str:
    return " ".join(f"{v:{spec}}" for v in vector.tolist())


def load_data(config: Config) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, config.data.val_fraction)
    return tok, train_ids, val_ids


def cost_of_one_prediction() -> None:
    section("1. O custo de uma previsão: −log p(token correto)")
    print(f"  {'p(correto)':>10} | {'−ln p (nats)':>12} | {'−log₂ p (bits)':>14}")
    for p in (1.0, 0.9, 0.5, 0.25, 0.1, 1 / 97, 0.001, 1e-6):
        print(f"  {p:>10.4g} | {0.0 - math.log(p):>12.3f} | {0.0 - math.log2(p):>14.3f}")


def step_by_step() -> None:
    section("2. Passo a passo numa posição, com um vocabulário de 4 tokens")
    tokens = "abcd"
    z = torch.tensor([2.0, 1.0, 0.1, -1.0])
    target = 2  # o próximo token correto é 'c'
    exp_z = torch.exp(z)
    probs = exp_z / exp_z.sum()
    lse = torch.logsumexp(z, dim=0)
    log_probs = z - lse
    loss = -log_probs[target]

    print(f"tokens:              {'      '.join(tokens)}")
    print(f"logits z:            {fmt(z)}")
    print(f"exp(z):              {fmt(exp_z)}   soma = {exp_z.sum():.3f}")
    print(f"p = exp(z) / soma:   {fmt(probs)}   soma = {probs.sum():.3f}")
    print(f"logsumexp(z) = ln {exp_z.sum():.3f} = {lse:.3f}")
    print(f"log p = z − {lse:.3f}:  {fmt(log_probs)}")
    print(f"alvo = {tokens[target]!r}: loss = −log p[{target}] = {loss:.3f}")
    reference = F.cross_entropy(z[None], torch.tensor([target]))
    print(
        f"F.cross_entropy: {reference:.3f}; cross_entropy do projeto: "
        f"{cross_entropy(z[None], torch.tensor([target])):.3f}"
    )

    print("\na loss se o alvo fosse cada um dos tokens:")
    for i, token in enumerate(tokens):
        print(f"  alvo {token!r}: p = {probs[i]:.3f}  loss = {-log_probs[i]:.3f}")

    print("\nnum batch, a loss é a média das posições:")
    losses = [-log_probs[i] for i in (0, 2, 3)]
    mean = sum(losses) / len(losses)
    print(f"  alvos 'a', 'c', 'd' -> ({fmt(torch.stack(losses), '.3f')}) / 3 = {mean:.3f}")


def numerical_stability() -> None:
    section("3. Por que logsumexp: logits grandes")
    z = torch.tensor([1000.0, 0.0, -1000.0])
    print(f"logits: {fmt(z, '.0f')}")
    print(f"  exp(z) / soma(exp(z)):         {fmt(torch.exp(z) / torch.exp(z).sum(), '.3f')}")
    print(f"  log(softmax(z)):               {fmt(torch.log(torch.softmax(z, dim=0)), '.1f')}")
    print(f"  z − logsumexp(z) (o projeto):  {fmt(z - torch.logsumexp(z, dim=0), '.1f')}")
    print(f"  maior float32 finito: {torch.finfo(torch.float32).max:.3e}; exp(1000) não cabe")


def gradient(config: Config, tok: CharTokenizer, train_ids: torch.Tensor) -> None:
    section("4. O gradiente nos logits é (p − one_hot(alvo)) / N")
    vocab_size = tok.vocab_size
    loader = create_dataloader(
        train_ids,
        context_length=config.model.context_length,
        stride=config.data.stride,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    x, y = next(iter(loader))
    torch.manual_seed(0)
    model = GPT(config.model, vocab_size)
    logits = model(x)  # [B, T, V]
    logits.retain_grad()
    loss = cross_entropy(logits, y)
    loss.backward()

    n = y.numel()
    probs = torch.softmax(logits, dim=-1).detach()
    expected = (probs - F.one_hot(y, vocab_size)) / n
    print(f"batch {tuple(x.shape)}: loss = {loss:.4f}; N = B·T = {n:,} posições")
    matches = torch.allclose(logits.grad, expected, atol=1e-9)
    print(f"logits.grad == (softmax − one_hot) / N: {matches}")

    b, t = 0, 10
    g = logits.grad[b, t] * n  # desfaz a divisão por N: gradiente "de uma posição"
    p = probs[b, t]
    target = int(y[b, t])
    context = tok.decode(x[b, : t + 1])
    print(
        f"\nposição {t} da sequência {b}: contexto {context[-20:]!r}, "
        f"alvo {tok.vocab.get_token(target)!r}"
    )
    print(f"  alvo:  p = {p[target]:.4f}  gradiente = p − 1 = {g[target]:+.4f}  (sobe)")
    others = [i for i in p.argsort(descending=True).tolist() if i != target][:3]
    for i in others:
        name = repr(tok.vocab.get_token(i))
        print(f"  {name:>5}: p = {p[i]:.4f}  gradiente = p     = {g[i]:+.4f}  (desce)")
    print(f"  soma do gradiente nesta posição: {g.sum():+.1e}")


def ngram_loss(
    train_ids: torch.Tensor, val_ids: torch.Tensor, vocab_size: int, order: int, alpha: float
) -> float:
    """Loss na validação de um modelo de contagem que olha `order − 1` caracteres para trás."""
    context = order - 1

    def codes(ids: torch.Tensor) -> torch.Tensor:
        # Um número único para cada (contexto, próximo): c₁·V² + c₂·V + próximo, por exemplo.
        n = len(ids) - context
        code = torch.zeros(n, dtype=torch.long)
        for k in range(order):
            code = code * vocab_size + ids[k : k + n]
        return code

    counts = torch.bincount(codes(train_ids), minlength=vocab_size**order).float()
    counts = counts.view(-1, vocab_size)  # [V^contexto, V]: uma linha por contexto
    # Suavização add-α: nenhum caractere fica com probabilidade zero.
    probs = (counts + alpha) / (counts.sum(dim=1, keepdim=True) + alpha * vocab_size)
    return -probs.view(-1).log()[codes(val_ids)].mean().item()


def evaluate(model: GPT, val_ids: torch.Tensor, seq_len: int) -> float:
    loader = create_dataloader(
        val_ids, context_length=seq_len, stride=seq_len, batch_size=32, shuffle=False
    )
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            total += cross_entropy(model(xb), yb).item() * yb.numel()
            count += yb.numel()
    return total / count


def rulers(
    config: Config, tok: CharTokenizer, train_ids: torch.Tensor, val_ids: torch.Tensor
) -> None:
    section("5. Réguas para interpretar a loss (medidas na validação)")
    vocab_size, seq_len = tok.vocab_size, config.model.context_length

    print("modelos de contagem (n-gramas) com suavização add-α, loss por α:")
    print(f"  {'':<30} {'α = 1':>8} {'α = 0,1':>8} {'α = 0,01':>9}")
    best = {}
    names = {
        1: "unigrama (nenhum contexto)",
        2: "bigrama (1 caractere antes)",
        3: "trigrama (2 caracteres antes)",
    }
    for order in (1, 2, 3):
        values = [ngram_loss(train_ids, val_ids, vocab_size, order, a) for a in (1.0, 0.1, 0.01)]
        best[order] = min(values)
        print(f"  {names[order]:<30} " + " ".join(f"{v:>8.3f}" for v in values))

    torch.manual_seed(0)
    model = GPT(config.model, vocab_size).eval()
    at_init = evaluate(model, val_ids, seq_len)

    steps = 150
    torch.manual_seed(0)
    model = GPT(replace(config.model, dropout=0.0), vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = create_dataloader(
        train_ids, context_length=seq_len, stride=seq_len, batch_size=16, shuffle=True, seed=0
    )
    for step, (xb, yb) in enumerate(loader):
        if step == steps:
            break
        loss = cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after_training = evaluate(model.eval(), val_ids, seq_len)

    print("\nmelhor α de cada n-grama, comparado com outras referências:")
    print(f"  {'régua':<36} {'loss (nats)':>11} {'perplexidade':>13} {'bits/token':>15}")
    rows = [
        (f"chute uniforme (ln {vocab_size})", math.log(vocab_size)),
        ("GPT recém-inicializado", at_init),
        ("unigrama", best[1]),
        ("bigrama", best[2]),
        (f"GPT após {steps} passos (prévia)", after_training),
        ("trigrama", best[3]),
    ]
    for name, value in rows:
        print(f"  {name:<36} {value:>11.3f} {math.exp(value):>13.2f} {value / math.log(2):>15.3f}")


def cost() -> None:
    section("6. Custo: cross_entropy do projeto × F.cross_entropy (forward + backward)")
    torch.manual_seed(0)
    logits = torch.randn(32, 128, 97, requires_grad=True)
    targets = torch.randint(0, 97, (32, 128))
    variants = [
        ("cross_entropy do projeto", lambda: cross_entropy(logits, targets)),
        ("F.cross_entropy", lambda: F.cross_entropy(logits.view(-1, 97), targets.view(-1))),
    ]
    for name, compute in variants:
        for _ in range(10):
            compute().backward()
        start = time.perf_counter()
        for _ in range(100):
            compute().backward()
        ms = (time.perf_counter() - start) / 100 * 1000
        print(f"  {name:<26} {ms:.3f} ms por chamada, entrada [32, 128, 97]")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    tok, train_ids, val_ids = load_data(config)
    cost_of_one_prediction()
    step_by_step()
    numerical_stability()
    gradient(config, tok, train_ids)
    rulers(config, tok, train_ids, val_ids)
    cost()


if __name__ == "__main__":
    main()
