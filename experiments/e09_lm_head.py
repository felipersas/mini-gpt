"""Experimento: do estado final aos logits, às probabilidades e ao weight tying.

Rode com:  uv run python -m experiments.e09_lm_head
"""

import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.gpt import GPT
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def load_data(config: Config) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor, torch.Tensor]:
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, _ = split_train_val(ids, config.data.val_fraction)
    loader = create_dataloader(
        train_ids,
        context_length=config.model.context_length,
        stride=config.data.stride,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    x, y = next(iter(loader))
    return tok, train_ids, x, y


def top_tokens(tok: CharTokenizer, probs: torch.Tensor, k: int = 5) -> str:
    values, indices = probs.topk(k)
    pairs = zip(values.tolist(), indices.tolist(), strict=True)
    return "  ".join(f"{tok.vocab.get_token(i)!r} {p:.3f}" for p, i in pairs)


def logits_to_probabilities(config: Config, tok: CharTokenizer, x: torch.Tensor) -> None:
    section("1. Do estado final aos logits, e dos logits às probabilidades")
    torch.manual_seed(0)
    model = GPT(config.model, tok.vocab_size).eval()
    hidden = model.hidden_states(x)  # [B, T, d_model]
    logits = model(x)  # [B, T, vocab_size]

    t = 10
    h, z = hidden[0, t], logits[0, t]  # [d_model], [vocab_size]
    W = model.lm_head.proj.weight  # [vocab_size, d_model]
    current = tok.vocab.get_token(int(x[0, t]))
    print(
        f"contexto até a posição {t}: {tok.decode(x[0, : t + 1])!r}  (caractere atual {current!r})"
    )
    print(f"h = estado final:  {tuple(h.shape)}, norma {h.norm():.2f}")
    print(f"W = lm_head:       {tuple(W.shape)}, linha i = vetor do token i")
    matches = torch.allclose(z, W @ h, atol=1e-5)
    print(f"logits = W · h:    {tuple(z.shape)}, iguais a W @ h: {matches}")
    print(f"  média {z.mean():+.3f}, desvio {z.std():.3f}, mín {z.min():+.3f}, máx {z.max():+.3f}")
    print(f"  logit do caractere atual {current!r}: {z[x[0, t]]:+.3f}")

    probs = torch.softmax(z, dim=-1)
    uniform = 1 / tok.vocab_size
    print(f"\nsoftmax: soma {probs.sum():.6f}, maior {probs.max():.4f}, menor {probs.min():.4f}")
    print(f"  (chute uniforme = 1/{tok.vocab_size} = {uniform:.4f})")
    print(f"  top-5: {top_tokens(tok, probs)}")
    same = torch.allclose(torch.softmax(z + 7.0, dim=-1), probs, atol=1e-6)
    print(f"  softmax(logits + 7) == softmax(logits): {same}")
    same_log = torch.allclose(F.log_softmax(z, dim=-1), probs.log(), atol=1e-5)
    print(f"  log_softmax == log(softmax): {same_log}")

    sums = torch.softmax(logits, dim=-1).sum(dim=-1)  # [B, T]
    print(f"\nbatch inteiro: logits {tuple(logits.shape)}")
    print(
        f"  cada uma das {sums.numel():,} posições tem uma distribuição que soma 1: "
        f"{torch.allclose(sums, torch.ones_like(sums))}"
    )


def initial_loss(config: Config, tok: CharTokenizer, x: torch.Tensor, y: torch.Tensor) -> None:
    section("2. A loss inicial depende da escala dos logits")
    vocab_size = tok.vocab_size
    print(f"chute uniforme: ln {vocab_size} = {math.log(vocab_size):.3f}")
    print("logits ~ N(0, s²), com s pequeno: loss ≈ ln V + s²/2")
    print("loss = −média de log p(token correto)  (a cross-entropy, ver docs/10-loss.md)\n")
    variants = [
        ("LM head std 0,02, sem tying", False, 0.02),
        ("weight tying (padrão)", True, None),
        ("LM head com init padrão do nn.Linear", False, "default"),
        ("LM head std 1,0", False, 1.0),
    ]
    print(f"  {'':<38} {'s':>6} {'ln V + s²/2':>12} {'loss':>7} {'P(caractere atual)':>19}")
    for name, tie, std in variants:
        torch.manual_seed(0)
        model = GPT(replace(config.model, tie_weights=tie), vocab_size).eval()
        if std == "default":
            model.lm_head.proj.reset_parameters()
        elif std is not None:
            nn.init.normal_(model.lm_head.proj.weight, std=std)
        log_probs = F.log_softmax(model(x), dim=-1)  # [B, T, V]
        s = model(x).std(dim=-1).mean()  # desvio entre os V logits, média nas posições
        loss = -log_probs.gather(-1, y.unsqueeze(-1)).mean()
        p_current = log_probs.gather(-1, x.unsqueeze(-1)).exp().mean()
        predicted = f"{math.log(vocab_size) + s.item() ** 2 / 2:.3f}" if s < 1 else "—"
        print(f"  {name:<38} {s:>6.3f} {predicted:>12} {loss:>7.3f} {p_current:>19.4f}")
    print(f"\n  (P uniforme = {1 / vocab_size:.4f})")
    repeats = (x == y).float().mean()
    print(f"  neste batch, o próximo caractere repete o atual em {repeats:.2%} das posições")


def weight_tying(config: Config, tok: CharTokenizer, x: torch.Tensor, y: torch.Tensor) -> None:
    section("3. Weight tying: a mesma matriz na entrada e na saída")
    vocab_size = tok.vocab_size
    print(f"caracteres distintos na entrada deste batch: {len(x.unique())} de {vocab_size}\n")
    for tie in (False, True):
        torch.manual_seed(0)
        model = GPT(replace(config.model, tie_weights=tie), vocab_size)
        emb, head = model.embeddings.token_embedding.weight, model.lm_head.proj.weight
        F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).backward()
        rows = emb.grad.abs().sum(dim=1) > 0
        plus = emb.grad[tok.vocab.get_id("+")].norm()
        label = "com tying" if tie else "sem tying"
        print(f"  {label}: {model.num_parameters():,} parâmetros | mesma matriz: {head is emb}")
        print(f"    linhas do token_embedding com gradiente: {rows.sum()} de {vocab_size}")
        print(f"    norma do gradiente na linha de '+': {plus:.2e}")


def short_training(config: Config, tok: CharTokenizer, train_ids: torch.Tensor) -> GPT:
    steps = 150
    section(
        f"4. Mini-treino com e sem weight tying ({steps} passos, batch [16, 128], AdamW lr 1e-3)"
    )
    vocab_size, seq_len = tok.vocab_size, config.model.context_length
    checkpoints = (0, 50, 100, steps)
    print(f"  {'':<10}" + "".join(f"{f'passo {s}':>11}" for s in checkpoints))
    trained = None
    for tie in (False, True):
        torch.manual_seed(0)
        model = GPT(replace(config.model, tie_weights=tie, dropout=0.0), vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = create_dataloader(
            train_ids, context_length=seq_len, stride=seq_len, batch_size=16, shuffle=True, seed=0
        )
        losses = []
        for step, (xb, yb) in enumerate(loader):
            loss = F.cross_entropy(model(xb).view(-1, vocab_size), yb.view(-1))
            if step in checkpoints:
                losses.append(loss.item())
            if step == steps:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        label = "com tying" if tie else "sem tying"
        print(f"  {label:<10}" + "".join(f"{v:>11.3f}" for v in losses))
        trained = model
    return trained


def predictions(model: GPT, tok: CharTokenizer) -> None:
    section("5. O que o modelo com tying prevê depois do mini-treino")
    model.eval()
    for prompt in ("qu", "Capit", "respond", "Dom Casmurr", "olhos de ressac"):
        probs = torch.softmax(model(torch.tensor([tok.encode(prompt)]))[0, -1], dim=-1)
        print(f"  {prompt!r:<18} -> {top_tokens(tok, probs)}")


def logits_memory() -> None:
    section("6. O tamanho do tensor de logits")
    for name, batch, seq_len, vocab_size in [
        ("Mini-GPT", 32, 128, 97),
        ("GPT-2 small", 8, 1024, 50257),
    ]:
        n = batch * seq_len * vocab_size
        shape = f"[B={batch}, T={seq_len}, V={vocab_size}]"
        print(f"  {name:<12} {shape:<26} {n:>13,} floats = {n * 4 / 2**20:>8,.1f} MiB (float32)")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    tok, train_ids, x, y = load_data(config)
    with torch.no_grad():
        logits_to_probabilities(config, tok, x)
        initial_loss(config, tok, x, y)
    weight_tying(config, tok, x, y)
    trained = short_training(config, tok, train_ids)
    with torch.no_grad():
        predictions(trained, tok)
        logits_memory()


if __name__ == "__main__":
    main()
