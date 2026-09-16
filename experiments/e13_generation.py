"""Experimento: do modelo treinado ao texto, com greedy, temperature, top-k e top-p.

Pré-requisito: uv run python train.py   (salva checkpoints/latest.pt)
Rode com:      uv run python -m experiments.e13_generation
"""

import math
import time
from pathlib import Path

import torch

from config import load_config
from generation.generate import generate, top_k_filter, top_p_filter
from model.gpt import GPT
from tokenizer import CharTokenizer
from training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
PROMPT = "Capitú"
LENGTH = 200


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


@torch.no_grad()
def next_token_logits(model: GPT, tok: CharTokenizer, context: str) -> torch.Tensor:
    return model(torch.tensor([tok.encode(context)]))[0, -1]  # [vocab_size]


def top(tok: CharTokenizer, probs: torch.Tensor, k: int = 6) -> str:
    values, indices = probs.topk(k)
    pairs = zip(values.tolist(), indices.tolist(), strict=True)
    return "  ".join(f"{tok.vocab.get_token(i)!r} {p:.3f}" for p, i in pairs)


def effective_choices(probs: torch.Tensor) -> float:
    """e^H, com H = −Σ p ln p: com k opções igualmente prováveis, dá exatamente k."""
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return math.exp(entropy.item())


def longest_copied(text: str, corpus: str) -> int:
    """Tamanho do maior trecho do texto que aparece literalmente no corpus."""
    best = 0
    for start in range(len(text)):
        length = best + 1
        while start + length <= len(text) and text[start : start + length] in corpus:
            best = length
            length += 1
    return best


def stats(text: str, corpus: str) -> str:
    words = text.split()
    distinct = len(set(words)) / max(1, len(words))
    return (
        f"palavras distintas {distinct:.0%} | maior trecho idêntico ao livro:"
        f" {longest_copied(text, corpus)} caracteres"
    )


def sample(model: GPT, tok: CharTokenizer, prompt: str, seed: int = 0, **options) -> str:
    generator = torch.Generator().manual_seed(seed)
    ids = generate(model, tok.encode(prompt), LENGTH, generator=generator, **options)
    return tok.decode(ids)


def one_step(model: GPT, tok: CharTokenizer) -> None:
    section(f"1. Um passo de geração: o que vem depois de {PROMPT!r}?")
    ids = tok.encode(PROMPT)
    logits = next_token_logits(model, tok, PROMPT)
    probs = torch.softmax(logits, dim=-1)
    print(f"prompt {PROMPT!r} -> IDs {ids} -> logits {tuple(logits.shape)} na última posição")
    print(f"os mais prováveis: {top(tok, probs, 8)}")
    print(f"opções efetivas (e^entropia): {effective_choices(probs):.2f} de {tok.vocab_size}")


def greedy(model: GPT, tok: CharTokenizer, corpus: str) -> None:
    section("2. Greedy: sempre o mais provável")
    text = sample(model, tok, PROMPT, temperature=0.0)
    print(repr(text))
    print(stats(text, corpus))


def temperature(model: GPT, tok: CharTokenizer, corpus: str) -> None:
    section("3. Temperature")
    context = "olhos de "
    logits = next_token_logits(model, tok, context)
    print(f"distribuição do próximo caractere depois de {context!r}:")
    for value in (0.2, 0.7, 1.0, 1.5):
        probs = torch.softmax(logits / value, dim=-1)
        print(
            f"  T = {value}: opções efetivas {effective_choices(probs):5.2f} | {top(tok, probs, 5)}"
        )

    print(f"\n{LENGTH} caracteres a partir de {PROMPT!r} (mesma seed):")
    for value in (0.2, 0.7, 1.0, 1.5):
        text = sample(model, tok, PROMPT, temperature=value)
        print(f"\n  T = {value}: {text!r}")
        print(f"  {stats(text, corpus)}")


def top_k(model: GPT, tok: CharTokenizer, corpus: str) -> None:
    section("4. Top-k: só os k mais prováveis (temperature 1,0)")
    logits = next_token_logits(model, tok, "olhos de ")
    probs = torch.softmax(logits, dim=-1)
    for k in (1, 5, 20):
        kept = torch.softmax(top_k_filter(logits, k), dim=-1) > 0
        text = sample(model, tok, PROMPT, top_k=k)
        print(
            f"\n  k = {k}: depois de 'olhos de ', mantém {kept.sum()} tokens com"
            f" {probs[kept].sum():.1%} da probabilidade"
        )
        print(f"  {text!r}")
        print(f"  {stats(text, corpus)}")


def top_p(model: GPT, tok: CharTokenizer, corpus: str) -> None:
    section("5. Top-p: o número de candidatos se adapta à confiança do modelo")
    contexts = ["qu", "olhos de ", "Capitú e "]
    print(f"  {'':<8}" + "".join(f"{repr(c):>14}" for c in contexts) + "   (tokens mantidos)")
    for p in (0.5, 0.9, 0.95):
        counts = []
        for context in contexts:
            filtered = top_p_filter(next_token_logits(model, tok, context), p)
            counts.append(int(torch.isfinite(filtered).sum()))
        print(f"  p = {p:<4}" + "".join(f"{n:>14}" for n in counts))
    text = sample(model, tok, PROMPT, top_p=0.9)
    print(f"\n  temperature 1,0 e p = 0,9: {text!r}")
    print(f"  {stats(text, corpus)}")


def combined(model: GPT, tok: CharTokenizer, corpus: str) -> None:
    section("6. Juntando: temperature 0,8 + top-p 0,95 (padrão do inference.py)")
    for seed, prompt in enumerate(["o gato", "Capitú", "Escobar"]):
        text = sample(model, tok, prompt, seed=seed, temperature=0.8, top_p=0.95)
        print(f"\n  {text!r}")
        print(f"  {stats(text, corpus)}")


def cost(model: GPT, tok: CharTokenizer) -> None:
    section("7. Custo: cada caractere novo é um forward completo")
    for new_tokens in (64, 256):
        start = time.perf_counter()
        generate(model, tok.encode(PROMPT), new_tokens, temperature=0.0)
        seconds = time.perf_counter() - start
        speed = new_tokens / seconds
        print(f"  {new_tokens:>3} caracteres: {seconds:.2f} s ({speed:.0f} caracteres/s)")
    print(f"  o contexto cresce até {model.config.context_length} tokens e depois é cortado")


def main() -> None:
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint não encontrado: {CHECKPOINT}\nRode: uv run python train.py")
    model, tok = load_checkpoint(CHECKPOINT)
    config = load_config(ROOT / "configs" / "tiny.yaml")
    corpus = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    print(f"modelo carregado de {CHECKPOINT.name}: {model.num_parameters():,} parâmetros")

    one_step(model, tok)
    greedy(model, tok, corpus)
    temperature(model, tok, corpus)
    top_k(model, tok, corpus)
    top_p(model, tok, corpus)
    combined(model, tok, corpus)
    cost(model, tok)


if __name__ == "__main__":
    main()
