"""Experimento: do corpus aos batches (x, y) usados no treinamento.

Rode com:  uv run python -m experiments.e02_dataset
"""

from collections import Counter
from pathlib import Path

import torch

from config import load_config
from data.dataset import NextTokenDataset
from data.loader import create_dataloader, split_train_val
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    context_length = config.model.context_length
    batch_size = config.training.batch_size

    section("Corpus")
    print(f"caracteres: {len(text):,}")
    print(f"parágrafos: {text.count(chr(10) * 2) + 1:,}")
    print(f"vocab_size: {tok.vocab_size}")
    counts = Counter(text)
    print("mais frequentes:", "  ".join(f"{c!r} {n:,}" for c, n in counts.most_common(8)))
    rare = sorted((n, c) for c, n in counts.items() if n < 10)
    print("raros (< 10x):  ", "  ".join(f"{c!r} {n}" for n, c in rare))

    section(f"Uma janela = {context_length} exemplos de previsão")
    x, y = NextTokenDataset(ids, context_length, stride=context_length)[0]
    print(f"x = {tok.decode(x)[:60]!r}...")
    print(f"y = {tok.decode(y)[:60]!r}...")
    for t in range(14):
        context = tok.decode(x[: t + 1])
        print(f"  t={t:<2} {context!r:>18} -> {tok.decode(y[t : t + 1])!r}")

    section("Treino × validação")
    train_ids, val_ids = split_train_val(ids, config.data.val_fraction)
    print(f"treino:    {len(train_ids):,} tokens")
    print(f"validação: {len(val_ids):,} tokens")
    print(f"fim do treino:       ...{tok.decode(train_ids[-50:])!r}")
    print(f"início da validação:    {tok.decode(val_ids[:50])!r}...")

    section(f"Efeito do stride (treino, context_length={context_length}, batch={batch_size})")
    print(f"{'stride':>6} | {'janelas':>8} | {'batches/época':>13} | {'alvos por época':>15}")
    for stride in (1, 16, 64, context_length):
        windows = len(NextTokenDataset(train_ids, context_length, stride))
        print(
            f"{stride:>6} | {windows:>8,} | {windows // batch_size:>13,}"
            f" | {windows * context_length:>15,}"
        )

    section("Batches")
    loader = create_dataloader(
        train_ids,
        context_length=context_length,
        stride=config.data.stride,
        batch_size=batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    x, y = next(iter(loader))
    print(f"x: {tuple(x.shape)} {x.dtype}")
    print(f"y: {tuple(y.shape)} {y.dtype}")
    print(f"batches por época: {len(loader)} | tokens por batch: {x.numel():,}")
    for row in range(3):
        print(f"  x[{row}] = {tok.decode(x[row])[:70]!r}...")


if __name__ == "__main__":
    main()
