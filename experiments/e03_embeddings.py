"""Experimento: como IDs viram vetores e por que a posição importa.

Rode com:  uv run python -m experiments.e03_embeddings
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from config import load_config
from data.loader import create_dataloader, split_train_val
from model.embeddings import Embeddings
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def fmt(vector: torch.Tensor) -> str:
    return " ".join(f"{v:+.3f}" for v in vector.tolist())


def main() -> None:
    torch.manual_seed(0)
    config = load_config(ROOT / "configs" / "tiny.yaml")
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    d_model, context_length = config.model.d_model, config.model.context_length

    emb = Embeddings(tok.vocab_size, context_length, d_model, dropout=0.0)
    token_weight = emb.token_embedding.weight
    position_weight = emb.position_embedding.weight

    section("Parâmetros")
    for name, weight in [("token", token_weight), ("position", position_weight)]:
        print(f"{name}_embedding.weight: {tuple(weight.shape)} = {weight.numel():,} parâmetros")

    section("Lookup = linha da matriz = one-hot × matriz")
    word = "casa"
    word_ids = torch.tensor([tok.encode(word)])  # [1, 4]
    lookup = emb.token_embedding(word_ids)  # [1, 4, d_model]
    one_hot = F.one_hot(word_ids, tok.vocab_size).float()  # [1, 4, vocab_size]
    print(f"IDs de {word!r}: {word_ids[0].tolist()}")
    projected = one_hot @ token_weight  # [1, 4, d_model]
    print(f"one-hot: {tuple(one_hot.shape)} -> one_hot @ weight: {tuple(projected.shape)}")
    print(f"embedding(ids) == weight[ids]:      {torch.equal(lookup, token_weight[word_ids])}")
    print(f"embedding(ids) == one_hot @ weight: {torch.allclose(lookup, one_hot @ token_weight)}")
    print(f"vetor de 'c' (8 de {d_model} dims): {fmt(lookup[0, 0, :8])}")

    section("Mesmo caractere, posições diferentes ('a' em 1 e em 3)")
    out = emb(word_ids)  # [1, 4, d_model]
    cos_token = F.cosine_similarity(lookup[0, 1], lookup[0, 3], dim=0)
    cos_total = F.cosine_similarity(out[0, 1], out[0, 3], dim=0)
    print(f"só token:        cos = {cos_token:.3f}")
    print(f"token + posição: cos = {cos_total:.3f}")

    section("Geometria na inicialização")
    unit = F.normalize(token_weight, dim=1)
    cosines = (unit @ unit.T)[~torch.eye(tok.vocab_size, dtype=torch.bool)]
    print(
        f"cosseno entre tokens distintos: média {cosines.mean():+.3f}, desvio {cosines.std():.3f}"
    )
    print(f"  referência para vetores aleatórios: desvio ≈ 1/√d = {d_model**-0.5:.3f}")
    print(f"norma média de um vetor de token: {token_weight.norm(dim=1).mean():.3f}")
    print(f"  referência: 0.02·√d = {0.02 * d_model**0.5:.3f}")

    section("Batch real")
    train_ids, _ = split_train_val(ids, config.data.val_fraction)
    loader = create_dataloader(
        train_ids,
        context_length=context_length,
        stride=config.data.stride,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    x, _ = next(iter(loader))
    out = emb(x)
    print(f"x {tuple(x.shape)} {x.dtype} -> embeddings {tuple(out.shape)} {out.dtype}")

    section("Gradiente esparso (um batch, loss arbitrária)")
    out.pow(2).mean().backward()
    token_rows = token_weight.grad.abs().sum(dim=1) > 0
    position_rows = position_weight.grad.abs().sum(dim=1) > 0
    print(f"caracteres distintos no batch: {len(x.unique())} de {tok.vocab_size}")
    print(f"linhas de token_embedding com gradiente: {token_rows.sum().item()}")
    missing = [tok.vocab.get_token(i) for i in range(tok.vocab_size) if not token_rows[i]]
    print("sem gradiente:", " ".join(repr(c) for c in missing))
    n_positions = position_rows.sum().item()
    print(f"linhas de position_embedding com gradiente: {n_positions} de {context_length}")

    section(f"Em quantos dos {len(loader)} batches de uma época cada token aparece")
    batches_with_token = torch.zeros(tok.vocab_size, dtype=torch.long)
    for xb, _ in loader:
        batches_with_token[xb.unique()] += 1
    for char in [" ", "a", "x", "?", "É", "W", "+"]:
        print(f"  {char!r:>5}: {batches_with_token[tok.vocab.get_id(char)].item():>3}")


if __name__ == "__main__":
    main()
