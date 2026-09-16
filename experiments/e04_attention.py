"""Experimento: a scaled dot-product attention passo a passo.

Rode com:  uv run python -m experiments.e04_attention
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.attention import (
    CausalSelfAttention,
    attention_weights,
    causal_mask,
    scaled_dot_product_attention,
)
from model.embeddings import Embeddings
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def show(name: str, matrix: torch.Tensor, labels: str) -> None:
    """Imprime uma matriz 2-D, uma linha por caractere de `labels`."""
    print(f"\n{name}  {tuple(matrix.shape)}")
    for label, row in zip(labels, matrix.tolist(), strict=True):
        cells = " ".join(f"{v:+7.3f}" if math.isfinite(v) else f"{v:>7}" for v in row)
        print(f"  {label!r:>5} {cells}")


def step_by_step() -> None:
    section("1. Passo a passo: 'gato', d_model = 6, d_k = 3")
    torch.manual_seed(0)
    chars, d_model, d_k = "gato", 6, 3
    seq_len = len(chars)

    # X faz o papel dos embeddings: um vetor por posição.
    X = torch.randn(seq_len, d_model)
    # Escala 1/√d_model para que Q e K tenham entradas com variância ~1.
    W_q, W_k, W_v = (torch.randn(d_model, d_k) / math.sqrt(d_model) for _ in range(3))

    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    show("X: entrada", X, chars)
    show("Q = X W_q: o que cada posição procura", Q, chars)
    show("K = X W_k: como cada posição se anuncia", K, chars)
    show("V = X W_v: o que cada posição entrega", V, chars)

    scores = Q @ K.T
    show("scores = Q Kᵀ: linha i, coluna j = afinidade de i por j", scores, chars)
    scaled = scores / math.sqrt(d_k)
    show(f"scores / √d_k   (√{d_k} = {math.sqrt(d_k):.3f})", scaled, chars)
    masked = scaled.masked_fill(~causal_mask(seq_len), float("-inf"))
    show("máscara causal: -inf onde j > i", masked, chars)
    weights = torch.softmax(masked, dim=-1)
    show("pesos = softmax de cada linha", weights, chars)
    print("  soma de cada linha:", [round(s, 6) for s in weights.sum(dim=-1).tolist()])
    output = weights @ V
    show("saída = pesos @ V", output, chars)

    i = 2
    terms = " + ".join(f"{weights[i, j]:.3f}·V[{chars[j]!r}]" for j in range(i + 1))
    print(f"\n  saída[{chars[i]!r}] = {terms}")

    reference, _ = scaled_dot_product_attention(Q, K, V, causal_mask(seq_len))
    pytorch = F.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=True)[0]
    print(f"  igual à função do projeto:          {torch.allclose(output, reference)}")
    print(f"  igual a F.scaled_dot_product_attention: {torch.allclose(output, pytorch, atol=1e-6)}")


def why_scale() -> None:
    section("2. Por que dividir por √d_k")
    torch.manual_seed(0)
    seq_len = 16
    print(f"q, k ~ N(0, 1); {seq_len} posições visíveis por linha; peso uniforme = {1 / seq_len}")
    print("1 − Σp² mede quanto a softmax ainda responde a mudanças nos scores (0 = saturada)\n")
    header = f"{'d_k':>5} | {'var(q·k)':>8} | {'var(q·k/√d_k)':>13} | "
    print(header + f"{'maior peso':^19} | {'1 − Σp²':^17}")
    print(f"{'':>5} | {'':>8} | {'':>13} | {'sem escala':>10} {'com':>8} | {'sem':>8} {'com':>8}")
    for d_k in (4, 16, 64, 256, 1024):
        q, k = torch.randn(256, seq_len, d_k), torch.randn(256, seq_len, d_k)
        raw = q @ k.transpose(-2, -1)  # [256, seq_len, seq_len]
        scaled = raw / math.sqrt(d_k)
        p_raw, p_scaled = torch.softmax(raw, dim=-1), torch.softmax(scaled, dim=-1)
        max_raw, max_scaled = p_raw.amax(-1).mean(), p_scaled.amax(-1).mean()
        sens_raw = (1 - p_raw.pow(2).sum(-1)).mean()
        sens_scaled = (1 - p_scaled.pow(2).sum(-1)).mean()
        print(
            f"{d_k:>5} | {raw.var():>8.1f} | {scaled.var():>13.3f} | "
            f"{max_raw:>10.3f} {max_scaled:>8.3f} | {sens_raw:>8.3f} {sens_scaled:>8.3f}"
        )


def causality() -> None:
    section("3. Causal mask: mudar o futuro não muda o passado")
    torch.manual_seed(0)
    text_a, text_b = "o gato dorme", "o gato corre"
    tok = CharTokenizer.from_text(text_a + text_b)
    emb = Embeddings(tok.vocab_size, context_length=len(text_a), d_model=16, dropout=0.0)
    attn = CausalSelfAttention(d_model=16, head_dim=8, dropout=0.0)
    # Escalas maiores que as da inicialização, para os pesos não saírem quase uniformes
    # (ver seção 5) e o efeito de cada posição ficar visível.
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        torch.nn.init.normal_(proj.weight, std=16**-0.5)
    xa = F.layer_norm(emb(torch.tensor([tok.encode(text_a)])), (16,))
    xb = F.layer_norm(emb(torch.tensor([tok.encode(text_b)])), (16,))

    def without_mask(x: torch.Tensor) -> torch.Tensor:
        out, _ = scaled_dot_product_attention(attn.q_proj(x), attn.k_proj(x), attn.v_proj(x))
        return out

    diff_masked = (attn(xa) - attn(xb)).abs().amax(dim=-1)[0]
    diff_unmasked = (without_mask(xa) - without_mask(xb)).abs().amax(dim=-1)[0]

    print(f"A = {text_a!r}, B = {text_b!r}")
    print("maior diferença |saída_A − saída_B| em cada posição:\n")
    print(f"  {'t':>2}  {'A':>3} {'B':>3}  {'com máscara':>12}  {'sem máscara':>12}")
    for t, (a, b) in enumerate(zip(text_a, text_b, strict=True)):
        mark = "   <- entrada diferente" if a != b else ""
        print(
            f"  {t:>2}  {a!r:>3} {b!r:>3}  {diff_masked[t]:>12.2e}  {diff_unmasked[t]:>12.2e}{mark}"
        )


def previous_token_head() -> None:
    section("4. Uma head construída à mão: 'copie o caractere anterior'")
    text = "o gato"
    chars = sorted(set(text))
    vocab_size, seq_len = len(chars), len(text)
    token_ids = torch.tensor([chars.index(c) for c in text])

    # Entrada = [one-hot do caractere | one-hot da posição]
    X = torch.cat([F.one_hot(token_ids, vocab_size).float(), torch.eye(seq_len)], dim=1)
    d_model, d_k = X.shape[1], seq_len
    print(
        f"vocabulário {chars}; X: {tuple(X.shape)} = [{vocab_size} dims de caractere | "
        f"{seq_len} dims de posição]"
    )

    strength = 8 * math.sqrt(d_k)  # após dividir por √d_k, o score do alvo vale 8
    W_q = torch.zeros(d_model, d_k)
    W_k = torch.zeros(d_model, d_k)
    for p in range(seq_len):
        W_k[vocab_size + p, p] = 1.0  # key:   "estou na posição p"
        if p > 0:
            W_q[vocab_size + p, p - 1] = strength  # query: "procuro a posição p − 1"
    W_v = torch.zeros(d_model, vocab_size)
    W_v[:vocab_size] = torch.eye(vocab_size)  # value: "carrego o meu caractere"

    output, weights = scaled_dot_product_attention(X @ W_q, X @ W_k, X @ W_v, causal_mask(seq_len))
    show("pesos", weights, text)
    print(f"\n  {'t':>2} {'char':>5}   {'olha para':<10} {'peso':>5}   copia")
    for t in range(seq_len):
        j = int(weights[t].argmax())
        copied = chars[int(output[t].argmax())]
        print(f"  {t:>2} {text[t]!r:>5}   t={j} {text[j]!r:<6} {weights[t, j]:>5.3f}   {copied!r}")


def at_initialization(config: Config) -> None:
    section("5. Um módulo recém-criado, num batch real do corpus")
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, _ = split_train_val(ids, config.data.val_fraction)
    context_length, d_model = config.model.context_length, config.model.d_model
    batch_size = config.training.batch_size
    loader = create_dataloader(
        train_ids,
        context_length=context_length,
        stride=config.data.stride,
        batch_size=batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    x, _ = next(iter(loader))

    torch.manual_seed(0)
    emb = Embeddings(tok.vocab_size, context_length, d_model, dropout=0.0)
    attn = CausalSelfAttention(d_model, config.model.head_dim, dropout=0.0)
    h = emb(x)
    print(f"x {tuple(x.shape)} -> embeddings {tuple(h.shape)} -> attention {tuple(attn(h).shape)}")

    mask = causal_mask(context_length)
    visible = torch.arange(1, context_length + 1)  # posições visíveis em cada linha
    inputs = {
        "embeddings (norma ~0,3)": h,
        "embeddings normalizados (como após LayerNorm)": F.layer_norm(h, (d_model,)),
    }
    print(
        f"\nentropia / entropia máxima (1 = uniforme); peso uniforme na última linha = "
        f"{1 / context_length:.4f}"
    )
    for name, inp in inputs.items():
        w = attention_weights(attn.q_proj(inp), attn.k_proj(inp), mask)  # [B, T, T]
        entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=-1)  # [B, T]
        ratio = (entropy[:, 1:] / visible[1:].log()).mean()
        last_row_max = w[:, -1].amax(dim=-1).mean()
        print(f"  {name:<46} entropia {ratio:.4f} | maior peso na última linha {last_row_max:.4f}")

    section("6. Custo: a matriz de pesos tem T × T entradas")
    for seq_len in (128, 512, 2048, 8192):
        elements = batch_size * seq_len * seq_len
        mib = elements * 4 / 2**20
        shape = f"[B={batch_size}, T, T]"
        print(f"  T = {seq_len:>5}: {shape} = {elements:>14,} floats = {mib:>9,.1f} MiB")
    print("  (float32, uma head, uma camada)")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    with torch.no_grad():
        step_by_step()
        why_scale()
        causality()
        previous_token_head()
        at_initialization(config)


if __name__ == "__main__":
    main()
