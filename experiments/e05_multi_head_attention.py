"""Experimento: por que várias heads e como elas viram um único tensor.

Rode com:  uv run python -m experiments.e05_multi_head_attention
"""

import math
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.attention import (
    LoopMultiHeadAttention,
    MultiHeadAttention,
    attention_weights,
    causal_mask,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)
from model.embeddings import Embeddings
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def copy_heads(loop: LoopMultiHeadAttention, fused: MultiHeadAttention) -> None:
    """Empilha as matrizes de cada head nas projeções da versão com tensor único."""
    for name in ("q_proj", "k_proj", "v_proj"):
        # cada head: [head_dim, d_model]; empilhadas: [num_heads * head_dim, d_model]
        stacked = torch.cat([getattr(head, name).weight for head in loop.heads], dim=0)
        getattr(fused, name).weight.copy_(stacked)
    fused.out_proj.weight.copy_(loop.out_proj.weight)


def one_head_vs_two() -> None:
    section("1. Uma head não consegue olhar para dois lugares ao mesmo tempo")
    chars = sorted(set("gato"))
    vocab_size, seq_len = len(chars), 4
    d_model, d_k = vocab_size + seq_len, seq_len
    strength = 8 * math.sqrt(d_k)

    def features(text: str) -> torch.Tensor:
        ids = torch.tensor([chars.index(c) for c in text])
        # [one-hot do caractere | one-hot da posição]
        return torch.cat([F.one_hot(ids, vocab_size).float(), torch.eye(seq_len)], dim=1)

    def head_output(X: torch.Tensor, offsets: list[int]) -> torch.Tensor:
        """Head construída à mão: procura as posições p − offset e copia o caractere."""
        W_q, W_k = torch.zeros(d_model, d_k), torch.zeros(d_model, d_k)
        for p in range(seq_len):
            W_k[vocab_size + p, p] = 1.0
            for offset in offsets:
                if p >= offset:
                    W_q[vocab_size + p, p - offset] = strength
        W_v = torch.zeros(d_model, vocab_size)
        W_v[:vocab_size] = torch.eye(vocab_size)
        out, _ = scaled_dot_product_attention(X @ W_q, X @ W_k, X @ W_v, causal_mask(seq_len))
        return out[-1]  # só a última posição

    def describe(vector: torch.Tensor) -> str:
        values = vector.tolist()
        return " ".join(f"{chars[i]}={v:.2f}" for i, v in enumerate(values) if v > 0.01)

    single_outputs, double_outputs = [], []
    for text in ("gato", "gtao"):
        X = features(text)
        single = head_output(X, offsets=[1, 2])
        double = torch.cat([head_output(X, offsets=[2]), head_output(X, offsets=[1])])
        single_outputs.append(single)
        double_outputs.append(double)

        print(f"\n{text!r}: na última posição, t−2 = {text[1]!r} e t−1 = {text[2]!r}")
        print(f"  1 head procurando t−2 e t−1: [{describe(single)}]")
        head_a, head_b = describe(double[:vocab_size]), describe(double[vocab_size:])
        print(f"  2 heads, concatenadas:        [{head_a}] | [{head_b}]")

    same_single = torch.allclose(*single_outputs)
    same_double = torch.allclose(*double_outputs)
    print(f"\nsaída igual para 'gato' e 'gtao'?  1 head: {same_single}   2 heads: {same_double}")


def split_and_merge() -> None:
    section("2. Dividir d_model entre as heads: view + transpose")
    batch, seq_len, d_model, num_heads = 1, 3, 8, 2
    head_dim = d_model // num_heads
    x = torch.arange(batch * seq_len * d_model, dtype=torch.float32)
    x = x.view(batch, seq_len, d_model)
    print(f"x {tuple(x.shape)}: cada linha é uma posição, cada coluna uma feature")
    for t in range(seq_len):
        print(f"  t={t}: {x[0, t].int().tolist()}")

    viewed = x.view(batch, seq_len, num_heads, head_dim)
    split = viewed.transpose(1, 2)
    print()
    for name, tensor in [("x", x), ("view", viewed), ("transpose", split)]:
        shape, stride = str(tuple(tensor.shape)), str(tensor.stride())
        print(
            f"  {name:>9}: shape {shape:<14} stride {stride:<16} contíguo: {tensor.is_contiguous()}"
        )

    print()
    for i in range(num_heads):
        block = f"features {i * head_dim}..{(i + 1) * head_dim - 1}"
        print(f"  head {i} ({block}): {split[0, i].int().tolist()}")
    print(f"  split_heads(x) dá o mesmo: {torch.equal(split_heads(x, num_heads), split)}")

    attention_output = split.contiguous()  # a attention devolve um tensor novo, contíguo
    try:
        attention_output.transpose(1, 2).view(batch, seq_len, d_model)
    except RuntimeError as error:
        print(f"\n  view após transpose, sem contiguous(): RuntimeError: {str(error)[:60]}...")
    merged = merge_heads(attention_output)
    print(f"  merge_heads (com contiguous) reconstrói x: {torch.equal(merged, x)}")


def real_batch(config: Config) -> torch.Tensor:
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
    x, _ = next(iter(loader))
    torch.manual_seed(0)
    emb = Embeddings(tok.vocab_size, config.model.context_length, config.model.d_model, 0.0)
    # Normalizado como após a LayerNorm, para os números terem escala legível.
    return F.layer_norm(emb(x), (config.model.d_model,))


def equivalence(h: torch.Tensor, config: Config) -> MultiHeadAttention:
    section("3. Lista de heads × tensor único × nn.MultiheadAttention do PyTorch")
    d_model, num_heads = config.model.d_model, config.model.num_heads
    torch.manual_seed(0)
    loop = LoopMultiHeadAttention(d_model, num_heads, dropout=0.0).eval()
    fused = MultiHeadAttention(d_model, num_heads, dropout=0.0).eval()
    copy_heads(loop, fused)

    torch_mha = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True).eval()
    qkv = torch.cat([fused.q_proj.weight, fused.k_proj.weight, fused.v_proj.weight])
    torch_mha.in_proj_weight.copy_(qkv)
    torch_mha.out_proj.weight.copy_(fused.out_proj.weight)

    seq_len = h.shape[1]
    out_loop, out_fused = loop(h), fused(h)
    out_torch, _ = torch_mha(h, h, h, attn_mask=~causal_mask(seq_len), need_weights=False)
    print(f"entrada {tuple(h.shape)}, d_model = {d_model}, num_heads = {num_heads}")
    print(f"  max |lista − tensor único|:    {(out_loop - out_fused).abs().max():.2e}")
    print(f"  max |tensor único − PyTorch|:  {(out_fused - out_torch).abs().max():.2e}")

    print("\nparâmetros:")
    for name, module in [("lista de heads", loop), ("tensor único", fused)]:
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<15} {n_params:,}  (4 · d_model² = {4 * d_model**2:,})")
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        print(f"  {name}.weight: {tuple(getattr(fused, name).weight.shape)}")
    print(f"  cada head na lista: q/k/v_proj.weight {tuple(loop.heads[0].q_proj.weight.shape)}")
    return fused


def forward_walkthrough(h: torch.Tensor, mha: MultiHeadAttention) -> None:
    section("4. O forward de MultiHeadAttention, etapa por etapa")
    seq_len = h.shape[1]
    q = mha.q_proj(h)
    q_heads = split_heads(q, mha.num_heads)
    k_heads = split_heads(mha.k_proj(h), mha.num_heads)
    v_heads = split_heads(mha.v_proj(h), mha.num_heads)
    weights = attention_weights(q_heads, k_heads, causal_mask(seq_len))
    heads = weights @ v_heads
    merged = merge_heads(heads)
    out = mha.out_proj(merged)
    steps = [
        ("x", h, "[B, T, d_model]"),
        ("q_proj(x)", q, "[B, T, d_model]"),
        ("split_heads", q_heads, "[B, heads, T, head_dim]"),
        ("pesos", weights, "[B, heads, T, T]"),
        ("pesos @ v", heads, "[B, heads, T, head_dim]"),
        ("merge_heads", merged, "[B, T, d_model]"),
        ("out_proj", out, "[B, T, d_model]"),
    ]
    for name, tensor, meaning in steps:
        print(f"  {name:<12} {str(tuple(tensor.shape)):<20} {meaning}")
    print(f"  igual a mha(x): {torch.allclose(out, mha(h))}")

    print("\ncada head tem sua própria matriz de pesos [T, T]; linha 5 de cada uma (6 visíveis):")
    for i in range(mha.num_heads):
        row = " ".join(f"{w:.3f}" for w in weights[0, i, 5, :6].tolist())
        print(f"  head {i}: {row}")


def sdpa_forward(mha: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    """Mesmo layout de MultiHeadAttention, com o kernel otimizado do PyTorch na attention."""
    q = split_heads(mha.q_proj(x), mha.num_heads)
    k = split_heads(mha.k_proj(x), mha.num_heads)
    v = split_heads(mha.v_proj(x), mha.num_heads)
    heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return mha.out_proj(merge_heads(heads))


def measure_ms(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, reps: int) -> float:
    for _ in range(10):  # aquecimento
        fn(x)
    start = time.perf_counter()
    for _ in range(reps):
        fn(x)
    return (time.perf_counter() - start) / reps * 1000


def timing(d_model: int) -> None:
    section("5. Tempo de forward na CPU, em ms (varia entre execuções e máquinas)")
    print(f"{'entrada':<14} {'heads':>5} | {'lista':>7} | {'único':>7} | {'único + SDPA':>12}")
    for batch, seq_len, reps in ((1, 16, 300), (32, 128, 50)):
        x = torch.randn(batch, seq_len, d_model)
        for num_heads in (4, 16):
            torch.manual_seed(0)
            loop = LoopMultiHeadAttention(d_model, num_heads, dropout=0.0).eval()
            fused = MultiHeadAttention(d_model, num_heads, dropout=0.0).eval()
            ms = [measure_ms(fn, x, reps) for fn in (loop, fused, partial(sdpa_forward, fused))]
            shape = f"[{batch}, {seq_len}, {d_model}]"
            print(f"{shape:<14} {num_heads:>5} | {ms[0]:>7.3f} | {ms[1]:>7.3f} | {ms[2]:>12.3f}")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    with torch.no_grad():
        one_head_vs_two()
        split_and_merge()
        h = real_batch(config)
        mha = equivalence(h, config)
        forward_walkthrough(h, mha)
        timing(config.model.d_model)


if __name__ == "__main__":
    main()
