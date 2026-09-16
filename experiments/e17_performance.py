"""Experimento: performance — attention eficiente (SDPA) e mixed precision (bfloat16).

Rode com: uv run python -m experiments.e17_performance
"""

import copy
import time

import torch
import torch.nn.functional as F

from config import ModelConfig
from device import autocast, device_memory_mb, select_device, synchronize
from model import GPT
from model.attention import MultiHeadAttention
from training.trainer import configure_optimizer, train_step

DEVICE = select_device("auto")
AVAILABLE = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]
ACCELERATOR = next((name for name, ok in AVAILABLE if ok), None)


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def timed(fn, steps: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    synchronize(DEVICE)
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    synchronize(DEVICE)
    return (time.perf_counter() - start) / steps


def correctness() -> None:
    section("1. SDPA calcula a mesma coisa que a attention manual")
    d_model, num_heads = 64, 4
    torch.manual_seed(0)
    manual = MultiHeadAttention(d_model, num_heads, dropout=0.0, backend="manual")
    sdpa = MultiHeadAttention(d_model, num_heads, dropout=0.0, backend="sdpa")
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(4, 32, d_model)
    diff = (manual(x) - sdpa(x)).abs().max().item()
    print(f"  mesmos pesos, mesma entrada: diferença máxima manual vs. sdpa = {diff:.2e}")


def attention_timing() -> None:
    section("2. Tempo da attention: manual vs. SDPA, por tamanho de contexto")
    if ACCELERATOR is None:
        print("  nenhum acelerador neste ambiente: comparação na CPU (menos representativa)")
    device = DEVICE
    batch, num_heads, head_dim = 30, 4, 64
    print(f"  device: {device} | batch {batch} | heads {num_heads} | head_dim {head_dim}")
    print(f"  {'contexto':>8} {'manual (ms)':>13} {'sdpa (ms)':>11} {'speedup':>9}")
    for seq_len in (128, 512, 1024):
        q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        def manual_step(q=q, k=k, v=v, mask=mask):
            scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            out = torch.softmax(scores, dim=-1) @ v
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        def sdpa_step(q=q, k=k, v=v):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        manual_ms = timed(manual_step, steps=10) * 1000
        sdpa_ms = timed(sdpa_step, steps=10) * 1000
        print(f"  {seq_len:>8} {manual_ms:>13.2f} {sdpa_ms:>11.2f} {manual_ms / sdpa_ms:>8.2f}x")
    print("\n  a vantagem do SDPA é não materializar a matriz [T, T] inteira (memória, seção 3),")
    print("  e escolher o kernel mais rápido do backend; o tempo medido aqui depende de qual")
    print("  kernel o backend deste device tem disponível (ver docs/17-performance.md).")


def attention_memory() -> None:
    section("3. Memória da attention: manual vs. SDPA")
    device = DEVICE
    batch, num_heads, head_dim = 30, 4, 64
    print(f"  {'contexto':>8} {'manual (MB)':>13} {'sdpa (MB)':>11}")
    for seq_len in (512, 1024, 2048):
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        def run(fn, seq_len=seq_len):
            q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
            k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
            v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
            fn(q, k, v).sum().backward()
            synchronize(device)
            return device_memory_mb(device)

        def manual(q, k, v, mask=mask):
            scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            return torch.softmax(scores, dim=-1) @ v

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        manual_mb, sdpa_mb = run(manual), run(sdpa)
        print(f"  {seq_len:>8} {manual_mb:>13.1f} {sdpa_mb:>11.1f}")
    print("\n  números idênticos em todo tamanho = a métrica é grossa demais neste device")
    print("  (device_memory_mb no MPS mede o reservado pelo driver, docs/16-gpu.md), não os")
    print("  dois backends usando de fato a mesma memória.")


def mixed_precision_correctness() -> None:
    section("4. Mixed precision: mesma loss (dentro do ruído), pesos continuam em float32")
    torch.manual_seed(0)
    config = ModelConfig(context_length=64, d_model=128, num_heads=4, num_layers=4, dropout=0.1)
    model = GPT(config, vocab_size=97).to(DEVICE)
    x = torch.randint(0, 97, (8, 64), device=DEVICE)
    y = torch.randint(0, 97, (8, 64), device=DEVICE)

    model.eval()
    with torch.no_grad(), autocast(DEVICE, enabled=False):
        logits_fp32 = model(x)
    with torch.no_grad(), autocast(DEVICE, enabled=True):
        logits_bf16 = model(x)
    diff = (logits_fp32 - logits_bf16.float()).abs().mean().item()
    print(f"  dtype dos logits: float32 -> {logits_fp32.dtype} | bfloat16 -> {logits_bf16.dtype}")
    print(f"  diferença média nos logits (fp32 vs. bf16): {diff:.4f}")

    model_copy = copy.deepcopy(model)
    optimizer = configure_optimizer(model, 1e-3, 0.0)
    optimizer_copy = configure_optimizer(model_copy, 1e-3, 0.0)
    full = train_step(model, x, y, optimizer, gradient_clip=1.0)
    mixed = train_step(model_copy, x, y, optimizer_copy, gradient_clip=1.0, mixed_precision=True)
    print(f"  loss de um passo: float32 {full.loss:.4f} | bfloat16 {mixed.loss:.4f}")
    param_dtype = model_copy.embeddings.token_embedding.weight.dtype
    print(f"  parâmetros continuam float32 com mixed_precision=True: {param_dtype}")


def mixed_precision_timing() -> None:
    section("5. Tempo de um passo de treino: combinando backend e precisão")
    config = ModelConfig(context_length=128, d_model=256, num_heads=4, num_layers=8, dropout=0.1)
    batch = 30

    print(f"  device {DEVICE} | modelo: d_model {config.d_model}, {config.num_layers} camadas")
    print(f"  {'attention':>10} {'precisão':>10} {'ms/passo':>10}")
    baseline = None
    for backend in ("manual", "sdpa"):
        for mixed in (False, True):
            torch.manual_seed(0)
            model_config = ModelConfig(
                config.context_length,
                config.d_model,
                config.num_heads,
                config.num_layers,
                config.dropout,
                attention_backend=backend,
            )
            model = GPT(model_config, vocab_size=97).to(DEVICE)
            optimizer = configure_optimizer(model, 1e-3, 0.0)
            x = torch.randint(0, 97, (batch, config.context_length), device=DEVICE)
            y = torch.randint(0, 97, (batch, config.context_length), device=DEVICE)

            def step(model=model, x=x, y=y, optimizer=optimizer, mixed=mixed):
                train_step(model, x, y, optimizer, gradient_clip=1.0, mixed_precision=mixed)

            ms = timed(step, steps=8) * 1000
            precisao = "bfloat16" if mixed else "float32"
            speedup = f"({baseline / ms:.2f}x)" if baseline else ""
            print(f"  {backend:>10} {precisao:>10} {ms:>10.1f}  {speedup}")
            if baseline is None:
                baseline = ms


def main() -> None:
    correctness()
    attention_timing()
    attention_memory()
    mixed_precision_correctness()
    mixed_precision_timing()


if __name__ == "__main__":
    main()
