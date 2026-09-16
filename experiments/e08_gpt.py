"""Experimento: o GPT montado — embeddings, blocos empilhados e LayerNorm final.

Rode com:  uv run python -m experiments.e08_gpt
"""

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import Config, ModelConfig, load_config
from data.loader import create_dataloader, split_train_val
from model.gpt import GPT
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_batch(config: Config) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
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
    return tok, train_ids, x


def anatomy(config: Config, vocab_size: int) -> None:
    section("1. Anatomia do modelo (tiny.yaml, vocab_size vindo do tokenizer)")
    torch.manual_seed(0)
    model = GPT(config.model, vocab_size)
    print(model)

    m = config.model
    d, T, L = m.d_model, m.context_length, m.num_layers
    print("\nparâmetros:")
    rows = [
        ("token_embedding", model.embeddings.token_embedding, f"V·d = {vocab_size}·{d}"),
        ("position_embedding", model.embeddings.position_embedding, f"T·d = {T}·{d}"),
        *[(f"blocks[{i}]", block, "12·d² + 9·d") for i, block in enumerate(model.blocks)],
        ("ln_final", model.ln_final, "2·d"),
    ]
    total = model.num_parameters()
    for name, module, formula in rows:
        n_params = count(module)
        print(f"  {name:<20} {n_params:>9,}  {n_params / total:6.1%}   {formula}")
    formula_total = vocab_size * d + T * d + L * (12 * d**2 + 9 * d) + 2 * d
    print(f"  {'total':<20} {total:>9,}  (V·d + T·d + L·(12·d² + 9·d) + 2·d = {formula_total:,})")


def forward_walkthrough(config: Config, x: torch.Tensor, vocab_size: int) -> None:
    section("2. O forward completo, camada por camada, num batch real")
    torch.manual_seed(0)
    model = GPT(config.model, vocab_size).eval()

    print(f"  {'etapa':<18} {'shape':<18} {'norma média':>12} {'‖Δ‖ / ‖entrada‖':>17}")
    print(f"  {'ids':<18} {str(tuple(x.shape)):<18} {'—':>12} {'—':>17}")
    h = model.embeddings(x)
    print(
        f"  {'embeddings':<18} {str(tuple(h.shape)):<18} {h.norm(dim=-1).mean():>12.3f} {'—':>17}"
    )
    for i, block in enumerate(model.blocks):
        new_h = block(h)
        change = (new_h - h).norm(dim=-1).mean() / h.norm(dim=-1).mean()
        h = new_h
        name = f"bloco {i}"
        print(
            f"  {name:<18} {str(tuple(h.shape)):<18} {h.norm(dim=-1).mean():>12.3f} {change:>17.3f}"
        )
    out = model.ln_final(h)
    print(
        f"  {'ln_final':<18} {str(tuple(out.shape)):<18} {out.norm(dim=-1).mean():>12.3f} {'—':>17}"
    )
    print(f"\n  igual a model.hidden_states(ids): {torch.allclose(out, model.hidden_states(x))}")
    print(f"  √d_model = {config.model.d_model**0.5:.2f}")


def causality(config: Config, tok: CharTokenizer) -> None:
    section("3. O modelo inteiro é causal")
    torch.manual_seed(0)
    model = GPT(config.model, tok.vocab_size).eval()
    text_a, text_b = "o gato dorme", "o gato corre"
    out_a = model.hidden_states(torch.tensor([tok.encode(text_a)]))
    out_b = model.hidden_states(torch.tensor([tok.encode(text_b)]))
    diff = (out_a - out_b).abs().amax(dim=-1)[0]
    print(f"A = {text_a!r}, B = {text_b!r}; {config.model.num_layers} blocos")
    print("maior |saída_A − saída_B| em cada posição:")
    for t, (a, b) in enumerate(zip(text_a, text_b, strict=True)):
        mark = "   <- entrada diferente" if a != b else ""
        print(f"  t={t:<2} {a!r} {b!r}  {diff[t]:.2e}{mark}")

    ids = torch.tensor([tok.encode(text_a)])
    captured = {}

    def keep_embeddings(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        output.retain_grad()
        captured["embeddings"] = output

    handle = model.embeddings.register_forward_hook(keep_embeddings)
    out = model.hidden_states(ids)
    handle.remove()
    (out[0, 5] * torch.randn(config.model.d_model)).sum().backward()
    reached = captured["embeddings"].grad[0].norm(dim=-1)
    positions = (reached > 0).nonzero().flatten().tolist()
    print(f"\no gradiente da saída na posição 5 chega aos embeddings das posições: {positions}")


def sum_pitfall(config: Config, x: torch.Tensor, vocab_size: int) -> None:
    section("4. Uma armadilha: out.sum() depois da LayerNorm final")
    for name, make_loss in [
        ("out.sum()", lambda out: out.sum()),
        ("(out · pesos aleatórios).sum()", lambda out: (out * torch.randn_like(out)).sum()),
    ]:
        torch.manual_seed(0)
        model = GPT(config.model, vocab_size).eval()
        make_loss(model.hidden_states(x)).backward()
        token_grad = model.embeddings.token_embedding.weight.grad.abs().max()
        final_grad = model.ln_final.bias.grad.abs().max()
        grads = f"token_embedding {token_grad:.2e} | ln_final.bias {final_grad:.2e}"
        print(f"  {name:<32} max |grad| {grads}")
    print("  cada vetor após a LayerNorm tem soma 0 (se γ = 1): out.sum() só depende de β")


def stream_before_final_norm(model: GPT, x: torch.Tensor) -> torch.Tensor:
    h = model.embeddings(x)
    for block in model.blocks:
        h = block(h)
    return h  # [batch, seq_len, d_model]


def norm_stats(h: torch.Tensor) -> str:
    norms = h.norm(dim=-1)
    return f"{norms.mean():>8.3f} {norms.min():>8.3f} {norms.max():>8.3f}"


def final_layer_norm(
    config: Config, x: torch.Tensor, train_ids: torch.Tensor, vocab_size: int
) -> None:
    section("5. Por que a LayerNorm final: a escala do residual stream")
    columns = f"{'média':>8} {'mín':>8} {'máx':>8}"
    print("norma por posição do stream antes de ln_final, e a média depois dela\n")
    print("na inicialização, variando a profundidade:")
    print(f"  {'L':>5} | {columns} | {'depois':>7}")
    with torch.no_grad():
        for num_layers in (1, 4, 16):
            torch.manual_seed(0)
            model = GPT(replace(config.model, num_layers=num_layers), vocab_size).eval()
            h = stream_before_final_norm(model, x)
            after = model.ln_final(h).norm(dim=-1).mean()
            print(f"  {num_layers:>5} | {norm_stats(h)} | {after:>7.3f}")

    d_model, seq_len = config.model.d_model, config.model.context_length
    print(
        "\ndurante um mini-treino (tiny.yaml + Linear(d_model, vocab) provisório, AdamW lr 1e-3):"
    )
    torch.manual_seed(0)
    model = GPT(replace(config.model, dropout=0.0), vocab_size)
    head = nn.Linear(d_model, vocab_size)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3)
    loader = create_dataloader(
        train_ids, context_length=seq_len, stride=seq_len, batch_size=16, shuffle=True, seed=0
    )
    print(f"  {'passo':>5} | {columns} | {'depois':>7} | {'loss':>6}")
    for step, (xb, yb) in enumerate(loader):
        loss = F.cross_entropy(head(model.hidden_states(xb)).view(-1, vocab_size), yb.view(-1))
        if step in (0, 50, 100, 150):
            with torch.no_grad():
                h = stream_before_final_norm(model, x)
                after = model.ln_final(h).norm(dim=-1).mean()
            print(f"  {step:>5} | {norm_stats(h)} | {after:>7.3f} | {loss.item():>6.3f}")
        if step == 150:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def scaling() -> None:
    section("6. Contando parâmetros: do Mini-GPT ao tamanho do GPT-2 small")
    configs = [
        (
            "tiny.yaml",
            97,
            ModelConfig(context_length=128, d_model=128, num_heads=4, num_layers=4, dropout=0.1),
        ),
        (
            "d = 256, L = 8",
            97,
            ModelConfig(context_length=256, d_model=256, num_heads=8, num_layers=8, dropout=0.1),
        ),
        (
            "GPT-2 small",
            50257,
            ModelConfig(context_length=1024, d_model=768, num_heads=12, num_layers=12, dropout=0.1),
        ),
    ]
    header = f"  {'':<15} {'V':>6} {'T':>5} {'d':>4} {'L':>3} | "
    print(header + f"{'embeddings':>11} {'blocos':>11} {'total':>12}")
    for name, vocab_size, cfg in configs:
        # Dispositivo "meta": cria os tensores só com shape, sem alocar memória.
        with torch.device("meta"):
            model = GPT(cfg, vocab_size)
        sizes = f"{vocab_size:>6} {cfg.context_length:>5} {cfg.d_model:>4} {cfg.num_layers:>3}"
        counts = f"{count(model.embeddings):>11,} {count(model.blocks):>11,}"
        print(f"  {name:<15} {sizes} | {counts} {model.num_parameters():>12,}")
    print(
        "\n  GPT-2 small oficial: 124.439.808 parâmetros. A diferença de 36.864 = 12 blocos × 4·768"
    )
    print(
        "  são os biases da attention, que não usamos. O LM head do GPT-2 reusa o token embedding."
    )


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    tok, train_ids, x = load_batch(config)
    with torch.no_grad():
        anatomy(config, tok.vocab_size)
        forward_walkthrough(config, x, tok.vocab_size)
    causality(config, tok)
    sum_pitfall(config, x, tok.vocab_size)
    final_layer_norm(config, x, train_ids, tok.vocab_size)
    with torch.no_grad():
        scaling()


if __name__ == "__main__":
    main()
