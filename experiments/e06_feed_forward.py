"""Experimento: o feed-forward, a não-linearidade e o processamento por posição.

Rode com:  uv run python -m experiments.e06_feed_forward
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.embeddings import Embeddings
from model.feed_forward import FeedForward
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def randomize(module: nn.Module, std: float) -> None:
    """Pesos e biases ~ N(0, std²): deixa os biases visíveis nas contas."""
    with torch.no_grad():
        for param in module.parameters():
            nn.init.normal_(param, std=std)


def gelu_vs_relu() -> None:
    section("1. GELU × ReLU")
    x = torch.tensor([-3.0, -2.0, -1.0, -0.75, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], requires_grad=True)
    relu, gelu = F.relu(x), F.gelu(x)
    (d_relu,) = torch.autograd.grad(relu.sum(), x)
    (d_gelu,) = torch.autograd.grad(gelu.sum(), x)
    phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # P(Z <= x), Z ~ N(0, 1)

    print("GELU(x) = x · Φ(x), onde Φ é a CDF da normal padrão\n")
    print(f"{'x':>6} | {'ReLU':>6} {'GELU':>7} | {'Φ(x)':>6} | {'ReLU′':>6} {'GELU′':>7}")
    columns = zip(
        x.tolist(),
        relu.tolist(),
        gelu.tolist(),
        phi.tolist(),
        d_relu.tolist(),
        d_gelu.tolist(),
        strict=True,
    )
    for xi, r, g, p, dr, dg in columns:
        print(f"{xi:>6.2f} | {r:>6.3f} {g:>+7.3f} | {p:>6.3f} | {dr:>6.2f} {dg:>+7.3f}")

    grid = torch.linspace(-3, 3, 60001)
    values = F.gelu(grid)
    i = values.argmin()
    print(f"\nmínimo da GELU: {values[i]:.4f} em x = {grid[i]:.3f}  (a ReLU nunca é negativa)")
    print("para x < 0: gradiente da ReLU = 0; o da GELU continua diferente de zero")


def why_nonlinearity() -> None:
    section("2. Sem não-linearidade, duas camadas lineares viram uma")
    torch.manual_seed(0)
    ff = FeedForward(d_model=8, d_ff=32, dropout=0.0)
    randomize(ff, std=1.0)
    x = torch.randn(5, 8)

    W1, b1 = ff.up_proj.weight, ff.up_proj.bias  # [32, 8], [32]
    W2, b2 = ff.down_proj.weight, ff.down_proj.bias  # [8, 32], [8]
    W, b = W2 @ W1, W2 @ b1 + b2  # [8, 8], [8]
    two_layers = ff.down_proj(ff.up_proj(x))
    one_layer = x @ W.T + b
    print("down_proj(up_proj(x)) sem GELU:")
    print(
        f"  igual a uma única camada x·(W2·W1)ᵀ + (W2·b1 + b2): "
        f"{torch.allclose(two_layers, one_layer, rtol=1e-4, atol=1e-4)}"
    )
    print(f"  W2·W1 tem shape {tuple(W.shape)}: expandir para 32 dims não acrescentou nada")

    print("\nXOR: saída 1 quando exatamente uma das entradas é 1")
    print("o mesmo FeedForward, com d_model = 2 e d_ff = 16; a 1ª coordenada da saída é o logit")
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    for name, activation in [("sem ativação", nn.Identity()), ("com GELU", nn.GELU())]:
        torch.manual_seed(0)
        model = FeedForward(d_model=2, d_ff=16, dropout=0.0)
        model.activation = activation
        randomize(model, std=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(500):
            loss = F.binary_cross_entropy_with_logits(model(inputs)[:, 0], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(model(inputs)[:, 0])
        answers = "  ".join(
            f"{a:.0f}{b:.0f}→{p:.2f}" for (a, b), p in zip(inputs, probs, strict=True)
        )
        print(f"  {name:<13} loss {loss.item():.3f} | P(saída = 1): {answers}")
    print(f"  (responder 0.5 para tudo dá loss ln 2 = {math.log(2):.3f})")


def key_value_memory() -> None:
    section("3. Uma soma de neurônios: cada um detecta um padrão e escreve um vetor")
    torch.manual_seed(0)
    d_model, d_ff = 8, 32
    ff = FeedForward(d_model, d_ff, dropout=0.0)
    randomize(ff, std=0.5)
    x = torch.randn(d_model)

    keys = ff.up_proj.weight  # [d_ff, d_model]: linha i = chave do neurônio i
    values = ff.down_proj.weight  # [d_model, d_ff]: coluna i = valor do neurônio i
    pre = keys @ x + ff.up_proj.bias  # [d_ff]: k_i · x + b_i
    act = F.gelu(pre)  # [d_ff]
    total = ff.down_proj.bias + sum(act[i] * values[:, i] for i in range(d_ff))
    print(f"ff(x) == b2 + Σ_i GELU(k_i·x + b_i) · v_i: {torch.allclose(total, ff(x), atol=1e-5)}")

    print("\nos 5 neurônios com maior |ativação| para este x:")
    for i in act.abs().argsort(descending=True)[:5].tolist():
        print(
            f"  neurônio {i:>2}: k·x + b = {pre[i]:+.3f} -> GELU = {act[i]:+.3f}  "
            f"(escreve {act[i]:+.3f} × v_{i})"
        )
    print(
        f"ativos (> 0,1): {(act > 0.1).sum()} de {d_ff}; "
        f"quase desligados (|a| < 0,05): {(act.abs() < 0.05).sum()}"
    )


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
    # Normalizado como após a LayerNorm.
    return F.layer_norm(emb(x), (config.model.d_model,))


def position_wise(config: Config) -> None:
    section("4. Cada posição é processada sozinha, com os mesmos pesos")
    h = real_batch(config)
    d_model, d_ff = config.model.d_model, config.model.d_ff
    torch.manual_seed(0)
    ff = FeedForward(d_model, d_ff, dropout=0.0)

    pre = ff.up_proj(h)
    out = ff(h)
    print(f"entrada   {tuple(h.shape)}")
    print(f"up_proj   {tuple(pre.shape)}   (expande para d_ff = 4 · {d_model})")
    print(f"saída     {tuple(out.shape)}")
    print(f"pré-ativações negativas: {(pre < 0).float().mean():.1%}")

    changed = h.clone()
    changed[:, 5] = torch.randn(d_model)
    positions = (ff(changed) - out).abs().amax(dim=(0, 2)).nonzero().flatten().tolist()
    print(f"\nmudando só a entrada da posição 5, saídas alteradas nas posições: {positions}")

    repeated = h.clone()
    repeated[0, 9] = repeated[0, 3]
    same = ff(repeated)
    print(
        f"mesmo vetor nas posições 3 e 9 -> mesma saída: {torch.allclose(same[0, 3], same[0, 9])}"
    )

    flat = ff(h.reshape(-1, d_model)).view_as(out)
    print(f"ff([B, T, d]) == ff([B·T, d]) remontado: {torch.allclose(flat, out, atol=1e-6)}")


def parameter_budget(config: Config) -> None:
    section("5. Parâmetros e contas por token: feed-forward × attention")
    d, seq_len = config.model.d_model, config.model.context_length
    attention = 4 * d * d
    print(f"attention (Q, K, V, W_O, sem bias): 4·d² = {attention:,} parâmetros\n")
    print(f"{'d_ff':>12} | {'parâmetros':>10} | {'× attention':>11}")
    for mult in (1, 2, 4, 8):
        ff = FeedForward(d, mult * d, dropout=0.0)
        n_params = sum(p.numel() for p in ff.parameters())
        print(f"{f'{mult}·d = {mult * d}':>12} | {n_params:>10,} | {n_params / attention:>11.2f}")

    ffn_mults = 2 * d * 4 * d
    attn_mults = 4 * d * d + 2 * seq_len * d
    print(f"\nmultiplicações por token no forward (T = {seq_len}):")
    print(f"  feed-forward: 2 · d · 4d               = {ffn_mults:,}")
    print(f"  attention:    4d² (projeções) + 2·T·d (scores e soma ponderada) = {attn_mults:,}")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    gelu_vs_relu()
    why_nonlinearity()
    with torch.no_grad():
        key_value_memory()
        position_wise(config)
        parameter_budget(config)


if __name__ == "__main__":
    main()
