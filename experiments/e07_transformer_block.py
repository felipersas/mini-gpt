"""Experimento: LayerNorm, conexões residuais e o Transformer Block completo.

Rode com:  uv run python -m experiments.e07_transformer_block
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.embeddings import Embeddings
from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def fmt(vector: torch.Tensor) -> str:
    return " ".join(f"{v:+.3f}" for v in vector.tolist())


def layer_norm_step_by_step() -> None:
    section("1. LayerNorm passo a passo, num vetor de 6 features")
    x = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 7.0])
    gamma = torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5, 0.5])
    beta = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    mean = x.mean()
    var = (x - mean).pow(2).mean()
    x_hat = (x - mean) / torch.sqrt(var + 1e-5)
    y = gamma * x_hat + beta
    print(f"x                          = {fmt(x)}")
    print(f"média = {mean:.3f}   variância = {var:.3f}   desvio = {var.sqrt():.3f}")
    print(f"x̂ = (x − média) / √(var + eps) = {fmt(x_hat)}")
    print(f"    média de x̂ = {x_hat.mean():+.3f}, variância de x̂ = {x_hat.pow(2).mean():.3f}")
    print(f"γ                          = {fmt(gamma)}")
    print(f"β                          = {fmt(beta)}")
    print(f"y = γ · x̂ + β              = {fmt(y)}")

    ln, reference = LayerNorm(6), nn.LayerNorm(6)
    for module in (ln, reference):
        module.weight.copy_(gamma)
        module.bias.copy_(beta)
    print(f"\nLayerNorm do projeto == nn.LayerNorm: {torch.allclose(ln(x), reference(x))}")
    print(
        f"LayerNorm(3·x + 10) == LayerNorm(x):  {torch.allclose(ln(3 * x + 10), ln(x), atol=1e-5)}"
    )


def layer_norm_vs_batch_norm() -> None:
    section("2. Por que LayerNorm e não BatchNorm: a causalidade")
    torch.manual_seed(0)
    d_model, seq_len = 8, 6
    x = torch.randn(1, seq_len, d_model)
    changed = x.clone()
    changed[:, 4:] = torch.randn(1, 2, d_model) * 5  # só as posições 4 e 5 mudam

    layer_norm = LayerNorm(d_model)
    batch_norm = nn.BatchNorm1d(d_model)  # em modo treino: estatísticas do próprio batch

    def apply_batch_norm(t: torch.Tensor) -> torch.Tensor:
        # Cada posição vira uma "amostra"; a média e a variância misturam todas as posições.
        return batch_norm(t.view(-1, d_model)).view_as(t)

    print("maior |diferença| na saída de cada posição (entradas diferentes só em 4 e 5):")
    for name, norm in [("LayerNorm", layer_norm), ("BatchNorm", apply_batch_norm)]:
        diff = (norm(x) - norm(changed)).abs().amax(dim=-1)[0]
        print(f"  {name:<10} " + "  ".join(f"t{t}={v:.2f}" for t, v in enumerate(diff.tolist())))


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
    return emb(x)


def block_walkthrough(config: Config) -> None:
    section("3. O bloco, etapa por etapa, num batch real (recém-inicializado)")
    m = config.model
    h = real_batch(config)
    torch.manual_seed(0)
    block = TransformerBlock(m.d_model, m.num_heads, m.d_ff, dropout=0.0, num_layers=m.num_layers)

    normed_1 = block.ln_1(h)
    attention_out = block.attention(normed_1)
    after_attention = h + attention_out
    normed_2 = block.ln_2(after_attention)
    ff_out = block.feed_forward(normed_2)
    out = after_attention + ff_out

    steps = [
        ("x (residual stream)", h),
        ("ln_1(x)", normed_1),
        ("attention(ln_1(x))", attention_out),
        ("x + attention", after_attention),
        ("ln_2(x + attention)", normed_2),
        ("feed_forward(ln_2(...))", ff_out),
        ("saída do bloco", out),
    ]
    print(f"  {'etapa':<25} {'shape':<16} {'norma média por posição':>23}")
    for name, tensor in steps:
        norm = tensor.norm(dim=-1).mean()
        print(f"  {name:<25} {str(tuple(tensor.shape)):<16} {norm:>23.3f}")
    print(f"\n  igual a block(x): {torch.allclose(out, block(h))}")
    print(f"  √d_model = {math.sqrt(m.d_model):.2f} (norma de um vetor com média 0 e variância 1)")
    change = (out - h).norm(dim=-1).mean() / h.norm(dim=-1).mean()
    print(f"  ‖saída − x‖ / ‖x‖ = {change:.3f}")


class PostNormBlock(TransformerBlock):
    """Post-norm, como no Transformer original: x = LayerNorm(x + subcamada(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(x + self.attention(x))
        return self.ln_2(x + self.feed_forward(x))


class NoResidualBlock(TransformerBlock):
    """Pre-norm sem conexões residuais: x = subcamada(LayerNorm(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.ln_1(x))
        return self.feed_forward(self.ln_2(x))


VARIANTS = {
    "pre-norm (projeto)": TransformerBlock,
    "post-norm": PostNormBlock,
    "sem residual": NoResidualBlock,
}


def gradient_flow() -> None:
    num_layers, d_model, num_heads = 16, 64, 4
    section(f"4. O gradiente atravessando {num_layers} blocos (na inicialização)")
    torch.manual_seed(0)
    x0 = torch.randn(8, 32, d_model) * 0.03  # escala dos embeddings
    readout = torch.randn(8, 32, d_model)  # loss = Σ saída · readout, logo ∂loss/∂saída = readout
    shown = [0, 1, 2, 4, 8, 12, 15]

    norms, cosines = {}, {}
    for name, cls in VARIANTS.items():
        torch.manual_seed(0)
        blocks = [cls(d_model, num_heads, 4 * d_model, 0.0, num_layers) for _ in range(num_layers)]
        h = x0.clone().requires_grad_()
        inputs = []
        for block in blocks:
            h.retain_grad()
            inputs.append(h)
            h = block(h)
        (h * readout).sum().backward()
        grads = [inputs[i].grad.flatten() for i in shown]
        norms[name] = [g.norm().item() for g in grads]
        cosines[name] = [F.cosine_similarity(g, readout.flatten(), dim=0).item() for g in grads]

    header = f"  {'':<20}" + "".join(f"{f'ℓ={i}':>9}" for i in shown)
    print("norma ‖∂loss/∂x_ℓ‖, com x_ℓ = entrada do bloco ℓ:")
    print(header)
    for name, values in norms.items():
        print(f"  {name:<20}" + "".join(f"{v:>9.1f}" for v in values))
    print("\ndireção: cosseno entre ∂loss/∂x_ℓ e ∂loss/∂saída (1 = o sinal de erro chega intacto):")
    print(header)
    for name, values in cosines.items():
        print(f"  {name:<20}" + "".join(f"{v:>+9.3f}" for v in values))


def short_training(config: Config) -> None:
    num_layers, d_model, num_heads, steps = 16, 64, 4, 150
    section(f"5. Mini-treino: {num_layers} blocos, d_model = {d_model}, {steps} passos")
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, _ = split_train_val(ids, config.data.val_fraction)
    print("prévia das próximas fases: embeddings -> blocos -> LayerNorm -> Linear(d_model, vocab)")
    print("cross-entropy, AdamW lr 1e-3, batch [16, 64]")
    print(f"loss de um chute uniforme: ln {tok.vocab_size} = {math.log(tok.vocab_size):.2f}\n")

    checkpoints = (0, 25, 50, 100, 150)
    print(f"  {'':<20}" + "".join(f"{f'passo {s}':>11}" for s in checkpoints))
    for name, cls in VARIANTS.items():
        torch.manual_seed(0)
        model = nn.ModuleDict(
            {
                "emb": Embeddings(tok.vocab_size, 64, d_model, 0.0),
                "blocks": nn.Sequential(
                    *(
                        cls(d_model, num_heads, 4 * d_model, 0.0, num_layers)
                        for _ in range(num_layers)
                    )
                ),
                "ln": LayerNorm(d_model),
                "head": nn.Linear(d_model, tok.vocab_size),
            }
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = create_dataloader(
            train_ids, context_length=64, stride=64, batch_size=16, shuffle=True, seed=0
        )
        batches = iter(loader)
        losses = []
        for step in range(steps + 1):
            x, y = next(batches)
            logits = model["head"](model["ln"](model["blocks"](model["emb"](x))))  # [B, T, vocab]
            loss = F.cross_entropy(logits.view(-1, tok.vocab_size), y.view(-1))
            if step in checkpoints:
                losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  {name:<20}" + "".join(f"{v:>11.3f}" for v in losses))


def depth_scaling() -> None:
    section("6. Inicialização por profundidade: quanto L blocos somam ao residual stream")
    d_model, num_heads = 64, 4
    torch.manual_seed(0)
    x = torch.randn(8, 32, d_model)  # residual stream com variância 1 por coordenada
    print("desvio padrão de (saída − entrada) depois de L blocos pre-norm:\n")
    print(f"{'L':>4} | {'W_O e down_proj std 0,02':>25} | {'std 0,02/√(2L)':>15}")
    for num_layers in (1, 4, 16, 64):
        stds = []
        for scaled in (False, True):
            torch.manual_seed(0)
            blocks = nn.Sequential(
                *(
                    TransformerBlock(d_model, num_heads, 4 * d_model, 0.0, num_layers)
                    for _ in range(num_layers)
                )
            )
            if not scaled:
                for block in blocks:
                    nn.init.normal_(block.attention.out_proj.weight, std=0.02)
                    nn.init.normal_(block.feed_forward.down_proj.weight, std=0.02)
            stds.append((blocks(x) - x).std().item())
        print(f"{num_layers:>4} | {stds[0]:>25.4f} | {stds[1]:>15.4f}")


def parameter_count(config: Config) -> None:
    section("7. Parâmetros de um bloco (tiny.yaml)")
    m = config.model
    block = TransformerBlock(m.d_model, m.num_heads, m.d_ff, 0.0, m.num_layers)
    total = sum(p.numel() for p in block.parameters())
    for name, module in block.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<13} {n_params:>8,}  ({n_params / total:5.1%})")
    formula = 12 * m.d_model**2 + 9 * m.d_model
    print(f"  {'total':<13} {total:>8,}  (12·d² + 9·d = {formula:,})")
    print(f"  {m.num_layers} blocos:     {m.num_layers * total:>8,}")


def main() -> None:
    config = load_config(ROOT / "configs" / "tiny.yaml")
    with torch.no_grad():
        layer_norm_step_by_step()
        layer_norm_vs_batch_norm()
        block_walkthrough(config)
    gradient_flow()
    short_training(config)
    with torch.no_grad():
        depth_scaling()
        parameter_count(config)


if __name__ == "__main__":
    main()
