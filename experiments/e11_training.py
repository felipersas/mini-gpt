"""Experimento: as peças do training loop, uma de cada vez.

Rode com:  uv run python -m experiments.e11_training
Comparação de learning rate, agenda e stride (~15 min):
           uv run python -m experiments.e11_training --comparar
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from config import Config, load_config
from data.loader import create_dataloader, split_train_val
from model.gpt import GPT
from tokenizer import CharTokenizer
from training.loss import cross_entropy
from training.trainer import configure_optimizer, evaluate, learning_rate_at, train_step

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def fmt(vector: torch.Tensor, spec: str = "+.5f") -> str:
    return " ".join(f"{v:{spec}}" for v in vector.tolist())


def load_data(config: Config) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, config.data.val_fraction)
    return tok, train_ids, val_ids


def train_loader_for(config: Config, train_ids: torch.Tensor, stride: int | None = None):
    return create_dataloader(
        train_ids,
        context_length=config.model.context_length,
        stride=stride or config.data.stride,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
    )


def one_step(config: Config, vocab_size: int, x: torch.Tensor, y: torch.Tensor) -> None:
    section("1. Um passo de treino, peça por peça")
    torch.manual_seed(0)
    model = GPT(replace(config.model, dropout=0.0), vocab_size)
    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.01)
    w = model.blocks[0].attention.q_proj.weight  # [128, 128]
    before = w[0, :4].detach().clone()

    logits = model(x)
    loss = cross_entropy(logits, y)
    print(f"1. forward:   ids {tuple(x.shape)} -> logits {tuple(logits.shape)}")
    print(f"2. loss:      {loss.item():.4f}")
    print(f"   q_proj.grad antes do backward: {w.grad}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    first_grads = fmt(w.grad[0, :4], "+.2e")
    print(f"3. backward:  q_proj.grad {tuple(w.grad.shape)}; w.grad[0, :4] = {first_grads}")
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"4. clipping:  norma de todos os gradientes = {norm:.3f} (limite 1,0)")
    optimizer.step()
    after = w[0, :4].detach()
    print(f"5. step:      w[0, :4] antes   {fmt(before)}")
    print(f"              w[0, :4] depois  {fmt(after)}")
    print(f"              diferença        {fmt(after - before, '+.2e')}")
    with torch.no_grad():
        new_loss = cross_entropy(model(x), y)
    print(f"\nloss no mesmo batch: {loss.item():.4f} antes do passo, {new_loss.item():.4f} depois")


def why_zero_grad(config: Config, vocab_size: int, x: torch.Tensor, y: torch.Tensor) -> None:
    section("2. Por que zero_grad: sem ele, os gradientes se acumulam")
    torch.manual_seed(0)
    model = GPT(replace(config.model, dropout=0.0), vocab_size)
    w = model.blocks[0].attention.q_proj.weight
    for i in range(1, 4):
        cross_entropy(model(x), y).backward()  # mesmo batch, sem zerar .grad
        print(f"  backward nº {i}: norma de q_proj.grad = {w.grad.norm():.4e}")
    print("  cada backward SOMA ao que já está em .grad; zero_grad limpa antes do próximo passo")


def adam_versus_sgd(config: Config, vocab_size: int, x: torch.Tensor, y: torch.Tensor) -> None:
    section("3. AdamW × SGD: quanto cada peso anda no primeiro passo (lr = 1e-3)")
    names = [
        "embeddings.token_embedding.weight",
        "embeddings.position_embedding.weight",
        "blocks.0.attention.q_proj.weight",
        "blocks.3.feed_forward.down_proj.bias",
        "ln_final.weight",
    ]
    moves = {}
    for label in ("SGD", "AdamW"):
        torch.manual_seed(0)
        model = GPT(replace(config.model, dropout=0.0), vocab_size)
        params = dict(model.named_parameters())
        before = {name: params[name].detach().clone() for name in names}
        if label == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        optimizer.zero_grad()
        cross_entropy(model(x), y).backward()
        grads = {name: params[name].grad.detach().clone() for name in names}
        optimizer.step()
        moves[label] = {
            name: (params[name].detach() - before[name]).abs()[grads[name] != 0].mean().item()
            for name in names
        }
    print(f"  {'parâmetro':<38} {'|grad| médio':>13} {'|Δ| SGD':>10} {'|Δ| AdamW':>10}")
    for name in names:
        grad = grads[name]
        print(
            f"  {name:<38} {grad[grad != 0].abs().mean():>13.2e}"
            f" {moves['SGD'][name]:>10.2e} {moves['AdamW'][name]:>10.2e}"
        )
    print("  SGD anda lr · |grad|; o primeiro passo do AdamW anda ≈ lr em todo peso com gradiente")


def gradient_clipping(config: Config, vocab_size: int, train_ids: torch.Tensor) -> None:
    steps = 100
    section(f"4. Gradient clipping: a norma do gradiente nos primeiros {steps} passos (limite 1,0)")
    torch.manual_seed(0)
    model = GPT(config.model, vocab_size)
    optimizer = configure_optimizer(model, 1e-3, config.training.weight_decay)
    norms = []
    for step, (xb, yb) in enumerate(train_loader_for(config, train_ids)):
        if step == steps:
            break
        norms.append(train_step(model, xb, yb, optimizer, gradient_clip=1.0).grad_norm)
    values = torch.tensor(norms)
    clipped = (values > 1.0).float().mean()
    print(f"  passos 1–5:  {' '.join(f'{v:.2f}' for v in norms[:5])}")
    print(f"  passos 96–100: {' '.join(f'{v:.2f}' for v in norms[-5:])}")
    print(
        f"  mínima {values.min():.2f} | mediana {values.median():.2f} | máxima {values.max():.2f}"
        f" | passos cortados: {clipped:.0%}"
    )


def weight_decay_groups(config: Config, vocab_size: int) -> None:
    section("5. Weight decay: quais parâmetros encolhem")
    model = GPT(config.model, vocab_size)
    names = {id(p): name for name, p in model.named_parameters()}
    optimizer = configure_optimizer(model, 3e-4, config.training.weight_decay)
    for group in optimizer.param_groups:
        params = group["params"]
        examples = ", ".join(names[id(p)] for p in params[:3])
        print(
            f"  weight_decay = {group['weight_decay']}: {len(params)} tensores,"
            f" {sum(p.numel() for p in params):,} parâmetros"
        )
        print(f"    ex.: {examples}, ...")


def schedule(config: Config, batches_per_epoch: int) -> None:
    section("6. A agenda do learning rate no tiny.yaml")
    t = config.training
    total = t.epochs * batches_per_epoch
    print(
        f"  {t.epochs} épocas × {batches_per_epoch} batches = {total} passos"
        f" | warmup {t.warmup_steps}"
        f" | máximo {t.learning_rate} | mínimo {t.min_learning_rate}\n"
    )
    marks = sorted(
        {
            0,
            t.warmup_steps // 2,
            t.warmup_steps - 1,
            t.warmup_steps,
            (t.warmup_steps + total) // 2,
            total - 1,
        }
    )
    for step in marks:
        value = learning_rate_at(
            step,
            max_lr=t.learning_rate,
            min_lr=t.min_learning_rate,
            warmup_steps=t.warmup_steps,
            total_steps=total,
        )
        bar = "█" * round(40 * value / t.learning_rate)
        print(f"  passo {step:>5}: lr {value:.2e} {bar}")


def compare(
    config: Config, vocab_size: int, train_ids: torch.Tensor, val_ids: torch.Tensor
) -> None:
    steps, warmup = 1500, 100
    section(f"7. Comparação: learning rate, agenda e stride ({steps} passos cada)")
    seq_len = config.model.context_length
    val_loader = create_dataloader(
        val_ids, context_length=seq_len, stride=seq_len, batch_size=32, shuffle=False
    )
    runs = [
        ("lr 3e-4 constante, stride 128", 3e-4, False, 128),
        ("lr 1e-3 constante, stride 128", 1e-3, False, 128),
        ("lr 1e-3 cosseno,   stride 128", 1e-3, True, 128),
        ("lr 1e-3 cosseno,   stride 32", 1e-3, True, 32),
    ]
    for name, max_lr, cosine, stride in runs:
        torch.manual_seed(config.training.seed)
        model = GPT(config.model, vocab_size)
        optimizer = configure_optimizer(model, max_lr, config.training.weight_decay)
        loader = train_loader_for(config, train_ids, stride)
        step, line = 0, []
        while step < steps:
            for xb, yb in loader:
                lr = (
                    learning_rate_at(
                        step,
                        max_lr=max_lr,
                        min_lr=max_lr / 10,
                        warmup_steps=warmup,
                        total_steps=steps,
                    )
                    if cosine
                    else max_lr
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                train_step(model, xb, yb, optimizer, config.training.gradient_clip)
                step += 1
                if step % 250 == 0:
                    line.append(evaluate(model, val_loader))
                if step == steps:
                    break
        print(f"  {name:<32} validação a cada 250 passos: {' '.join(f'{v:.3f}' for v in line)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparar", action="store_true", help="roda a comparação longa (seção 7)")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "tiny.yaml")
    tok, train_ids, val_ids = load_data(config)
    loader = train_loader_for(config, train_ids)
    x, y = next(iter(loader))

    one_step(config, tok.vocab_size, x, y)
    why_zero_grad(config, tok.vocab_size, x, y)
    adam_versus_sgd(config, tok.vocab_size, x, y)
    gradient_clipping(config, tok.vocab_size, train_ids)
    weight_decay_groups(config, tok.vocab_size)
    schedule(config, len(loader))
    if args.comparar:
        compare(config, tok.vocab_size, train_ids, val_ids)
    else:
        print("\n(seção 7, a comparação longa, roda com --comparar)")


if __name__ == "__main__":
    main()
