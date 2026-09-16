"""Experimento: checkpoints — o que guardar para continuar o treino, e o que falta sem cada parte.

Pré-requisito: uv run python train.py   (salva latest.pt, best.pt e checkpoint_<passo>.pt)
Rode com:      uv run python -m experiments.e15_checkpoints
"""

import copy
import io
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from data.loader import create_dataloader, split_train_val
from generation.generate import generate
from tokenizer import CharTokenizer
from training.checkpoint import (
    TrainingCheckpoint,
    load_checkpoint,
    load_training_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from training.trainer import TrainingProgress, configure_optimizer, evaluate, train, train_step

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
MEGABYTE = 2**20


class Stop(Exception):
    """Interrompe o treino de propósito, ao fim da época."""


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def serialized_mb(value) -> float:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getbuffer().nbytes / MEGABYTE


def make_loaders(config: Config, tokenizer: CharTokenizer):
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, config.data.val_fraction)
    seq_len, batch_size = config.model.context_length, config.training.batch_size
    train_loader = create_dataloader(
        train_ids,
        context_length=seq_len,
        stride=config.data.stride,
        batch_size=batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    val_loader = create_dataloader(
        val_ids, context_length=seq_len, stride=seq_len, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def anatomy() -> None:
    section("1. O que há dentro de um checkpoint de treino (checkpoints/latest.pt)")
    path = CHECKPOINTS / "latest.pt"
    checkpoint = read_checkpoint(path)
    optimizer_state = checkpoint["optimizer_state"]
    descriptions = {
        "model_config": "hiperparâmetros da arquitetura",
        "vocab": f"os {len(checkpoint['vocab'])} tokens, na ordem dos IDs",
        "model_state": f"{len(checkpoint['model_state'])} tensores de pesos, por nome",
        "config": "a config inteira: dados, modelo e treino",
        "optimizer_state": f"m e v do AdamW para {len(optimizer_state['state'])} tensores",
        "progress": "passo, época, posição no loader, histórico, estados aleatórios",
    }
    print(f"  {'chave':<16} {'tamanho':>8}  conteúdo")
    for key, value in checkpoint.items():
        print(f"  {key:<16} {serialized_mb(value):>5.2f} MB  {descriptions[key]}")
    print(f"  {'arquivo inteiro':<16} {path.stat().st_size / MEGABYTE:>5.2f} MB")

    model, tokenizer = load_checkpoint(path)
    with tempfile.TemporaryDirectory() as directory:
        weights_only = Path(directory) / "pesos.pt"
        save_checkpoint(weights_only, model, tokenizer)
        size = weights_only.stat().st_size / MEGABYTE
    print(f"  {'só os pesos':<16} {size:>5.2f} MB  (o formato usado na geração de texto)")

    print("\n  optimizer_state:")
    for i, group in enumerate(optimizer_state["param_groups"]):
        print(
            f"    grupo {i}: {len(group['params'])} tensores | lr {group['lr']}"
            f" | betas {group['betas']} | eps {group['eps']}"
            f" | weight_decay {group['weight_decay']}"
        )
    first_name = next(name for name, p in model.named_parameters() if p.dim() >= 2)
    first = optimizer_state["state"][0]
    shapes = ", ".join(
        f"{key} {tuple(value.shape)}" for key, value in first.items() if value.dim() > 0
    )
    print(f"    estado 0 ({first_name}): {shapes} | step {first['step'].item():.0f}")

    print("\n  progress:")
    for key, value in checkpoint["progress"].items():
        if isinstance(value, torch.Tensor):
            shown = f"tensor de {value.numel():,} bytes"
        elif isinstance(value, list):
            shown = f"lista com {len(value)} itens"
        elif isinstance(value, float):
            shown = f"{value:.1f}"
        else:
            shown = repr(value)
        print(f"    {key:<20} {shown}")


def saved_checkpoints() -> None:
    section("2. Os checkpoints salvos pelo train.py")
    loaded = [(path, load_training_checkpoint(path)) for path in CHECKPOINTS.glob("*.pt")]
    loaded.sort(key=lambda item: (item[1].progress.step, item[0].name))
    first = loaded[0][1]
    _, val_loader = make_loaders(first.config, first.tokenizer)

    print(
        f"  {'arquivo':<20} {'tamanho':>8} {'passo':>6} {'época':>6} {'batches feitos':>15}"
        f" {'validação agora':>16}"
    )
    for path, checkpoint in loaded:
        progress = checkpoint.progress
        val_loss = evaluate(checkpoint.model, val_loader)
        print(
            f"  {path.name:<20} {path.stat().st_size / MEGABYTE:>5.1f} MB {progress.step:>6,}"
            f" {progress.epoch:>6} {progress.batches_done:>15} {val_loss:>16.3f}"
        )

    prompt = "Capitú"
    print(
        f"\n  a mesma geração em cada checkpoint ({prompt!r}, seed 0, temperature 0.8, top-p 0.95):"
    )
    for path, checkpoint in loaded:
        tokenizer = checkpoint.tokenizer
        ids = generate(
            checkpoint.model,
            tokenizer.encode(prompt),
            80,
            temperature=0.8,
            top_k=None,
            top_p=0.95,
            eos_id=tokenizer.vocab.eos_id,
            generator=torch.Generator().manual_seed(0),
        )
        print(f"  {path.name:<20} {tokenizer.decode(ids).replace(chr(10), ' ')}")


def resume_run(
    checkpoint: TrainingCheckpoint,
    progress: TrainingProgress,
    *,
    restore_optimizer: bool,
    epochs: int,
) -> tuple[TrainingProgress, TrainingProgress]:
    """Treina a partir do checkpoint até o fim da época em andamento.

    Devolve o progresso logo antes do último passo (com as losses de cada passo da época) e o
    progresso no fim da época (com o registro da época no histórico).
    """
    torch.manual_seed(0)  # só importa para as variantes que não restauram o estado aleatório
    train_loader, val_loader = make_loaders(checkpoint.config, checkpoint.tokenizer)
    model = copy.deepcopy(checkpoint.model)
    training = checkpoint.config.training
    optimizer = configure_optimizer(model, training.learning_rate, training.weight_decay)
    if restore_optimizer:
        # Cópia: o AdamW altera m e v no lugar, e o mesmo checkpoint serve a várias variantes.
        optimizer.load_state_dict(copy.deepcopy(checkpoint.optimizer_state))
    states: dict[str, TrainingProgress] = {}

    def watch(state: TrainingProgress) -> None:
        if state.batches_done == 0:
            states["end"] = state
            raise Stop
        states["last"] = state

    try:
        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            epochs=epochs,
            max_lr=training.learning_rate,
            min_lr=training.min_learning_rate,
            warmup_steps=training.warmup_steps,
            gradient_clip=training.gradient_clip,
            log_every=10**9,
            progress=progress,
            checkpoint_every=1,
            on_checkpoint=watch,
            log=lambda line: None,
        )
    except Stop:
        pass
    return states["last"], states["end"]


def resume_variants() -> None:
    section("3. Retomando do checkpoint_1000.pt: o que falta sem cada parte")
    checkpoint = load_training_checkpoint(CHECKPOINTS / "checkpoint_1000.pt")
    base = checkpoint.progress
    history = load_training_checkpoint(CHECKPOINTS / "latest.pt").progress.history
    original = history[base.epoch - 1]
    print(f"  checkpoint: passo {base.step}, época {base.epoch}, batch {base.batches_done}")
    print(
        f"  treino original sem pausa, época {original.epoch}: loss de treino"
        f" {original.train_loss:.6f} | validação {original.val_loss:.6f}\n"
    )

    restart_epoch = {
        "batches_done": 0,
        "epoch_losses": [],
        "epoch_grad_norms": [],
        "epoch_tokens": 0,
        "epoch_train_seconds": 0.0,
        "loader_rng_state": None,
    }
    variants = [
        ("completo", base, True),
        ("sem o estado do AdamW", base, False),
        ("sem o estado aleatório", replace(base, rng_state=None), True),
        ("sem a posição no loader", replace(base, **restart_epoch), True),
        ("só os pesos", TrainingProgress(), False),
    ]
    print(
        f"  {'retomada':<24} {'passos':>6} {'loss 10 1ºs passos':>19} {'loss treino':>12}"
        f" {'validação':>10} {'− original':>11}"
    )
    for name, progress, restore_optimizer in variants:
        last, end = resume_run(
            checkpoint,
            progress,
            restore_optimizer=restore_optimizer,
            epochs=checkpoint.config.training.epochs,
        )
        record = end.history[-1]
        start = progress.batches_done
        first_steps = last.epoch_losses[start : start + 10]
        print(
            f"  {name:<24} {end.step - progress.step:>6} {sum(first_steps) / 10:>19.3f}"
            f" {record.train_loss:>12.6f} {record.val_loss:>10.6f}"
            f" {record.val_loss - original.val_loss:>+11.6f}"
        )
        if name == "completo":
            exact = (record.train_loss, record.val_loss) == (original.train_loss, original.val_loss)
    print(f"\n  retomada completa == treino original, bit a bit: {exact}")


def first_step_size(checkpoint: TrainingCheckpoint, batch, *, restore_optimizer: bool) -> float:
    """Quanto cada peso anda, em média, no primeiro passo depois de retomar, em unidades de lr."""
    model = copy.deepcopy(checkpoint.model)
    training = checkpoint.config.training
    optimizer = configure_optimizer(model, training.learning_rate, training.weight_decay)
    if restore_optimizer:
        optimizer.load_state_dict(copy.deepcopy(checkpoint.optimizer_state))
    before = [p.detach().clone() for p in model.parameters()]
    torch.manual_seed(0)  # as mesmas máscaras de dropout nas duas variantes
    train_step(model, *batch, optimizer, training.gradient_clip)
    changes = [
        (p.detach() - b).abs().flatten() for p, b in zip(model.parameters(), before, strict=True)
    ]
    return torch.cat(changes).mean().item() / training.learning_rate


def continue_after_the_end() -> None:
    section("4. Uma época a mais depois do fim (latest.pt, época 31)")
    checkpoint = load_training_checkpoint(CHECKPOINTS / "latest.pt")
    last_record = checkpoint.progress.history[-1]
    print(
        f"  latest.pt: passo {checkpoint.progress.step}, {last_record.epoch} épocas completas,"
        f" validação {last_record.val_loss:.3f}\n"
    )
    train_loader, _ = make_loaders(checkpoint.config, checkpoint.tokenizer)
    batch = next(iter(train_loader))
    print(
        f"  {'retomada':<24} {'|Δw| no 1º passo':>17} {'loss dos 5 primeiros passos':>32}"
        f" {'validação depois':>17}"
    )
    for name, restore_optimizer in [("completo", True), ("sem o estado do AdamW", False)]:
        size = first_step_size(checkpoint, batch, restore_optimizer=restore_optimizer)
        last, end = resume_run(
            checkpoint,
            checkpoint.progress,
            restore_optimizer=restore_optimizer,
            epochs=last_record.epoch + 1,
        )
        first_steps = " ".join(f"{loss:.3f}" for loss in last.epoch_losses[:5])
        print(
            f"  {name:<24} {size:>11.2f} × lr {first_steps:>32} {end.history[-1].val_loss:>17.3f}"
        )


def broken_file() -> None:
    section("5. Um checkpoint gravado pela metade")
    data = (CHECKPOINTS / "latest.pt").read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "latest.pt"
        path.write_bytes(data[: len(data) // 2])
        try:
            read_checkpoint(path)
            print("  carregou, inesperadamente")
        except Exception as error:
            message = str(error).splitlines()[0][:70]
            size = len(data) // 2 / MEGABYTE
            print(f"  metade do arquivo ({size:.1f} MB): {type(error).__name__}: {message}")


def vocabulary_matters() -> None:
    section("6. Por que o vocabulário vai dentro do checkpoint")
    checkpoint = load_training_checkpoint(CHECKPOINTS / "latest.pt")
    tokenizer = checkpoint.tokenizer
    text = (ROOT / checkpoint.config.data.corpus_path).read_text(encoding="utf-8")
    new_char = next(c for c in "@#$%&*+=" if c not in tokenizer.vocab.id_to_token)
    other = CharTokenizer.from_text(text + new_char)
    moved = [
        token
        for token in tokenizer.vocab.id_to_token
        if other.vocab.get_id(token) != tokenizer.vocab.get_id(token)
    ]
    print(
        f"  o corpus ganha um {new_char!r}: vocab_size {tokenizer.vocab_size} -> {other.vocab_size}"
    )
    print(f"  tokens que mudaram de ID: {len(moved)} de {tokenizer.vocab_size}")
    word = "Capitú"
    old_ids = tokenizer.encode(word)
    print(f"  {word!r} com o vocabulário do checkpoint: {old_ids}")
    print(f"  {word!r} com o vocabulário do corpus novo: {other.encode(word)}")
    print(f"  os IDs do checkpoint lidos com o vocabulário novo: {other.decode(old_ids)!r}")


def main() -> None:
    required = [CHECKPOINTS / "latest.pt", CHECKPOINTS / "checkpoint_1000.pt"]
    if not all(path.is_file() for path in required):
        raise SystemExit("Checkpoints de treino não encontrados.\nRode: uv run python train.py")
    anatomy()
    saved_checkpoints()
    resume_variants()
    continue_after_the_end()
    broken_file()
    vocabulary_matters()


if __name__ == "__main__":
    main()
