"""Ponto de entrada do treinamento."""

import argparse
import math
from pathlib import Path

import torch

from config import load_config
from data.loader import create_dataloader, split_train_val
from device import describe, select_device
from model import GPT
from tokenizer import CharTokenizer
from training.checkpoint import load_training_checkpoint, save_checkpoint
from training.metrics import bits_per_token, perplexity, save_history
from training.trainer import TrainingProgress, configure_optimizer, evaluate, train

ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "configs" / "tiny.yaml"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o Mini-GPT.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="arquivo YAML de configuração")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="onde salvar os checkpoints e o histórico",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="checkpoint de onde continuar o treino (usa a config salva nele)",
    )
    parser.add_argument("--epochs", type=int, help="total de épocas, no lugar do valor da config")
    parser.add_argument(
        "--device", default="auto", help="auto (CUDA, depois MPS, depois CPU), cuda, mps ou cpu"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="força bfloat16 num acelerador, no lugar do valor da config",
    )
    args = parser.parse_args()

    resumed = load_training_checkpoint(args.resume) if args.resume else None
    config = resumed.config if resumed else load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.mixed_precision:
        config.training.mixed_precision = True
    training = config.training

    corpus_path = ROOT / config.data.corpus_path
    if not corpus_path.is_file():
        raise SystemExit(
            f"Corpus não encontrado: {corpus_path}\nRode: uv run python scripts/prepare_corpus.py"
        )
    text = corpus_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    if resumed and resumed.tokenizer.vocab.id_to_token != tokenizer.vocab.id_to_token:
        raise SystemExit("O corpus mudou desde o checkpoint: os IDs não correspondem aos pesos.")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, config.data.val_fraction)

    context_length = config.model.context_length
    train_loader = create_dataloader(
        train_ids,
        context_length=context_length,
        stride=config.data.stride,
        batch_size=training.batch_size,
        shuffle=True,
        seed=training.seed,
    )
    # Mesmo texto do treino, em ordem fixa: mede a loss de treino sem dropout ao fim de cada época.
    train_eval_loader = create_dataloader(
        train_ids,
        context_length=context_length,
        stride=context_length,
        batch_size=training.batch_size,
        shuffle=False,
    )
    # Validação sem sobreposição: cada token é avaliado uma única vez.
    val_loader = create_dataloader(
        val_ids,
        context_length=context_length,
        stride=context_length,
        batch_size=training.batch_size,
        shuffle=False,
    )

    device = select_device(args.device)
    progress: TrainingProgress | None = None
    if resumed:
        model = resumed.model.to(device)
        # O otimizador vem depois de mover o modelo: o estado carregado segue o device dos pesos.
        optimizer = configure_optimizer(model, training.learning_rate, training.weight_decay)
        optimizer.load_state_dict(resumed.optimizer_state)  # m e v de cada parâmetro
        progress = resumed.progress
    else:
        torch.manual_seed(training.seed)
        # Criado na CPU e depois movido: os pesos iniciais são os mesmos em qualquer device.
        model = GPT(config.model, vocab_size=tokenizer.vocab_size).to(device)
        optimizer = configure_optimizer(model, training.learning_rate, training.weight_decay)
    total_steps = training.epochs * len(train_loader)

    tying = "com" if config.model.tie_weights else "sem"
    amp = "bfloat16" if training.mixed_precision else "float32"
    print(f"PyTorch {torch.__version__} | device: {describe(device)}")
    print(
        f"corpus:     {corpus_path.name} ({len(text):,} caracteres,"
        f" vocab_size {tokenizer.vocab_size})"
    )
    print(
        f"dados:      treino {len(train_ids):,} tokens, {len(train_loader)} batches/época"
        f" | validação {len(val_ids):,} tokens, {len(val_loader)} batches"
    )
    print(
        f"modelo:     GPT com {model.num_parameters():,} parâmetros ({tying} weight tying)"
        f" | attention {config.model.attention_backend} | precisão {amp}"
    )
    print(
        f"treino:     {training.epochs} épocas = {total_steps:,} passos"
        f" | lr {training.learning_rate} -> {training.min_learning_rate}"
        f" (warmup {training.warmup_steps})"
        f" | weight decay {training.weight_decay} | clip {training.gradient_clip}"
    )
    checkpoint_dir = args.checkpoint_dir
    print(
        f"checkpoints: {checkpoint_dir} | latest.pt a cada época"
        f" | checkpoint_<passo>.pt a cada {training.checkpoint_every} passos | best.pt"
    )
    if progress is not None:
        print(
            f"retomando:  {args.resume} | passo {progress.step:,} | época {progress.epoch}"
            f" ({progress.batches_done} batches já feitos)\n"
        )
    else:
        initial = evaluate(model, val_loader, training.mixed_precision)
        print(
            f"loss de validação antes do treino: {initial:.3f} (chute uniforme: "
            f"{math.log(tokenizer.vocab_size):.3f})\n"
        )

    def on_checkpoint(state: TrainingProgress) -> None:
        names = ["latest.pt"]
        if training.checkpoint_every and state.step % training.checkpoint_every == 0:
            names.append(f"checkpoint_{state.step}.pt")
        if state.batches_done == 0:  # fim de época: há um novo registro no histórico
            save_history(checkpoint_dir / "history.json", state.history)
            if state.history[-1].val_loss <= min(r.val_loss for r in state.history):
                names.append("best.pt")
        for name in names:
            save_checkpoint(
                checkpoint_dir / name,
                model,
                tokenizer,
                config=config,
                optimizer=optimizer,
                progress=state,
            )
        print(f"   salvo (passo {state.step}): {', '.join(names)}")

    history = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=training.epochs,
        max_lr=training.learning_rate,
        min_lr=training.min_learning_rate,
        warmup_steps=training.warmup_steps,
        gradient_clip=training.gradient_clip,
        log_every=training.log_every,
        train_eval_loader=train_eval_loader,
        progress=progress,
        checkpoint_every=training.checkpoint_every,
        on_checkpoint=on_checkpoint,
        mixed_precision=training.mixed_precision,
    )
    if not history:
        raise SystemExit("Nenhuma época completa: nada a relatar.")

    final = history[-1]
    best = min(history, key=lambda record: record.val_loss)
    print(
        f"\nfim: loss de validação {final.val_loss:.3f}"
        f" | perplexidade {perplexity(final.val_loss):.2f}"
        f" | {bits_per_token(final.val_loss):.3f} bits/caractere | {final.seconds:.0f} s"
    )
    if final.train_eval_loss is not None:
        print(
            f"loss de treino sem dropout: {final.train_eval_loss:.3f}"
            f" | validação − treino: {final.val_loss - final.train_eval_loss:+.3f}"
        )
    print(f"melhor época: {best.epoch} (validação {best.val_loss:.3f}) -> {checkpoint_dir}/best.pt")
    print(f"último estado: {checkpoint_dir}/latest.pt | histórico: {checkpoint_dir}/history.json")


if __name__ == "__main__":
    main()
