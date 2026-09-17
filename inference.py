"""Gera texto com um modelo treinado (checkpoint salvo pelo train.py)."""

import argparse
from pathlib import Path

import torch

from device import select_device
from generation.generate import generate, trim_trailing_partial_word
from training.checkpoint import load_checkpoint

ROOT = Path(__file__).parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera texto com o Mini-GPT.")
    parser.add_argument("prompt", nargs="?", default="o gato", help="texto inicial")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="modelo treinado")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="tokens a gerar")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=None, help="só os k mais prováveis")
    parser.add_argument("--top-p", type=float, default=0.95, help="massa de probabilidade")
    parser.add_argument("--seed", type=int, default=None, help="para repetir a mesma geração")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps ou cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint não encontrado: {checkpoint}\nTreine antes: uv run python train.py"
        )
    model, tokenizer = load_checkpoint(checkpoint)
    model.to(select_device(args.device))
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    ids = generate(
        model,
        tokenizer.encode(args.prompt),
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tokenizer.vocab.eos_id,
        generator=generator,
    )
    print(trim_trailing_partial_word(tokenizer.decode(ids)))


if __name__ == "__main__":
    main()
