"""Experimento: GPU — escolha automática de device, mesmos números, e o que muda ao acelerar.

Pré-requisito: uv run python train.py   (usa checkpoints/latest.pt para a seção 6)
Rode com:      uv run python -m experiments.e16_gpu
"""

import copy
import time
from pathlib import Path

import torch

from config import ModelConfig
from data.loader import create_dataloader, split_train_val
from device import (
    describe,
    device_memory_mb,
    get_device_rng_state,
    model_device,
    select_device,
    set_device_rng_state,
    synchronize,
)
from model import GPT
from tokenizer import CharTokenizer
from training.checkpoint import load_training_checkpoint
from training.trainer import configure_optimizer, evaluate, train_step

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
AVAILABLE = [("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())]
ACCELERATORS = [name for name, available in AVAILABLE if available]


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def detection() -> None:
    section("1. Detectando o hardware disponível")
    print(f"  PyTorch {torch.__version__}")
    print(f"  torch.cuda.is_available()          {torch.cuda.is_available()}")
    print(f"  torch.backends.mps.is_available()  {torch.backends.mps.is_available()}")
    device = select_device("auto")
    print(f"  select_device('auto')              -> {describe(device)}")
    for name in ("cpu", "cuda", "mps"):
        try:
            select_device(name)
            print(f"  select_device({name!r})".ljust(37) + " -> ok")
        except ValueError as error:
            print(f"  select_device({name!r})".ljust(37) + f" -> ValueError: {error}")


def where_the_tensors_live() -> None:
    section("2. O modelo e os dados precisam estar no mesmo device")
    torch.manual_seed(0)
    config = ModelConfig(context_length=8, d_model=16, num_heads=2, num_layers=2, dropout=0.0)
    model = GPT(config, vocab_size=10)
    print(f"  criado com GPT(...): model_device -> {model_device(model)}")
    if not ACCELERATORS:
        print("  nenhum acelerador neste ambiente: pulando o resto da seção")
        return
    name = ACCELERATORS[0]
    model.to(name)
    print(f"  depois de model.to({name!r}): model_device -> {model_device(model)}")

    x_cpu = torch.randint(0, 10, (2, 8))
    try:
        model(x_cpu)
        print("  model(x_cpu) rodou sem erro (inesperado)")
    except RuntimeError as error:
        print(f"  model(x_cpu) com os pesos em {name!r}: RuntimeError")
        print(f"    {str(error).splitlines()[0]}")
    x_device = x_cpu.to(name)
    logits = model(x_device)
    print(f"  model(x_cpu.to({name!r})): ok, logits em {logits.device}")


def same_numbers_everywhere() -> None:
    section("3. O mesmo treino dá a mesma loss em qualquer device")
    config = ModelConfig(context_length=8, d_model=16, num_heads=2, num_layers=2, dropout=0.0)
    torch.manual_seed(0)
    cpu_model = GPT(config, vocab_size=10)
    ids = torch.arange(10).repeat(8)
    loader = create_dataloader(ids, context_length=8, stride=8, batch_size=4, shuffle=False)
    x, y = next(iter(loader))

    def run(model: GPT) -> tuple[float, float]:
        optimizer = configure_optimizer(model, 1e-3, 0.0)
        result = train_step(model, x, y, optimizer, gradient_clip=1.0)
        grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
        return result.loss, grad_norm

    cpu_loss, cpu_grad = run(copy.deepcopy(cpu_model))
    print(f"  {'device':<8} {'loss':>10} {'norma do grad':>15}")
    print(f"  {'cpu':<8} {cpu_loss:>10.6f} {cpu_grad:>15.6f}")
    for name in ACCELERATORS:
        model = copy.deepcopy(cpu_model).to(name)
        loss, grad = run(model)
        same = abs(loss - cpu_loss) < 1e-4
        print(f"  {name:<8} {loss:>10.6f} {grad:>15.6f}   (== cpu, ~1e-4: {same})")


def timing_a_bigger_model() -> tuple[GPT, torch.Tensor, torch.Tensor]:
    config = ModelConfig(context_length=256, d_model=256, num_heads=4, num_layers=4, dropout=0.1)
    torch.manual_seed(0)
    model = GPT(config, vocab_size=97)
    x = torch.randint(0, 97, (30, 256))
    y = torch.randint(0, 97, (30, 256))
    return model, x, y


def timed_steps(
    model: GPT, x: torch.Tensor, y: torch.Tensor, device: torch.device, steps: int
) -> float:
    model = copy.deepcopy(model).to(device)
    x, y = x.to(device), y.to(device)
    optimizer = configure_optimizer(model, 1e-3, 0.0)
    for _ in range(3):  # aquecimento: primeira chamada compila/aloca kernels
        train_step(model, x, y, optimizer, gradient_clip=1.0)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, x, y, optimizer, gradient_clip=1.0)
    synchronize(device)
    return (time.perf_counter() - start) / steps


def synchronize_and_speed() -> None:
    section("4. Por que sincronizar antes de cronometrar")
    if not ACCELERATORS:
        print("  nenhum acelerador neste ambiente: pulando esta seção")
        return
    name = ACCELERATORS[0]
    device = torch.device(name)
    model, x, y = timing_a_bigger_model()
    accel_model = copy.deepcopy(model).to(device).eval()
    x_device = x.to(device)
    with torch.no_grad():
        for _ in range(3):  # aquecimento: primeira chamada compila/aloca kernels
            accel_model(x_device)

        start = time.perf_counter()
        for _ in range(50):
            accel_model(x_device)  # só enfileira: a CPU não espera o resultado
        before_sync = time.perf_counter() - start
        synchronize(device)  # agora sim: espera a fila esvaziar
        after_sync = time.perf_counter() - start

    print(f"  {'medição':<32} {'ms (50 forwards)':>17}  observação")
    print(f"  {'sem esperar a fila':<32} {before_sync * 1000:>17.1f}  fila ainda cheia")
    print(f"  {'depois de synchronize()':<32} {after_sync * 1000:>17.1f}  tempo real")

    cpu_ms = timed_steps(model, x, y, torch.device("cpu"), steps=5) * 1000
    accel_ms = timed_steps(model, x, y, device, steps=20) * 1000
    print("\n  um passo de treino completo já sincroniza sozinho (train_step lê .item() no fim):")
    print("  modelo do tiny.yaml (30 x 256, d_model 256, 4 heads, 4 layers):")
    print(f"  {'cpu':<8} {cpu_ms:>10.1f} ms/passo")
    print(f"  {name:<8} {accel_ms:>10.1f} ms/passo  (speedup {cpu_ms / accel_ms:.2f}x)")


def rng_per_device() -> None:
    section("5. Cada device sorteia o dropout com o seu próprio gerador")
    if not ACCELERATORS:
        print("  nenhum acelerador neste ambiente: pulando esta seção")
        return
    name = ACCELERATORS[0]
    device = torch.device(name)
    config = ModelConfig(context_length=8, d_model=16, num_heads=2, num_layers=2, dropout=0.5)
    torch.manual_seed(0)
    model = GPT(config, vocab_size=10).to(device).train()
    x = torch.randint(0, 10, (4, 8)).to(device)

    state = get_device_rng_state(device)
    first = model(x)
    second = model(x)
    set_device_rng_state(device, state)
    third = model(x)
    repeats = torch.equal(first, second)
    restored = torch.equal(first, third)
    print(f"  duas chamadas seguidas, mesmo estado inicial: idênticas -> {repeats}")
    print(f"  restaurando o estado do device antes da 3ª chamada: repete a 1ª -> {restored}")

    cpu_before = torch.get_rng_state()
    model(x)  # o dropout deste forward sorteia no gerador do device, não no da CPU
    cpu_untouched = torch.equal(cpu_before, torch.get_rng_state())
    print(f"  chamar o modelo no {name!r} não mexe no gerador global da CPU: {cpu_untouched}")


def device_memory() -> None:
    section("6. Memória do acelerador")
    if not ACCELERATORS:
        print("  nenhum acelerador neste ambiente: pulando esta seção")
        return
    name = ACCELERATORS[0]
    device = torch.device(name)
    model, x, y = timing_a_bigger_model()
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    optimizer = configure_optimizer(model, 1e-3, 0.0)
    print(f"  {'depois de':<20} {'device_memory_mb':>17}")
    print(f"  {'mover o modelo':<20} {device_memory_mb(device) or 0:>17.1f}")
    for i in range(3):
        train_step(model, x, y, optimizer, gradient_clip=1.0)
        print(f"  {f'passo {i + 1}':<20} {device_memory_mb(device) or 0:>17.1f}")


def resuming_across_devices() -> None:
    section("7. Retomando o treino: o mesmo device, ou outro")
    if not CHECKPOINT.is_file():
        print("  checkpoints/latest.pt não encontrado: rode uv run python train.py")
        print("  pulando esta seção")
        return
    checkpoint = load_training_checkpoint(CHECKPOINT)
    saved_on = checkpoint.progress.device_type
    print(f"  checkpoint salvo com device_type = {saved_on!r}")

    text = (ROOT / checkpoint.config.data.corpus_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, checkpoint.config.data.val_fraction)
    seq_len = checkpoint.config.model.context_length
    batch_size = checkpoint.config.training.batch_size
    val_loader = create_dataloader(
        val_ids, context_length=seq_len, stride=seq_len, batch_size=batch_size, shuffle=False
    )
    val_loss = evaluate(checkpoint.model, val_loader)
    print(f"  validação com os pesos como estão: {val_loss:.3f}")

    for name in ["cpu", saved_on]:
        matches = name == saved_on
        where = "mesmo device do checkpoint" if matches else "device diferente"
        action = "usa device_rng_state" if matches else "avisa e ignora o rng do acelerador"
        print(f"  retomar em {name!r}: {where} -> {action}")


def main() -> None:
    detection()
    where_the_tensors_live()
    same_numbers_everywhere()
    synchronize_and_speed()
    rng_per_device()
    device_memory()
    resuming_across_devices()


if __name__ == "__main__":
    main()
