"""Dispositivos: onde o modelo roda (CUDA, MPS ou CPU) e o que muda em cada um."""

from contextlib import AbstractContextManager, nullcontext

import torch


def select_device(name: str = "auto") -> torch.device:
    """`auto` escolhe CUDA, depois MPS (GPU da Apple), depois CPU. Um nome explícito é conferido."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA não está disponível neste ambiente")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS não está disponível neste ambiente")
    return device


def model_device(model: torch.nn.Module) -> torch.device:
    """O device dos parâmetros: é para lá que as entradas do modelo precisam ir."""
    return next(model.parameters()).device


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (GPU da Apple, via Metal)"
    return f"cpu ({torch.get_num_threads()} threads)"


def synchronize(device: torch.device) -> None:
    """Espera o acelerador terminar o trabalho na fila.

    Na GPU, as operações só são enfileiradas; sem sincronizar, um cronômetro mede o envio.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_device_rng_state(device: torch.device) -> torch.Tensor | None:
    """Estado do gerador aleatório do acelerador. Na CPU é None: o gerador global já cobre."""
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


def autocast(device: torch.device, enabled: bool) -> AbstractContextManager:
    """Contexto de mixed precision: forward em bfloat16; pesos e gradientes seguem em float32.

    bfloat16 tem o mesmo alcance de expoente do float32 (só menos casas de mantissa), então, ao
    contrário do float16, não corre risco de estourar (overflow) — dispensa o GradScaler.
    """
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def device_memory_mb(device: torch.device) -> float | None:
    """Memória do acelerador, em MB: pico de tensores (CUDA) ou reservado pelo driver (MPS)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None
