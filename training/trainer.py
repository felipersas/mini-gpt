"""Training loop: forward, loss, backward, clipping e passo do AdamW."""

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from device import (
    autocast,
    device_memory_mb,
    get_device_rng_state,
    model_device,
    set_device_rng_state,
)
from training.loss import cross_entropy
from training.metrics import peak_memory_mb, perplexity


def configure_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.AdamW:
    """AdamW com weight decay só nos parâmetros com 2 ou mais dimensões (matrizes).

    Biases e os γ/β das LayerNorms são vetores e ficam sem decay.
    """
    params = list(model.parameters())  # tensores compartilhados (weight tying) aparecem uma vez
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate)


def learning_rate_at(
    step: int, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int
) -> float:
    """Warmup linear até max_lr e depois decaimento cosseno até min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    # cos(π · progress) vai de 1 a −1: o lr vai de max_lr a min_lr.
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@dataclass
class StepResult:
    loss: float
    grad_norm: float  # norma de todos os gradientes juntos, antes do clipping


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float,
    mixed_precision: bool = False,
) -> StepResult:
    """Um passo de otimização em um batch."""
    model.train()  # dropout ligado
    device = model_device(model)
    x, y = x.to(device), y.to(device)  # o DataLoader entrega os batches na CPU
    with autocast(device, mixed_precision):
        logits = model(x)  # [batch, seq_len, vocab_size]
        loss = cross_entropy(logits, y)  # escalar; PyTorch mantém o logsumexp em float32
    # Sem isto, o backward somaria os gradientes novos aos do passo anterior.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # preenche param.grad de todos os parâmetros
    # Se a norma total passar de gradient_clip, todos os gradientes são reescalados para ela.
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()  # atualiza os parâmetros
    # .item() copia os valores para a CPU e, na GPU, espera o passo terminar.
    return StepResult(loss=loss.item(), grad_norm=grad_norm.item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, mixed_precision: bool = False) -> float:
    """Loss média por token em todo o loader, sem dropout e sem gradientes."""
    was_training = model.training
    model.eval()
    device = model_device(model)
    total, count = 0.0, 0
    with autocast(device, mixed_precision):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += cross_entropy(model(x), y).item() * y.numel()
            count += y.numel()
    model.train(was_training)
    return total / count


@dataclass
class EpochRecord:
    epoch: int
    step: int
    train_loss: float  # média das losses dos batches da época, com dropout e pesos mudando
    train_eval_loss: float | None  # loss no treino ao fim da época, sem dropout (se medida)
    val_loss: float
    learning_rate: float  # lr do último passo da época
    grad_norm: float  # média, na época, da norma do gradiente antes do clipping
    tokens_per_second: float  # só o tempo dos passos de treino, sem a avaliação
    peak_memory_mb: float  # pico de memória do processo até o fim da época
    seconds: float  # tempo acumulado desde o início do treino
    device_memory_mb: float | None = None  # memória do acelerador ao fim da época; None na CPU


@dataclass
class TrainingProgress:
    """Onde o treino está: o que, além dos pesos e do otimizador, é preciso para continuar dele."""

    step: int = 0  # passos de otimização já dados
    epoch: int = 1  # época em andamento
    batches_done: int = 0  # batches da época em andamento já usados
    # Acumulados da época em andamento, para o EpochRecord sair igual ao de um treino sem pausa.
    epoch_losses: list[float] = field(default_factory=list)
    epoch_grad_norms: list[float] = field(default_factory=list)
    epoch_tokens: int = 0
    epoch_train_seconds: float = 0.0
    seconds: float = 0.0  # tempo acumulado desde o início do treino
    history: list[EpochRecord] = field(default_factory=list)
    # Estado do gerador que sorteia a ordem das janelas, no início da época em andamento.
    loader_rng_state: torch.Tensor | None = None
    # Estado do gerador global da CPU, que sorteia as máscaras de dropout com o modelo na CPU.
    rng_state: torch.Tensor | None = None
    # Numa GPU, o dropout usa o gerador do próprio acelerador.
    device_rng_state: torch.Tensor | None = None
    device_type: str = "cpu"  # onde o treino estava rodando quando o progresso foi salvo


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    gradient_clip: float,
    log_every: int,
    train_eval_loader: DataLoader | None = None,
    progress: TrainingProgress | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[TrainingProgress], None] | None = None,
    log: Callable[[str], None] = print,
    mixed_precision: bool = False,
) -> list[EpochRecord]:
    """Treina até completar `epochs` passadas no train_loader, avaliando ao fim de cada uma.

    O treino roda no device do modelo. Com `train_eval_loader`, também mede a loss de treino sem
    dropout, comparável à de validação. Com `progress` (vindo de um checkpoint), continua
    exatamente do ponto em que o treino parou. `on_checkpoint` recebe uma cópia do progresso a cada
    `checkpoint_every` passos e ao fim de cada época. Com `mixed_precision`, forward e loss rodam em
    bfloat16 no acelerador; os pesos e o `optimizer.step()` continuam em float32.
    """
    device = model_device(model)
    total_steps = epochs * len(train_loader)
    generator = train_loader.generator  # sorteia a ordem das janelas a cada época
    progress = copy.deepcopy(progress) if progress is not None else TrainingProgress()
    if progress.loader_rng_state is not None and generator is not None:
        generator.set_state(progress.loader_rng_state)
    if progress.rng_state is not None:
        torch.set_rng_state(progress.rng_state)
    if progress.step > 0 and progress.device_type != device.type:
        log(
            f"aviso: o progresso foi salvo em '{progress.device_type}' e o treino continua em"
            f" '{device.type}': os números não serão idênticos aos de um treino sem pausa"
        )
    elif progress.device_rng_state is not None:
        set_device_rng_state(device, progress.device_rng_state)
    start = time.perf_counter() - progress.seconds

    def checkpoint() -> None:
        if on_checkpoint is not None:
            progress.seconds = time.perf_counter() - start
            progress.rng_state = torch.get_rng_state()
            progress.device_rng_state = get_device_rng_state(device)
            progress.device_type = device.type
            on_checkpoint(copy.deepcopy(progress))

    while progress.epoch <= epochs:
        if generator is not None:
            # A ordem das janelas desta época é sorteada a partir deste estado.
            progress.loader_rng_state = generator.get_state()
        batches = iter(train_loader)
        # Ao retomar no meio de uma época: a mesma ordem, pulando os batches já usados.
        for _ in range(progress.batches_done):
            next(batches)

        for x, y in batches:
            lr = learning_rate_at(
                progress.step,
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            step_start = time.perf_counter()
            result = train_step(model, x, y, optimizer, gradient_clip, mixed_precision)
            step_seconds = time.perf_counter() - step_start
            progress.step += 1
            progress.batches_done += 1
            progress.epoch_losses.append(result.loss)
            progress.epoch_grad_norms.append(result.grad_norm)
            progress.epoch_tokens += y.numel()
            progress.epoch_train_seconds += step_seconds

            if progress.step == 1 or progress.step % log_every == 0:
                log(
                    f"passo {progress.step:>5}/{total_steps} | época {progress.epoch:>2}"
                    f" | loss {result.loss:.3f} | lr {lr:.2e} | grad {result.grad_norm:.2f}"
                    f" | {y.numel() / step_seconds:,.0f} tokens/s"
                )
            end_of_epoch = progress.batches_done == len(train_loader)
            if checkpoint_every and progress.step % checkpoint_every == 0 and not end_of_epoch:
                checkpoint()

        val_loss = evaluate(model, val_loader, mixed_precision)
        train_eval_loss = (
            evaluate(model, train_eval_loader, mixed_precision)
            if train_eval_loader is not None
            else None
        )
        record = EpochRecord(
            epoch=progress.epoch,
            step=progress.step,
            train_loss=sum(progress.epoch_losses) / len(progress.epoch_losses),
            train_eval_loss=train_eval_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
            grad_norm=sum(progress.epoch_grad_norms) / len(progress.epoch_grad_norms),
            tokens_per_second=progress.epoch_tokens / progress.epoch_train_seconds,
            peak_memory_mb=peak_memory_mb(),
            seconds=time.perf_counter() - start,
            device_memory_mb=device_memory_mb(device),
        )
        progress.history.append(record)

        train_part = f"loss treino {record.train_loss:.3f}"
        if train_eval_loss is not None:
            train_part += f" (sem dropout {train_eval_loss:.3f})"
        memory = f"{record.peak_memory_mb:.0f} MB"
        if record.device_memory_mb is not None:
            memory += f" (+ {record.device_memory_mb:.0f} MB na {device.type})"
        log(
            f"== época {record.epoch:>2}/{epochs} | {train_part} | loss validação {val_loss:.3f}"
            f" (perplexidade {perplexity(val_loss):.2f}) | {record.tokens_per_second:,.0f} tokens/s"
            f" | {memory} | {record.seconds:.0f} s"
        )

        progress.epoch += 1
        progress.batches_done = 0
        progress.epoch_losses, progress.epoch_grad_norms = [], []
        progress.epoch_tokens, progress.epoch_train_seconds = 0, 0.0
        if generator is not None:
            progress.loader_rng_state = generator.get_state()  # a próxima época começa daqui
        checkpoint()
    return progress.history
