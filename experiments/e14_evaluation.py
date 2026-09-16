"""Experimento: avaliando o modelo treinado — curvas, onde ele erra e para onde a attention olha.

Pré-requisito: uv run python train.py   (salva checkpoints/latest.pt e checkpoints/history.json)
Rode com:      uv run python -m experiments.e14_evaluation
"""

import math
from pathlib import Path

import torch

from config import load_config
from data.loader import create_dataloader, split_train_val
from experiments.e10_loss import ngram_loss
from model.attention import attention_weights, causal_mask, merge_heads, split_heads
from model.gpt import GPT
from tokenizer import CharTokenizer
from training.checkpoint import load_checkpoint
from training.loss import cross_entropy, token_losses
from training.metrics import bits_per_token, load_history, perplexity
from training.trainer import evaluate

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
HISTORY = ROOT / "checkpoints" / "history.json"
PUNCTUATION = set(".,;:!?-'()«»\"")


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def bar(value: float, scale: float, width: int = 30) -> str:
    return "█" * max(0, round(width * value / scale))


def final_metrics(
    model: GPT, tok: CharTokenizer, train_ids, val_ids, train_loader, val_loader
) -> None:
    section("1. As métricas do modelo treinado")
    print(f"  {'':<22} {'loss (nats)':>11} {'perplexidade':>13} {'bits/caractere':>15}")
    losses = {}
    for name, loader in [("treino, sem dropout", train_loader), ("validação", val_loader)]:
        loss = evaluate(model, loader)
        losses[name] = loss
        print(f"  {name:<22} {loss:>11.3f} {perplexity(loss):>13.2f} {bits_per_token(loss):>15.3f}")
    gap = losses["validação"] - losses["treino, sem dropout"]
    print(f"  diferença validação − treino: {gap:+.3f}")

    print("\n  réguas na validação (docs/10-loss.md):")
    rulers = [("chute uniforme", math.log(tok.vocab_size))]
    for order, name in [(1, "unigrama"), (2, "bigrama"), (3, "trigrama")]:
        best = min(ngram_loss(train_ids, val_ids, tok.vocab_size, order, a) for a in (1, 0.1, 0.01))
        rulers.append((name, best))
    rulers.append(("GPT treinado", losses["validação"]))
    for name, loss in rulers:
        print(f"  {name:<22} {loss:>11.3f} {perplexity(loss):>13.2f} {bits_per_token(loss):>15.3f}")


def curves(history: list[dict]) -> None:
    section("2. Curvas de aprendizado (checkpoints/history.json)")
    header = (
        f"  {'época':>5} | {'treino c/ dropout':>17} | {'treino s/ dropout':>17} | {'validação':>9}"
        f" | {'grad':>5} | {'tokens/s':>8} | {'memória':>8}"
    )
    print(header)
    shown = {1, 2, 3, 5, 10, 15, 20, 25, 27, 28, 29, 30}
    for r in history:
        if r["epoch"] in shown:
            print(
                f"  {r['epoch']:>5} | {r['train_loss']:>17.3f} | {r['train_eval_loss']:>17.3f}"
                f" | {r['val_loss']:>9.3f} | {r['grad_norm']:>5.2f}"
                f" | {r['tokens_per_second']:>8,.0f} | {r['peak_memory_mb']:>5.0f} MB"
            )

    train_values = [r["train_eval_loss"] for r in history]
    val_values = [r["val_loss"] for r in history]
    top, bottom, height = max(train_values + val_values), min(train_values + val_values), 12
    rows = [[" "] * len(history) for _ in range(height)]
    for mark, values in (("t", train_values), ("v", val_values)):
        for i, value in enumerate(values):
            row = round((top - value) / (top - bottom) * (height - 1))
            rows[row][i] = "*" if rows[row][i] not in (" ", mark) else mark
    print("\n  loss por época (t = treino sem dropout, v = validação, * = as duas)")
    for r, row in enumerate(rows):
        level = top - (top - bottom) * r / (height - 1)
        print(f"  {level:5.2f} │ " + " ".join(row))
    print("        └─" + "──" * len(history))
    print("          " + "".join(f"{e:<10}" for e in range(1, len(history) + 1, 5)))


@torch.no_grad()
def predictions(model: GPT, loader) -> dict[str, torch.Tensor]:
    """Entradas, alvos, loss por token, confiança e acerto de cada posição da validação."""
    model.eval()
    parts: dict[str, list[torch.Tensor]] = {"x": [], "y": [], "loss": [], "conf": [], "hit": []}
    for x, y in loader:
        logits = model(x)  # [B, T, V]
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)  # [B, T]
        parts["x"].append(x)
        parts["y"].append(y)
        parts["loss"].append(token_losses(logits, y))
        parts["conf"].append(confidence)
        parts["hit"].append(predicted == y)
    return {name: torch.cat(values) for name, values in parts.items()}


def by_position(result: dict[str, torch.Tensor]) -> None:
    section("3. Onde o modelo erra: a posição dentro da janela")
    losses = result["loss"]  # [janelas, T]
    print("  a posição t prevê o próximo caractere vendo só t + 1 caracteres da janela")
    print(f"  (cada posição tem {losses.shape[0]} janelas; posições vizinhas são agrupadas)\n")
    print(f"  {'posições':>9} {'contexto':>11} {'loss':>6} {'perplexidade':>13}")
    ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 128)]
    for start, end in ranges:
        loss = losses[:, start:end].mean().item()
        single = end == start + 1
        positions = f"{start}" if single else f"{start}–{end - 1}"
        context = f"{start + 1}" if single else f"{start + 1}–{end}"
        stats = f"{loss:>6.3f} {perplexity(loss):>13.2f}"
        print(f"  {positions:>9} {context:>11} {stats}  {bar(loss, 3.0)}")


def by_character(tok: CharTokenizer, result: dict[str, torch.Tensor]) -> None:
    section("4. Onde o modelo erra: o tipo de caractere a prever")
    previous = [tok.vocab.get_token(i) for i in result["x"].flatten().tolist()]
    target = [tok.vocab.get_token(i) for i in result["y"].flatten().tolist()]
    losses = result["loss"].flatten().tolist()

    def category(prev: str, char: str) -> str:
        if char == " ":
            return "espaço"
        if char == "\n":
            return "quebra de linha"
        if char in PUNCTUATION:
            return "pontuação"
        if char.isalpha():
            return "1ª letra da palavra" if not prev.isalpha() else "letra no meio da palavra"
        return "outros"

    groups: dict[str, list[float]] = {}
    for prev, char, loss in zip(previous, target, losses, strict=True):
        groups.setdefault(category(prev, char), []).append(loss)
    print(f"  {'alvo':<26} {'posições':>9} {'loss':>6} {'perplexidade':>13}")
    for name, values in sorted(groups.items(), key=lambda item: -sum(item[1]) / len(item[1])):
        mean = sum(values) / len(values)
        stats = f"{len(values):>9,} {mean:>6.3f} {perplexity(mean):>13.2f}"
        print(f"  {name:<26} {stats}  {bar(mean, 3.5)}")

    per_char: dict[str, list[float]] = {}
    for char, loss in zip(target, losses, strict=True):
        per_char.setdefault(char, []).append(loss)
    frequent = {c: sum(v) / len(v) for c, v in per_char.items() if len(v) >= 100}
    ranked = sorted(frequent.items(), key=lambda item: item[1])
    print("\n  caracteres com 100+ ocorrências na validação:")
    print("  mais fáceis:  " + "  ".join(f"{c!r} {v:.2f}" for c, v in ranked[:6]))
    print("  mais difíceis:" + "  ".join(f"{c!r} {v:.2f}" for c, v in ranked[-6:]))


def calibration(result: dict[str, torch.Tensor]) -> None:
    section("5. Calibração: quando o modelo diz 80%, ele acerta 80%?")
    confidence, hit = result["conf"].flatten(), result["hit"].flatten().float()
    print(f"  {'confiança':>11} {'posições':>9} {'confiança média':>16} {'acerto real':>12}")
    ece = 0.0
    for low in [i / 10 for i in range(10)]:
        mask = (confidence >= low) & (confidence < low + 0.1 + (1e-6 if low == 0.9 else 0))
        if mask.sum() == 0:
            continue
        mean_confidence, accuracy = confidence[mask].mean().item(), hit[mask].mean().item()
        # Média das distâncias |acerto − confiança|, pesada pela fração de posições na faixa.
        ece += mask.float().mean().item() * abs(accuracy - mean_confidence)
        print(
            f"  {low:.1f}–{low + 0.1:.1f} {int(mask.sum()):>9,} {mean_confidence:>16.3f}"
            f" {accuracy:>12.3f}"
        )
    print(f"  acurácia geral (o mais provável é o certo): {hit.mean():.1%}")
    print(f"  erro de calibração esperado (ECE): {ece:.3f}")


@torch.no_grad()
def attention_maps(model: GPT, x: torch.Tensor, check: bool = False) -> list[torch.Tensor]:
    """Pesos de attention [B, heads, T, T] de cada bloco, recalculados a partir da entrada."""
    model.eval()
    inputs: dict[int, torch.Tensor] = {}
    hooks = [
        block.attention.register_forward_pre_hook(
            lambda module, args, index=index: inputs.__setitem__(index, args[0])
        )
        for index, block in enumerate(model.blocks)
    ]
    model(x)
    for hook in hooks:
        hook.remove()

    maps = []
    for index, block in enumerate(model.blocks):
        attention, h = block.attention, inputs[index]
        q = split_heads(attention.q_proj(h), attention.num_heads)
        k = split_heads(attention.k_proj(h), attention.num_heads)
        v = split_heads(attention.v_proj(h), attention.num_heads)
        weights = attention_weights(q, k, causal_mask(h.shape[1]))
        if check and index == 0:
            rebuilt = attention.out_proj(merge_heads(weights @ v))
            same = torch.allclose(rebuilt, attention(h), atol=1e-5)
            print(f"  pesos reconstruídos reproduzem a saída da attention: {same}")
        maps.append(weights)
    return maps


def attention_patterns(model: GPT, tok: CharTokenizer, loader) -> None:
    section("6. Para onde cada head olha (batch de validação)")
    x, _ = next(iter(loader))
    maps = attention_maps(model, x, check=True)
    seq_len = x.shape[1]
    distance = torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]  # i − j
    distance = distance.clamp_min(0).float()

    columns = f"{'no anterior':>12} {'em si':>7} {'no 1º':>7} {'distância média':>16}"
    print(f"\n  {'bloco.head':>10} {columns}")
    uniform = torch.tril(torch.ones(seq_len, seq_len)) / torch.arange(1, seq_len + 1)[:, None]
    rows = [("uniforme", uniform[None, None])] + [
        (f"{b}.{h}", maps[b][:, h : h + 1])
        for b in range(len(maps))
        for h in range(maps[b].shape[1])
    ]
    stats = {}
    for name, w in rows:
        previous = torch.diagonal(w, offset=-1, dim1=-2, dim2=-1).mean().item()
        itself = torch.diagonal(w, dim1=-2, dim2=-1)[..., 1:].mean().item()
        first = w[..., 1:, 0].mean().item()
        mean_distance = (w * distance).sum(dim=-1)[..., 1:].mean().item()
        stats[name] = (previous, mean_distance)
        print(f"  {name:>10} {previous:>12.3f} {itself:>7.3f} {first:>7.3f} {mean_distance:>16.1f}")

    heads = {name: s for name, s in stats.items() if name != "uniforme"}
    previous_head = max(heads, key=lambda name: heads[name][0])
    far_head = max(heads, key=lambda name: heads[name][1])
    text = "Capitú e Escobar"
    ids = torch.tensor([tok.encode(text)])
    text_maps = attention_maps(model, ids)
    for label, name in [
        ("mais focada no anterior", previous_head),
        ("que olha mais longe", far_head),
    ]:
        b, h = (int(part) for part in name.split("."))
        weights = text_maps[b][0, h]  # [T, T]
        print(f"\n  head {name} ({label}), em {text!r}; linha = quem olha, coluna = para onde")
        print("  █ ≥ 0,75   ▓ ≥ 0,5   ▒ ≥ 0,25   ░ ≥ 0,1\n")
        print("        " + " ".join(text))
        for i, char in enumerate(text):
            cells = [shade(weights[i, j].item()) if j <= i else "·" for j in range(len(text))]
            print(f"    {char!r:>3} " + " ".join(cells))


def shade(value: float) -> str:
    """Um caractere de preenchimento para um peso de attention entre 0 e 1."""
    for limit, char in ((0.75, "█"), (0.5, "▓"), (0.25, "▒"), (0.1, "░")):
        if value >= limit:
            return char
    return " "


def saved_activations(model: GPT, x: torch.Tensor, y: torch.Tensor) -> dict[tuple, int]:
    """Bytes que o autograd guarda para o backward de um batch, agrupados pelo shape do tensor."""
    parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen: set[int] = set()
    sizes: dict[tuple, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        # Chamado para cada tensor que o autograd guarda. Views do mesmo storage contam uma vez.
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in seen and pointer not in parameter_storages:
            seen.add(pointer)
            shape = tuple(tensor.shape)
            sizes[shape] = sizes.get(shape, 0) + tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = cross_entropy(model(x), y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return sizes


def memory(model: GPT, config, train_loader, history: list[dict]) -> None:
    section("7. Memória do treino")
    n = model.num_parameters()
    megabyte = 2**20
    x, y = next(iter(train_loader))
    model.train()
    sizes = saved_activations(model, x, y)

    batch, seq_len = x.shape
    h, d = config.model.num_heads, config.model.d_model
    vocab_size = model.lm_head.proj.out_features
    # Shapes exatos: como aqui d = T = 128, comparar só as últimas dimensões seria ambíguo.
    families = [
        (
            "[B, T, d]: entradas de LayerNorm, Linear, dropout",
            {(batch, seq_len, d), (batch * seq_len, d)},
        ),
        (
            "[B, h, T, T]: pesos de attention",
            {(batch, h, seq_len, seq_len), (batch * h, seq_len, seq_len)},
        ),
        (
            "[B, T, 4d]: camada escondida do FFN",
            {(batch, seq_len, 4 * d), (batch * seq_len, 4 * d)},
        ),
        (
            "[B, h, T, d_h]: Q, K e V por head",
            {(batch * h, seq_len, d // h), (batch * h, d // h, seq_len)},
        ),
        ("[B, T, V]: logits e log-probabilidades", {(batch, seq_len, vocab_size)}),
    ]
    rows = [
        ("parâmetros (float32)", n * 4),
        ("gradientes (um por parâmetro)", n * 4),
        ("AdamW: m e v (dois por parâmetro)", 2 * n * 4),
    ]
    counted: set[tuple] = set()
    for name, shapes in families:
        rows.append((f"ativações {name}", sum(sizes.get(shape, 0) for shape in shapes)))
        counted |= shapes
    rows.append(
        (
            "ativações: o resto (desvios da LayerNorm, IDs...)",
            sum(size for shape, size in sizes.items() if shape not in counted),
        )
    )

    total = sum(size for _, size in rows)
    for name, size in rows:
        print(f"  {name:<60} {size / megabyte:>6.1f} MB  {bar(size, total)}")
    activations = sum(sizes.values())
    print(f"  {'total das ativações':<60} {activations / megabyte:>6.1f} MB")
    print(f"  {'total do modelo e do treino':<60} {total / megabyte:>6.1f} MB")

    model.eval()
    without_dropout = sum(saved_activations(model, x, y).values()) / megabyte
    print(f"\n  ativações do mesmo batch sem dropout (model.eval()): {without_dropout:.1f} MB")
    attention_bytes = batch * h * seq_len**2 * 4
    peak = history[-1]["peak_memory_mb"]
    print(f"  uma matriz de pesos de attention [B, h, T, T]: {attention_bytes / megabyte:.1f} MB")
    print(f"  pico de memória do processo no treino (history.json): {peak:.0f} MB")
    print("  (inclui o próprio Python, as bibliotecas do PyTorch e os dados carregados)")


def main() -> None:
    if not CHECKPOINT.is_file() or not HISTORY.is_file():
        raise SystemExit("Checkpoint ou histórico não encontrado.\nRode: uv run python train.py")
    config = load_config(ROOT / "configs" / "tiny.yaml")
    model, tok = load_checkpoint(CHECKPOINT)
    text = (ROOT / config.data.corpus_path).read_text(encoding="utf-8")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, config.data.val_fraction)
    seq_len, batch_size = config.model.context_length, config.training.batch_size
    train_loader = create_dataloader(
        train_ids, context_length=seq_len, stride=seq_len, batch_size=batch_size, shuffle=False
    )
    val_loader = create_dataloader(
        val_ids, context_length=seq_len, stride=seq_len, batch_size=batch_size, shuffle=False
    )

    final_metrics(model, tok, train_ids, val_ids, train_loader, val_loader)
    curves(load_history(HISTORY))
    result = predictions(model, val_loader)
    by_position(result)
    by_character(tok, result)
    calibration(result)
    attention_patterns(model, tok, val_loader)
    memory(model, config, train_loader, load_history(HISTORY))


if __name__ == "__main__":
    main()
