# Performance

> **Como acelerar o treino sem mudar o que o modelo calcula?**

Código: `backend` em [`model/attention.py`](../model/attention.py) (`MultiHeadAttention`); `autocast`
em [`device.py`](../device.py); `mixed_precision` em [`training/trainer.py`](../training/trainer.py);
`attention_backend` e `mixed_precision` em [`config.py`](../config.py) e
[`configs/tiny.yaml`](../configs/tiny.yaml); `--mixed-precision` em [`train.py`](../train.py). Testes:
[`tests/test_multi_head_attention.py`](../tests/test_multi_head_attention.py) (seção "backend sdpa"),
[`tests/test_device.py`](../tests/test_device.py) (seção "mixed precision") e
[`tests/test_config.py`](../tests/test_config.py). Experimento:
`uv run python -m experiments.e17_performance`. Notação geral: [00-notacao.md](00-notacao.md). Visão
geral: [architecture.md](architecture.md).

---

## Para que serve

### O problema

Existem várias otimizações possíveis de performance a considerar num treino como o do Mini-GPT: mixed
precision, gradient accumulation, batching otimizado, attention eficiente, KV cache, compilação do
modelo e otimização de memória. Este documento cobre as **duas escolhidas antes de retreinar um modelo
maior**: **attention eficiente** e **mixed precision**. As outras cinco ficam para depois (seção 6).

As duas têm uma coisa em comum: **prometem deixar o treino mais rápido sem mudar o resultado
matemático**. Isso é diferente de "melhorar o modelo" ([11-training.md](11-training.md),
[14-evaluation.md](14-evaluation.md)) — nenhuma das duas muda a loss que o modelo converge, só quanto
tempo (e memória) cada passo consome para chegar lá.

- **Attention eficiente.** A `MultiHeadAttention` de [04-attention.md](04-attention.md) e
  [05-multi-head-attention.md](05-multi-head-attention.md) monta a matriz `[T, T]` de scores por
  inteiro na memória (`scores = q @ k.transpose(...)`, depois `softmax`, depois `@ v`). É didático —
  cada operação aparece separada — mas gasta memória proporcional a $T^2$ e faz o PyTorch percorrer a
  memória várias vezes. `torch.nn.functional.scaled_dot_product_attention` ("SDPA") calcula a mesma
  fórmula (04-attention.md) dentro de um único kernel otimizado, sem nunca guardar a matriz `[T, T]`
  inteira.
- **Mixed precision.** Até aqui, todo tensor era `float32` (32 bits por número). `bfloat16` usa só 16
  bits, mas com o **mesmo alcance de expoente** do `float32` — só menos casas decimais de precisão.
  Fazer o forward em `bfloat16` processa metade dos bytes por número, o que costuma acelerar contas
  limitadas por largura de banda de memória (comum em modelos pequenos como o `tiny.yaml`).

Termos usados daqui em diante:

- **Kernel:** a rotina de baixo nível que executa uma operação (16-gpu.md).
- **Fundir kernels** (*kernel fusion*): fazer várias operações matemáticas dentro de uma única
  passagem pela memória, em vez de uma operação por vez com resultados intermediários salvos.
- **Largura de banda de memória** (*memory bandwidth*): quantos bytes por segundo o device consegue
  ler/escrever. Em modelos pequenos, mover os números costuma custar mais tempo que fazer a conta com
  eles — daí "limitado por memória" (*memory-bound*), o oposto de "limitado por computação"
  (*compute-bound*).
- **Precisão mista** (*mixed precision*): parte do forward roda num tipo de menos bits (`bfloat16`),
  mas os pesos "mestres" e os gradientes continuam em `float32`.
- **bfloat16** ("brain float16"): 1 bit de sinal, 8 de expoente (igual ao `float32`), 7 de mantissa
  (contra 23 do `float32`). Por ter o mesmo expoente, não têm o problema de overflow do `float16`
  (que usa só 5 bits de expoente).

### Uma analogia

A attention manual é como copiar cada etapa de uma conta longa num caderno separado — cada passo fica
visível, mas você gasta papel e tempo virando página. O SDPA é a mesma conta feita de cabeça por
alguém treinado: o resultado é idêntico, só o caminho interno é mais direto. Mixed precision é como
fazer as contas de meio-termo com uma calculadora de 3 dígitos em vez de 8: mais rápido de digitar,
ligeiramente menos preciso, mas o suficiente quando o próximo passo (o gradiente, calculado em
`float32`) volta a ter toda a precisão.

### O que entra e o que sai

| | o quê | exemplo |
|---|---|---|
| entra | `attention_backend` (config) | `"manual"` (padrão) ou `"sdpa"` |
| entra | `mixed_precision` (config) ou `--mixed-precision` (CLI) | `false` (padrão) ou `true` |
| sai | a mesma sequência de treino, dentro de uma tolerância pequena | loss e gradientes comparáveis aos de 16-gpu.md |
| sai | tempo por passo, possivelmente menor | seção 4 mede os dois juntos |

### Onde isso entra no pipeline

```text
configs/tiny.yaml
  model.attention_backend  ──► GPT ──► TransformerBlock ──► MultiHeadAttention(backend=...)
  training.mixed_precision ──► train.py ──► train() ──► train_step / evaluate ──► autocast(device, ...)
```

Nenhum dos dois muda a **arquitetura** (num_parameters, seção anterior) nem o **checkpoint** (os pesos
continuam `float32`, 15-checkpoints.md) — só o caminho que o forward/backward percorre.

### O que daria errado sem esta parte

- **Nada quebraria.** Ao contrário dos outros componentes, este é inteiramente opcional: o `tiny.yaml`
  continua com `attention_backend: manual` e `mixed_precision: false` por padrão (seção 5 explica por
  quê, com números medidos nesta máquina).
- **Sem attention eficiente, o custo de memória cresce com $T^2$** — um modelo com contexto maior
  ficaria limitado pela matriz de scores antes de ficar limitado pelos pesos.
- **Sem mixed precision, cada número ocupa o dobro dos bytes** no forward — em GPUs com suporte
  dedicado a `bfloat16`/`float16` (a maioria das NVIDIA recentes), isso deixa dinheiro na mesa.

---

## Notação deste documento

| termo | como se lê | o que significa | no código |
|---|---|---|---|
| SDPA | "essidipiêi" | `scaled_dot_product_attention`, a função do PyTorch | `torch.nn.functional.scaled_dot_product_attention` |
| FP32 | "efe pê trinta e dois" | `float32`: 1 + 8 + 23 bits (sinal, expoente, mantissa) | `torch.float32` |
| BF16 | "bê efe dezesseis" | `bfloat16`: 1 + 8 + 7 bits | `torch.bfloat16` |
| $T$ | "tê" | tamanho do contexto (00-notacao.md) | `context_length`, `seq_len` |
| *memory-bound* | | o tempo é dominado por mover dados, não por calcular com eles | — |
| *compute-bound* | | o tempo é dominado pelas operações aritméticas em si | — |

---

## 1. SDPA: mesma fórmula, um kernel só

[04-attention.md](04-attention.md) define `Attention(Q, K, V) = softmax(QKᵀ/√d_k) V`, com máscara
causal. É exatamente essa
fórmula que `F.scaled_dot_product_attention(q, k, v, is_causal=True)` calcula — só que sem passar por
um tensor `[B, h, T, T]` intermediário guardado inteiro na memória entre a `softmax` e a multiplicação
por `V`.

```python
if self.backend == "sdpa":
    drop = self.dropout.p if self.training else 0.0
    heads = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
else:
    mask = causal_mask(seq_len, device=x.device)
    weights = attention_weights(q, k, mask)   # materializa [B, h, T, T]
    weights = self.dropout(weights)
    heads = weights @ v
```

Seção 1 do experimento, com os mesmos pesos nos dois backends:

```text
  mesmos pesos, mesma entrada: diferença máxima manual vs. sdpa = 8.38e-09
```

A diferença é ruído de arredondamento (16-gpu.md, seção 6) — a mesma ordem de grandeza de comparar dois
devices, porque o SDPA soma os termos numa ordem diferente da versão manual, não porque calcula outra
coisa. `tests/test_sdpa_backend_matches_manual_backend` confere isso com tolerância, e
`test_sdpa_backend_is_also_causal` confirma que a máscara causal (posições futuras não vazam para o
passado) continua valendo.

`MultiHeadAttention(d_model, num_heads, dropout, backend="sdpa")` é a única mudança de interface; o
resto do modelo (`TransformerBlock`, `GPT`) só repassa `config.attention_backend` adiante.

---

## 2. Medindo a attention: manual vs. SDPA

Seção 2 do experimento, isolando só a attention (sem o resto do bloco), em três tamanhos de contexto:

```text
  device: mps | batch 30 | heads 4 | head_dim 64
  contexto   manual (ms)   sdpa (ms)   speedup
       128          3.78        3.71     1.02x
       512         39.57       44.66     0.89x
      1024        195.06      143.42     1.36x
```

**Nesta máquina (MPS), não há um padrão confiável.** Rodei este experimento várias vezes durante o
desenvolvimento deste recurso; o "speedup" de cada linha variou entre 0,69x e 1,36x **entre execuções do
mesmo código, no mesmo tamanho** — a variação de uma execução para a outra é maior que a diferença
entre manual e SDPA dentro de uma mesma execução. Isso é ruído de medição, não um padrão real. Não
invalida o SDPA: quer dizer que **o kernel de attention do backend MPS deste PyTorch, nestes tamanhos,
já é comparável ao manual**. A vantagem teórica do SDPA — não alocar o tensor `[B, h, T, T]` — é
sobretudo de **memória**, não necessariamente de tempo. Seção 3 do experimento:

```text
  contexto   manual (MB)   sdpa (MB)
       512        3174.7      3174.7
      1024        3174.7      3174.7
      2048        8994.7      8994.7
```

Os números são **idênticos** entre os dois backends, em todo tamanho — mas 16-gpu.md, seção 8, já
tinha explicado por quê: no MPS, `device_memory_mb` lê o que o *driver* reservou para o processo, não
o pico real de tensores vivos. Essa métrica é grossa demais para provar a economia de memória do SDPA
nesta máquina; ela existe e é bem documentada no PyTorch, só não é visível com as ferramentas
disponíveis aqui.

**Onde o SDPA compensa de verdade:** em GPUs NVIDIA, ele despacha para kernels de *Flash Attention*
(quando as condições batem: `dropout_p` compatível, sem máscara customizada, etc.), que são
relatados na literatura como várias vezes mais rápidos e com uso de memória linear em $T$, não
quadrático. Este projeto não tem CUDA disponível para medir isso (16-gpu.md) — o que se pode afirmar
com os dados daqui é: **correto em qualquer device** (seção 1), **neutro em velocidade neste Mac**, e a
API certa para usar se um dia o treino rodar numa GPU NVIDIA.

---

## 3. Mixed precision: bfloat16 no forward, float32 nos pesos

```python
def autocast(device: torch.device, enabled: bool) -> AbstractContextManager:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
```

`torch.autocast` decide, operação por operação, se ela roda em `bfloat16` (multiplicações de matriz,
a maior parte do custo) ou se **continua em `float32`** (reduções sensíveis a precisão, como o
`logsumexp` de [09-lm-head.md](09-lm-head.md) e [10-loss.md](10-loss.md)). Isso acontece
**automaticamente** — o código de `training/loss.py` não mudou uma linha:

```python
def train_step(model, x, y, optimizer, gradient_clip, mixed_precision=False):
    ...
    with autocast(device, mixed_precision):
        logits = model(x)
        loss = cross_entropy(logits, y)   # PyTorch mantém a redução em float32 sozinho
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                        # fora do autocast: gradientes em float32
    ...
```

Seção 4 do experimento:

```text
  dtype dos logits: float32 -> torch.float32 | bfloat16 -> torch.bfloat16
  diferença média nos logits (fp32 vs. bf16): 0.0009
  loss de um passo: float32 4.6052 | bfloat16 4.6088
  parâmetros continuam float32 com mixed_precision=True: torch.float32
```

- **Os logits saem em `bfloat16`**, mas a diferença média para a versão `float32` é pequena (0,0009,
  numa escala de logits de alguns pontos) — consistente com bfloat16 ter só ~3 casas decimais de
  precisão de mantissa, mas o mesmo alcance de expoente do float32.
- **A loss muda na quarta casa decimal** (4,6052 → 4,6088), do mesmo tamanho do ruído entre devices
  visto em 16-gpu.md.
- **Os parâmetros nunca saem de `float32`.** É por isso que não precisa de `GradScaler` (usado com
  `float16`, que tem só 5 bits de expoente e pode estourar): `bfloat16` tem a mesma faixa dinâmica do
  `float32`, só menos dígitos — o risco que o `GradScaler` existe para evitar (gradiente virar `inf`)
  não se aplica aqui.

`tests/test_mixed_precision_train_step_keeps_weights_and_grads_in_float32` confere isso: depois de um
passo com `mixed_precision=True`, todo parâmetro continua `float32` e todo gradiente é finito (sem
`inf`/`NaN`).

---

## 4. As duas juntas, no tamanho do retreino planejado

Seção 5 do experimento, um passo de treino completo (forward + backward + `optimizer.step`) com o
modelo maior planejado para o próximo retreino (`d_model=256`, 8 camadas):

```text
  device mps | modelo: d_model 256, 8 camadas
   attention   precisão   ms/passo
      manual    float32      228.5
      manual   bfloat16      220.6  (1.04x)
        sdpa    float32      230.7  (0.99x)
        sdpa   bfloat16      226.8  (1.01x)
```

**As quatro combinações ficam dentro de ±5% umas das outras — ruído de medição, não um vencedor
claro.** Rodei isso três vezes (não só a tabela acima) e a ordem das quatro linhas mudou entre as
execuções. A conclusão honesta para **este hardware, neste tamanho de modelo**: nem SDPA nem mixed
precision aceleram o treino de forma mensurável no MPS deste Mac.

**Por que implementar mesmo assim?**

1. **Correção comprovada** (seções 1 e 3): trocar de backend ou ligar mixed precision não muda o que o
   modelo aprende, só como ele calcula.
2. **O ganho depende do hardware e do tamanho.** A literatura (fora deste projeto) documenta ganhos
   grandes de SDPA/Flash Attention e mixed precision em GPUs NVIDIA com núcleos dedicados a
   `float16`/`bfloat16` (Tensor Cores) — o Mini-GPT não tem acesso a esse hardware para medir, mas o
   código já está pronto para ele.
3. **`tiny.yaml` continua com os padrões antigos** (`attention_backend: manual`,
   `mixed_precision: false`) porque é o que os dados desta máquina sustentam — mudar o padrão sem
   ganho medido seria complexidade sem benefício (a mesma filosofia de 12-overfitting.md: não
   adicionar o que não se prova necessário).

---

## 5. Usando

```yaml
# configs/tiny.yaml
model:
  attention_backend: sdpa     # ou "manual" (padrão)
training:
  mixed_precision: true        # ou false (padrão)
```

```bash
uv run python train.py                       # usa o que estiver no YAML
uv run python train.py --mixed-precision      # força bfloat16, sem editar o YAML
uv run python -m experiments.e17_performance  # as seções deste documento
```

| opção | onde | padrão | efeito |
|---|---|---|---|
| `attention_backend` | `configs/*.yaml`, `model:` | `manual` | `manual` ou `sdpa` |
| `mixed_precision` | `configs/*.yaml`, `training:` | `false` | liga/desliga o autocast bfloat16 |
| `--mixed-precision` | CLI do `train.py` | — | força `true`, no lugar do valor da config |

O log do `train.py` mostra a escolha:

```text
modelo:     GPT com 820.096 parâmetros (com weight tying) | attention manual | precisão float32
```

---

## 6. O que fica para depois

Há mais cinco possibilidades na lista original de otimizações; nenhuma foi implementada aqui:

| item | por que ficou de fora agora |
|---|---|
| **KV cache** | só acelera `inference.py` (geração), não o treino — o foco imediato era o retreino do modelo maior |
| **Gradient accumulation** | simula batches maiores sem mais memória; só vale a pena se o batch atual já estiver no limite da memória do device |
| **Batching otimizado** | o `DataLoader` atual (02-dataset.md) já é simples e rápido para este corpus; não há gargalo medido ali |
| **Compilação do modelo** (`torch.compile`) | suporte a MPS ainda é experimental nesta versão do PyTorch; vale medir separadamente |
| **Otimização de memória** | sem um modelo grande o bastante para esbarrar em limites de memória, não há problema real para resolver ainda |

---

## 7. O que cada teste garante

[`tests/test_multi_head_attention.py`](../tests/test_multi_head_attention.py):

| teste | propriedade |
|---|---|
| `test_invalid_backend_is_rejected` | um `backend` desconhecido levanta `ValueError` |
| `test_sdpa_backend_matches_manual_backend` | os dois backends, com os mesmos pesos, dão a mesma saída (tolerância) |
| `test_sdpa_backend_is_also_causal` | mudar posições futuras não muda a saída das posições passadas, também no SDPA |
| `test_sdpa_backend_gradients_reach_all_projections` | `q_proj`, `k_proj`, `v_proj`, `out_proj` recebem gradiente com o backend SDPA |

[`tests/test_device.py`](../tests/test_device.py):

| teste | propriedade |
|---|---|
| `test_autocast_disabled_keeps_float32` | `autocast(device, False)` é um no-op (não muda o dtype) |
| `test_autocast_enabled_casts_matmul_to_bfloat16` | `autocast(device, True)` muda o dtype de uma multiplicação de matriz |
| `test_mixed_precision_train_step_keeps_weights_and_grads_in_float32` | depois de um passo com `mixed_precision=True`, pesos e gradientes continuam `float32` e finitos |
| `test_mixed_precision_loss_close_to_full_precision` | a loss de um passo com mixed precision fica perto da versão `float32` (tolerância maior, só roda com acelerador) |

[`tests/test_config.py`](../tests/test_config.py):

| teste | propriedade |
|---|---|
| `test_attention_backend_defaults_to_manual_and_is_validated` | o padrão é `"manual"`; um valor inválido levanta `ValueError` |
| `test_mixed_precision_defaults_to_false` | o `tiny.yaml` carrega com `mixed_precision: false` |

---

## Resumo

1. Este documento troca **como** o forward/backward é calculado, nunca **o quê** — a loss e os
   gradientes continuam corretos (seções 1 e 3), dentro do mesmo ruído de arredondamento já visto entre
   devices (16-gpu.md).
2. **SDPA** (`torch.nn.functional.scaled_dot_product_attention`) calcula a mesma fórmula de
   04-attention.md num kernel só, sem materializar a matriz `[T, T]` de scores — uma vantagem sobretudo
   de **memória**, que este projeto não conseguiu medir diretamente no MPS (a métrica disponível é
   grossa demais, 16-gpu.md).
3. **Mixed precision** roda o forward em `bfloat16` (mesmo alcance de expoente do `float32`, menos
   mantissa) via `torch.autocast`; os pesos e o passo do otimizador continuam `float32`, e reduções
   sensíveis (a loss) o PyTorch mantém em `float32` sozinho — sem precisar de `GradScaler`.
4. **Medido nesta máquina (MPS, Apple Silicon), nenhum dos dois acelera o treino de forma confiável** —
   as quatro combinações de backend × precisão ficaram dentro de ±5%, ordem que mudou entre execuções.
   Por isso `configs/tiny.yaml` continua com os padrões antigos (`manual`, `float32`); os dois recursos
   ficam disponíveis, testados e documentados, para quando o hardware (uma GPU NVIDIA) ou o tamanho do
   modelo mudarem essa conta.
5. KV cache, gradient accumulation, batching, `torch.compile` e otimização de memória ficam para
   depois — cada um só compensa a complexidade quando há um gargalo real para resolver.

---

## Para checar o entendimento

1. Por que `MultiHeadAttention(backend="sdpa")` e `backend="manual"` não dão exatamente o mesmo
   resultado bit a bit, mesmo com os mesmos pesos e a mesma entrada?
2. O que `is_causal=True` no `F.scaled_dot_product_attention` substitui, comparado à versão manual?
3. Por que a vantagem de memória do SDPA (não guardar a matriz `[T, T]`) cresce com o **quadrado** do
   contexto, e não linearmente?
4. `device_memory_mb` deu o mesmo valor para manual e SDPA, em todos os tamanhos de contexto testados.
   Isso prova que os dois usam a mesma memória? Por que não, nesta máquina?
5. Por que `bfloat16` dispensa o `GradScaler` que o `float16` precisa?
6. Na seção 3, os logits em `bfloat16` diferem da versão `float32` por só ~0,0009 em média, mas os
   parâmetros do modelo nunca chegam a existir em `bfloat16`. Onde exatamente a conversão de tipo
   acontece, e onde ela volta para `float32`?
7. Por que `cross_entropy` (que usa `torch.logsumexp`) continua numericamente estável dentro de um
   bloco `autocast`, mesmo os logits de entrada estando em `bfloat16`?
8. A seção 4 mostrou as quatro combinações (manual/sdpa × float32/bfloat16) dentro de ±5% umas das
   outras, com a ordem mudando entre execuções. O que isso diz sobre onde está o gargalo real do
   treino **nesta máquina**, com este tamanho de modelo — e por que isso pode ser diferente numa GPU
   NVIDIA?
9. Por que faz sentido manter `attention_backend: manual` e `mixed_precision: false` como padrão no
   `tiny.yaml`, mesmo com os dois recursos implementados e testados?
10. KV cache estava na lista de possibilidades, mas não foi implementado aqui. O que ele aceleraria,
    e por que isso não ajudaria o retreino do modelo maior que vem a seguir?
