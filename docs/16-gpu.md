# GPU

> **Como usar CUDA ou MPS sem duplicar código, e sem perder a reprodutibilidade ao trocar de device?**

Código: [`device.py`](../device.py); `select_device` em [`train.py`](../train.py) e
[`inference.py`](../inference.py); `model_device`, `device_memory_mb` e o `device_rng_state` em
[`training/trainer.py`](../training/trainer.py). Testes: [`tests/test_device.py`](../tests/test_device.py)
e a parametrização por device em [`tests/test_checkpoint.py`](../tests/test_checkpoint.py). Experimento:
`uv run python -m experiments.e16_gpu` (mais completo com `checkpoints/latest.pt`, de
`uv run python train.py`). Notação geral: [00-notacao.md](00-notacao.md). Visão geral:
[architecture.md](architecture.md).

---

## Para que serve

### O problema

Até aqui, tudo rodou na CPU. Um treino do Mini-GPT leva minutos; um modelo maior, em CPU, levaria dias.
A conta é sempre a mesma: forward e backward são, sobretudo, multiplicações de matrizes grandes
([04-attention.md](04-attention.md), [06-feed-forward.md](06-feed-forward.md)), e **cada número do
resultado pode ser calculado independentemente dos outros**. Uma
GPU tem milhares de núcleos simples que fazem exatamente esse tipo de conta em paralelo; uma CPU tem
poucos núcleos, cada um muito mais versátil.

Só que "rodar na GPU" não é gratuito:

- **nem toda máquina tem uma.** O código não pode presumir `cuda`; este Mac, por exemplo, não tem CUDA,
  mas tem **MPS** (Metal Performance Shaders, a GPU integrada da Apple);
- **os tensores do modelo e os do batch precisam estar no mesmo lugar.** Multiplicar um tensor na CPU
  por um na GPU é um erro (seção 2);
- **GPU e CPU trabalham de forma assíncrona.** A CPU manda o trabalho e segue em frente sem esperar o
  resultado (seção 4); cronometrar sem saber disso mede a velocidade errada;
- **cada acelerador tem o seu próprio gerador de números aleatórios**, separado do gerador global da
  CPU ([15-checkpoints.md](15-checkpoints.md)). Sem cuidar disso, retomar um treino na GPU deixaria de
  reproduzir as máscaras de dropout.

Este documento resolve os quatro pontos com um módulo pequeno, `device.py`, usado em todo lugar que já
existia (`train.py`, `inference.py`, `training/trainer.py`) sem mudar o que cada um faz — só **onde**
roda.

Termos usados daqui em diante:

- **Device** (dispositivo): onde um tensor mora e onde uma operação roda — `cpu`, `cuda` ou `mps`.
- **Acelerador**: um device diferente da CPU (`cuda` ou `mps` no Mini-GPT).
- **CUDA**: a plataforma da NVIDIA para programar a GPU.
- **MPS**: a plataforma da Apple (Metal) para programar a GPU dos chips M1/M2/M3/…
- **Kernel**: uma rotina que roda no acelerador (por exemplo, "multiplicar duas matrizes").
- **Enfileirar** (*queue*, *dispatch*): a CPU pede ao acelerador para rodar um kernel e segue em frente,
  sem esperar o resultado (seção 4).
- **Sincronizar**: esperar até que todos os kernels enfileirados até agora tenham terminado.
- **Determinístico**: dá sempre o mesmo resultado, nas mesmas condições.

### Uma analogia

Pense na CPU como um chef sozinho na cozinha e na GPU/MPS como uma linha de mil ajudantes, cada um
cortando um único legume. Para uma receita simples (poucos legumes), chamar mil ajudantes, explicar a
tarefa a cada um e esperar todos terminarem **demora mais** do que o chef cortar sozinho. Para uma
receita com toneladas de legumes idênticos, a linha vence com folga. A GPU também tem esse
"custo de organizar a linha" (seção 4); ele só compensa quando há trabalho suficiente para diluir esse
custo (seção 3 do experimento mostra isso com o modelo pequeno do `tiny.yaml`).

### O que entra e o que sai

| | o quê | exemplo |
|---|---|---|
| entra | `--device` (CLI) | `auto` (padrão), `cuda`, `mps` ou `cpu` |
| entra | o modelo, criado na CPU | `GPT(...)` |
| entra | os batches, vindos do `DataLoader` | sempre na CPU ([02-dataset.md](02-dataset.md)) |
| sai | o modelo e os batches, no mesmo device | `.to(device)` |
| sai | logs com o device escolhido e a memória usada | `describe(device)`, `device_memory_mb` |
| sai | um checkpoint que sabe em que device foi salvo | `progress.device_type`, `progress.device_rng_state` |

### Onde isso entra no pipeline

```text
train.py / inference.py
  │
  │  select_device(args.device)          ← escolhe o device uma vez, no início
  ▼
model = GPT(...).to(device)              ← os pesos se mudam para lá
  │
  ▼
training/trainer.py: train_step, evaluate, generate
  │  x, y = x.to(device), y.to(device)   ← cada batch (da CPU) se muda antes de entrar no modelo
  │  device_memory_mb(device)            ← memória do acelerador, no EpochRecord (14-evaluation.md)
  │  get_device_rng_state / set_device_rng_state  ← RNG do device no checkpoint (15-checkpoints.md)
  ▼
training/checkpoint.py: progress.device_type, progress.device_rng_state
```

### O que daria errado sem esta parte

- **O código só funcionaria numa máquina específica.** Hardcoded `cuda` quebraria neste Mac (sem CUDA);
  hardcoded `cpu` deixaria uma GPU disponível parada.
- **Um erro confuso ao misturar devices.** `RuntimeError` de baixo nível, sem dizer que o problema é
  "modelo aqui, dado ali" (seção 2).
- **Tempos de treino enganosos.** Sem sincronizar, um cronômetro mediria só "o tempo de enfileirar", não
  o tempo real do trabalho (seção 4).
- **Retomar o treino numa GPU deixaria de reproduzir o dropout.** O gerador global da CPU não é o que
  sorteia as máscaras quando o modelo está no acelerador (seção 5).

---

## Notação deste documento

Este documento é mais sobre engenharia do que sobre matemática nova; poucos símbolos aparecem além dos
já definidos em [00-notacao.md](00-notacao.md).

| termo | como se lê | o que significa | no código |
|---|---|---|---|
| `device` | "dispositivo" | onde um tensor mora / uma operação roda | `torch.device` |
| `device.type` | | a família do device, sem o índice | `"cpu"`, `"cuda"`, `"mps"` |
| `cuda:0`, `mps:0` | | um device específico, com índice (uma máquina pode ter várias GPUs) | `torch.device("cuda", 0)` |
| RSS | "pico de memória residente" | memória do **processo inteiro**, medida pelo sistema operacional ([14-evaluation.md](14-evaluation.md)) | `peak_memory_mb()` |
| `device_memory_mb` | | memória só do acelerador: pico dos tensores (CUDA) ou total reservado (MPS) | `device_memory_mb(device)` |
| ULP | "unidade na última posição" | a menor diferença representável entre dois floats vizinhos | ~1e-8 num float32 perto de 0,2 |

---

## 1. Por que uma GPU acelera (ou não) uma multiplicação de matrizes

Uma camada linear ([06-feed-forward.md](06-feed-forward.md)) calcula `Y = X @ W.T`. Cada elemento de
`Y` é um produto escalar (00-notacao.md,
seção 5.2) entre uma linha de `X` e uma linha de `W` — uma soma de produtos que **não depende de nenhum
outro elemento de `Y`**. Uma CPU calcula esses elementos em sequência (ou em poucos, com alguns núcleos);
uma GPU/MPS calcula milhares deles ao mesmo tempo, um por núcleo.

Isso só ajuda se houver **elementos suficientes para preencher os núcleos** e se o tempo de computação for
maior que o custo fixo de organizar o trabalho (seção 3 abaixo). O `tiny.yaml` (30 sequências, 256
posições, `d_model` 256) é pequeno para os padrões de GPU: as matrizes cabem folgadamente, mas o número
de operações é baixo o bastante para que o custo de despachar cada kernel pese no resultado final.

---

## 2. Escolhendo o device: `select_device`

```python
def select_device(name: str = "auto") -> torch.device:
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
```

**A ordem `auto` é CUDA, depois MPS, depois CPU:** numa máquina com as duas (raro), CUDA costuma ser mais
madura e rápida para treino; MPS é a opção de quem tem um Mac Apple Silicon; CPU sempre funciona, como
último recurso. Pedir um device explícito que não existe **falha alto e claro**, em vez de rodar na CPU
silenciosamente.

Seção 1 do experimento, nesta máquina:

```text
  PyTorch 2.14.0
  torch.cuda.is_available()          False
  torch.backends.mps.is_available()  True
  select_device('auto')              -> mps (GPU da Apple, via Metal)
  select_device('cpu')                -> ok
  select_device('cuda')               -> ValueError: CUDA não está disponível neste ambiente
  select_device('mps')                -> ok
```

`train.py` e `inference.py` chamam `select_device(args.device)` uma única vez, logo no início, e usam o
resultado para criar ou mover o modelo (`GPT(...).to(device)`). Todo o resto do código nunca escreve
`"cuda"` ou `"mps"` diretamente — só conhece `device`.

---

## 3. Onde o modelo e os dados precisam estar

`model_device` lê o device **dos parâmetros**, e não guarda estado à parte:

```python
def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device
```

Toda operação do PyTorch exige que os tensores envolvidos estejam no **mesmo** device. Seção 2 do
experimento, tentando passar um tensor da CPU para um modelo movido ao MPS:

```text
  criado com GPT(...): model_device -> cpu
  depois de model.to('mps'): model_device -> mps:0
  model(x_cpu) com os pesos em 'mps': RuntimeError
    Passed CPU tensor to MPS op
  model(x_cpu.to('mps')): ok, logits em mps:0
```

É exatamente esse erro que `train_step` e `evaluate` evitam, movendo cada batch para o device do modelo
antes do forward ([`training/trainer.py`](../training/trainer.py)):

```python
def train_step(model, x, y, optimizer, gradient_clip):
    device = model_device(model)
    x, y = x.to(device), y.to(device)   # o DataLoader entrega os batches na CPU
    ...
```

**Por que o `DataLoader` continua entregando tensores na CPU?** Porque ele roda em processos separados
([02-dataset.md](02-dataset.md)), e mover para a GPU ali complicaria a paralelização sem ganho: o
`.to(device)` de um batch (KB a poucos MB) é rápido comparado ao forward/backward do modelo.

---

## 4. Os mesmos números, em qualquer device

Um passo de treino no MPS deveria dar a mesma loss que na CPU, a menos de ruído de ponto flutuante. Seção
3 do experimento, um mesmo modelo e batch, treinados um passo em cada device:

```text
  device         loss   norma do grad
  cpu        2.318176        0.999999
  mps        2.318176        0.999999   (== cpu, ~1e-4: True)
```

`tests/test_device.py` confere isso com `torch.testing.assert_close(rtol=1e-4, atol=1e-6)` — uma
tolerância pequena, não zero. A razão está na seção 6: os *kernels* de cada backend (CPU, CUDA, MPS) não
somam os termos de uma multiplicação de matrizes exatamente na mesma ordem, e ponto flutuante **não é
associativo** ($$(a+b)+c$$ pode diferir de $$a+(b+c)$$ na última casa decimal). O resultado é correto e
praticamente idêntico, mas não *bit a bit* idêntico entre devices.

---

## 5. Execução assíncrona: por que sincronizar antes de cronometrar

A CPU e o acelerador trabalham em paralelo: quando o código chama uma operação num tensor MPS ou CUDA, a
CPU **enfileira** o kernel e segue para a próxima linha, sem esperar o resultado. Um cronômetro que para
logo depois de enfileirar mede só isso — enfileirar — e não o trabalho de verdade.

Seção 4 do experimento, 50 forwards seguidos, sem tocar em nenhum resultado (por isso a CPU nunca precisa
esperar):

```text
  medição                           ms (50 forwards)  observação
  sem esperar a fila                          4728.4  fila ainda cheia
  depois de synchronize()                    11077.6  tempo real
```

**Ler o número errado teria concluído "50 forwards levam 4,7 s".** O tempo real, medido depois de
`synchronize()`, é 11,1 s — mais que o dobro. `synchronize` só existe para isso:

```python
def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    # cpu: cada operação já é síncrona, nada a esperar
```

**Por que `train_step` não precisa chamar `synchronize` explicitamente?** Porque ele termina lendo
`loss.item()` e `grad_norm.item()` — copiar um valor do device para um número Python **força** a CPU a
esperar o resultado. `.item()`, `.cpu()` e `print(tensor)` sincronizam por conta própria. É só em medições
de tempo *sem* ler nenhum resultado (como o forward puro da seção 4) que o cuidado é necessário.

Com isso em mente, os tempos por passo do modelo do `tiny.yaml` já são reais:

```text
  modelo do tiny.yaml (30 x 256, d_model 256, 4 heads, 4 layers):
  cpu           721.8 ms/passo
  mps           608.3 ms/passo  (speedup 1.19x)
```

**Speedup modesto, e isso é esperado.** Como a seção 1 previu: o `tiny.yaml` é pequeno demais para que o
paralelismo do MPS compense o custo de organizar o trabalho. Um modelo com mais camadas, mais `d_model`
ou batches maiores mudaria essa conta — é o tipo de decisão que [17-performance.md](17-performance.md)
volta a olhar.

---

## 6. RNG por device: por que o dropout precisa do seu próprio gerador

Relembrando os checkpoints ([15-checkpoints.md](15-checkpoints.md)): um gerador pseudoaleatório é uma
máquina determinística — o mesmo **estado** sempre produz a mesma sequência. A CPU e cada acelerador têm geradores **separados**: sortear uma máscara
de dropout no MPS não consome nenhum número do gerador global da CPU, e vice-versa.

```python
def get_device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None  # cpu: o gerador global (torch.get_rng_state()) já cobre
```

Seção 5 do experimento, chamando o mesmo modelo (com dropout) três vezes:

```text
  duas chamadas seguidas, mesmo estado inicial: idênticas -> False
  restaurando o estado do device antes da 3ª chamada: repete a 1ª -> True
  chamar o modelo no 'mps' não mexe no gerador global da CPU: True
```

- **Duas chamadas seguidas dão saídas diferentes:** cada uma sorteia uma máscara de dropout nova.
- **Guardar o estado do device e restaurá-lo repete exatamente a mesma máscara** — é o mesmo mecanismo
  de `torch.get_rng_state()` / `torch.set_rng_state()` da CPU (15-checkpoints.md), só que para o
  acelerador.
- **O gerador global da CPU não muda** ao rodar no MPS: são geradores independentes.

Por isso `TrainingProgress` (15-checkpoints.md) guarda **três** coisas relacionadas a aleatoriedade, não
duas: `loader_rng_state` (ordem das janelas), `rng_state` (dropout na CPU) e, desde este documento,
`device_rng_state` mais `device_type` (dropout no acelerador, e em qual).

---

## 7. Retomando o treino ao trocar de device

```python
if progress.step > 0 and progress.device_type != device.type:
    log(f"aviso: o progresso foi salvo em '{progress.device_type}' e o treino continua em"
        f" '{device.type}': os números não serão idênticos aos de um treino sem pausa")
elif progress.device_rng_state is not None:
    set_device_rng_state(device, progress.device_rng_state)
```

Um `device_rng_state` salvo no MPS não tem como ser restaurado no gerador do CUDA (são bibliotecas e
formatos diferentes) nem faria sentido na CPU. Quando o device muda, `train` **avisa** e segue em frente
sem restaurar esse estado — os pesos e o estado do AdamW continuam corretos, só as máscaras de dropout do
próximo forward serão outras.

Seção 7 do experimento, com o `checkpoints/latest.pt` de um treino real (salvo no MPS):

```text
  checkpoint salvo com device_type = 'mps'
  validação com os pesos como estão: 1.419
  retomar em 'cpu': device diferente -> avisa e ignora o rng do acelerador
  retomar em 'mps': mesmo device do checkpoint -> usa device_rng_state
```

`tests/test_checkpoint.py::test_accelerator_checkpoint_loads_on_cpu_and_warns_when_resumed_there` cobre
esse aviso; `test_resumed_training_matches_uninterrupted_training` cobre a retomada normal, no mesmo
device onde foi salva.

### Um detalhe sobre "exato": CPU determinística, acelerador não

O documento de checkpoints provou que retomar na CPU reproduz o treino **bit a bit**. No MPS (e, de
forma semelhante, no
CUDA), isso deixa de valer: mesmo com tudo restaurado certo — pesos, estado do AdamW, RNG do loader e do
device —, os *kernels* de matmul/attention do acelerador não garantem somar os termos sempre na mesma
ordem entre duas execuções. A diferença fica no nível do último bit de um `float32`, ULP:

```text
embeddings.token_embedding.weight  diferença máxima: 7.45e-09
blocks.0.attention.q_proj.weight   diferença máxima: 6.05e-09
lm_head.proj.weight                diferença máxima: 7.45e-09
```

Por isso `test_resumed_training_matches_uninterrupted_training` usa `torch.equal` (igualdade exata) só
para `device == "cpu"`, e `torch.testing.assert_close(rtol=0, atol=1e-6)` para acelerador — a mesma
distinção que a seção 4 já tinha mostrado para um único passo. A retomada continua **correta**: nada de
RNG, otimizador ou posição no loader se perde; só a garantia de bit-a-bit é uma propriedade da CPU, não do
acelerador.

---

## 8. Memória do acelerador

```python
def device_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20   # pico dos tensores
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20        # total reservado pelo driver
    return None
```

Seção 6 do experimento, movendo o modelo e dando três passos de treino:

```text
  depois de             device_memory_mb
  mover o modelo                  2226.7
  passo 1                         2226.7
  passo 2                         2226.7
  passo 3                         2226.7
```

**Por que o número não cresce a cada passo, mesmo com ativações e gradientes novos sendo alocados?**
`driver_allocated_memory()` (MPS) reporta o que o *driver* reservou para o processo, e o alocador do MPS
reutiliza blocos já reservados em vez de pedir mais ao sistema a cada tensor — o mesmo padrão do
alocador de memória da CPU. No CUDA, `max_memory_allocated` mede outra coisa: o **pico** de bytes
efetivamente ocupados por tensores (não o que o driver reservou), e cresceria a cada passo até
estabilizar. É por isso que [14-evaluation.md](14-evaluation.md) já tinha `peak_memory_mb()` (RSS do
processo inteiro, via SO) e este documento soma `device_memory_mb` a ele no `EpochRecord`, só quando não
é `None`:

```text
577 MB (+ 2227 MB na mps)
```

---

## 9. Usando

```bash
uv run python train.py --device auto                 # padrão: CUDA, depois MPS, depois CPU
uv run python train.py --device cpu                   # força CPU, mesmo com GPU disponível
uv run python inference.py "Capitú" --device mps
uv run python -m experiments.e16_gpu                   # as seções deste documento
```

| opção | padrão | efeito |
|---|---|---|
| `--device` | `auto` | `auto`, `cuda`, `mps` ou `cpu`; um nome explícito indisponível falha com `ValueError` |

Nenhuma outra opção muda: o restante do `train.py` e do `inference.py` é igual em qualquer device.

---

## 10. O que cada teste garante

[`tests/test_device.py`](../tests/test_device.py):

| teste | propriedade |
|---|---|
| `test_auto_prefers_cuda_then_mps_then_cpu` | a ordem de `select_device("auto")` é CUDA, depois MPS, depois CPU |
| `test_unavailable_device_is_rejected` | pedir `cuda`/`mps` sem hardware correspondente levanta `ValueError` |
| `test_cpu_is_always_available_and_has_no_separate_rng_state` | `select_device("cpu")` sempre funciona; `get_device_rng_state` devolve `None` na CPU |
| `test_model_device_follows_the_parameters` | `model_device` lê o device dos parâmetros, sem estado à parte |
| `test_train_step_on_accelerator_matches_cpu` | um passo de treino dá a mesma loss e os mesmos gradientes (tolerância) na CPU e no acelerador |
| `test_evaluate_on_accelerator_matches_cpu` | a loss de avaliação bate entre devices |
| `test_greedy_generation_on_accelerator_matches_cpu` | geração gulosa ([13-generation.md](13-generation.md)) dá a mesma sequência de IDs em qualquer device |
| `test_accelerator_rng_state_reproduces_dropout` | salvar e restaurar `device_rng_state` repete a mesma máscara de dropout no acelerador |

Os quatro últimos só rodam se houver um acelerador disponível (`needs_accelerator`); nesta máquina, rodam
no MPS.

[`tests/test_checkpoint.py`](../tests/test_checkpoint.py), parametrizados por `device` (seção 7 acima):

| teste | propriedade |
|---|---|
| `test_resumed_training_matches_uninterrupted_training[*-mps]` | retomar no mesmo acelerador reproduz o treino, com tolerância de arredondamento |
| `test_accelerator_checkpoint_loads_on_cpu_and_warns_when_resumed_there` | um checkpoint salvo num acelerador carrega na CPU e avisa que o RNG do device não pôde ser restaurado |

---

## Resumo

1. GPUs (CUDA) e a GPU integrada da Apple (MPS) aceleram o treino porque forward e backward são, em
   grande parte, multiplicações de matrizes com elementos independentes entre si — paralelizáveis. O
   ganho só aparece quando há trabalho suficiente para compensar o custo de organizar a execução.
2. `select_device("auto")` escolhe CUDA, depois MPS, depois CPU; um nome explícito indisponível falha
   claramente, em vez de cair silenciosamente na CPU.
3. Modelo e dados precisam estar no mesmo device; `model_device` + `.to(device)` em cada batch evitam o
   erro. O `DataLoader` continua entregando tensores na CPU (02-dataset.md); só o batch usado se move.
4. Os mesmos passos de treino dão a mesma loss em qualquer device, dentro de uma tolerância pequena — os
   backends não somam os termos de uma multiplicação de matrizes exatamente na mesma ordem.
5. CPU e acelerador trabalham de forma assíncrona: sem `synchronize()`, um cronômetro mede só o tempo de
   enfileirar (4,7 s medidos, contra 11,1 s reais). `.item()` e `.cpu()` já sincronizam sozinhos, por isso
   `train_step` não precisa chamar `synchronize` explicitamente.
6. Cada device tem seu próprio gerador de números aleatórios; o dropout no acelerador não consome o
   gerador global da CPU. `TrainingProgress` guarda `device_rng_state` e `device_type` além do `rng_state`
   da CPU (15-checkpoints.md).
7. Retomar no mesmo device reproduz o treino, mas só **exatamente** (bit a bit) na CPU; num acelerador, a
   diferença fica no ULP do `float32` (~1e-8), por isso os testes comparam com uma tolerância mínima ali.
   Retomar num device diferente do salvo avisa e ignora o RNG do acelerador antigo.
8. `device_memory_mb` complementa o `peak_memory_mb` do processo (14-evaluation.md) com a memória do
   próprio acelerador — mas mede coisas diferentes em CUDA (pico de tensores) e MPS (total reservado pelo
   driver).

---

## Para checar o entendimento

1. Por que `select_device("cuda")` numa máquina sem GPU NVIDIA levanta um erro, em vez de cair para a
   CPU como o `"auto"` faz?
2. `model(x_cpu)` com os pesos no MPS deu `RuntimeError: Passed CPU tensor to MPS op`. Por que
   `train_step` nunca cai nesse erro, mesmo o `DataLoader` sempre entregando tensores na CPU?
3. A seção 4 mediu 4,7 s "sem esperar a fila" contra 11,1 s "depois de `synchronize()`", para os mesmos 50
   forwards. Por que o primeiro número não é zero, se a CPU só está enfileirando trabalho?
4. Por que `train_step` não chama `synchronize()` em nenhum lugar do código, mesmo rodando no MPS?
5. O speedup do MPS sobre a CPU, para o modelo do `tiny.yaml`, foi de só 1,19x. O que teria que mudar na
   configuração para esse número crescer?
6. Por que o gerador de números aleatórios do MPS é separado do gerador global da CPU? O que aconteceria
   com a reprodutibilidade do dropout se os dois compartilhassem o mesmo estado?
7. Um checkpoint foi salvo com `device_type = "mps"`. O que muda ao retomar o treino em `"cpu"`, em vez de
   `"mps"`? O treino ainda funciona? Os números batem com um treino sem pausa?
8. Por que `test_resumed_training_matches_uninterrupted_training` usa `torch.equal` só quando
   `device == "cpu"`, e uma tolerância (`atol=1e-6`) nos outros casos? Isso é sinal de um bug no RNG do
   device?
9. `device_memory_mb` ficou constante (2226,7 MB) nos três passos medidos, mesmo com ativações e
   gradientes novos sendo criados a cada passo. Isso quer dizer que nenhuma memória nova foi usada? O que
   esse número mede, exatamente, no MPS?
10. Por que a soma `a + b + c` pode dar resultados diferentes em `float32` dependendo da ordem, e por que
    isso é suficiente para explicar a diferença de ~1e-8 entre CPU e MPS na seção 4?
