# Checkpoints

> **Como parar o treino e continuar depois exatamente do mesmo ponto?**

Código: [`training/checkpoint.py`](../training/checkpoint.py), `TrainingProgress` e `train` em
[`training/trainer.py`](../training/trainer.py), `checkpoint_every` em [`config.py`](../config.py) e
[`configs/tiny.yaml`](../configs/tiny.yaml), e as opções `--resume`, `--epochs` e `--checkpoint-dir`
do [`train.py`](../train.py). Testes: [`tests/test_checkpoint.py`](../tests/test_checkpoint.py) e
[`tests/test_config.py`](../tests/test_config.py). Experimento:
`uv run python -m experiments.e15_checkpoints` (precisa dos checkpoints salvos por
`uv run python train.py`). Notação geral: [00-notacao.md](00-notacao.md). Visão geral:
[architecture.md](architecture.md).

---

## Para que serve

### O problema

O treino do Mini-GPT leva ~8 minutos. Um modelo grande treina por semanas. Nesse tempo, muita coisa
pode acontecer:

- **o processo cai:** falta de luz, o notebook fecha, um erro no meio do caminho;
- **você quer treinar mais:** as 30 épocas acabaram, e a validação ainda estava caindo;
- **o fim não é o melhor:** se o treino passar do ponto (overfitting, ver [14-evaluation.md](14-evaluation.md)), o modelo final é pior
  que o de algumas épocas antes;
- **você quer comparar momentos:** como o modelo escrevia no passo 1000? E no 2000?

O checkpoint de [13-generation.md](13-generation.md) guardava só **os pesos** (com a config da
arquitetura e o vocabulário). Isso basta para gerar texto, mas não para continuar o treino como se
nada tivesse acontecido. O checkpoint precisa permitir três coisas: **continuar o treinamento,
executar inferência e recuperar o modelo**. Este documento descobre o que mais precisa ser guardado, e
prova que a retomada é exata.

Termos usados daqui em diante:

- **Checkpoint:** um arquivo com um "retrato" do treino num certo momento.
- **Retomar** (*resume*): carregar um checkpoint e continuar o treino a partir dele.
- **Estado do otimizador:** as médias $m$ e $v$ que o AdamW acumula para cada peso (ver [11-training.md](11-training.md)).
- **Gerador pseudoaleatório** (RNG, *random number generator*): o algoritmo que produz os números
  "sorteados". Ele é determinístico: a partir do mesmo **estado** interno, gera sempre a mesma
  sequência.
- **Reprodutível bit a bit:** dá exatamente os mesmos números, até a última casa decimal.
- **Serializar:** transformar objetos da memória (tensores, dicionários) em bytes num arquivo.
  `torch.save` faz isso, e `torch.load` faz o caminho inverso.
- **Gravação atômica:** uma escrita que acontece por inteiro ou não acontece; nunca pela metade.

### Uma analogia

Pense no *save* de um videogame:

- Salvar **só o personagem** (os pesos) e carregar depois te põe no começo da fase, sem os itens que
  você juntou e com os inimigos em outras posições.
- Um *save* completo guarda também **onde você está no mapa** (o passo e a posição dentro da época),
  **o inventário** (o estado do AdamW), **a semente do mundo** (os estados aleatórios) e **o diário de
  missões** (o histórico).
- E ninguém quer que uma queda de energia **durante** o salvamento destrua o *save* anterior. É a
  gravação atômica.

### O que entra e o que sai

| | o quê | exemplo |
|---|---|---|
| entra | o modelo, o tokenizer, a config | GPT com 820.096 parâmetros, vocabulário de 97 tokens, `tiny.yaml` |
| entra | o otimizador | AdamW com $m$ e $v$ para cada peso |
| entra | o progresso | passo 1000, época 13, 16 batches feitos, histórico, estados aleatórios |
| sai | `latest.pt` | o estado mais recente, regravado ao fim de cada época (9,5 MB) |
| sai | `best.pt` | o estado da época com a menor loss de validação |
| sai | `checkpoint_<passo>.pt` | um retrato fixo a cada 1000 passos |
| sai | `history.json` | os registros das épocas (14-evaluation.md) |

### Onde este documento entra

```text
tiny.yaml + corpus → tokenizer → loaders → train() ──────────────────► history.json
                                             │
                                             │  ao fim de cada época e a cada 1000 passos  ← este documento
                                             ▼
                        save_checkpoint → latest.pt · best.pt · checkpoint_<passo>.pt
                                             │
                        ┌────────────────────┴─────────────────────┐
                        ▼                                          ▼
       train.py --resume: continua o treino       inference.py, e13, e14: geram e avaliam
       exatamente de onde parou                   (load_checkpoint lê só os pesos)
```

### O que daria errado sem esta parte

- **Uma queda perderia o treino inteiro.** Com `latest.pt` a cada época, perde-se no máximo uma época
  (82 passos, ~15 s aqui).
- **Treinar mais exigiria começar do zero**, ou continuar de forma inexata (seção 6).
- **O melhor modelo poderia se perder** se o treino passasse do ponto.
- **Os resultados deixariam de ser reproduzíveis.** Um treino interrompido e retomado daria números
  diferentes do mesmo treino sem pausa, e comparações entre configurações ([11-training.md](11-training.md)) ficariam
  contaminadas por essa diferença.

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $w$ | "dáblio" | um peso do modelo | 820.096 no total | `model.state_dict()` |
| $g_t$ | "gê tê" | gradiente de um peso no passo $t$ | | `p.grad` |
| $t$ | "tê" | passo de otimização (quantos `optimizer.step()` já houve) | 2460 no fim | `step` |
| $m_t$, $v_t$ | "eme tê", "vê tê" | médias do AdamW: do gradiente e do gradiente ao quadrado | uma de cada por peso | `exp_avg`, `exp_avg_sq` |
| $\hat m_t$, $\hat v_t$ | "eme chapéu", "vê chapéu" | as médias com correção de viés | | calculadas dentro do AdamW |
| $\beta_1$, $\beta_2$ | "beta um", "beta dois" | quanto das médias antigas se mantém a cada passo | 0,9 e 0,999 | `betas` |
| $\eta$ | "eta" | learning rate | 0,001 | `lr` |
| $\lambda$ | "lambda" | weight decay | 0,01 (matrizes) | `weight_decay` |
| $\epsilon$ | "épsilon" | número minúsculo que evita divisão por zero | $10^{-8}$ | `eps` |
| $\lvert \Delta w \rvert$ | "módulo de delta dáblio" | quanto um peso anda num passo | 0,19 × $\eta$ em média | `(p - antes).abs()` |
| $N_b$ | "ene bê" | batches por época | 82 | `len(train_loader)` |
| $e$, $b$ | "é", "bê" | época em andamento e batches já feitos nela | 13 e 16 no passo 1000 | `epoch`, `batches_done` |

---

## 1. O que precisa ser salvo

### 1.1 Três usos, três conteúdos

| uso | o que é indispensável | onde fica no arquivo |
|---|---|---|
| **gerar texto** | arquitetura, vocabulário e pesos ([13-generation.md](13-generation.md)) | `model_config`, `vocab`, `model_state` |
| **recuperar o melhor modelo** | o mesmo, mais saber qual época foi a melhor | `best.pt`, escolhido pelo histórico |
| **continuar o treino** | tudo acima, mais a config inteira, o otimizador e o progresso | `config`, `optimizer_state`, `progress` |

Seção 1 do experimento, abrindo o `latest.pt` do fim do treino:

```text
  chave             tamanho  conteúdo
  model_config      0.00 MB  hiperparâmetros da arquitetura
  vocab             0.00 MB  os 97 tokens, na ordem dos IDs
  model_state       3.15 MB  53 tensores de pesos, por nome
  config            0.00 MB  a config inteira: dados, modelo e treino
  optimizer_state   6.30 MB  m e v do AdamW para 52 tensores
  progress          0.01 MB  passo, época, posição no loader, histórico, estados aleatórios
  arquivo inteiro   9.46 MB
  só os pesos       3.15 MB  (o formato usado na geração de texto)
```

- **Os pesos são um terço do arquivo.** O estado do AdamW ocupa o dobro deles: duas médias, $m$ e $v$,
  para cada um dos 820.096 pesos.
- **53 tensores de pesos, mas 52 no otimizador.** Com weight tying ([09-lm-head.md](09-lm-head.md)),
  o embedding e o LM head são o mesmo tensor, que aparece com dois nomes no `state_dict` e é guardado
  uma vez só.
- **O resto é pequeno:** a config, o vocabulário e o progresso somam ~10 KB.
- **Os dois formatos funcionam para gerar texto.** `load_checkpoint` lê só as três primeiras chaves,
  então o `inference.py` e os experimentos de geração e avaliação (13-generation.md, 14-evaluation.md)
  continuam funcionando com os arquivos novos.

### 1.2 O estado do AdamW

Relembrando o treino ([11-training.md](11-training.md)), o AdamW guarda duas médias móveis por peso e usa as duas para decidir o tamanho
do passo:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t
\qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2
$$

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}
\qquad
\hat v_t = \frac{v_t}{1 - \beta_2^t}
\qquad
\Delta w = -\eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda w \right)
$$

**Lendo as fórmulas:**

- $m_t$ é a média dos gradientes recentes, **com sinal**: gradientes que alternam entre positivo e
  negativo se cancelam.
- $v_t$ é a média dos gradientes **ao quadrado**: nunca se cancela, mede o tamanho típico do gradiente.
- $\hat m_t$ e $\hat v_t$ corrigem o fato de as médias começarem em zero. A correção depende de $t$,
  e é por isso que o AdamW também guarda o número do passo (`step`).
- O passo $\Delta w$ é proporcional a $\hat m_t / \sqrt{\hat v_t}$: **a direção consistente dividida
  pelo tamanho típico**.

**Por que perder esse estado muda o treino.** Dois exemplos para um peso com gradientes de tamanho 0,1:

| gradientes recentes | $m$ | $v$ | $\hat m / \sqrt{\hat v}$ | passo |
|---|---:|---:|---:|---|
| sempre +0,1 (direção firme) | ≈ 0,1 | ≈ 0,01 | ≈ 1 | $\approx \eta$ |
| alternando +0,1 e −0,1 (ruído) | ≈ 0 | ≈ 0,01 | ≈ 0 | quase nenhum |

Com o estado acumulado, pesos cujo gradiente é só ruído quase não andam. Agora imagine **zerar** $m$ e
$v$. No primeiro passo depois disso ($t = 1$):

$$
\hat m_1 = \frac{(1 - \beta_1)\, g_1}{1 - \beta_1} = g_1
\qquad
\hat v_1 = g_1^2
\qquad
\frac{\hat m_1}{\sqrt{\hat v_1}} = \frac{g_1}{\lvert g_1 \rvert} = \pm 1
$$

**Lendo:** sem histórico, o AdamW não tem como saber o que é ruído, e **todo peso anda exatamente**
$\eta$, na direção do gradiente daquele batch.

Seção 4 do experimento, medindo o primeiro passo depois de retomar o `latest.pt`:

```text
  retomada                  |Δw| no 1º passo      loss dos 5 primeiros passos  validação depois
  completo                        0.19 × lr    1.398 1.425 1.435 1.406 1.434             1.448
  sem o estado do AdamW           1.00 × lr    1.398 1.426 1.453 1.425 1.447             1.446
```

- **Com o estado, cada peso anda em média 0,19 × lr;** sem ele, exatamente **1,00 × lr**: passos 5
  vezes maiores, como a conta previa.
- **A loss sente os passos grandes:** no terceiro passo, 1,453 contra 1,435; no quarto, 1,425 contra
  1,406.
- **O efeito é passageiro:** em algumas dezenas de passos, $m$ e $v$ se reconstroem, e depois de uma
  época inteira as validações (1,446 e 1,448) são iguais dentro do ruído. A seção 6 volta a isso.

### 1.3 O progresso: `TrainingProgress`

Além dos pesos e do otimizador, o `train` precisa saber **onde estava**
([`training/trainer.py`](../training/trainer.py)):

| campo | por que é preciso | no `latest.pt` |
|---|---|---|
| `step` | o lr de cada passo depende do passo global (warmup e agenda em [11-training.md](11-training.md)) | 2460 |
| `epoch` | a época em andamento; ao fim da 30ª, já aponta para a 31ª | 31 |
| `batches_done` | quantos batches da época em andamento já foram usados | 0 |
| `epoch_losses`, `epoch_grad_norms`, `epoch_tokens`, `epoch_train_seconds` | acumulados da época em andamento, para o `EpochRecord` sair igual ao de um treino sem pausa | vazios |
| `seconds` | tempo total, que continua contando depois de retomar | 497,6 |
| `history` | os `EpochRecord` das épocas completas (curvas e escolha do `best.pt`) | 30 registros |
| `loader_rng_state` | o estado do gerador que sorteia a ordem das janelas | 5.056 bytes |
| `rng_state` | o estado do gerador global, que sorteia as máscaras de dropout | 5.056 bytes |

Seção 1 do experimento:

```text
  progress:
    step                 2460
    epoch                31
    batches_done         0
    epoch_losses         lista com 0 itens
    epoch_grad_norms     lista com 0 itens
    epoch_tokens         0
    epoch_train_seconds  0.0
    seconds              497.6
    history              lista com 30 itens
    loader_rng_state     tensor de 5,056 bytes
    rng_state            tensor de 5,056 bytes
```

### 1.4 Os dois geradores aleatórios

O treino usa aleatoriedade em dois lugares, e cada um tem o seu gerador:

1. **A ordem das janelas.** A cada época, o `DataLoader` sorteia uma nova permutação das 2.635 janelas
   ([02-dataset.md](02-dataset.md)), usando o gerador criado com a `seed` da config.
2. **As máscaras de dropout.** Cada forward de treino sorteia quais ativações zerar, usando o gerador
   global do PyTorch.

Um gerador pseudoaleatório é uma máquina determinística: dado o seu **estado** interno (aqui, 5.056
bytes), a sequência de números que ele produz está completamente decidida. Guardar o estado e
restaurá-lo depois (`get_state` / `set_state`, `torch.get_rng_state` / `torch.set_rng_state`) faz o
gerador continuar a mesma sequência, como se o processo nunca tivesse parado.

Um detalhe importante: o `DataLoader` sorteia a permutação da época **inteira** quando a época começa.
Por isso o checkpoint guarda o estado do gerador **no início da época em andamento**, e não o estado no
momento do salvamento: é a partir dele que a mesma ordem pode ser sorteada de novo.

---

## 2. Como o treino é retomado

### 2.1 Quando salvar

`train` chama `on_checkpoint` em dois momentos:

- **ao fim de cada época**, depois da avaliação: há um registro novo no histórico;
- **a cada `checkpoint_every` passos** (1000 no `tiny.yaml`), mesmo no meio de uma época.

O `train` não sabe nada de arquivos: ele entrega uma **cópia** do progresso ao `on_checkpoint`, e quem
chamou decide o que gravar. O coração do loop:

```python
if progress.loader_rng_state is not None:
    generator.set_state(progress.loader_rng_state)     # retomando: mesma ordem de janelas
if progress.rng_state is not None:
    torch.set_rng_state(progress.rng_state)            # retomando: mesmas máscaras de dropout

while progress.epoch <= epochs:
    progress.loader_rng_state = generator.get_state()  # a ordem desta época sai deste estado
    batches = iter(train_loader)
    for _ in range(progress.batches_done):             # retomando no meio: pula o que já foi feito
        next(batches)

    for x, y in batches:
        ...                                            # lr, train_step, acumulados
        if checkpoint_every and progress.step % checkpoint_every == 0 and not end_of_epoch:
            checkpoint()

    ...                                                # avaliação e EpochRecord
    progress.epoch += 1
    progress.batches_done = 0                          # e zera os acumulados da época
    progress.loader_rng_state = generator.get_state()  # a próxima época começa daqui
    checkpoint()
```

`checkpoint()` registra o tempo e o estado do gerador global logo antes de entregar a cópia:

```python
def checkpoint() -> None:
    if on_checkpoint is not None:
        progress.seconds = time.perf_counter() - start
        progress.rng_state = torch.get_rng_state()
        on_checkpoint(copy.deepcopy(progress))
```

### 2.2 Retomar no meio de uma época

O `checkpoint_1000.pt` foi salvo no meio da época 13: $1000 = 12 \times 82 + 16$.

```text
época 13: 82 batches, na ordem sorteada a partir de loader_rng_state
  ┌─────────────────────────┬──────────────────────────────────────────────────┐
  │ b1  b2  ...  b16        │ b17  b18  ...                               b82 │
  └─────────────────────────┴──────────────────────────────────────────────────┘
    já usados antes de salvar ▲ ao retomar, o treino continua daqui
    (pulados, sem treinar)
```

Ao retomar, o gerador volta ao estado do início da época 13, sorteia a **mesma** permutação, e os 16
primeiros batches são lidos e descartados (ler um batch não consome números aleatórios). O gerador
global volta ao estado do passo 1000, e o batch 17 recebe as mesmas máscaras de dropout que recebeu no
treino original.

### 2.3 O `train.py`

A função que o `train.py` passa como `on_checkpoint` decide os nomes:

```python
def on_checkpoint(state: TrainingProgress) -> None:
    names = ["latest.pt"]
    if training.checkpoint_every and state.step % training.checkpoint_every == 0:
        names.append(f"checkpoint_{state.step}.pt")
    if state.batches_done == 0:  # fim de época: há um novo registro no histórico
        save_history(checkpoint_dir / "history.json", state.history)
        if state.history[-1].val_loss <= min(r.val_loss for r in state.history):
            names.append("best.pt")
    for name in names:
        save_checkpoint(checkpoint_dir / name, model, tokenizer,
                        config=config, optimizer=optimizer, progress=state)
```

O log do treino mostra cada salvamento:

```text
checkpoints: /Users/felipersas/Documents/Studies/Mini-GPT/checkpoints | latest.pt a cada época | checkpoint_<passo>.pt a cada 1000 passos | best.pt
loss de validação antes do treino: 4.603 (chute uniforme: 4.575)

passo     1/2460 | época  1 | loss 4.608 | lr 1.00e-05 | grad 4.16 | 27,413 tokens/s
== época  1/30 | loss treino 3.219 (sem dropout 2.465) | loss validação 2.473 (perplexidade 11.86) | 47,795 tokens/s | 577 MB | 9 s
   salvo (passo 82): latest.pt, best.pt
```

Para retomar, o `train.py` lê o checkpoint, recria o otimizador sobre os pesos carregados e restaura
o estado dele:

```python
resumed = load_training_checkpoint(args.resume) if args.resume else None
config = resumed.config if resumed else load_config(args.config)
...
model = resumed.model
optimizer = configure_optimizer(model, training.learning_rate, training.weight_decay)
optimizer.load_state_dict(resumed.optimizer_state)  # m e v de cada parâmetro
progress = resumed.progress
```

Três decisões:

- **A config vem do checkpoint**, e não do `--config`: retomar com outro `d_model` ou outro batch não
  faria sentido.
- **`--epochs` pode aumentar o total**, para treinar mais. Cuidado com agendas de lr: o total de passos
  muda, e um decaimento cosseno (11-training.md) passaria a decair mais devagar. Com o lr constante do
  `tiny.yaml`, isso não muda nada.
- **O vocabulário é conferido.** Se o corpus mudou desde o checkpoint, o `train.py` se recusa a
  continuar (seção 7).

---

## 3. Salvando com segurança: gravação atômica

`torch.save` leva algum tempo para escrever 9,5 MB. Se o processo morrer no meio, o arquivo fica pela
metade. Se esse arquivo for o `latest.pt`, **o checkpoint anterior também se perde**, porque foi
sobrescrito.

A solução é gravar em outro arquivo e só depois trocar o nome:

```python
temporary = path.with_name(path.name + ".tmp")
try:
    torch.save(checkpoint, temporary)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
temporary.replace(path)
```

- **Até `replace`,** o `latest.pt` antigo continua intacto. Uma queda deixa, no máximo, um `.tmp`
  inútil.
- **`replace` é atômico** no mesmo sistema de arquivos: o nome passa a apontar para o arquivo novo de
  uma vez. Não existe um instante em que `latest.pt` esteja pela metade.
- **`BaseException`** inclui o Ctrl+C (`KeyboardInterrupt`): mesmo interrompido, o temporário é
  apagado.

Seção 5 do experimento, tentando carregar a primeira metade de um checkpoint:

```text
  metade do arquivo (4.7 MB): RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central
```

O formato do `torch.save` é um arquivo zip, e o índice do zip (o *central directory*) fica no fim.
Metade de um arquivo não tem esse índice e não pode ser lida. O teste
`test_failed_save_keeps_the_previous_checkpoint` simula um disco cheio no meio da escrita e confere que
o checkpoint anterior continua carregando.

**E o `weights_only=True`?** Um checkpoint salvo com `torch.save` pode conter objetos Python
arbitrários, e carregá-lo sem cuidado executaria código embutido no arquivo. `weights_only=True` só
aceita tensores e tipos simples (números, strings, listas, dicionários). Por isso o progresso é salvo
como dicionário (`asdict`) e convertido de volta em `TrainingProgress` ao carregar.

---

## 4. Os arquivos: `latest`, `best` e `checkpoint_<passo>`

Seção 2 do experimento, avaliando cada arquivo na validação:

```text
  arquivo               tamanho  passo  época  batches feitos  validação agora
  checkpoint_1000.pt     9.5 MB  1,000     13              16            1.758
  checkpoint_2000.pt     9.5 MB  2,000     25              32            1.499
  best.pt                9.5 MB  2,460     31               0            1.452
  latest.pt              9.5 MB  2,460     31               0            1.452

  a mesma geração em cada checkpoint ('Capitú', seed 0, temperature 0.8, top-p 0.95):
  checkpoint_1000.pt   Capitú algunstiva da caminidas a grasinas. Tambem dos ser não da recomo fui um a bente
  checkpoint_2000.pt   Capitú alguns simplicos. Eram alma, só os amigos, dolhos nas casas do ceu acces do cha
  best.pt              Capitú se faça que ver as destas relações. Alguma dos seus seus articas fui casada, co
  latest.pt            Capitú se faça que ver as destas relações. Alguma dos seus seus articas fui casada, co
```

- **`best.pt` e `latest.pt` são o mesmo estado** neste treino, porque a validação caiu até a última
  época ([14-evaluation.md](14-evaluation.md)). Num treino que passasse do ponto, o `best.pt` ficaria parado na melhor época enquanto
  o `latest.pt` seguiria em frente.
- **Os checkpoints periódicos mostram a evolução.** No passo 1000, quase nenhuma palavra existe
  ("algunstiva", "caminidas"); no passo 2000, a maioria das palavras é real ("alguns", "amigos",
  "casas"); no fim, trechos inteiros soam como português ("se faça que ver as destas relações").
- **O `best.pt` é regravado a cada nova melhor época,** não só no fim. O log da retomada da seção 5
  mostra isso: na época 29, a validação subiu de 1,463 para 1,464, e só o `latest.pt` foi salvo.
- **Espaço em disco:** 4 arquivos de 9,5 MB. Com checkpoints periódicos num treino longo, é comum
  apagar os mais antigos; aqui, com 2.460 passos, são só dois.

Qualquer um deles serve para gerar texto:

```bash
uv run python inference.py "Capitú" --checkpoint checkpoints/checkpoint_1000.pt
```

---

## 5. Retomar é exato: as provas

**Prova 1: o teste.** `test_resumed_training_matches_uninterrupted_training` treina um modelo minúsculo
(com dropout) por 2 épocas sem pausa. Depois treina outro igual, "derruba" o processo logo depois de
salvar um checkpoint, carrega o checkpoint com o gerador global embaralhado (como num processo novo) e
termina o treino. Os pesos finais são comparados com `torch.equal`, que exige igualdade exata, e as
losses de cada época com `==`. O teste roda com a queda no fim da época 1 e no meio da época 2.

**Prova 2: o `train.py` de verdade.** Retomando do `checkpoint_2000.pt` do treino real:

```bash
uv run python train.py --resume checkpoints/checkpoint_2000.pt --checkpoint-dir /tmp/retomada
```

```text
retomando:  checkpoints/checkpoint_2000.pt | passo 2,000 | época 25 (32 batches já feitos)

== época 25/30 | loss treino 1.505 (sem dropout 1.365) | loss validação 1.491 (perplexidade 4.44) | 12,039 tokens/s | 570 MB | 434 s
   salvo (passo 2050): latest.pt, best.pt
== época 26/30 | loss treino 1.491 (sem dropout 1.347) | loss validação 1.477 (perplexidade 4.38) | 14,544 tokens/s | 570 MB | 463 s
   salvo (passo 2132): latest.pt, best.pt
== época 27/30 | loss treino 1.476 (sem dropout 1.333) | loss validação 1.476 (perplexidade 4.37) | 14,741 tokens/s | 570 MB | 492 s
   salvo (passo 2214): latest.pt, best.pt
== época 28/30 | loss treino 1.464 (sem dropout 1.319) | loss validação 1.463 (perplexidade 4.32) | 15,318 tokens/s | 570 MB | 522 s
   salvo (passo 2296): latest.pt, best.pt
== época 29/30 | loss treino 1.452 (sem dropout 1.312) | loss validação 1.464 (perplexidade 4.32) | 14,203 tokens/s | 570 MB | 553 s
   salvo (passo 2378): latest.pt
== época 30/30 | loss treino 1.441 (sem dropout 1.296) | loss validação 1.452 (perplexidade 4.27) | 15,311 tokens/s | 570 MB | 580 s
   salvo (passo 2460): latest.pt, best.pt
```

As épocas 25 a 30 repetem os valores do treino sem pausa (a tabela de [14-evaluation.md](14-evaluation.md), seção 2.3). Comparando
o `latest.pt` final da retomada com o do treino original, tensor a tensor, **pesos, $m$ e $v$ do
AdamW, estados aleatórios, estado do loader e histórico são idênticos**. Só os tempos mudam (outro
processo, outra carga na máquina).

**Prova 3: o experimento.** A primeira linha da tabela da seção 6 retoma do `checkpoint_1000.pt` até o
fim da época 13, e as losses batem com as do treino original em todas as casas decimais.

---

## 6. O que falta sem cada parte

Seção 3 do experimento: a partir do `checkpoint_1000.pt`, cinco formas de retomar até o fim da época
13, cada uma deixando de restaurar uma parte.

```text
  checkpoint: passo 1000, época 13, batch 16
  treino original sem pausa, época 13: loss de treino 1.805137 | validação 1.730046

  retomada                 passos  loss 10 1ºs passos  loss treino  validação  − original
  completo                     66               1.809     1.805137   1.730046   +0.000000
  sem o estado do AdamW        66               1.834     1.807123   1.726596   -0.003450
  sem o estado aleatório       66               1.806     1.804974   1.731253   +0.001207
  sem a posição no loader      82               1.802     1.797020   1.721458   -0.008588
  só os pesos                  82               1.793     1.775738   1.721662   -0.008385

  retomada completa == treino original, bit a bit: True
```

| retomada | o que muda |
|---|---|
| completo | nada: é o treino original |
| sem o estado do AdamW | $m$ e $v$ recomeçam do zero: os primeiros passos são ~5× maiores (seção 1.2) |
| sem o estado aleatório | as máscaras de dropout são outras |
| sem a posição no loader | a época 13 recomeça do batch 1, numa ordem nova: 16 batches são vistos duas vezes, e a época tem 82 passos em vez de 66 |
| só os pesos | tudo acima, e o passo volta a 0: o warmup se repete (lr de 0,00001 subindo até 0,00082 em 82 passos) e o histórico se perde |

**O que a tabela mostra:**

- **Só a retomada completa reproduz o treino.** Qualquer parte faltando muda os números.
- **O tamanho do ruído.** Trocar apenas as máscaras de dropout mudou a validação em +0,0012. Diferenças
  dessa ordem não significam "melhor" ou "pior"; significam "outro sorteio".
- **Sem o AdamW, o começo piora:** a loss dos 10 primeiros passos foi 1,834, contra 1,809. Ao fim da
  época, a validação ficou 0,003 abaixo, dentro do ruído.
- **As variantes que recomeçam a época ficaram com loss menor, mas treinaram mais:** 82 passos em vez
  de 66. Com o modelo ainda longe de convergir, 16 passos a mais valem mais que qualquer diferença de
  método. Não é uma vantagem de retomar mal.
- **"Só os pesos" não desandou** por dois motivos que valem só para este treino: o warmup repetido usa
  lr pequeno justamente nos passos em que o AdamW está sem histórico, e o lr do `tiny.yaml` é constante,
  então recomeçar a agenda não muda nada depois do warmup.

**Então por que tanto cuidado?** Neste modelo pequeno e com lr constante, retomar de forma inexata custa
pouco em loss. O que se perde é outra coisa:

- **reprodutibilidade:** um treino retomado deixa de ser comparável ao mesmo treino sem pausa, e
  diferenças de 0,003 a 0,009 (o tamanho das comparações em [11-training.md](11-training.md)) passam a
  ser indistinguíveis de efeitos reais;
- **a agenda do lr:** com decaimento cosseno, voltar o passo a 0 levaria o lr de volta ao máximo no
  meio ou no fim do treino;
- **o histórico:** sem ele, as curvas de [14-evaluation.md](14-evaluation.md) e a escolha do `best.pt`
  recomeçam do zero.

---

## 7. Por que o vocabulário vai dentro do checkpoint

Seção 6 do experimento:

```text
  o corpus ganha um '@': vocab_size 97 -> 98
  tokens que mudaram de ID: 68 de 97
  'Capitú' com o vocabulário do checkpoint: [31, 53, 68, 61, 72, 96]
  'Capitú' com o vocabulário do corpus novo: [32, 54, 69, 62, 73, 97]
  os IDs do checkpoint lidos com o vocabulário novo: 'BZohsõ'
```

O tokenizer ordena os caracteres para montar o vocabulário ([01-tokenizer.md](01-tokenizer.md)). Um único `'@'` a mais no corpus
entra antes das letras e **empurra 68 dos 97 IDs** uma posição para a frente. Os pesos, porém, estão
presos aos IDs antigos: a linha 31 da matriz de embeddings aprendeu a representar `'C'`. Com o
vocabulário novo, o ID 31 passa a significar `'B'`, e "Capitú" vira "BZohsõ".

Por isso o checkpoint guarda o vocabulário, e o `train.py` se recusa a retomar se o vocabulário do
corpus atual for diferente:

```python
if resumed and resumed.tokenizer.vocab.id_to_token != tokenizer.vocab.id_to_token:
    raise SystemExit("O corpus mudou desde o checkpoint: os IDs não correspondem aos pesos.")
```

---

## 8. Usando

```bash
uv run python train.py                                             # treino novo, salva em checkpoints/
uv run python train.py --resume checkpoints/latest.pt              # continua de onde parou
uv run python train.py --resume checkpoints/latest.pt --epochs 40  # treina até a época 40
uv run python train.py --checkpoint-dir runs/teste                 # salva em outra pasta
uv run python inference.py "Capitú" --checkpoint checkpoints/best.pt
uv run python -m experiments.e15_checkpoints                       # as seções deste documento
```

| opção | padrão | efeito |
|---|---|---|
| `--config` | `configs/tiny.yaml` | config de um treino novo (ignorada ao retomar) |
| `--checkpoint-dir` | `checkpoints/` | onde gravar `latest.pt`, `best.pt`, `checkpoint_<passo>.pt` e `history.json` |
| `--resume` | nenhum | checkpoint de onde continuar |
| `--epochs` | o da config | total de épocas, para treinar além do planejado |

Na config, `checkpoint_every` define o intervalo dos checkpoints periódicos (0 desliga). O `latest.pt`
é gravado ao fim de toda época de qualquer forma.

---

## 9. O que cada teste garante

[`tests/test_checkpoint.py`](../tests/test_checkpoint.py):

| teste | propriedade |
|---|---|
| `test_round_trip_preserves_config_vocab_and_outputs` | salvar e carregar só os pesos mantém config, vocabulário e saídas |
| `test_loaded_model_keeps_weight_tying` | o LM head continua compartilhando a matriz do embedding |
| `test_loaded_model_is_ready_for_inference` | o modelo carregado vem em modo `eval` |
| `test_checkpoints_are_offered_periodically_and_at_every_epoch_end` | `on_checkpoint` é chamado nos passos 10, 20, 30, 40 e no fim de cada época, com época e batches corretos |
| `test_resumed_training_matches_uninterrupted_training` | cair e retomar (no fim de uma época ou no meio da seguinte) dá pesos e losses **idênticos** aos de um treino sem pausa |
| `test_resuming_without_the_optimizer_state_changes_the_result` | sem o estado do AdamW, os pesos finais mudam (o teste anterior não passa por acaso) |
| `test_training_checkpoint_restores_config_and_progress` | a config e o progresso (passo, época, batches, acumulados, histórico) voltam iguais |
| `test_training_checkpoint_also_works_for_inference` | `load_checkpoint` lê um checkpoint de treino e gera logits |
| `test_weights_only_checkpoint_cannot_resume_training` | um arquivo só com pesos gera um erro claro ao tentar retomar |
| `test_training_state_must_be_complete` | salvar só parte do estado de treino (otimizador sem progresso) é recusado |
| `test_failed_save_keeps_the_previous_checkpoint` | uma falha no meio da escrita não destrói o checkpoint anterior e não deixa o `.tmp` |

[`tests/test_config.py`](../tests/test_config.py):

| teste | propriedade |
|---|---|
| `test_tiny_config_loads` | agora também confere que `checkpoint_every` está definido |
| `test_config_round_trips_through_a_dict` | `config_from_dict(asdict(config))` devolve a mesma config, o caminho que ela faz para dentro e para fora do checkpoint |

---

## Resumo

1. Para **gerar texto**, bastam arquitetura, vocabulário e pesos (3,15 MB). Para **continuar o treino**
   exatamente, é preciso também a config, o estado do AdamW e o progresso (9,46 MB no total).
2. O AdamW guarda $m$ e $v$ para cada peso. Sem eles, o primeiro passo move todo peso exatamente
   $\eta$ (medido: 1,00 × lr, contra 0,19 × lr com o estado), e a loss piora por alguns passos.
3. O progresso inclui o passo, a posição dentro da época, os acumulados da época, o histórico e os
   estados dos **dois** geradores aleatórios: o que sorteia a ordem das janelas (guardado no início da
   época) e o do dropout.
4. A retomada é **exata**: o teste, o `train.py --resume` e o experimento reproduzem o treino sem pausa
   bit a bit, inclusive retomando no meio de uma época.
5. Retomar de forma incompleta muda pouco a loss neste modelo (diferenças do tamanho do ruído), mas
   destrói a reprodutibilidade, a agenda do lr e o histórico.
6. Os checkpoints são gravados em `.tmp` e renomeados: uma queda no meio da escrita nunca destrói o
   anterior. `latest.pt` a cada época, `best.pt` na melhor validação, `checkpoint_<passo>.pt` a cada
   1000 passos.
7. O vocabulário vai junto porque os pesos estão presos aos IDs: um caractere novo no corpus mudaria 68
   dos 97 IDs e transformaria "Capitú" em "BZohsõ".

---

## Para checar o entendimento

1. Quais chaves do checkpoint o `inference.py` usa? Por que ele consegue ler tanto o formato mais
   simples (13-generation.md) quanto o completo deste documento?
2. O estado do AdamW ocupa 6,30 MB, e os pesos, 3,15 MB. Por que exatamente o dobro?
3. Um peso recebeu gradientes +0,2, −0,2, +0,2, −0,2 por muitos passos. Quanto ele anda por passo com o
   estado do AdamW preservado? E no primeiro passo depois de zerar $m$ e $v$?
4. Por que o AdamW guarda o número do passo junto com $m$ e $v$? O que aconteceria com $\hat v$ se $m$
   e $v$ fossem restaurados, mas o passo voltasse a 1?
5. Por que o checkpoint guarda o estado do gerador do loader **no início** da época em andamento, e não
   no momento em que o arquivo é salvo?
6. Ao retomar no batch 17 de uma época, os 16 primeiros batches são lidos e descartados. Por que ler
   esses batches não muda nenhum número aleatório do treino?
7. Na seção 6, a variante "sem a posição no loader" terminou com validação 0,0086 menor que o treino
   original. Isso quer dizer que recomeçar a época é melhor? Por quê?
8. Com uma agenda de decaimento cosseno, o que aconteceria com o lr se o treino fosse retomado só com
   os pesos, perto do fim?
9. Por que gravar direto em `latest.pt` é perigoso? O que o `replace` garante?
10. Por que `weights_only=True` exige que o progresso seja salvo como dicionário, e não como o objeto
    `TrainingProgress`?
11. Num treino em que a validação começa a subir na época 20 e o treino vai até a 30, o que haverá em
    `best.pt` e em `latest.pt`? Qual você usaria para gerar texto?
12. O corpus ganhou um `'@'`. Por que o ID de `'C'` mudou, mas o de `' '` (espaço) não?
