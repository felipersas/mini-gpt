# Avaliação

> **Como saber se o modelo aprendeu de verdade, e onde ele ainda erra?**

Código: [`training/metrics.py`](../training/metrics.py), `token_losses` em
[`training/loss.py`](../training/loss.py), `EpochRecord` e `train_eval_loader` em
[`training/trainer.py`](../training/trainer.py), e [`train.py`](../train.py), que agora grava
`checkpoints/history.json`. Testes: [`tests/test_metrics.py`](../tests/test_metrics.py) e adições em
[`tests/test_loss.py`](../tests/test_loss.py) e [`tests/test_trainer.py`](../tests/test_trainer.py).
Experimento: `uv run python -m experiments.e14_evaluation` (precisa do modelo e do histórico salvos
por `uv run python train.py`). Notação geral: [00-notacao.md](00-notacao.md). Visão geral:
[architecture.md](architecture.md).

---

## Para que serve

### O problema

Ao fim do treino ([11-training.md](11-training.md)), ficou **um número**: loss de validação 1,452. Ele diz *quanto* o modelo
erra em média, mas não responde às perguntas que importam:

1. **Esse número é bom?** Comparado com o quê? E o que "1,452 nats" significa na prática?
2. **O modelo generaliza ou decora?** A loss de treino e a de validação estão sendo medidas do mesmo
   jeito?
3. **Onde ele erra?** No começo das janelas? Em que tipo de caractere?
4. **Ele sabe quando está certo?** Quando o modelo dá 90% a uma opção, acerta 90% das vezes?
5. **O que a attention aprendeu?** Para onde cada head olha?
6. **Quanto custa treinar?** Quanta memória, e em quê?

Avaliar é transformar a loss em respostas para essas perguntas. Este documento também encontrou um
**erro de interpretação** em [11-training.md](11-training.md) (seção 2.4), o que mostra por que medir
com cuidado importa.

Termos usados daqui em diante:

- **Métrica:** um número que resume algum aspecto do comportamento do modelo.
- **Nat e bit:** unidades de informação. A loss usa logaritmo natural, então é medida em **nats**.
  Com logaritmo na base 2, a unidade é o **bit**.
- **Perplexidade:** $e^{\text{loss}}$, lida como "entre quantas opções o modelo hesita" (ver
  [10-loss.md](10-loss.md)).
- **Curva de aprendizado:** a loss ao longo das épocas.
- **Gap de generalização:** a diferença entre a loss de validação e a de treino.
- **Overfitting:** quando o modelo continua melhorando no treino, mas **piora** na validação.
- **Acurácia:** a fração de posições em que o token mais provável é o correto.
- **Calibração:** o quanto a confiança do modelo corresponde à sua taxa real de acerto.
- **Mapa de attention:** a matriz de pesos de uma head, mostrando para onde cada posição olha.
- **Ativações:** os tensores intermediários do forward (saídas de cada camada).
- **Hook:** uma função que o PyTorch chama automaticamente em certo momento (antes de um módulo
  rodar, ao guardar um tensor para o backward), sem mudar o código do modelo.
- **RSS** (*resident set size*): a memória RAM realmente ocupada pelo processo.

### Uma analogia

Pense num aluno que tirou **7** num simulado:

- **7 é bom?** Depende da turma. Se chutando se tira 1 e os melhores métodos simples dão 5, 7 é
  ótimo. São as **réguas** (seção 1).
- **Ele aprendeu ou decorou?** Compare a nota nos exercícios que ele já tinha feito com a nota numa
  prova nova. E compare nas mesmas condições: não vale fazer uma prova com barulho e a outra em
  silêncio. São as **curvas** (seção 2).
- **Em que ele erra?** O boletim por matéria diz mais que a média. É a análise **por posição e por
  tipo de caractere** (seções 3 e 4).
- **Quando diz "tenho certeza", ele acerta?** É a **calibração** (seção 5).
- **Como ele resolve as questões?** Olhar o rascunho. São os **mapas de attention** (seção 6).

### O que entra e o que sai

| | o quê | exemplo |
|---|---|---|
| entra | o modelo treinado | `checkpoints/latest.pt` (ver [13-generation.md](13-generation.md)) |
| entra | o histórico do treino | `checkpoints/history.json`: 30 registros, um por época |
| entra | o texto de validação | 37.478 caracteres, avaliados em 288 janelas de 128 |
| sai | métricas | loss 1,452 · perplexidade 4,27 · 2,095 bits por caractere |
| sai | curvas | treino × validação, época a época |
| sai | diagnóstico | loss por posição, por tipo de caractere, calibração |
| sai | interpretação | estatísticas e mapas de attention de cada head |
| sai | custo | memória por componente, tokens por segundo |

### Onde este documento entra

```text
corpus → tokenizer → dataset → GPT → loss → training loop (11-training.md)
                                                 │
                                                 │  a cada época: EpochRecord     ← este documento
                                                 │  (inclui a loss de treino sem dropout)
                                                 ▼
                              checkpoints/latest.pt  +  checkpoints/history.json
                                                 │
                                                 ▼
          e14_evaluation: réguas, curvas, loss por posição e por caractere, ← este documento
                          calibração, mapas de attention, memória
```

### O que daria errado sem esta parte

- **Conclusões erradas.** Em [11-training.md](11-training.md), as curvas de treino e validação
  pareciam se cruzar na época 27. Era um efeito do dropout, não do modelo (seção 2.4).
- **Nenhuma ideia do que melhorar.** Saber que o modelo erra muito na primeira letra das palavras e
  pouco nos espaços orienta decisões melhor que um número médio.
- **Nenhum plano para escalar.** Sem saber que as ativações ocupam 96% da memória do treino, não dá
  para prever o que acontece ao aumentar o contexto (seção 7), nem para planejar otimizações de GPU e
  de performance depois.

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $\mathcal{L}$ | "éle caligráfico" | loss média (cross-entropy), em nats | 1,452 na validação | `val_loss` |
| $\ell_t$ | "éle tê" | loss de uma posição: $-\ln p(y_t)$ | um valor por posição, `[B, T]` | `token_losses(logits, y)` |
| $p(y_t)$ | "pê de ípsilon tê" | probabilidade que o modelo deu ao caractere correto na posição $t$ | entre 0 e 1 | `probs.gather(...)` |
| $\ln$, $\log_2$ | "logaritmo natural", "log na base 2" | $\ln 2 \approx 0{,}693$ converte entre as bases | | `math.log`, `math.log2` |
| $\text{PPL}$ | "perplexidade" | $e^{\mathcal{L}}$ | 4,27 | `perplexity(loss)` |
| $\text{BPC}$ | "bits por caractere" | $\mathcal{L} / \ln 2$ | 2,095 | `bits_per_token(loss)` |
| $\Delta$ | "delta" | gap: $\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{treino}}$ | +0,156 na época 30 | `val_loss - train_eval_loss` |
| $N$ | "ene" | número de posições avaliadas | 36.864 (288 janelas × 128) | `y.numel()` |
| $W$ | "dáblio" | número de janelas de validação | 288 | `losses.shape[0]` |
| $t$ | "tê" | posição dentro da janela; vê $t + 1$ caracteres | 0 a 127 | índice da coluna |
| $c$ | "cê" | confiança: a maior probabilidade, $\max_i p_i$ | entre 1/97 e 1 | `probs.max(dim=-1)` |
| $\hat y$ | "ípsilon chapéu" | o caractere previsto: $\arg\max_i p_i$ | um ID | `predicted` |
| $b$, $n_b$ | "bê", "ene bê" | uma faixa de confiança e quantas posições caem nela | 0,9–1,0 com 7.419 | `mask.sum()` |
| $\text{ECE}$ | "erro de calibração esperado" | média de $\lvert \text{acerto} - \text{confiança} \rvert$ nas faixas | ≈ 0,03 | `ece` |
| $a_{ij}$ | "a i jota" | peso de attention da posição $i$ para a posição $j$ | entre 0 e 1; cada linha soma 1 | `weights[..., i, j]` |
| $\bar D$ | "dê barra" | distância média para onde uma head olha: $\sum_j a_{ij}(i - j)$ | 1,6 a 13,8 | `(w * distance).sum(-1)` |
| $B, h, T, d, d_h, V$ | | batch, heads, contexto, `d_model`, dimensão por head, vocabulário | 32, 4, 128, 128, 32, 97 | `config` |
| MB | "megabyte" | $2^{20}$ bytes | | `2**20` |

---

## 1. As métricas: loss, perplexidade e bits

### 1.1 Três formas de escrever o mesmo número

A loss, a perplexidade e os bits por caractere carregam **a mesma informação**, em escalas diferentes:

$$
\text{PPL} = e^{\mathcal{L}}
\qquad\qquad
\text{BPC} = \frac{\mathcal{L}}{\ln 2}
$$

**Lendo a fórmula:**

- $\mathcal{L}$ é a loss média em nats. Com $\mathcal{L} = 1{,}452$:
- $\text{PPL} = e^{1{,}452} = 4{,}27$. Se o modelo escolhesse entre $k$ opções igualmente prováveis,
  a loss seria $\ln k$ e a perplexidade seria $k$. Então **4,27 é o número efetivo de opções** entre as
  quais o modelo hesita, em média, a cada caractere.
- $\text{BPC} = 1{,}452 / 0{,}693 = 2{,}095$. Trocar a base do logaritmo só muda a unidade: 1 nat vale
  $1/\ln 2 \approx 1{,}443$ bits.

**O que os bits significam.** Um compressor que usasse as probabilidades do modelo gastaria, em média,
**2,095 bits por caractere** para guardar o texto de validação. Sem modelo nenhum (97 caracteres
igualmente prováveis), seriam $\log_2 97 = 6{,}6$ bits. Mil caracteres de *Dom Casmurro* caberiam em
~2.095 bits (~262 bytes), contra 6.600 bits no chute uniforme: o modelo "entende" o texto o bastante
para comprimi-lo 3,15 vezes. Prever bem e comprimir bem são a mesma coisa.

**A perplexidade como média geométrica.** Como $\mathcal{L}$ é a média de $-\ln p(y_t)$:

$$
\text{PPL} = e^{\frac{1}{N}\sum_t -\ln p(y_t)} = \left(\prod_{t=1}^{N} \frac{1}{p(y_t)}\right)^{1/N}
$$

**Lendo a fórmula:** $1/p(y_t)$ é "de quantas opções o acerto parecia ser uma", para cada posição. Se o
modelo dá 0,5 ao caractere certo, isso vale 2; se dá 0,1, vale 10. A perplexidade é a **média
geométrica** desses valores, em que um único $p$ muito pequeno pesa bastante.

O código ([`training/metrics.py`](../training/metrics.py)):

```python
def perplexity(loss: float) -> float:
    return math.exp(loss)

def bits_per_token(loss: float) -> float:
    return loss / math.log(2)
```

### 1.2 O modelo contra as réguas

Seção 1 do experimento:

```text
                         loss (nats)  perplexidade  bits/caractere
  treino, sem dropout          1.296          3.66           1.870
  validação                    1.452          4.27           2.095
  diferença validação − treino: +0.156

  réguas na validação (docs/10-loss.md):
  chute uniforme               4.575         97.00           6.600
  unigrama                     3.058         21.29           4.412
  bigrama                      2.328         10.26           3.359
  trigrama                     1.923          6.84           2.774
  GPT treinado                 1.452          4.27           2.095
```

As réguas são os modelos de contagem de [10-loss.md](10-loss.md) (seção 6): o unigrama só conhece a frequência de cada
caractere; o bigrama usa 1 caractere de contexto; o trigrama, 2.

- **O GPT ganha do trigrama por 0,471 nats** (0,68 bit por caractere). A perplexidade cai de 6,84 para
  4,27: o modelo hesita entre ~4 opções onde o trigrama hesita entre ~7.
- **Esse ganho é maior que o do bigrama para o trigrama** (0,405 nats), mesmo com o trigrama já tendo
  pegado os padrões mais fáceis. Usar muito contexto de forma flexível vale mais que contar
  combinações curtas.
- **O treino tem loss menor que a validação** (1,296 contra 1,452). A diferença, +0,156, é o gap de
  generalização, analisado na seção 2.

---

## 2. Curvas de aprendizado

### 2.1 O histórico: `EpochRecord` e `history.json`

A cada época, `train` cria um `EpochRecord`, e o `train.py` grava a lista inteira em
`checkpoints/history.json`. Assim, as curvas podem ser analisadas depois, sem treinar de novo.

| campo | o que é | época 30 |
|---|---|---:|
| `epoch`, `step` | época e total de passos até aqui | 30, 2460 |
| `train_loss` | média das losses dos batches da época, **com dropout** e com os pesos mudando | 1,441 |
| `train_eval_loss` | loss no treino **ao fim da época, sem dropout** | 1,296 |
| `val_loss` | loss na validação ao fim da época, sem dropout | 1,452 |
| `learning_rate` | lr do último passo | 0,001 |
| `grad_norm` | média da norma do gradiente antes do clipping | 0,63 |
| `tokens_per_second` | tokens processados por segundo, só nos passos de treino | 29.484 |
| `peak_memory_mb` | pico de memória do processo até aqui | 598 |
| `seconds` | tempo acumulado desde o início | 446 |

O pico de memória vem de `resource.getrusage`, que devolve o maior RSS do processo até o momento:

```python
def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # O macOS informa em bytes; o Linux, em kilobytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
```

### 2.2 Duas losses de treino, e por que só uma é comparável

A loss de treino que o log de [11-training.md](11-training.md) mostrava (`train_loss`) **não é comparável** com a de validação,
por dois motivos:

1. **Dropout.** Durante o treino, 10% das ativações são zeradas ao acaso. O modelo trabalha "com
   parte do cérebro desligada" e erra mais. A validação é medida sem dropout (`model.eval()`).
2. **Pesos mudando.** `train_loss` é a média dos 82 batches da época, e o primeiro batch foi calculado
   com os pesos do começo da época, piores que os do fim. A validação usa só os pesos do fim.

Os dois efeitos empurram `train_loss` **para cima**. A solução é medir o treino exatamente como a
validação: ao fim da época, sem dropout, sem gradientes. É o `train_eval_loader` do `train.py`:

```python
# Mesmo texto do treino, em ordem fixa: mede a loss de treino sem dropout ao fim de cada época.
train_eval_loader = create_dataloader(
    train_ids, context_length=context_length, stride=context_length,
    batch_size=training.batch_size, shuffle=False,
)
```

Dentro de `train`, depois da validação:

```python
train_eval_loss = (
    evaluate(model, train_eval_loader) if train_eval_loader is not None else None
)
```

`evaluate` é a mesma função da validação (11-training.md): `model.eval()`, `torch.no_grad()` e média pesada
pelo número de tokens. O custo é uma passada só de forward pelo texto de treino a cada época.

**Lendo o gap:**

$$
\Delta = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{treino, sem dropout}}
$$

- $\Delta > 0$: o modelo se sai melhor no texto que viu no treino do que em texto novo.
- $\Delta \approx 0$: o que ele aprendeu vale igualmente para os dois textos.
- Na época 30: $\Delta = 1{,}452 - 1{,}296 = +0{,}156$.

### 2.3 As curvas

Seção 2 do experimento:

```text
  época | treino c/ dropout | treino s/ dropout | validação |  grad | tokens/s |  memória
      1 |             3.219 |             2.465 |     2.473 |  1.56 |   49,888 |   598 MB
      2 |             2.398 |             2.338 |     2.355 |  0.49 |   46,933 |   598 MB
      3 |             2.330 |             2.279 |     2.301 |  0.46 |   49,776 |   598 MB
      5 |             2.223 |             2.138 |     2.165 |  0.59 |   49,536 |   598 MB
     10 |             1.957 |             1.828 |     1.871 |  0.66 |   27,859 |   598 MB
     15 |             1.731 |             1.599 |     1.658 |  0.64 |   26,603 |   598 MB
     20 |             1.595 |             1.460 |     1.553 |  0.64 |   30,144 |   598 MB
     25 |             1.505 |             1.365 |     1.491 |  0.64 |   30,839 |   598 MB
     27 |             1.476 |             1.333 |     1.476 |  0.63 |   29,832 |   598 MB
     28 |             1.464 |             1.319 |     1.463 |  0.63 |   31,415 |   598 MB
     29 |             1.452 |             1.312 |     1.464 |  0.63 |   31,665 |   598 MB
     30 |             1.441 |             1.296 |     1.452 |  0.63 |   29,484 |   598 MB

  loss por época (t = treino sem dropout, v = validação, * = as duas)
   2.47 │ *
   2.37 │   *
   2.26 │     * v
   2.15 │       t * v
   2.05 │           t *
   1.94 │               * *
   1.83 │                   * v
   1.72 │                     t * * v
   1.62 │                           t * * v v v
   1.51 │                                 t t t * v v v v v v v v v
   1.40 │                                         t t t t t         v
   1.30 │                                                   t t t t t
        └─────────────────────────────────────────────────────────────
          1         6         11        16        21        26
```

(No gráfico, cada coluna é uma época e cada linha é uma faixa de loss; `*` indica que as duas curvas
caíram na mesma faixa.)

**O que as curvas mostram:**

- **Sem dropout, o treino fica abaixo da validação em todas as épocas.** O gap cresce quase sem parar:

  | época | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | $\Delta$ | +0,008 | +0,026 | +0,043 | +0,059 | +0,093 | +0,126 | +0,156 |

- **Por que o gap começa pequeno e cresce.** A validação são os últimos 10% do livro: outros capítulos,
  outras cenas. No começo, o modelo aprende regras que valem para o livro inteiro (ortografia,
  palavras comuns), e o gap é quase zero. Depois, passa a aprender também sequências específicas do
  texto de treino (frases, combinações de nomes e palavras), que ajudam pouco nos capítulos finais.
- **Isso ainda não é overfitting.** Overfitting é a validação **piorar** enquanto o treino melhora. Aqui
  a validação caiu em todas as épocas, com uma única exceção desprezível: 1,4634 → 1,4635 entre as
  épocas 28 e 29. A melhor época é a última. O gap crescente é um **aviso**: com mais épocas neste
  mesmo corpus, a validação tende a parar de cair e depois subir.
- **A loss com dropout é bem maior no começo** (3,219 contra 2,465 na época 1) por causa do motivo 2 da
  seção 2.2: a média da época 1 inclui os primeiros passos, com loss perto de 4,6. No fim, os pesos
  quase não mudam dentro de uma época, e a diferença (1,441 contra 1,296) é basicamente o dropout.
- **A norma do gradiente se estabiliza** em ~0,63 depois das primeiras épocas, abaixo do limite de
  clipping (1,0).
- **A velocidade caiu de ~50 mil para ~30 mil tokens/s a partir da época 10.** O modelo e os batches
  são os mesmos em todas as épocas; a queda veio de outros processos usando a CPU durante o treino.
  É uma medida sensível à carga da máquina, e é por isso que o treino levou 446 s, contra 285 s em
  [11-training.md](11-training.md).
- **A memória chegou ao pico (598 MB) já na primeira época** e não cresceu mais: todos os batches têm o
  mesmo tamanho, então cada passo precisa da mesma memória (seção 7).

### 2.4 Corrigindo uma leitura errada do treino

Em [11-training.md](11-training.md), a comparação era entre `train_loss` (com dropout) e a validação (sem dropout). Com ela,
a leitura foi: "o treino ficou acima da validação até a época 26; na 27 as curvas se cruzaram". Veja a
linha da época 27:

```text
     27 |             1.476 |             1.333 |     1.476 |
```

A loss com dropout empatou com a validação, mas a loss sem dropout já estava **0,143 abaixo**. O
"cruzamento" era só o ponto em que a penalidade do dropout (que diminui aos poucos) ficou do tamanho do
gap de generalização (que cresce aos poucos). Não havia nada de especial acontecendo com o modelo na
época 27.

**A lição:** só compare medidas feitas nas mesmas condições. O doc
[11-training.md](11-training.md) (seção 10.4) foi atualizado com essa correção.

---

## 3. Onde o modelo erra: a posição dentro da janela

### 3.1 A loss de cada posição: `token_losses`

A loss média esconde **como** o erro se distribui. Para enxergar isso, a cross-entropy foi separada em
duas funções ([`training/loss.py`](../training/loss.py)):

```python
def token_losses(logits, targets):
    """-log p(alvo) de cada posição: shape de targets."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

def cross_entropy(logits, targets):
    return token_losses(logits, targets).mean()
```

$$
\ell_t = -\ln p(y_t \mid x_{\le t})
\qquad\qquad
\mathcal{L} = \frac{1}{N} \sum_{\text{posições}} \ell_t
$$

**Lendo a fórmula:**

- $\ell_t$ é a penalidade de uma única previsão: o $-\ln$ da probabilidade que o modelo deu ao
  caractere correto $y_t$, tendo visto os caracteres até a posição $t$ ($x_{\le t}$).
- $\mathcal{L}$ é a média de todas as $N$ penalidades. É exatamente a loss definida em
  [10-loss.md](10-loss.md) e usada no treino ([11-training.md](11-training.md)); só que agora dá para
  agrupar os $\ell_t$ de outras formas antes de tirar a média.
- Exemplo: se o modelo deu 0,8 ao caractere certo, $\ell_t = -\ln 0{,}8 = 0{,}22$; se deu 0,05,
  $\ell_t = 3{,}0$.

O experimento roda o modelo em toda a validação e guarda, para cada posição, a entrada, o alvo,
$\ell_t$, a confiança e se acertou. São 288 janelas de 128 caracteres: 36.864 posições. (Com
`drop_last`, as 4 janelas que não completam um batch de 32 ficam de fora.)

### 3.2 A loss por posição

A posição $t$ da janela vê só $t + 1$ caracteres, por causa da causal mask (ver
[04-attention.md](04-attention.md)). A posição 0 vê um
único caractere; a posição 127 vê a janela inteira. A média de cada posição sobre as $W = 288$ janelas:

$$
\bar\ell_t = \frac{1}{W} \sum_{w=1}^{W} \ell_{w,t}
$$

Seção 3 do experimento (posições vizinhas agrupadas, para reduzir o ruído de só 288 exemplos por
posição):

```text
   posições    contexto   loss  perplexidade
          0           1  2.441         11.48  ████████████████████████
          1           2  2.120          8.33  █████████████████████
          2           3  1.758          5.80  ██████████████████
        3–4         4–5  1.627          5.09  ████████████████
        5–9        6–10  1.519          4.57  ███████████████
      10–19       11–20  1.455          4.29  ███████████████
      20–49       21–50  1.431          4.18  ██████████████
      50–99      51–100  1.418          4.13  ██████████████
    100–127     101–128  1.442          4.23  ██████████████
```

**O que a tabela mostra:**

- **Pouco contexto, muito erro.** Com 1 caractere, a perplexidade é 11,5; com 3, cai para 5,8.
- **A posição 0 é pior que o bigrama** (2,441 contra 2,328), mesmo tendo a mesma informação: um
  caractere. Duas razões prováveis: a posição 0 é só 1/128 das previsões do treino, então recebe pouco
  "treino dedicado"; e o $\alpha$ das réguas foi escolhido olhando a validação, o que as deixa um
  pouco otimistas (10-loss.md, seção 6.3). O mesmo acontece na posição 1 contra o trigrama (2,120 contra
  1,923).
- **Depois de ~20 caracteres, mais contexto ajuda pouco.** De 11–20 para 51–100, a loss só cai de
  1,455 para 1,418. O modelo usa principalmente o contexto próximo: a palavra atual e as anteriores.
  A seção 6 mostra o mesmo pelo lado da attention: nenhuma head olha, em média, mais que ~14
  caracteres para trás.
- **A leve subida em 100–127** (1,442) é pequena demais para concluir algo com 288 janelas.
- **Consequência para a métrica.** Como a validação usa janelas sem sobreposição, 1 em cada 128
  previsões é feita com um único caractere de contexto. A loss de 1,452 é um pouco **pessimista**:
  na geração ([13-generation.md](13-generation.md)), depois dos primeiros caracteres o modelo sempre tem contexto de sobra, e a
  loss das posições com 11 ou mais caracteres de contexto fica em ~1,43.

---

## 4. Onde o modelo erra: o tipo de caractere

### 4.1 As categorias

Cada posição é classificada pelo caractere que o modelo precisa prever (o alvo) e pelo caractere
anterior (a entrada da mesma posição):

| categoria | regra | exemplo (alvo em negrito) |
|---|---|---|
| espaço | o alvo é `' '` | "Capitú**␣**e" |
| quebra de linha | o alvo é `'\n'` | fim de parágrafo |
| pontuação | o alvo é `. , ; : ! ? - ' ( ) « » "` | "Escobar**,**" |
| 1ª letra da palavra | o alvo é letra e o anterior **não** é | "e **E**scobar" |
| letra no meio da palavra | o alvo é letra e o anterior também | "E**s**cobar" |
| outros | o resto (dígitos, símbolos raros) | |

### 4.2 O resultado

Seção 4 do experimento, mais uma coluna calculada a partir dela: a **fatia da loss total**,
$n \times \text{loss}$ de cada categoria dividido pela soma de todas.

```text
  alvo                        posições   loss  perplexidade
  outros                             6  4.399         81.40  ██████████████████████████████████████
  1ª letra da palavra            6,618  2.755         15.71  ████████████████████████
  pontuação                      1,644  2.354         10.52  ████████████████████
  letra no meio da palavra      22,006  1.290          3.63  ███████████
  quebra de linha                  326  0.477          1.61  ████
  espaço                         6,264  0.459          1.58  ████

  caracteres com 100+ ocorrências na validação:
  mais fáceis:  ' ' 0.46  '\n' 0.48  'ã' 0.93  'e' 1.10  'a' 1.17  'o' 1.30
  mais difíceis:'q' 2.44  'g' 2.71  'b' 2.98  'f' 3.22  'E' 3.63  ';' 4.11
```

| categoria | % das posições | fatia da loss total |
|---|---:|---:|
| letra no meio da palavra | 60% | 53% |
| 1ª letra da palavra | 18% | 34% |
| pontuação | 4,5% | 7% |
| espaço | 17% | 5% |
| quebra de linha | 0,9% | 0,3% |
| outros | 0,02% | 0,05% |

**O que isso mostra:**

- **Terminar palavras é fácil; começar é difícil.** No meio de uma palavra, o prefixo restringe muito as
  opções: depois de "Escob", quase só "a" serve (perplexidade 3,6 no geral). Um espaço tem perplexidade
  1,6: o modelo quase sempre sabe quando a palavra acabou.
- **A primeira letra é escolher a próxima palavra.** Perplexidade 15,7: é aí que está o "sentido" do
  texto, e é onde um modelo de 820 mil parâmetros com 128 caracteres de contexto mais sofre. São só
  18% das posições, mas **um terço da loss total**.
- **Pontuação é uma decisão de estilo.** Vírgula, ponto ou ponto e vírgula, e quando: `;` é o caractere
  mais difícil (4,11), porque quase sempre uma vírgula ou um ponto também caberiam.
- **Os mais fáceis são previsíveis pelo contexto imediato.** `'ã'` aparece quase sempre em poucas
  palavras ("não", "mãe", "irmã") e só depois de certas letras.
- **Os mais difíceis são inícios de palavra.** `'q'`, `'g'`, `'b'`, `'f'` aparecem muito no começo das
  palavras; `'E'` maiúsculo começa frases e nomes (Escobar), que o modelo precisa antecipar.
- **"outros" tem só 6 posições:** a loss alta (4,40) não permite conclusão nenhuma.
- **Onde melhorar:** para reduzir a loss de forma relevante, o modelo precisa escolher melhor as
  palavras. Isso pede mais contexto útil e mais capacidade, não ajustes de ortografia.

---

## 5. Calibração: quando o modelo diz 80%, ele acerta 80%?

### 5.1 A ideia

Em cada posição, o modelo dá uma distribuição $p$ sobre os 97 caracteres. Duas grandezas:

- **confiança** $c = \max_i p_i$: a probabilidade da opção favorita;
- **acerto**: se a favorita $\hat y = \arg\max_i p_i$ é o caractere correto $y$.

Um modelo **bem calibrado** acerta, entre todas as vezes em que teve confiança ~0,8, cerca de 80%
delas. Para medir, as posições são separadas em 10 faixas de confiança, e em cada faixa $b$:

$$
\text{conf}(b) = \frac{1}{n_b} \sum_{t \in b} c_t
\qquad
\text{acc}(b) = \frac{1}{n_b} \sum_{t \in b} \mathbb{1}[\hat y_t = y_t]
\qquad
\text{ECE} = \sum_{b} \frac{n_b}{N} \left\lvert \text{acc}(b) - \text{conf}(b) \right\rvert
$$

**Lendo a fórmula:**

- $n_b$ é o número de posições cuja confiança caiu na faixa $b$ (por exemplo, entre 0,6 e 0,7).
- $\text{conf}(b)$ é a confiança média nessa faixa; $\text{acc}(b)$ é a fração de acertos.
- $\mathbb{1}[\hat y_t = y_t]$ ("indicadora") vale 1 se o palpite acertou e 0 se errou.
- $\text{ECE}$ (*expected calibration error*) é a distância média entre acerto e confiança, pesando cada
  faixa pela fração de posições $n_b / N$. Zero é calibração perfeita.
- **Por que isso importa:** a geração (13-generation.md) **sorteia** a partir de $p$. Se o modelo diz 90% e acerta
  90%, sortear com essas probabilidades reproduz a incerteza real. Um modelo mal calibrado sorteia com
  probabilidades distorcidas.

### 5.2 O resultado

Seção 5 do experimento:

```text
    confiança  posições  confiança média  acerto real
  0.0–0.1       105            0.092        0.095
  0.1–0.2     4,694            0.152        0.159
  0.2–0.3     3,232            0.250        0.228
  0.3–0.4     4,015            0.351        0.325
  0.4–0.5     4,095            0.451        0.428
  0.5–0.6     3,955            0.549        0.511
  0.6–0.7     3,383            0.649        0.589
  0.7–0.8     3,183            0.749        0.699
  0.8–0.9     2,783            0.849        0.800
  0.9–1.0     7,419            0.971        0.957
  acurácia geral (o mais provável é o certo): 54.6%
  erro de calibração esperado (ECE): 0.028
```

- **O modelo é bem calibrado, com leve excesso de confiança.** Acerto e confiança andam juntos em
  todas as faixas. Da faixa 0,2 para cima, o acerto fica 2 a 6 pontos abaixo da confiança; a maior
  distância é na faixa 0,6–0,7 (64,9% de confiança, 58,9% de acerto). O ECE é 0,028: em média, a
  confiança erra a taxa real de acerto por ~3 pontos percentuais.
- **Uma explicação provável para o excesso de confiança** é o gap da seção 2: a cross-entropy
  treina o modelo para ser calibrado **no treino**, e na validação ele erra um pouco mais do que
  "espera".
- **20% das posições têm confiança acima de 90%** (7.419), com 95,7% de acerto. São, sobretudo, os
  espaços e os meios de palavra da seção 4.
- **Acurácia de 54,6% com loss 1,452.** O palpite favorito acerta só pouco mais da metade das vezes,
  mas a loss não mede só o favorito: ela premia dar probabilidade alta ao caractere certo mesmo quando
  ele não é o primeiro (10-loss.md, seção 1.3). Quando o modelo erra o palpite, o certo costuma estar
  entre as primeiras opções.

---

## 6. Para onde a attention olha

### 6.1 Extraindo os pesos sem mudar o modelo

`MultiHeadAttention` calcula os pesos $a_{ij}$ internamente e não os devolve. Para vê-los sem mexer no
código do modelo, o experimento usa **forward pre-hooks**: funções que o PyTorch chama logo antes do
`forward` de um módulo, recebendo a entrada dele.

```python
inputs = {}
hooks = [
    block.attention.register_forward_pre_hook(
        lambda module, args, index=index: inputs.__setitem__(index, args[0])
    )
    for index, block in enumerate(model.blocks)
]
model(x)                  # cada hook guarda a entrada da attention do seu bloco
for hook in hooks:
    hook.remove()         # hooks esquecidos continuariam rodando em todo forward
```

Com a entrada `h` de cada bloco, os pesos são recalculados com as mesmas funções de attention e
multi-head attention ([04-attention.md](04-attention.md), [05-multi-head-attention.md](05-multi-head-attention.md)):

```python
q = split_heads(attention.q_proj(h), attention.num_heads)
k = split_heads(attention.k_proj(h), attention.num_heads)
weights = attention_weights(q, k, causal_mask(h.shape[1]))    # [B, h, T, T]
```

Dois detalhes:

- **`index=index`** fixa o valor de `index` no momento em que cada lambda é criada. Sem isso, todas as
  lambdas veriam o último valor do loop (3), e as 4 entradas seriam gravadas no mesmo lugar.
- **A reconstrução é conferida:** com os pesos recalculados, `out_proj(merge_heads(weights @ v))`
  reproduz a saída da attention (o experimento imprime `True`). Sem dropout, não há diferença entre
  os pesos recalculados e os que o modelo usou.

### 6.2 Estatísticas de cada head

Para cada head, sobre um batch de validação, quatro números:

| coluna | o que mede | fórmula |
|---|---|---|
| no anterior | peso médio na posição imediatamente anterior | média de $a_{i,i-1}$ |
| em si | peso médio na própria posição (sem contar $i = 0$, que só pode olhar para si) | média de $a_{i,i}$ |
| no 1º | peso médio no primeiro caractere da janela | média de $a_{i,0}$, $i \ge 1$ |
| distância média | quão para trás a head olha | média de $\bar D_i$ |

$$
\bar D_i = \sum_{j \le i} a_{ij} \, (i - j)
$$

**Lendo a fórmula:** $i - j$ é quantas posições para trás está $j$; cada distância é pesada pela
attention que recebe. Exemplo: uma head que põe 0,9 no anterior (distância 1) e 0,1 num caractere 20
posições atrás tem $\bar D = 0{,}9 \times 1 + 0{,}1 \times 20 = 2{,}9$. **Pouco peso muito longe
pesa bastante na média.**

A linha "uniforme" é a régua: uma attention que distribui o peso igualmente entre as $i + 1$ posições
visíveis. Ela põe $1/(i+1)$ em cada posição (média 0,035) e tem distância média $i/2$ (média 32).

Seção 6 do experimento:

```text
  bloco.head  no anterior   em si   no 1º  distância média
    uniforme        0.035   0.035   0.035             32.0
         0.0        0.087   0.039   0.018              3.9
         0.1        0.973   0.000   0.013              2.7
         0.2        0.191   0.055   0.020              5.2
         0.3        0.944   0.009   0.009              1.6
         1.0        0.079   0.053   0.025             13.8
         1.1        0.240   0.089   0.015              5.8
         1.2        0.251   0.128   0.011              4.8
         1.3        0.194   0.102   0.018              6.5
         2.0        0.135   0.103   0.014              8.0
         2.1        0.226   0.104   0.015              6.3
         2.2        0.212   0.072   0.017              7.1
         2.3        0.172   0.098   0.018              5.7
         3.0        0.124   0.069   0.017             11.8
         3.1        0.168   0.120   0.014              9.6
         3.2        0.133   0.073   0.014             11.6
         3.3        0.110   0.052   0.016             11.3
```

**O que a tabela mostra:**

- **Duas previous-token heads no primeiro bloco.** As heads 0.1 e 0.3 põem 97,3% e 94,4% do peso no
  caractere anterior. É exatamente a head que construímos à mão em [04-attention.md](04-attention.md)
  e que [08-gpt.md](08-gpt.md) (seção 5) previa como o primeiro passo de um circuito de induction
  head. **Ninguém programou isso:** o
  gradiente descobriu que copiar o caractere anterior é útil.
- **Por que é útil.** Depois do bloco 0, cada posição carrega "qual é o meu caractere" (embedding) e
  "qual veio antes" (copiado pela head). Os blocos seguintes podem combinar isso para enxergar
  sequências mais longas.
- **A head 0.1 tem distância 2,7, e não ~1.** Os 2,7% que sobram vão para longe: só o primeiro
  caractere (1,3% do peso, a ~64 posições em média) já soma ~0,8 à distância. É o efeito "pouco peso
  muito longe" da fórmula.
- **Os blocos mais altos olham mais longe.** No bloco 0, as distâncias vão de 1,6 a 5,2; no bloco 3,
  de 9,6 a 11,8. As camadas superiores juntam contexto mais amplo, construído sobre o que as de baixo
  já montaram.
- **Mas ninguém olha muito longe.** A maior distância média (head 1.0, 13,8) é bem menor que a da
  attention uniforme (32). Faz sentido com a seção 3.2: depois de ~20 caracteres, mais contexto quase
  não ajuda este modelo.
- **Nenhuma head se concentra no primeiro caractere** ("no 1º" entre 0,009 e 0,025, abaixo do
  uniforme). Em modelos maiores, é comum algumas heads usarem a primeira posição como um "depósito"
  de attention quando não precisam de nada; aqui isso não aparece.

### 6.3 Mapas de attention

Os mapas mostram $a_{ij}$ para um texto curto. Cada linha é uma posição que olha; cada coluna, uma
posição para onde ela pode olhar. `·` marca as posições bloqueadas pela causal mask.

```text
  head 0.1 (mais focada no anterior), em 'Capitú e Escobar'; linha = quem olha, coluna = para onde
  █ ≥ 0,75   ▓ ≥ 0,5   ▒ ≥ 0,25   ░ ≥ 0,1

        C a p i t ú   e   E s c o b a r
    'C' █ · · · · · · · · · · · · · · ·
    'a' █   · · · · · · · · · · · · · ·
    'p'   █   · · · · · · · · · · · · ·
    'i'     █   · · · · · · · · · · · ·
    't'       █   · · · · · · · · · · ·
    'ú'         █   · · · · · · · · · ·
    ' '           █   · · · · · · · · ·
    'e'             █   · · · · · · · ·
    ' '               █   · · · · · · ·
    'E'                 █   · · · · · ·
    's'                   █   · · · · ·
    'c'                     █   · · · ·
    'o'                       █   · · ·
    'b'                         █   · ·
    'a'                           █   ·
    'r'                             █
```

A head 0.1 desenha uma **diagonal perfeita logo abaixo da principal**: cada caractere olha para o
anterior. A primeira linha (`'C'`) olha para si porque não tem outra opção.

```text
  head 1.0 (que olha mais longe), em 'Capitú e Escobar'; linha = quem olha, coluna = para onde
  █ ≥ 0,75   ▓ ≥ 0,5   ▒ ≥ 0,25   ░ ≥ 0,1

        C a p i t ú   e   E s c o b a r
    'C' █ · · · · · · · · · · · · · · ·
    'a' ▒ ▓ · · · · · · · · · · · · · ·
    'p' ▒ ▒ ▒ · · · · · · · · · · · · ·
    'i' ░ ▒ ▒ ░ · · · · · · · · · · · ·
    't' ░ ▒ ░ ░ ░ · · · · · · · · · · ·
    'ú' ░ ░ ▒ ░     · · · · · · · · · ·
    ' '   ▒ ▒ ░       · · · · · · · · ·
    'e'   ▒ ░   ░       · · · · · · · ·
    ' '   ░ ░ ░ ░     ░   · · · · · · ·
    'E'   ░ ░ ░ ░         ░ · · · · · ·
    's'       ░ ░         ░   · · · · ·
    'c'   ░   ░ ░         ░     · · · ·
    'o'       ░           ░     ░ · · ·
    'b'                   ░     ▒   · ·
    'a'                   ░     ░   ░ ·
    'r'                   ░     ░   ░
```

A head 1.0 **espalha** o peso por vários caracteres anteriores, sem nenhum dominante. Dentro de
"Escobar", todas as linhas dão algum peso ao `'E'` inicial; dentro de "Capitú", as primeiras letras
recebem mais peso. Isso sugere uma head que acompanha **o começo da palavra atual**, como
05-multi-head-attention.md imaginava ("uma head rastreia o caractere anterior, outra o início da
palavra"). Mas é um único
exemplo: confirmar exigiria medir o padrão em muitas palavras.

**Um cuidado com interpretação.** As estatísticas são médias, e o que uma head faz depende do
conteúdo. Heads "limpas" como a 0.1 são fáceis de nomear; a maioria (como as dos blocos 2 e 3) mistura
vários comportamentos e não tem um papel único.

---

## 7. Memória do treino

### 7.1 O que ocupa memória

Um passo de treino guarda quatro tipos de tensores:

| parte | por que existe | tamanho |
|---|---|---|
| **parâmetros** | os pesos do modelo | $n$ números |
| **gradientes** | `param.grad`, um por parâmetro (ver [11-training.md](11-training.md)) | $n$ números |
| **estado do AdamW** | as médias $m$ e $v$ de cada parâmetro (11-training.md) | $2n$ números |
| **ativações** | tensores do forward que o backward precisa | depende de $B$, $T$, $d$, $h$ |

Cada número em `float32` ocupa 4 bytes. Com $n = 820.096$:

$$
\text{memória} = n \times 4 \text{ bytes} = 3.280.384 \text{ bytes} = 3{,}1 \text{ MB}
$$

**Lendo a fórmula:** $n$ parâmetros × 4 bytes cada. Os gradientes ocupam o mesmo; o AdamW, o dobro.

**Por que o backward precisa das ativações.** A regra da cadeia (11-training.md) usa valores do forward. Uma
`Linear` $y = Wx$ precisa da entrada $x$ para calcular o gradiente de $W$ ($\partial \mathcal{L} /
\partial W = \delta \, x^\top$, onde $\delta$ é o gradiente que chega da camada de cima). O softmax
precisa da própria saída. Então o autograd **guarda** esses tensores durante o forward, e só os
libera depois do backward.

**Medindo.** `torch.autograd.graph.saved_tensors_hooks` chama uma função (`pack`) para cada tensor
que o autograd guarda. O experimento soma o tamanho de cada um, contando uma vez os tensores que
compartilham memória e ignorando os próprios parâmetros:

```python
def pack(tensor):
    # Chamado para cada tensor que o autograd guarda. Views do mesmo storage contam uma vez.
    pointer = tensor.untyped_storage().data_ptr()
    if pointer not in seen and pointer not in parameter_storages:
        seen.add(pointer)
        shape = tuple(tensor.shape)
        sizes[shape] = sizes.get(shape, 0) + tensor.untyped_storage().nbytes()
    return tensor

with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
    loss = cross_entropy(model(x), y)
```

### 7.2 O resultado

Seção 7 do experimento, com um batch de treino ($B = 32$, $T = 128$):

```text
  parâmetros (float32)                                            3.1 MB
  gradientes (um por parâmetro)                                   3.1 MB
  AdamW: m e v (dois por parâmetro)                               6.3 MB  █
  ativações [B, T, d]: entradas de LayerNorm, Linear, dropout    98.0 MB  ██████████
  ativações [B, h, T, T]: pesos de attention                     96.0 MB  ██████████
  ativações [B, T, 4d]: camada escondida do FFN                  64.0 MB  ██████
  ativações [B, h, T, d_h]: Q, K e V por head                    24.0 MB  ██
  ativações [B, T, V]: logits e log-probabilidades                3.0 MB
  ativações: o resto (desvios da LayerNorm, IDs...)               0.3 MB
  total das ativações                                           285.3 MB
  total do modelo e do treino                                   297.8 MB

  ativações do mesmo batch sem dropout (model.eval()): 203.3 MB
  uma matriz de pesos de attention [B, h, T, T]: 8.0 MB
  pico de memória do processo no treino (history.json): 598 MB
```

De onde vem cada linha (tamanho de um tensor = produto do shape × 4 bytes):

| família | um tensor | quantos | total |
|---|---|---:|---:|
| $[B, T, d]$ | $32 \times 128 \times 128 \times 4$ = 2,0 MB | 49 | 98 MB |
| $[B, h, T, T]$ | $32 \times 4 \times 128 \times 128 \times 4$ = 8,0 MB | 12 (3 por bloco) | 96 MB |
| $[B, T, 4d]$ | $32 \times 128 \times 512 \times 4$ = 8,0 MB | 8 (2 por bloco) | 64 MB |
| $[B \cdot h, T, d_h]$ | $128 \times 128 \times 32 \times 4$ = 2,0 MB | 12 (Q, K e V por bloco) | 24 MB |
| $[B, T, V]$ | $32 \times 128 \times 97 \times 4$ = 1,5 MB | 2 | 3 MB |

(Algumas famílias aparecem em mais de um formato, como $[B \cdot T, d]$ em vez de $[B, T, d]$: o
PyTorch achata as dimensões de lote em certas operações e guarda a versão achatada. Como aqui
$d = T = 128$, o experimento classifica cada tensor pelo shape **completo**; olhar só as últimas
dimensões confundiria $[B, T, d]$ com uma matriz $T \times T$.)

**O que os números mostram:**

- **As ativações são 96% da memória do treino** (285,3 de 297,8 MB). Os pesos do modelo, com gradientes
  e AdamW, são só 12,5 MB. Num modelo pequeno com batch razoável, o gargalo é o que se guarda para o
  backward, não o modelo.
- **A attention guarda 3 matrizes $T \times T$ por bloco** (8 MB cada), e o dropout é o responsável
  por duas delas. Sem dropout (`model.eval()`), as ativações caem para 203,3 MB. A diferença, 82 MB,
  fecha exatamente:
  - na attention, de 3 para 1 matriz por bloco: $4 \times 2 \times 8 = 64$ MB. Com dropout, ficam
    guardados a saída do softmax, a máscara sorteada e o resultado depois do dropout; sem dropout,
    basta a saída do softmax;
  - 9 máscaras $[B, T, d]$ dos outros dropouts (1 depois dos embeddings e 2 por bloco):
    $9 \times 2 = 18$ MB.

  O dropout custa memória, e não só regularização.

### 7.3 Como a memória cresce

Quase todas as ativações são proporcionais a $B \cdot T$; as da attention, a $B \cdot h \cdot T^2$:

| mudança | ativações lineares em $T$ (189 MB) | attention (96 MB) | total aproximado |
|---|---:|---:|---:|
| nenhuma | 189 MB | 96 MB | 285 MB |
| batch 64 ($B \times 2$) | 379 MB | 192 MB | 570 MB |
| contexto 512 ($T \times 4$) | 757 MB | 1.536 MB | ~2,3 GB |
| contexto 1024 ($T \times 8$) | 1.514 MB | 6.144 MB | ~7,7 GB |

**Lendo a tabela:** dobrar o batch dobra tudo. Multiplicar o contexto por 4 multiplica as ativações
comuns por 4, mas as da attention por $4^2 = 16$. Com contexto 128, a attention é um terço das
ativações; com 1024, seria **80%**. (São estimativas por proporção, não medições.)

É por isso que implementações de attention otimizadas, que evitam guardar a matriz $T \times T$
inteira (como o kernel SDPA mencionado em [05-multi-head-attention.md](05-multi-head-attention.md)),
importam tanto em modelos de contexto longo. Isso é assunto de [17-performance.md](17-performance.md).

### 7.4 Por que o processo usa 598 MB

A conta dá 297,8 MB, e o pico do processo foi 598 MB. A diferença não é erro:

- o **interpretador Python e as bibliotecas do PyTorch** carregadas ocupam uma parte fixa, de centenas
  de MB, antes de qualquer tensor ser criado;
- o **backward cria tensores temporários**: o gradiente de cada ativação existe por um instante enquanto
  a camada correspondente é processada;
- o **alocador do PyTorch** reserva memória em blocos e nem sempre a devolve ao sistema;
- os **dados** (o corpus tokenizado, com treino e validação) e a avaliação também ocupam memória.

O RSS mede tudo isso junto. A conta por componente mostra o que **cresce** com o modelo e com o
batch; o RSS mostra o que o computador realmente precisa ter livre.

---

## 8. Rodando a avaliação

```bash
uv run python train.py                            # treina, salva checkpoints/latest.pt e history.json
uv run python -m experiments.e14_evaluation       # as 7 seções deste documento
```

Ao fim, o `train.py` também imprime o gap:

```text
loss de treino sem dropout: 1.296 | validação − treino: +0.156
```

O experimento é determinístico, com exceção dos tempos, da velocidade e do pico de memória
registrados no histórico: rodar de novo com o mesmo checkpoint dá os mesmos números. Retreinar também
dá as mesmas losses, mas não os mesmos tempos: no retreino feito em [15-checkpoints.md](15-checkpoints.md),
o treino levou 498 s e o pico foi de 577 MB.

---

## 9. O que cada teste garante

[`tests/test_metrics.py`](../tests/test_metrics.py):

| teste | propriedade |
|---|---|
| `test_perplexity_of_a_uniform_guess_is_the_number_of_options` | loss $\ln 97$ → perplexidade 97 |
| `test_zero_loss_has_perplexity_one` | loss 0 → perplexidade 1 (certeza total) |
| `test_bits_per_token_converts_nats_to_bits` | $\ln 2$ nats = 1 bit; $\ln 97$ nats = $\log_2 97$ bits |
| `test_history_round_trip` | gravar e ler `history.json` devolve os mesmos campos |
| `test_peak_memory_is_measured` | o pico de memória é um número positivo (unidade certa no macOS e no Linux) |

[`tests/test_loss.py`](../tests/test_loss.py):

| teste | propriedade |
|---|---|
| `test_token_losses_keep_one_value_per_position_and_average_to_cross_entropy` | `token_losses` tem o shape dos alvos e sua média é a `cross_entropy` |

[`tests/test_trainer.py`](../tests/test_trainer.py):

| teste | propriedade |
|---|---|
| `test_train_runs_epochs_and_learns` | agora também confere `train_eval_loss`, o lr do último passo, a norma do gradiente, a velocidade e a memória |
| `test_train_without_train_eval_loader_skips_that_measurement` | sem `train_eval_loader`, o registro traz `train_eval_loss = None` |

---

## Resumo

1. Loss, perplexidade e bits são o mesmo número em escalas diferentes. O modelo tem loss **1,452**,
   hesita entre **4,27** opções e usaria **2,095 bits** por caractere; o trigrama fica em 1,923 / 6,84 /
   2,774.
2. A loss de treino do log (com dropout e com pesos mudando) não é comparável à de validação. Medida
   sem dropout, ela fica abaixo da validação **em todas as épocas**, com o gap crescendo de +0,008 para
   +0,156. O "cruzamento das curvas" visto no treino ([11-training.md](11-training.md)) era um efeito
   do dropout.
3. Ainda não há overfitting: a validação cai até a última época. O gap crescente é o aviso de que mais
   épocas neste corpus logo deixarão de ajudar.
4. O modelo erra muito com pouco contexto (perplexidade 11,5 com 1 caractere) e quase para de
   melhorar depois de ~20 caracteres.
5. Terminar palavras é fácil (espaço: perplexidade 1,6); começar é difícil (1ª letra: 15,7, um terço da
   loss total). O que falta ao modelo é escolher palavras.
6. O modelo é bem calibrado (ECE 0,028), com leve excesso de confiança, e acerta o favorito em 54,6%
   das posições.
7. O gradiente criou sozinho duas **previous-token heads** no primeiro bloco (97,3% e 94,4% no
   caractere anterior); os blocos de cima olham mais longe, mas nenhum passa de ~14 caracteres em
   média.
8. As ativações ocupam 96% da memória do treino (285 de 298 MB). A attention cresce com $T^2$, e o
   dropout custa 82 MB.

---

## Para checar o entendimento

1. Um modelo tem loss 1,0 nat. Qual é a perplexidade? E quantos bits por caractere?
2. Um modelo hesita, em média, entre 8 opções igualmente prováveis. Qual é a loss em nats e em bits?
3. Na época 1, a loss de treino com dropout foi 3,219 e a sem dropout, 2,465. Quais são os dois motivos
   dessa diferença? Qual deles praticamente desaparece no fim do treino?
4. O gap cresce de +0,008 para +0,156, e a validação continua caindo. Isso é overfitting? O que
   precisaria acontecer para ser?
5. Por que a posição 0 da janela tem loss pior que a do bigrama, se as duas usam um caractere de
   contexto?
6. A validação usa janelas sem sobreposição. Por que isso deixa a loss de 1,452 um pouco pessimista?
   Como você mediria a loss com contexto sempre cheio?
7. Por que prever a primeira letra de uma palavra é tão mais difícil que prever um espaço?
8. Na faixa de confiança 0,6–0,7, o modelo acertou 58,9%. Ele está confiante demais ou de menos? Por
   que isso importa para a geração?
9. A acurácia é só 54,6%. Como a loss pode ser 1,452, e não muito pior?
10. A head 0.1 põe 97,3% do peso no caractere anterior, mas tem distância média 2,7. Por quê?
11. Para que serve uma previous-token head no primeiro bloco? O que os blocos seguintes podem fazer com
    a informação que ela copia?
12. No código dos hooks, o que aconteceria sem `index=index` na lambda?
13. Por que desligar o dropout reduz as ativações em 82 MB? De onde vêm os 64 MB da attention?
14. Estime a memória das ativações com `context_length = 256`, mantendo o resto. Qual parte cresce mais
    rápido, e por quê?
15. O pico do processo foi 598 MB, e a conta por componente dá 297,8 MB. De onde vem a diferença?
