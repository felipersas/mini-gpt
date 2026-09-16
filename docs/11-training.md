# Training Loop

> **Como o erro altera os pesos?**

Código: [`training/trainer.py`](../training/trainer.py) e [`train.py`](../train.py). Testes:
[`tests/test_trainer.py`](../tests/test_trainer.py). Experimento:
`uv run python -m experiments.e11_training` (~30 s; a comparação longa, com `--comparar`, leva
~15 min). Notação geral: [00-notacao.md](00-notacao.md). Visão geral:
[architecture.md](architecture.md). Pré-requisito: [10-loss.md](10-loss.md).

---

## Para que serve

### O problema

Até aqui ([10-loss.md](10-loss.md)), o modelo sabe **calcular** o quanto erra: a loss. E o PyTorch sabe calcular o
**gradiente** da loss em relação a cada um dos 820.096 parâmetros. Mas nada disso muda os pesos.
O modelo continua tão ignorante quanto na inicialização (loss ≈ 4,57).

O training loop é a repetição, milhares de vezes, de um ciclo curto:

1. pegar um batch de textos;
2. calcular as previsões e a loss;
3. calcular o gradiente;
4. mexer um pouquinho em cada parâmetro na direção que diminui a loss.

Cada repetição é um **passo**. Depois de alguns milhares de passos, os parâmetros deixam de ser
números aleatórios e passam a codificar regularidades do texto.

Termos usados daqui em diante:

- **Passo** (*step*): uma atualização dos pesos, feita com um batch.
- **Época** (*epoch*): uma passada completa pelos batches do treino. No `tiny.yaml`, 82 passos.
- **Backpropagation:** o algoritmo que calcula o gradiente de todos os parâmetros de uma vez,
  aplicando a regra da cadeia de trás para frente (seção 2).
- **Otimizador:** a regra que transforma o gradiente numa mudança de pesos. Usamos o **AdamW**.
- **Learning rate** ($\eta$, taxa de aprendizado): o tamanho do passo.
- **Weight decay:** um pequeno encolhimento dos pesos a cada passo, para evitar que cresçam sem
  necessidade.
- **Gradient clipping:** um limite para o tamanho do gradiente, contra passos gigantes.
- **Warmup:** começar com learning rate pequeno e aumentar aos poucos.
- **Underfitting:** o modelo ainda não aprendeu o que poderia (a loss de validação ainda cai).
  **Overfitting:** o modelo decora o treino e piora na validação.
- **Tokens por segundo:** quantas previsões o treino processa por segundo; mede a velocidade.

### Uma analogia

Imagine descer uma montanha no meio de uma neblina densa, querendo chegar ao vale:

- você não vê o caminho, mas **sente a inclinação do chão** sob os pés: é o gradiente;
- a cada passo, anda **morro abaixo**: é a descida do gradiente;
- o **tamanho do passo** é o learning rate: passos grandes descem rápido, mas podem pular o
  vale; passos pequenos são seguros, mas lentos;
- o **AdamW** é uma bengala que lembra a direção dos últimos passos e ajusta o tamanho do passo
  para cada direção;
- o **clipping** é a regra "nunca dar um passo maior que um metro", útil quando o chão engana;
- o **warmup** é começar devagar até os olhos se acostumarem à neblina.

A "montanha" tem 820.096 dimensões, uma por parâmetro, e a altura em cada ponto é a loss.

### O que entra e o que sai

| | o quê | no `tiny.yaml` |
|---|---|---|
| entra | o modelo inicializado | 820.096 parâmetros aleatórios, loss de validação 4,603 |
| entra | os batches de treino e de validação | 82 batches `[32, 128]` por época; 9 de validação |
| entra | os hiperparâmetros | lr 0,001, warmup 100, 30 épocas, weight decay 0,01, clip 1,0 |
| sai | o modelo treinado | os mesmos 820.096 parâmetros, agora ajustados |
| sai | o histórico | loss de treino e de validação ao fim de cada época |

### Onde isso entra no pipeline

```text
texto: "Uma noite destas, vindo da cidade..."
  │
  ▼  tokenizer (01-tokenizer.md)                str → IDs [374.785]
  ▼  dataset e batches (02-dataset.md)          IDs → x, y: [32, 128]
  ▼  embeddings (03-embeddings.md)              [32, 128] → [32, 128, 128]
  ▼  blocos Transformer × 4 (04 a 08)           [32, 128, 128] → [32, 128, 128]
  ▼  LM head (09-lm-head.md)                    [32, 128, 128] → logits [32, 128, 97]
  ▼  loss (10-loss.md)                          logits e y → um número
  ▼  backward + otimizador (este documento)     o número → ajuste dos 820.096 parâmetros
  │
  └──► próximo batch (repete 2.460 vezes)
```

### O que daria errado sem esta parte

- **Sem o loop,** o modelo nunca aprenderia: a loss ficaria em ~4,57.
- **Sem `zero_grad`,** os gradientes de todos os passos se somariam, e os passos cresceriam sem
  controle (seção 3).
- **Com learning rate grande demais,** a loss oscila ou explode; **pequeno demais,** o treino
  leva uma eternidade (seção 8).
- **Sem clipping,** um único batch com gradiente enorme (vimos um de norma 64,9) pode desfazer em
  um passo o que centenas de passos construíram (seção 6).
- **Sem avaliar na validação,** não daria para distinguir aprendizado de memorização.

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $\theta$ | "teta" | um parâmetro (ou todos eles, vistos como um vetor) | 820.096 números | `model.parameters()` |
| $t$ | "tê" | número do **passo** de treino (aqui **não** é posição na sequência) | de 1 a 2.460 | `step` |
| $\mathcal{L}$ | "ele caligráfico" | a loss do batch | ~4,57 no início | `loss` |
| $g$, $g_t$ | "gê", "gê tê" | gradiente $\partial \mathcal{L} / \partial \theta$ no passo $t$ | mesmo shape de cada parâmetro | `param.grad` |
| $\|g\|$ | "norma de gê" | tamanho de **todos** os gradientes juntos, como um único vetor | 5,44 no primeiro batch | `clip_grad_norm_(...)` |
| $c$ | "cê" | limite do gradient clipping | 1,0 | `gradient_clip` |
| $\eta$, $\eta_t$ | "eta", "eta tê" | learning rate (no passo $t$) | 0,001 | `lr`, `learning_rate` |
| $\eta_{\max}$, $\eta_{\min}$ | "eta máx", "eta mín" | learning rate no fim do warmup e no fim do treino | 0,001 e 0,001 | `max_lr`, `min_lr` |
| $t_w$ | "tê dáblio" | número de passos de warmup | 100 | `warmup_steps` |
| $t_{\text{total}}$ | "tê total" | número total de passos: épocas × batches por época | 30 × 82 = 2.460 | `total_steps` |
| $\lambda$ | "lambda" | weight decay | 0,01 | `weight_decay` |
| $m_t$ | "eme tê" | primeiro momento do Adam: média móvel dos gradientes | shape do parâmetro | estado do otimizador |
| $v_t$ | "vê tê" | segundo momento do Adam: média móvel dos gradientes ao quadrado. **Não** é o `vocab_size` | shape do parâmetro | estado do otimizador |
| $\hat m_t$, $\hat v_t$ | "eme chapéu", "vê chapéu" | $m_t$ e $v_t$ corrigidos do viés inicial | | |
| $\beta_1$, $\beta_2$ | "beta um", "beta dois" | quanto de memória cada média móvel tem. **Não** é o $\beta$ da LayerNorm | 0,9 e 0,999 | `betas` (padrão do AdamW) |
| $\epsilon$ | "épsilon" | número minúsculo para evitar divisão por zero | $10^{-8}$ | `eps` (padrão do AdamW) |
| $g^2$, $\sqrt{v}$ | | quadrado e raiz **elemento a elemento** (cada peso separado) | | `g * g`, `v.sqrt()` |
| $\text{progresso}$ | | fração do decaimento já percorrida, de 0 a 1 | | `progress` |
| $\pi$ | "pi" | 3,14159… | | `math.pi` |

---

## 1. O ciclo de um passo

### 1.1 As cinco operações

A função `train_step` faz exatamente isto:

```python
def train_step(model, x, y, optimizer, gradient_clip):
    model.train()                                   # dropout ligado
    logits = model(x)                               # 1. forward
    loss = cross_entropy(logits, y)                 # 2. loss
    optimizer.zero_grad(set_to_none=True)           # 3. limpa os gradientes antigos
    loss.backward()                                 # 4. backward: calcula os gradientes
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)   # 5. clipping
    optimizer.step()                                # 6. atualiza os pesos
    return StepResult(loss=loss.item(), grad_norm=grad_norm.item())
```

| linha | objetivo | o que faz |
|---|---|---|
| `model.train()` | colocar o modelo em modo de treino | liga o dropout (03 a 07) |
| `model(x)` | prever | forward completo: `[32, 128]` → logits `[32, 128, 97]` |
| `cross_entropy(logits, y)` | medir o erro | um escalar ([10-loss.md](10-loss.md)) |
| `zero_grad(set_to_none=True)` | começar limpo | apaga o `.grad` de cada parâmetro (seção 3) |
| `loss.backward()` | descobrir como cada peso afeta o erro | preenche `param.grad` de todos os parâmetros (seção 2) |
| `clip_grad_norm_` | evitar passos gigantes | se $\|g\| > c$, encolhe todos os gradientes (seção 6) |
| `optimizer.step()` | aprender | muda cada parâmetro usando o seu gradiente (seção 4) |
| `.item()` | registrar | converte os tensores em números Python para o log |

### 1.2 Um passo de verdade

Seção 1 do experimento, com o GPT recém-inicializado, um batch real e lr = 0,001. Acompanhamos
os 4 primeiros números da primeira linha de `q_proj.weight` do bloco 0:

```text
1. forward:   ids (32, 128) -> logits (32, 128, 97)
2. loss:      4.5629
   q_proj.grad antes do backward: None
3. backward:  q_proj.grad (128, 128); w.grad[0, :4] = +2.22e-04 -1.82e-05 +5.63e-05 +4.81e-05
4. clipping:  norma de todos os gradientes = 5.443 (limite 1,0)
5. step:      w[0, :4] antes   +0.01971 -0.01986 +0.00441 +0.00795
              w[0, :4] depois  +0.01871 -0.01886 +0.00341 +0.00695
              diferença        -1.00e-03 +9.97e-04 -9.99e-04 -9.99e-04

loss no mesmo batch: 4.5629 antes do passo, 4.0566 depois
```

- **Antes do backward, `.grad` é `None`:** o gradiente não existe até ser calculado.
- **O backward preenche `.grad` com o mesmo shape do peso,** `(128, 128)`: um número para cada
  peso.
- **A norma total dos gradientes é 5,443,** acima do limite 1,0: o clipping entra em ação.
- **Cada peso andou quase exatamente 0,001,** que é o learning rate. Os sinais são **opostos** aos
  do gradiente: $+2{,}22 \times 10^{-4}$ virou $-0{,}001$. A seção 4 explica por que o tamanho é
  sempre ≈ lr no primeiro passo do AdamW.
- **Um único passo derrubou a loss do batch de 4,563 para 4,057.** Nos próximos passos, cada
  batch é diferente e a queda fica bem menor.

---

## 2. Backpropagation: o que o `backward` faz

### 2.1 O grafo computacional

Durante o forward, o PyTorch **grava** cada operação feita com tensores que exigem gradiente
(como os parâmetros): multiplicações de matrizes, somas, softmax, LayerNorm. Essa gravação é o
**grafo computacional**: um mapa de "qual tensor foi calculado a partir de quais".

O `loss.backward()` percorre esse grafo **de trás para frente**, da loss até os parâmetros,
aplicando a **regra da cadeia** ([00-notacao.md](00-notacao.md), seção 7.3) em cada operação.

### 2.2 A regra da cadeia numa cadeia de operações

Para um peso $w$ no primeiro bloco, a loss depende dele através de uma longa cadeia: $w$ afeta a
saída do bloco 1, que afeta o bloco 2, …, que afeta os logits, que afetam a loss. Com
$h_1, h_2, \dots$ representando os valores intermediários:

$$
\frac{\partial \mathcal{L}}{\partial w}
= \frac{\partial \mathcal{L}}{\partial \text{logits}}
\cdot \frac{\partial \text{logits}}{\partial h_4}
\cdot \frac{\partial h_4}{\partial h_3}
\cdots
\frac{\partial h_1}{\partial w}
$$

**Lendo a fórmula:**

- cada fração é a derivada de **uma** operação: "quanto a saída desta etapa muda se a entrada
  mudar um pouco";
- $\frac{\partial \mathcal{L}}{\partial \text{logits}}$: o primeiro fator, já calculado em
  [10-loss.md](10-loss.md): $(p - \text{one\_hot}(y))/N$;
- o produto encadeia todas as etapas, da loss até $w$.

**Por que "de trás para frente":** o fator $\frac{\partial \mathcal{L}}{\partial h_4}$ serve para
**todos** os parâmetros que vêm antes de $h_4$. Calculando da loss para trás, cada produto parcial
é calculado uma única vez e reaproveitado. Por isso um backward custa só cerca de 2 vezes um
forward, mesmo produzindo 820.096 derivadas.

### 2.3 Como o erro altera os pesos

Juntando tudo, a resposta à pergunta deste documento:

1. o **forward** produz uma previsão, e a **loss** mede o erro ([10-loss.md](10-loss.md));
2. o **backward** distribui a "culpa" pelo erro entre os 820.096 parâmetros: $g = \partial
   \mathcal{L}/\partial \theta$ diz, para cada peso, em que direção e com que intensidade mexer
   nele aumentaria a loss;
3. o **otimizador** move cada peso **no sentido contrário**, um pouquinho;
4. no próximo batch, as previsões ficam um pouco melhores, e o ciclo recomeça.

Ninguém programa "se vier `'q'`, prever `'u'`". Essa regra aparece sozinha, porque cada passo
empurra os pesos na direção que torna as previsões corretas mais prováveis.

---

## 3. Por que `zero_grad`

O `backward` **soma** os gradientes novos ao que já está em `.grad`. Isso é útil em técnicas
como *gradient accumulation* ([17-performance.md](17-performance.md)), mas no treino normal
significaria somar os gradientes de
todos os passos anteriores. Seção 2 do experimento, com o **mesmo** batch três vezes e sem zerar:

```text
  backward nº 1: norma de q_proj.grad = 4.9665e-03
  backward nº 2: norma de q_proj.grad = 9.9331e-03
  backward nº 3: norma de q_proj.grad = 1.4900e-02
  cada backward SOMA ao que já está em .grad; zero_grad limpa antes do próximo passo
```

A norma dobra e triplica. Por isso `train_step` chama `optimizer.zero_grad(set_to_none=True)`
antes de cada `backward`. O `set_to_none=True` apaga o tensor em vez de preenchê-lo com zeros:
economiza memória e deixa claro que não existe gradiente até o próximo backward. O teste
`test_gradients_do_not_accumulate_between_steps` confere que dois passos seguidos, no mesmo
batch, deixam o mesmo gradiente.

---

## 4. O otimizador: de SGD a AdamW

### 4.1 SGD: descida do gradiente pura

A regra mais simples (*stochastic gradient descent*):

$$
\theta_t = \theta_{t-1} - \eta\, g_t
$$

**Lendo a fórmula:**

- $\theta_{t-1}$: o valor do parâmetro antes do passo $t$.
- $g_t$: o gradiente calculado no batch do passo $t$.
- $\eta\, g_t$: o passo, proporcional ao gradiente.
- o sinal de menos: andar **contra** o gradiente, onde a loss diminui.
- "estocástico": o gradiente vem de um batch sorteado, não do corpus inteiro, então é uma
  estimativa com ruído.

**O problema:** o tamanho do passo é proporcional ao tamanho do gradiente, e os gradientes do
modelo variam enormemente entre parâmetros (seção 4.3).

### 4.2 Adam e AdamW

O Adam (Kingma & Ba, 2015) mantém, **para cada peso**, duas médias móveis:

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2
\end{aligned}
$$

**Lendo a fórmula:**

- $m_t$ (**primeiro momento**): uma média dos gradientes recentes. Com $\beta_1 = 0{,}9$, cada passo
  guarda 90% da média antiga e mistura 10% do gradiente novo. Suaviza o ruído dos batches, como
  o **momento** de uma bola rolando ladeira abaixo.
- $v_t$ (**segundo momento**): uma média dos **quadrados** dos gradientes. Mede o tamanho típico
  do gradiente daquele peso. Com $\beta_2 = 0{,}999$, a memória é longa (~1.000 passos).
- $g_t^2$: o quadrado de cada gradiente, peso por peso.

Como $m_0 = v_0 = 0$, as médias começam puxadas para zero. A **correção de viés** compensa isso:

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

**Lendo a fórmula:**

- $\beta_1^t$: $\beta_1$ elevado ao número do passo (aqui o $t$ em cima **é** expoente). No passo 1,
  $1 - 0{,}9^1 = 0{,}1$, e dividir por 0,1 desfaz o fator 10% da média.
- com $t$ grande, $\beta^t \to 0$ e a correção deixa de ter efeito.

A atualização do **AdamW** (Loshchilov & Hutter, 2019):

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_{t-1} \right)
$$

**Lendo a fórmula:**

- $\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$: a direção média do gradiente **dividida pelo tamanho
  típico** do gradiente daquele peso. É uma normalização por peso: pesos com gradientes minúsculos e
  pesos com gradientes grandes andam passos de tamanho parecido.
- $\epsilon = 10^{-8}$: evita divisão por zero.
- $\lambda\, \theta_{t-1}$: o **weight decay** (seção 5). O "W" de AdamW indica que ele é aplicado
  **separado** da parte adaptativa ("decoupled"). No Adam original, o decay era somado ao gradiente
  e acabava também normalizado, o que o enfraquecia de forma desigual.
- $\eta$: o learning rate multiplica tudo.

### 4.3 O primeiro passo: todo peso anda ≈ $\eta$

No passo 1, $\hat m_1 = g_1$ e $\hat v_1 = g_1^2$. Então

$$
\frac{\hat m_1}{\sqrt{\hat v_1} + \epsilon} = \frac{g_1}{|g_1| + \epsilon} \approx \pm 1
$$

**Lendo a fórmula:**

- $|g_1|$: o valor absoluto do gradiente (sem sinal).
- $g_1 / |g_1|$ é $+1$ ou $-1$, conforme o sinal: o tamanho do gradiente **some**.
- logo, no primeiro passo, **cada peso anda exatamente $\eta$** (sem o decay), no sentido contrário
  ao do gradiente.

Seção 3 do experimento, comparando com o SGD (lr = 0,001, sem weight decay):

```text
  parâmetro                               |grad| médio    |Δ| SGD  |Δ| AdamW
  embeddings.token_embedding.weight           6.08e-03   6.08e-06   1.00e-03
  embeddings.position_embedding.weight        2.21e-03   2.21e-06   1.00e-03
  blocks.0.attention.q_proj.weight            2.76e-05   2.76e-08   9.97e-04
  blocks.3.feed_forward.down_proj.bias        7.62e-02   7.62e-05   1.00e-03
  ln_final.weight                             1.34e-03   1.34e-06   1.00e-03
  SGD anda lr · |grad|; o primeiro passo do AdamW anda ≈ lr em todo peso com gradiente
```

- **Os gradientes variam por um fator de ~3.000** entre parâmetros: de $2{,}8 \times 10^{-5}$ em
  `q_proj` a $7{,}6 \times 10^{-2}$ num bias do feed-forward.
- **Com SGD,** `q_proj` andaria $2{,}8 \times 10^{-8}$ por passo: precisaria de milhões de passos para
  sair do lugar. Aumentar o lr para ajudar `q_proj` faria o bias explodir.
- **Com AdamW,** todos andam ~$10^{-3}$. É por isso que o AdamW é o otimizador padrão de
  Transformers.

Nos passos seguintes, $\hat m_t$ e $\hat v_t$ acumulam histórico. Onde o gradiente muda de sinal com
frequência (ruído), $\hat m_t$ fica pequeno perto de $\sqrt{\hat v_t}$, e o passo encolhe; onde o
gradiente aponta sempre para o mesmo lado, o passo fica perto de $\eta$.

**Custo de memória:** o AdamW guarda $m$ e $v$ para cada parâmetro, o dobro dos pesos:
$2 \times 820.096 = 1{,}64$ milhão de números a mais.

---

## 5. Weight decay

### 5.1 O que faz

Isolando o termo de decay da fórmula do AdamW:

$$
\theta \leftarrow \theta - \eta\, \lambda\, \theta = (1 - \eta \lambda)\, \theta
$$

**Lendo a fórmula:**

- $(1 - \eta\lambda)$: um fator ligeiramente menor que 1. Com $\eta = 0{,}001$ e $\lambda = 0{,}01$, é
  $1 - 0{,}00001 = 0{,}99999$.
- a cada passo, todo peso com decay encolhe 0,001% em direção a zero, **independentemente** do
  gradiente.
- em 2.460 passos: $0{,}99999^{2460} \approx 0{,}976$. Um peso que não recebesse nenhum incentivo do
  gradiente perderia ~2,4% do valor.

O efeito é uma **pressão para pesos pequenos**. Um peso só se mantém grande se o gradiente
"defende" que ele é útil. Isso reduz a tendência de o modelo usar pesos enormes para decorar
detalhes do treino (uma forma de **regularização**).

### 5.2 Quais parâmetros encolhem

`configure_optimizer` divide os parâmetros em dois grupos pela dimensão do tensor:

```python
params = list(model.parameters())                 # com tying, a matriz compartilhada aparece uma vez
decay = [p for p in params if p.dim() >= 2]       # matrizes
no_decay = [p for p in params if p.dim() < 2]     # vetores
groups = [{"params": decay, "weight_decay": weight_decay},
          {"params": no_decay, "weight_decay": 0.0}]
```

Seção 5 do experimento:

```text
  weight_decay = 0.01: 26 tensores, 815,232 parâmetros
    ex.: embeddings.token_embedding.weight, embeddings.position_embedding.weight, blocks.0.attention.q_proj.weight, ...
  weight_decay = 0.0: 26 tensores, 4,864 parâmetros
    ex.: blocks.0.ln_1.weight, blocks.0.ln_1.bias, blocks.0.ln_2.weight, ...
```

- **Com decay:** as matrizes (embeddings, projeções da attention, camadas do feed-forward). São
  99,4% dos parâmetros, e é nelas que o modelo "armazena" o que aprende.
- **Sem decay:** biases e os $\gamma$/$\beta$ das LayerNorms. Encolher $\gamma$ em direção a zero
  apagaria a saída da LayerNorm, e encolher biases só desloca limiares; nenhum dos dois reduz a
  capacidade de decorar de forma útil. É a mesma divisão usada no GPT-2 e no nanoGPT.

---

## 6. Gradient clipping

### 6.1 A fórmula

Seja $\|g\|$ a norma de **todos** os gradientes do modelo, como se fossem um único vetor de 820.096
números. Se ela passar do limite $c$, todos os gradientes são multiplicados pelo mesmo fator:

$$
g \leftarrow g \cdot \min\left(1,\ \frac{c}{\|g\|}\right)
$$

**Lendo a fórmula:**

- $\|g\| = \sqrt{\sum_{\text{todos os pesos}} g_i^2}$: o tamanho do gradiente inteiro.
- $\frac{c}{\|g\|}$: o fator que faria a norma virar exatamente $c$.
- $\min(1, \dots)$: se a norma já é menor que $c$, o fator é 1 e nada muda.
- **a direção é preservada:** todos os componentes encolhem na mesma proporção; só o tamanho muda.

Exemplo: o primeiro batch da seção 1 tinha $\|g\| = 5{,}443$. Com $c = 1$, o fator é
$1/5{,}443 = 0{,}184$: cada gradiente vira 18,4% do original.

### 6.2 Na prática

Seção 4 do experimento: a norma (antes do clipping) nos 100 primeiros passos, com lr 0,001:

```text
  passos 1–5:  4.75 2.73 2.23 2.05 2.00
  passos 96–100: 0.45 0.37 0.48 0.39 0.44
  mínima 0.35 | mediana 0.51 | máxima 64.90 | passos cortados: 21%
```

- **No começo, os gradientes são grandes** (4,75 no primeiro passo) e caem rápido para ~0,4.
- **Houve um pico de 64,9** em algum desses passos. Sem clipping, esse passo seria ~65 vezes maior
  que o normal para o SGD. Com AdamW, o efeito é amortecido pela normalização, mas um gradiente
  desse tamanho ainda contamina $v_t$ por centenas de passos.
- **21% dos passos foram cortados,** quase todos no início. Depois, a norma fica abaixo de 1 e o
  clipping raramente age: ele funciona como um **seguro**, não como parte normal da conta.

O teste `test_clipping_limits_the_gradient_norm` confere que, com $c = 0{,}001$, a norma depois do
clipping é exatamente 0,001 e a devolvida (antes) é maior.

---

## 7. O learning rate e a sua agenda

### 7.1 Warmup e decaimento cosseno

`learning_rate_at` calcula o learning rate de cada passo em duas fases:

$$
\eta_t =
\begin{cases}
\eta_{\max} \cdot \dfrac{t + 1}{t_w} & \text{se } t < t_w \quad \text{(warmup)} \\[2ex]
\eta_{\min} + \dfrac{1}{2}\,(\eta_{\max} - \eta_{\min})\,\big(1 + \cos(\pi \cdot \text{progresso})\big) & \text{se } t \ge t_w \quad \text{(decaimento)}
\end{cases}
\qquad
\text{progresso} = \min\!\left(1,\ \frac{t - t_w}{t_{\text{total}} - t_w}\right)
$$

**Lendo a fórmula:**

- $t$: o passo atual, começando em 0 no código.
- **Warmup:** $\frac{t + 1}{t_w}$ cresce linearmente de $1/t_w$ até 1. O lr sobe de quase zero até
  $\eta_{\max}$ nos primeiros $t_w$ passos.
- **progresso:** 0 logo depois do warmup e 1 no último passo.
- $\cos(\pi \cdot \text{progresso})$: vai de $\cos 0 = 1$ a $\cos \pi = -1$.
- $\frac{1}{2}(1 + \cos(\dots))$: vai de 1 a 0, numa curva suave (rápida no meio, lenta nas pontas).
- o resultado vai de $\eta_{\max}$ a $\eta_{\min}$.
- $\min(1, \dots)$: depois do último passo previsto, o lr fica em $\eta_{\min}$.

Exemplo, com os valores dos testes ($\eta_{\max} = 1$, $\eta_{\min} = 0{,}1$, $t_w = 10$,
$t_{\text{total}} = 110$):

| passo $t$ | fase | conta | $\eta_t$ |
|---:|---|---|---:|
| 0 | warmup | $1 \cdot 1/10$ | 0,1 |
| 4 | warmup | $1 \cdot 5/10$ | 0,5 |
| 9 | warmup | $1 \cdot 10/10$ | 1,0 |
| 10 | decaimento | progresso 0: $0{,}1 + 0{,}45 \cdot 2$ | 1,0 |
| 60 | decaimento | progresso 0,5: $0{,}1 + 0{,}45 \cdot (1 + 0)$ | 0,55 |
| 110 | decaimento | progresso 1: $0{,}1 + 0{,}45 \cdot 0$ | 0,1 |

### 7.2 Por que warmup

- **Os primeiros gradientes são os maiores** (4,75 no passo 1, contra ~0,4 no passo 100; seção 6.2).
- **As médias do Adam ainda não têm histórico:** $\hat v_t$ se baseia em poucos gradientes, então a
  normalização é pouco confiável.

Começar com lr pequeno evita que os primeiros passos, os mais "desinformados", empurrem os pesos
para uma região ruim. 100 passos de warmup são ~4% do treino.

### 7.3 Por que decair (e por que não decaímos agora)

No fim de um treino longo, um lr menor permite **refinar**: passos grandes fazem os pesos
oscilarem em torno do mínimo, passos pequenos deixam assentar. Por isso warmup + cosseno é a agenda
padrão de GPTs.

**Mas medimos antes de adotar** (seção 8): com o orçamento de passos que o Mini-GPT treina na CPU,
o modelo ainda está longe de convergir, e decair o lr cedo **atrasou** o aprendizado. O `tiny.yaml`
usa `min_learning_rate: 0.001`, igual ao máximo: warmup de 100 passos e depois lr constante. A
função continua suportando o decaimento; basta um valor menor na config.

---

## 8. Comparando configurações

Antes de fixar o `tiny.yaml`, treinamos 4 configurações por 1.500 passos cada (batch 32, a mesma
seed, dropout 0,1, weight decay 0,01, clip 1,0), medindo a loss de validação a cada 250 passos. É a
seção 7 do experimento (`--comparar`).

| configuração | 250 | 500 | 750 | 1.000 | 1.250 | 1.500 |
|---|---:|---:|---:|---:|---:|---:|
| lr 3e-4 constante, stride 128 | 2,353 | 2,255 | 2,132 | 2,014 | 1,910 | 1,820 |
| **lr 1e-3 constante, stride 128** | 2,270 | 2,069 | 1,895 | 1,742 | 1,637 | **1,571** |
| lr 1e-3 cosseno até 1e-4, stride 128 | 2,296 | 2,096 | 1,941 | 1,820 | 1,743 | 1,705 |
| lr 1e-3 cosseno até 1e-4, stride 32 | 2,296 | 2,088 | 1,933 | 1,804 | 1,735 | 1,696 |

Referências de [10-loss.md](10-loss.md): bigrama 2,328; **trigrama 1,923**.

O que a tabela mostra:

- **Todas as configurações passam do trigrama** entre os passos 750 e 1.250: o modelo usa mais que 2
  caracteres de contexto.
- **lr 1e-3 aprende muito mais rápido que 3e-4:** 1,571 contra 1,820 no mesmo número de passos. Com
  3e-4, os passos são pequenos demais para este orçamento.
- **O cosseno perdeu para o lr constante** (1,705 contra 1,571). A loss ainda caía ~0,07 a cada 250
  passos quando o lr começou a diminuir: o modelo estava em **underfitting**, e reduzir o passo nessa
  fase só atrasa. O decaimento compensa quando o modelo já está perto do que consegue aprender.
- **Stride 32 ou 128 praticamente empatam** (1,696 contra 1,705). Mais janelas diferentes por época
  não fizeram diferença em 1.500 passos, provavelmente porque o modelo ainda não esgotou o que
  aprende das janelas de stride 128.
- **Não há overfitting:** em todas as linhas do log, a loss de validação ficou **abaixo** da de
  treino. Isso acontece porque o treino é medido com dropout ligado e a validação sem, e é um sinal
  de que o modelo ainda tem capacidade sobrando.

**Decisão para o `tiny.yaml`:** lr 1e-3 com warmup de 100 passos e depois constante, stride 128 e 30
épocas (2.460 passos), para aproveitar o fato de a loss ainda estar caindo.

---

## 9. O loop completo

### 9.1 A função `train`

```python
def train(model, train_loader, val_loader, optimizer, *, epochs, max_lr, min_lr,
          warmup_steps, gradient_clip, log_every, log=print):
    total_steps = epochs * len(train_loader)
    for epoch in range(1, epochs + 1):
        for x, y in train_loader:                       # outra ordem de janelas a cada época
            lr = learning_rate_at(step, ...)
            for group in optimizer.param_groups:        # aplica o lr deste passo
                group["lr"] = lr
            result = train_step(model, x, y, optimizer, gradient_clip)
            ...                                         # log a cada log_every passos
        val_loss = evaluate(model, val_loader)          # uma avaliação por época
        history.append(EpochRecord(...))
```

Parte por parte:

- **Duas voltas aninhadas:** a de fora conta épocas; a de dentro percorre os batches. A cada nova
  volta, o `DataLoader` sorteia outra ordem de janelas (o gerador com seed avança).
- **O lr é aplicado a cada passo** escrevendo em `param_groups`. O otimizador guarda um lr por grupo
  (lembre os dois grupos da seção 5), e os dois recebem o mesmo valor.
- **O log** mostra, a cada `log_every` passos: passo, época, loss do batch, lr, norma do gradiente e
  tokens por segundo — as observações que a seção 14 usa para diagnosticar o treino: loss, learning
  rate, gradient norm e velocidade.
- **`EpochRecord`** guarda, por época: a média das losses de treino, a loss de validação e o tempo
  acumulado. `train` devolve a lista desses registros: o histórico do treino.

### 9.2 A função `evaluate`

```python
@torch.no_grad()
def evaluate(model, loader):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        total += cross_entropy(model(x), y).item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count
```

- **`@torch.no_grad()`:** o PyTorch não grava o grafo computacional. Não há backward na avaliação,
  então gravar seria desperdício de memória e tempo.
- **`model.eval()`:** desliga o dropout. A avaliação mede o modelo "de verdade", sem ruído
  artificial, e é determinística.
- **Média ponderada por tokens:** cada batch contribui com `loss × número de tokens`. Com batches de
  tamanhos diferentes, a média simples das losses daria peso errado aos menores.
- **`model.train(was_training)`:** devolve o modo em que o modelo estava, para não surpreender quem
  chamou.

### 9.3 Por que a loss de treino fica acima da de validação

As duas não são medidas nas mesmas condições:

| | loss de treino (no log) | loss de validação |
|---|---|---|
| dropout | ligado | desligado |
| pesos | mudando durante a época (média de 82 momentos) | os do fim da época |
| texto | os 90% iniciais do livro | os 10% finais |

Com dropout ligado, 10% das ativações são zeradas ao acaso: o modelo prevê pior de propósito. Por
isso a loss de treino pode ficar acima da de validação sem que haja nada errado. O sinal de
**overfitting** é outro: a loss de validação **parar de cair e começar a subir** enquanto a de
treino continua caindo.

---

## 10. O treino completo

### 10.1 O que o `train.py` faz

1. carrega a config, o corpus e o tokenizer;
2. divide treino e validação e cria os dois `DataLoader`s ([02-dataset.md](02-dataset.md));
3. cria o modelo com a seed da config e o otimizador com `configure_optimizer`;
4. mede a loss de validação **antes** de treinar, como ponto de partida;
5. chama `train` com os hiperparâmetros do `tiny.yaml`;
6. no fim, mostra a loss final, a perplexidade, os bits por caractere e a melhor época.

### 10.2 O início do log

Saída de `uv run python train.py`:

```text
PyTorch 2.14.0 | device: cpu
corpus:     dom_casmurro.txt (374,785 caracteres, vocab_size 97)
dados:      treino 337,307 tokens, 82 batches/época | validação 37,478 tokens, 9 batches
modelo:     GPT com 820,096 parâmetros (com weight tying)
treino:     30 épocas = 2,460 passos | lr 0.001 -> 0.001 (warmup 100) | weight decay 0.01 | clip 1.0
loss de validação antes do treino: 4.603 (chute uniforme: 4.575)

passo     1/2460 | época  1 | loss 4.608 | lr 1.00e-05 | grad 4.16 | 20,221 tokens/s
passo   100/2460 | época  2 | loss 2.426 | lr 1.00e-03 | grad 0.42 | 41,162 tokens/s
passo   200/2460 | época  3 | loss 2.328 | lr 1.00e-03 | grad 0.52 | 44,015 tokens/s
passo   300/2460 | época  4 | loss 2.301 | lr 1.00e-03 | grad 0.54 | 39,048 tokens/s
passo   400/2460 | época  5 | loss 2.239 | lr 1.00e-03 | grad 0.58 | 42,710 tokens/s
```

Como ler cada campo de uma linha de passo:

| campo | exemplo (passo 100) | significado |
|---|---|---|
| `passo 100/2460` | | passo atual e total previsto |
| `época 2` | | em que passada pelo treino estamos |
| `loss 2.426` | | cross-entropy **deste batch**, com dropout ligado; oscila de batch para batch |
| `lr 1.00e-03` | | learning rate aplicado neste passo; aqui o warmup acabou de terminar |
| `grad 0.42` | | norma total dos gradientes antes do clipping |
| `41,162 tokens/s` | | velocidade: $32 \times 128 = 4.096$ tokens divididos pelo tempo do passo |

- **Passo 1:** loss 4,608 (o chute uniforme) e lr $10^{-5}$, o primeiro degrau do warmup:
  $0{,}001 \times 1/100$.
- **Passo 100:** em 100 passos a loss do batch caiu de 4,6 para 2,4, e a norma do gradiente de 4,16
  para 0,42.
- **Velocidade:** ~40 mil tokens por segundo na CPU (o primeiro passo é mais lento porque o PyTorch
  ainda está alocando memória).

### 10.3 A evolução por época

```text
== época  1/30 | loss treino 3.219 | loss validação 2.473 | 19 s
== época  2/30 | loss treino 2.398 | loss validação 2.355 | 28 s
== época  3/30 | loss treino 2.330 | loss validação 2.301 | 37 s
== época  4/30 | loss treino 2.275 | loss validação 2.221 | 45 s
== época  5/30 | loss treino 2.223 | loss validação 2.165 | 54 s
== época  6/30 | loss treino 2.171 | loss validação 2.106 | 62 s
== época  7/30 | loss treino 2.121 | loss validação 2.036 | 71 s
== época  8/30 | loss treino 2.068 | loss validação 1.985 | 79 s
== época  9/30 | loss treino 2.014 | loss validação 1.931 | 86 s
== época 10/30 | loss treino 1.957 | loss validação 1.871 | 95 s
== época 11/30 | loss treino 1.898 | loss validação 1.821 | 103 s
== época 12/30 | loss treino 1.849 | loss validação 1.771 | 111 s
== época 13/30 | loss treino 1.805 | loss validação 1.730 | 119 s
== época 14/30 | loss treino 1.766 | loss validação 1.695 | 127 s
== época 15/30 | loss treino 1.731 | loss validação 1.658 | 135 s
== época 16/30 | loss treino 1.696 | loss validação 1.634 | 144 s
== época 17/30 | loss treino 1.669 | loss validação 1.609 | 153 s
== época 18/30 | loss treino 1.643 | loss validação 1.587 | 161 s
== época 19/30 | loss treino 1.616 | loss validação 1.564 | 169 s
== época 20/30 | loss treino 1.595 | loss validação 1.553 | 178 s
== época 21/30 | loss treino 1.573 | loss validação 1.541 | 187 s
== época 22/30 | loss treino 1.555 | loss validação 1.524 | 196 s
== época 23/30 | loss treino 1.536 | loss validação 1.509 | 204 s
== época 24/30 | loss treino 1.519 | loss validação 1.500 | 212 s
== época 25/30 | loss treino 1.505 | loss validação 1.491 | 220 s
== época 26/30 | loss treino 1.491 | loss validação 1.477 | 229 s
== época 27/30 | loss treino 1.476 | loss validação 1.476 | 237 s
== época 28/30 | loss treino 1.464 | loss validação 1.463 | 247 s
== época 29/30 | loss treino 1.452 | loss validação 1.464 | 266 s
== época 30/30 | loss treino 1.441 | loss validação 1.452 | 285 s

fim: loss de validação 1.452 | perplexidade 4.27 | 2.095 bits/caractere | 285 s
melhor época: 30 (validação 1.452)
```

A mesma evolução, comparada com as réguas de [10-loss.md](10-loss.md):

| marco | época | loss de validação |
|---|---:|---:|
| antes do treino | 0 | 4,603 |
| passa do unigrama (3,058) | 1 | 2,473 |
| passa do bigrama (2,328) | 3 | 2,301 |
| passa do trigrama (1,923) | 10 | 1,871 |
| fim | 30 | **1,452** |

### 10.4 O que o resultado mostra

**O modelo aprendeu, e usa contexto.** A loss de validação caiu de 4,603 para 1,452. A perplexidade
foi de ~97 para **4,27**: em média, o modelo hesita entre ~4 caracteres, contra 6,8 do trigrama.
Nenhum modelo de contagem com 1 ou 2 caracteres de contexto chega perto disso.

**O ritmo está diminuindo.** Entre as épocas 10 e 20, a validação caiu 0,318 (1,871 → 1,553); entre
20 e 30, caiu 0,101 (1,553 → 1,452). O modelo está se aproximando do que consegue aprender com esta
arquitetura, este corpus e este lr. Numa próxima rodada mais longa, o decaimento do lr (seção 7.3)
passa a fazer sentido.

**As curvas se cruzaram no fim.** Até a época 26, a loss de treino ficou acima da de validação (o
efeito do dropout, seção 9.3). Na época 27 elas empataram (1,476), e nas épocas 29 e 30 o treino
ficou abaixo (1,441 contra 1,452). Na época 29 a validação chegou a subir um pouco (1,463 → 1,464)
antes de voltar a cair. Isso ainda **não** é overfitting, porque a melhor validação é a da última
época. Mas é o sinal de que o modelo começa a ajustar-se mais ao treino do que generaliza, e de que
treinar muito além disso, com o mesmo corpus, tende a piorar a validação. Acompanhar esse ponto é
um dos objetivos da avaliação ([14-evaluation.md](14-evaluation.md)) e dos checkpoints
([15-checkpoints.md](15-checkpoints.md)), que vão permitir guardar a melhor época.

> **Correção feita em [14-evaluation.md](14-evaluation.md).** O cruzamento acima compara a loss de
> treino **com dropout** com a de validação **sem dropout**, então parte dele é efeito do dropout.
> Em 14-evaluation.md, o treino passou a ser
> medido também sem dropout, ao fim de cada época. Nessa comparação justa, a loss de treino fica
> **abaixo** da de validação desde a primeira época, e a diferença cresce continuamente: +0,008 na
> época 1, +0,043 na 10, +0,093 na 20 e +0,156 na 30. Não há um "ponto de cruzamento": parte do que o
> modelo aprende é específica do texto de treino desde o início, e essa parte cresce. Como a
> validação continua caindo, isso ainda não é overfitting. Detalhes em
> [14-evaluation.md](14-evaluation.md), seção 2.

**Custo:** 285 segundos na CPU para 2.460 passos, cerca de 9,5 s por época, incluindo a avaliação.

**Uma nota sobre a velocidade:** o log de um dos passos finais mostra 14.364 tokens/s, contra ~40 mil
nos demais. É uma medição de um único passo, sensível a qualquer outra atividade do computador
naquele instante. As épocas 29 e 30 também levaram mais tempo (19 s cada, contra ~8,5 s), o que
confirma uma lentidão momentânea da máquina, e não do modelo.

---

## 11. O que cada teste garante

[`tests/test_trainer.py`](../tests/test_trainer.py). Os testes usam um modelo minúsculo e uma
sequência previsível (o próximo token é sempre o atual + 1), que qualquer modelo funcional aprende
em poucos passos.

| teste | propriedade |
|---|---|
| `test_weight_decay_only_on_matrices` | decay só em tensores com 2+ dimensões; biases e LayerNorm sem decay |
| `test_every_parameter_is_optimized_exactly_once` | nenhum parâmetro fica de fora ou repetido (inclusive com tying) |
| `test_warmup_grows_linearly_to_max` | warmup: 0,1 → 0,5 → 1,0 |
| `test_cosine_decay_goes_from_max_to_min` | decaimento: 1,0 → 0,55 → 0,1, e fica em 0,1 depois |
| `test_decay_is_monotonic` | o lr nunca sobe durante o decaimento |
| `test_without_warmup_starts_at_max` | com `warmup_steps = 0`, começa em $\eta_{\max}$ |
| `test_train_step_changes_parameters` | um passo muda todos os parâmetros |
| `test_gradients_do_not_accumulate_between_steps` | `zero_grad` impede a soma entre passos |
| `test_clipping_limits_the_gradient_norm` | a norma depois do clipping é exatamente $c$ |
| `test_repeated_steps_reduce_the_loss` | 40 passos no mesmo batch reduzem a loss a menos da metade |
| `test_evaluate_matches_manual_average` | `evaluate` = média das losses dos batches |
| `test_evaluate_does_not_touch_parameters_or_mode` | avaliar não muda pesos, não cria gradientes e restaura o modo |
| `test_train_runs_epochs_and_learns` | 3 épocas: histórico correto, log por época e validação caindo a menos da metade |

---

## Resumo

1. Um passo de treino é: forward → loss → `zero_grad` → `backward` → clipping → `optimizer.step()`.
   O training loop repete isso por épocas, avaliando na validação ao fim de cada uma.
2. O `backward` aplica a regra da cadeia de trás para frente pelo grafo gravado no forward e
   calcula, de uma vez, o gradiente dos 820.096 parâmetros. O otimizador move cada peso contra o seu
   gradiente: é assim que o erro altera os pesos.
3. O AdamW normaliza o passo de cada peso pelo tamanho típico do seu gradiente: no primeiro passo,
   todo peso anda ≈ lr, enquanto com SGD os passos variariam por um fator de milhares.
4. Weight decay encolhe as matrizes a cada passo; gradient clipping limita a norma total a 1,0 (21%
   dos primeiros 100 passos foram cortados, com um pico de 64,9); warmup começa com passos pequenos.
5. Medimos antes de escolher: lr 1e-3 constante venceu 3e-4 e o decaimento cosseno neste orçamento,
   todas as configurações passaram do trigrama, e não houve overfitting.
6. O treino completo (30 épocas, 2.460 passos, 285 s na CPU) levou a validação de 4,603 a **1,452**
   (perplexidade 4,27), passando do trigrama na época 10. No fim, o ritmo de melhora caiu. As curvas
   de treino (com dropout) e de validação se cruzaram, mas, medida sem dropout
   ([14-evaluation.md](14-evaluation.md)), a loss de treino fica abaixo da de validação desde o
   início, com a diferença crescendo até +0,156.

---

## Para checar o entendimento

1. O que aconteceria se `optimizer.zero_grad()` fosse chamado **depois** de `loss.backward()`, e não
   antes?
2. Por que o backward custa só ~2 vezes o forward, mesmo calculando 820.096 derivadas?
3. No primeiro passo do AdamW, um peso com gradiente $10^{-6}$ e outro com gradiente $10^{-1}$ andam
   quanto, com lr = 0,001? E com SGD?
4. Calcule $\hat m_1$ e $\hat v_1$ a partir de $m_1 = (1 - \beta_1) g_1$ e $v_1 = (1 - \beta_2) g_1^2$.
   Por que a correção de viés é necessária?
5. Com $\eta = 0{,}001$ e $\lambda = 0{,}01$, quanto um peso encolhe por passo, só pelo weight decay?
   Por que não aplicar decay nos $\gamma$ da LayerNorm?
6. Os gradientes de um passo têm norma 8. Com $c = 1$, por quanto cada gradiente é multiplicado? A
   direção muda?
7. Usando a fórmula do cosseno, qual o lr no meio do decaimento, com $\eta_{\max} = 0{,}001$ e
   $\eta_{\min} = 0{,}0001$?
8. Por que o decaimento cosseno perdeu para o lr constante na comparação da seção 8? Em que situação
   você esperaria o contrário?
9. A loss de treino no log está acima da de validação. Isso indica um problema? Qual seria o sinal
   de overfitting?
10. Por que `evaluate` usa `@torch.no_grad()` e `model.eval()`? O que cada um evita?
11. No treino completo, a validação caiu 0,318 entre as épocas 10 e 20, e só 0,101 entre 20 e 30. O
    que você mudaria numa rodada de 60 épocas: o learning rate, a agenda, o dropout? Por quê?
12. Nas épocas 29 e 30, a loss de treino do log ficou abaixo da de validação. Por que essa comparação
    é enganosa (veja a correção na seção 10.4)? E por que uma diferença entre treino e validação,
    sozinha, ainda não prova overfitting? Que evidência adicional você procuraria?
