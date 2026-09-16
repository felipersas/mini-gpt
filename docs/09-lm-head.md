# Language Model Head

> **Como um vetor vira uma previsão sobre o próximo token?**

Código: [`model/lm_head.py`](../model/lm_head.py) e [`model/gpt.py`](../model/gpt.py). Testes:
[`tests/test_lm_head.py`](../tests/test_lm_head.py). Experimento:
`uv run python -m experiments.e09_lm_head` (~20 s, por causa dos mini-treinos). Visão geral:
[architecture.md](architecture.md). Guia de notação: [00-notacao.md](00-notacao.md).

---

## Para que serve

### O problema

Depois do corpo do GPT ([08-gpt.md](08-gpt.md)), cada posição do texto termina com um vetor $h$
de 128 números: o **estado final**. Ele resume o que o modelo calculou sobre o texto até aquele
ponto.

Mas 128 números soltos não respondem à pergunta que interessa: **qual é o próximo
caractere?** A resposta precisa ter outro formato: **um número para cada um dos 97 caracteres
do vocabulário**, dizendo o quanto aquele caractere é um bom candidato.

O **LM head** (*language model head*, "cabeça do modelo de linguagem") faz essa conversão. É a
última camada do GPT.

- **Entrada:** o estado final, $d = 128$ números por posição.
- **Saída:** os **logits**. Um logit é um **score** (uma nota) por token do vocabulário: $V = 97$
  números por posição. Logit alto = token favorecido. Logits podem ser negativos e não somam
  nada em particular.
- **Depois dele:** a **softmax** transforma os 97 logits em 97 **probabilidades**: números entre
  0 e 1 que somam 1.

### Uma analogia

Imagine 97 "detectores", um por caractere. Cada detector guarda um **vetor-modelo** de 128
números (uma linha da matriz do LM head). Ele compara o estado final com o seu vetor-modelo e
dá uma nota: quanto mais parecidos, maior a nota. A comparação é o **produto escalar**
([00-notacao.md](00-notacao.md), seção 5.2). A softmax converte as 97 notas em porcentagens.

Com essa imagem, todo o trabalho dos blocos anteriores tem um objetivo claro: transformar o
contexto num estado final parecido com o vetor-modelo do caractere que vem a seguir.

### O que entra e o que sai

| tensor | shape genérico | no `tiny.yaml` (batch de treino) | significado |
|---|---|---|---|
| entrada: estado final | `[B, T, d]` | `(32, 128, 128)` | um vetor de 128 números por posição |
| parâmetro: `lm_head.proj.weight` | `[V, d]` | `(97, 128)` | uma linha (vetor-modelo) por token |
| saída: logits | `[B, T, V]` | `(32, 128, 97)` | 97 scores por posição |
| depois da softmax | `[B, T, V]` | `(32, 128, 97)` | 97 probabilidades por posição, somando 1 |

São $32 \times 128 = 4.096$ posições no batch, e **cada uma** recebe a sua própria previsão.

### Onde fica no pipeline

```text
texto (str)
  │  tokenizer (01-tokenizer.md)
  ▼
ids [N]
  │  batches (02-dataset.md)
  ▼
x [B, T] = (32, 128)
  │  embeddings (03-embeddings.md)
  ▼
[B, T, d] = (32, 128, 128)
  │  4× Transformer block (04 a 07)
  ▼
[B, T, d] = (32, 128, 128)
  │  LayerNorm final (08-gpt.md)
  ▼
h = estado final [B, T, d] = (32, 128, 128)
  │  LM head                                        ← este documento
  ▼
z = logits [B, T, V] = (32, 128, 97)
  │  softmax (este documento)  ou  cross_entropy (10-loss.md)
  ▼
p = probabilidades [B, T, V]   ou   ℒ = loss (um número)
```

A **loss** $\mathcal{L}$ ([10-loss.md](10-loss.md)) é o número que o treino tenta diminuir. A
usada em modelos de linguagem é a **cross-entropy**: $-\log$ da probabilidade que o modelo deu
ao caractere certo.

### O que daria errado

- **Sem LM head**, o modelo devolveria vetores de 128 números, e o alvo `y` é um ID entre 0 e
  96. Não haveria como comparar a saída com o alvo (loss, [10-loss.md](10-loss.md)), nem como
  escolher um caractere para gerar texto ([13-generation.md](13-generation.md)).
- **Com inicialização grande**, os logits começariam enormes, e o modelo apostaria quase tudo
  num caractere arbitrário antes de aprender qualquer coisa. Na seção 3, pesos com desvio 1,0
  levam a uma loss inicial de **29**; com desvio 0,02, a loss fica em **4,6**, perto do **chute
  uniforme** (dar a mesma probabilidade, 1/97, a todo caractere).

### Mapa do documento

| seção | assunto |
|---|---|
| 1 | logits: um produto escalar por token |
| 2 | softmax, log-probabilidades e `log_softmax` |
| 3 | por que a loss inicial é $\approx \ln V + s^2/2$ |
| 4 | **weight tying**: usar a mesma matriz no token embedding (entrada) e no LM head (saída) |
| 5 | o que o modelo prevê depois de 150 passos de treino |
| 6 | a memória ocupada pelos logits |
| 7 | o código, parte por parte |
| 8 | os testes |

### Onde estamos

Depois do corpo do GPT ([08-gpt.md](08-gpt.md)), cada posição termina com um vetor $h$ de 128
dimensões que resume o que o modelo calculou sobre o texto até ali. Mas o objetivo é **prever o
próximo caractere**: é preciso uma distribuição de probabilidade sobre os 97 tokens do
vocabulário, em cada posição.

O LM head faz essa última conversão:

```text
estado final [B, T, d]  ──LM head──►  logits [B, T, V]  ──softmax──►  probabilidades [B, T, V]
```

A partir daqui, `model(ids)` devolve **logits**. Os estados finais continuam acessíveis
em `model.hidden_states(ids)`.

---

## Notação deste documento

Os símbolos gerais (soma, produto escalar, exponencial, logaritmo, softmax, esperança, normal,
derivada) têm explicação com exemplos no [guia de notação](00-notacao.md). A tabela abaixo
define **todos** os símbolos usados aqui.

**Cuidado com três letras reaproveitadas:**

- $h$ aqui é o **estado final** de uma posição (um vetor). **Não** é o número de heads da
  attention ([05-multi-head-attention.md](05-multi-head-attention.md)).
- $s$ aqui é o **desvio padrão dos logits**. **Não** é o stride de [02-dataset.md](02-dataset.md).
- $\sigma$ aqui é o **desvio padrão por coordenada do residual stream**. **Não** é a função de
  ativação de [06-feed-forward.md](06-feed-forward.md).

### Tamanhos e índices

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $B$ | "bê" | sequências por batch | 32 | `batch_size` |
| $T$ | "tê" | posições por sequência | 128 | `context_length` |
| $d$ | "dê" | largura dos vetores do modelo | 128 | `d_model` |
| $V$ | "vê" | tamanho do vocabulário | 97 | `vocab_size`, `tok.vocab_size` |
| $i$, $j$ | "i", "jota" | índice de um token do vocabulário | 0 a 96 | `logits[..., i]` |
| $t$ | "tê" | índice de posição na sequência; $(b, t)$ = sequência $b$, posição $t$ | 0 a 127 | `x[b, t]` |

### Dados e parâmetros

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $h$ | "agá" | **estado final** de uma posição: saída da LayerNorm final (não é número de heads) | vetor `(128,)`; norma ≈ 11,29 na inicialização | `model.hidden_states(ids)[b, t]`; `h` em `LMHead.forward` |
| $h_0$ | "agá zero" | residual stream na entrada do primeiro bloco: $E_{x_t} + P_t$ | `(128,)` | `model.embeddings(ids)` |
| $x_t$ | "xis tê" | ID do caractere na posição $t$ (o **caractere atual**) | inteiro de 0 a 96 | `x[b, t]` |
| $y$ | "ípsilon" | ID do **próximo** caractere, o correto (o **alvo**) | inteiro de 0 a 96 | `y[b, t]` |
| $W$ | "dáblio" | matriz do LM head | `[V, d]` = `(97, 128)` | `lm_head.proj.weight` |
| $w_i$ | "dáblio i" | linha $i$ de $W$: o vetor-modelo do token $i$ | `(128,)` | `W[i]` |
| $E$ | "é" | matriz do token embedding ([03-embeddings.md](03-embeddings.md)) | `[V, d]` = `(97, 128)` | `embeddings.token_embedding.weight` |
| $E_i$ | "é i" | linha $i$ de $E$: o embedding do token $i$ | `(128,)`; $\lVert E_i \rVert^2 \approx 0{,}051$ | `E[i]` |
| $P_t$ | "pê tê" | positional embedding da posição $t$ (também [03-embeddings.md](03-embeddings.md)) | `(128,)` | `position_embedding.weight[t]` |
| $z$ | "zê" | **logits** de uma posição | `(97,)`; no batch, `(32, 128, 97)` | `logits = model(ids)` |
| $z_i$ | "zê i" | logit do token $i$ | um número; $\approx 0 \pm 0{,}25$ na inicialização | `logits[..., i]` |
| $z_y$ | "zê ípsilon" | logit do token correto | um número | `logits.gather(-1, y.unsqueeze(-1))` |
| $p$ | "pê" | probabilidades de uma posição (saída da softmax) | `(97,)`, somam 1 | `torch.softmax(logits, dim=-1)` |
| $p_i$ | "pê i" | probabilidade do token $i$ | 1/97 = 0,0103 no chute uniforme | `probs[..., i]` |
| $p_y$ | "pê ípsilon" | probabilidade dada ao token correto | um número entre 0 e 1 | `probs.gather(-1, y.unsqueeze(-1))` |
| $c$ | "cê" | uma constante somada a todos os logits | 7 no experimento | `logits + 7.0` |
| $m$ | "eme" | o maior logit de uma posição | 2 em $z = (2, 1, 0)$ | `z.max(dim=-1)` |
| $\eta$ | "eta" | learning rate: tamanho do passo do otimizador | exemplo da seção 4.4 | `lr` |

### Operações e funções

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $w_i \cdot h$ | "w i escalar h" | produto escalar: multiplica coordenada a coordenada e soma | um número | `torch.dot(W[i], h)` |
| $W h$ | "W vezes h" | todos os $V$ produtos escalares de uma vez | `(97,)` | `W @ h`, `self.proj(h)` |
| $E^\top$ | "E transposta" | $E$ com linhas e colunas trocadas | `(128, 97)` | `E.T` |
| $\mathbb{R}^{V \times d}$ | "erre V por d" | conjunto das matrizes de números reais com $V$ linhas e $d$ colunas | shape `(97, 128)` | |
| $e^{a}$, $\exp(a)$ | "e elevado a a" | exponencial; $e \approx 2{,}718$ | $e^1 = 2{,}718$ | `torch.exp`, `math.exp` |
| $\ln$, $\log$ | "logaritmo natural" | inverso da exponencial ($\log$ sem base = natural) | $\ln 97 = 4{,}575$ | `torch.log`, `math.log` |
| $\sum_j$ | "soma sobre j" | soma do termo para todos os tokens $j$ | 97 termos | `.sum(dim=-1)` |
| $\text{softmax}(z)_i$ | "softmax de z, componente i" | $e^{z_i} / \sum_j e^{z_j}$ | $(0{,}665;\ 0{,}245;\ 0{,}090)$ para $z = (2, 1, 0)$ | `torch.softmax(z, dim=-1)` |
| $\log p_i$ | "log de p i" | **log-probabilidade**: logaritmo da probabilidade; sempre $\le 0$ | $-4{,}575$ no chute uniforme | `F.log_softmax(z, dim=-1)` |
| $\text{logsumexp}(z)$ | "log-sum-exp de z" | $\ln \sum_j e^{z_j}$: log da soma das exponenciais | 2,408 para $z = (2, 1, 0)$ | `torch.logsumexp(z, dim=-1)` |
| $\max_j z_j$ | "máximo de z j" | o maior dos logits | | `z.max(dim=-1)` |
| $\mathcal{L}$ | "ele cursivo", "a loss" | erro de uma previsão: $-\log p_y$ (cross-entropy) | ≈ 4,6 na inicialização | `F.cross_entropy` ([10-loss.md](10-loss.md)) |
| $\mathcal{L}_{\text{inicial}}$ | "loss inicial" | a loss antes de qualquer treino | 4,563 a 4,735 (seção 3.2) | |
| $\mathbb{E}[\cdot]$ | "esperança de" | média teórica de uma quantidade aleatória | $\mathbb{E}[z_y] = 0$ | ≈ `.mean()` sobre muitas amostras |
| $\mathcal{N}(0, s^2)$ | "normal de média 0 e variância s ao quadrado" | distribuição "curva de sino" centrada em 0 | | `torch.randn(...) * s` |
| $z_j \sim \mathcal{N}(0, s^2)$ | "z j segue uma normal" | $z_j$ é sorteado dessa distribuição | | |
| $s$ | "ésse" | **desvio padrão dos $V$ logits** de uma posição (não é stride) | 0,227 / 0,246 / 0,583 / 11,366 | `model(x).std(dim=-1).mean()` (experimento, linha 109) |
| $s^2$ | "ésse ao quadrado" | variância dos logits | 0,0515 com $s = 0{,}227$ | |
| $e^{s^2/2}$ | "e elevado a s ao quadrado sobre dois" | valor esperado de $e^{z}$ quando $z \sim \mathcal{N}(0, s^2)$ | 1,026 com $s = 0{,}227$ | `math.exp(s**2 / 2)` |
| $\int \dots \, dz$ | "integral em z" | soma contínua sobre todos os valores de $z$ (só na derivação opcional da seção 3.1) | | |
| $\pi$ | "pi" | 3,14159… (aparece na fórmula da normal) | | `math.pi` |
| $\mathbb{1}[i = y]$ | "indicadora de i igual a y" | 1 se $i$ é o token correto, 0 se não | $(0, 1, 0)$ para $y = 1$ e $V = 3$ | `F.one_hot(y, V)` |
| $\dfrac{\partial \mathcal{L}}{\partial z_i}$ | "d parcial de L em relação a z i" | quanto a loss muda se $z_i$ subir um pouquinho | $p_i - \mathbb{1}[i = y]$ | `logits.grad` |
| $\dfrac{\partial \mathcal{L}}{\partial w_i}$ | "d parcial de L em relação a w i" | gradiente da linha $i$ do LM head | `(128,)` | `lm_head.proj.weight.grad[i]` |
| $\sum_{\text{posições}}$ | "soma sobre as posições" | soma sobre todas as posições $(b, t)$ do batch | 4.096 termos | |
| $\sigma$ | "sigma" | desvio padrão por coordenada do residual stream antes da LayerNorm final (não é ativação) | ≈ 0,049 na inicialização | ≈ `x.std(dim=-1)` |
| $\lVert v \rVert$ | "norma de v" | comprimento do vetor $v$ | $\lVert h \rVert \approx 11{,}29$ | `v.norm()` |
| $\approx$, $\Rightarrow$ | "aproximadamente", "implica" | | | |

---

## 1. Um score por token

**O que calcula.** O logit de cada token é o produto escalar entre o estado final e a linha
daquele token na matriz $W$. O LM head é uma camada linear **sem bias** (sem vetor somado ao
resultado):

$$
z = W h, \qquad z_i = w_i \cdot h, \qquad i = 0, \dots, V - 1
$$

**Lendo a fórmula:**

- $h \in \mathbb{R}^d$: o estado final da posição, 128 números.
- $W \in \mathbb{R}^{V \times d}$: a matriz do LM head, com uma linha por token.
- $w_i$: a linha $i$ de $W$, o vetor-modelo do token $i$.
- $w_i \cdot h$: multiplica as 128 coordenadas de $w_i$ e de $h$, par a par, e soma. É grande
  quando os dois vetores apontam na mesma direção.
- $z_i$: o logit do token $i$, um número.
- $z = Wh$: os 97 produtos escalares calculados de uma vez, num vetor de 97 logits.
- $i = 0, \dots, V - 1$: a fórmula vale para cada um dos 97 tokens.

**Exemplo pequeno** ($d = 2$, $V = 3$ tokens). Estado final $h = (2, 1)$ e linhas
$w_0 = (1, 0)$, $w_1 = (0, 1)$, $w_2 = (1, -2)$:

- $z_0 = 1 \cdot 2 + 0 \cdot 1 = 2$;
- $z_1 = 0 \cdot 2 + 1 \cdot 1 = 1$;
- $z_2 = 1 \cdot 2 + (-2) \cdot 1 = 0$.

Logo $z = (2, 1, 0)$: o estado final "se parece" mais com o vetor do token 0. Esse $z$ vai
reaparecer nas seções 2, 3 e 4.

Cada linha $w_i$ da matriz é um vetor associado ao token $i$, e o **logit** $z_i$ é o produto
escalar entre o estado final e esse vetor. Para o modelo dar um logit alto ao token certo, o
estado final precisa **apontar na direção** de $w_{\text{próximo token}}$. Todo o trabalho dos
blocos anteriores pode ser lido assim: transformar o contexto num vetor que aponte para o
próximo caractere.

Seção 1 do experimento (modelo recém-inicializado, posição 10 da primeira sequência):

```text
contexto até a posição 10: 'alguma cous'  (caractere atual 's')
h = estado final:  (128,), norma 11.29
W = lm_head:       (97, 128), linha i = vetor do token i
logits = W · h:    (97,), iguais a W @ h: True
  média -0.009, desvio 0.254, mín -0.460, máx +1.496
  logit do caractere atual 's': +1.496
```

Como ler:

- **Norma 11,29:** a LayerNorm final deixa $\lVert h \rVert \approx \sqrt{128} = 11{,}31$ ([08-gpt.md](08-gpt.md)).
- **`iguais a W @ h: True`:** a saída do módulo é exatamente a fórmula $z = Wh$.
- **Média −0,009 e desvio 0,254:** na inicialização, os logits são pequenos e centrados em 0.

| tensor | shape |
|---|---|
| estado final | `[B, T, d]` = `(32, 128, 128)` |
| `lm_head.proj.weight` | `[V, d]` = `(97, 128)` |
| logits | `[B, T, V]` = `(32, 128, 97)` |

O maior logit é justamente o do caractere atual `'s'`. Isso não é coincidência: a seção 4.5
explica.

---

## 2. Dos logits às probabilidades

### 2.1 Softmax

**O que calcula.** Transforma os 97 logits de uma posição em 97 probabilidades. É necessária
porque logits podem ser negativos e não somam 1; a loss e a geração precisam de uma
distribuição de probabilidade.

$$
p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=0}^{V-1} e^{z_j}}
$$

**Lendo a fórmula:**

- $p_i$: a probabilidade de o token $i$ ser o próximo.
- $e^{z_i}$: a exponencial do logit. Torna tudo positivo e preserva a ordem (logit maior,
  exponencial maior). Cada +1 no logit multiplica esse valor por $e \approx 2{,}718$.
- $\sum_{j=0}^{V-1} e^{z_j}$: a soma das exponenciais dos 97 logits. Dividir por ela faz o total
  dar 1. É o **mesmo** denominador para todo $i$.
- $j$: o índice que percorre todos os tokens dentro da soma. Usa outra letra para não confundir
  com o $i$ do numerador.

**Exemplo** com $z = (2, 1, 0)$ da seção 1:

- $e^2 = 7{,}389$, $e^1 = 2{,}718$, $e^0 = 1$; soma $= 11{,}107$;
- $p = (7{,}389;\ 2{,}718;\ 1) / 11{,}107 = (0{,}665;\ 0{,}245;\ 0{,}090)$, que soma 1.

**No código:** `torch.softmax(z, dim=-1)` (experimento, linha 70). `dim=-1` significa "ao longo
da última dimensão", a dos $V$ tokens: cada posição é normalizada separadamente.

```text
softmax: soma 1.000000, maior 0.0446, menor 0.0063
  (chute uniforme = 1/97 = 0.0103)
  top-5: 's' 0.045  'A' 0.015  "'" 0.015  'r' 0.014  'v' 0.014
  softmax(logits + 7) == softmax(logits): True
  log_softmax == log(softmax): True

batch inteiro: logits (32, 128, 97)
  cada uma das 4,096 posições tem uma distribuição que soma 1: True
```

- **Toda posição recebe uma distribuição completa** sobre os 97 tokens: são 4.096
  previsões num único forward, uma para cada `y[b, t]` do dataset ([02-dataset.md](02-dataset.md)).
- **Na inicialização a distribuição é quase uniforme** (maior 0,045, menor 0,006, contra
  0,0103 do uniforme): o modelo ainda não sabe nada.
- **`top-5`** são os 5 tokens com maior probabilidade. O `'s'` lidera com 0,045, cerca de 4× o
  uniforme (seção 4.5).
- As duas linhas com `True` são explicadas nas seções 2.2 e 2.3.

### 2.2 Por que "logit"

**De onde vem o nome.** Na regressão logística, "logit" é o logaritmo da razão de chances. Em
redes neurais, o termo passou a significar qualquer score que entra numa softmax.

**Somar uma constante não muda nada.** Somar o mesmo número $c$ a todos os logits deixa as
probabilidades iguais:

$$
\text{softmax}(z + c) = \text{softmax}(z)
$$

**Lendo a fórmula:**

- $z + c$: somar $c$ a cada logit $z_i$.
- Por que vale: $e^{z_i + c} = e^{z_i} \cdot e^{c}$. O fator $e^c$ aparece no numerador e em
  todos os termos do denominador, e cancela.
- **Exemplo:** $z + 10 = (12, 11, 10)$ dá a mesma softmax $(0{,}665;\ 0{,}245;\ 0{,}090)$. No
  experimento, a linha `softmax(logits + 7) == softmax(logits): True` (linha 75).

Logo, os logits são **log-probabilidades não normalizadas**. Tirando o log da softmax e isolando
$z_i$:

$$
z_i = \log p_i + \underbrace{\log \textstyle\sum_j e^{z_j}}_{\text{igual para todo } i}
$$

**Lendo a fórmula:**

- $\log p_i$: a **log-probabilidade** do token $i$. Como $0 < p_i \le 1$, ela é sempre $\le 0$:
  vale 0 para probabilidade 1 e fica muito negativa para probabilidades pequenas.
- $\log \sum_j e^{z_j}$: o log do denominador da softmax. A chave embaixo lembra que é o mesmo
  número para todos os tokens.
- "não normalizadas": o logit é a log-probabilidade **mais** uma constante. Falta subtraí-la.
- **Exemplo:** para $z = (2, 1, 0)$, $\log \sum_j e^{z_j} = \ln 11{,}107 = 2{,}408$. Então
  $\log p = (2 - 2{,}408;\ 1 - 2{,}408;\ 0 - 2{,}408) = (-0{,}408;\ -1{,}408;\ -2{,}408)$. Confere:
  $\ln 0{,}665 = -0{,}408$.

Só as **diferenças** entre logits importam: $z_i - z_j = \log(p_i / p_j)$.

**Lendo a fórmula:** a diferença entre dois logits é o log da razão entre as duas
probabilidades. **Exemplo:** $z_0 - z_1 = 2 - 1 = 1$, então $p_0 / p_1 = e^1 = 2{,}718$. De fato,
$0{,}665 / 0{,}245 = 2{,}718$. Regra prática: **cada ponto de vantagem no logit multiplica a
probabilidade relativa por 2,718**.

### 2.3 `log_softmax` em vez de `log(softmax)`

**O problema.** A loss ([10-loss.md](10-loss.md)) precisa de $\log p_{\text{alvo}}$. Calcular `softmax` e depois
`log` funciona aqui, mas é numericamente frágil por causa do **underflow**: quando um número é
pequeno demais para o formato do computador, ele é arredondado para 0.

Se um logit está 800 abaixo do maior, $e^{-800}$ vira 0 em float32, e o log vira $-\infty$. Em
float32 (números de 32 bits), isso já acontece bem antes: $e^{-103} \approx 1{,}4 \times 10^{-45}$
ainda é representável, mas $e^{-104}$ já vira 0.

```python
torch.log(torch.softmax(torch.tensor([0., -800.]), -1))   # tensor([0., -inf])
F.log_softmax(torch.tensor([0., -800.]), -1)              # tensor([   0., -800.])
```

**A solução.** O `log_softmax` calcula direto

$$
\log p_i = z_i - \log \sum_j e^{z_j}
$$

**Lendo a fórmula:** é a mesma igualdade da seção 2.2, reorganizada. A log-probabilidade é o
logit menos o log-sum-exp. Nenhuma probabilidade é formada no caminho.

O termo $\log \sum_j e^{z_j}$ é calculado com o truque *log-sum-exp* (subtrai o maior logit antes
de exponenciar), sem nunca formar probabilidades minúsculas:

$$
\text{logsumexp}(z) = \ln \sum_j e^{z_j} = m + \ln \sum_j e^{z_j - m}, \qquad m = \max_j z_j
$$

**Lendo a fórmula:**

- $m$: o maior logit da posição.
- $z_j - m$: cada logit menos o maior. Todos ficam $\le 0$, então cada $e^{z_j - m}$ fica entre 0
  e 1: nada explode.
- O maior termo da soma é $e^{m - m} = e^0 = 1$, então a soma é $\ge 1$ e o log nunca é $-\infty$.
- Por que as duas formas são iguais: $\sum_j e^{z_j} = e^m \sum_j e^{z_j - m}$, e
  $\ln(e^m \cdot S) = m + \ln S$.

**Exemplos:**

- $z = (2, 1, 0)$: $m = 2$; $e^0 + e^{-1} + e^{-2} = 1 + 0{,}368 + 0{,}135 = 1{,}503$;
  $\ln 1{,}503 = 0{,}408$; $\text{logsumexp} = 2 + 0{,}408 = 2{,}408$. Igual a $\ln 11{,}107$.
- $z = (0, -800)$: $m = 0$; a soma é $1 + e^{-800}$, que vira $1 + 0 = 1$ em float32. O
  $\text{logsumexp}$ é 0, e $\log p_1 = -800 - 0 = -800$. O termo minúsculo se perdeu, mas dentro
  de uma soma que já vale 1, sem log de zero.

`F.cross_entropy` faz isso internamente.

---

## 3. A loss inicial depende da escala dos logits

**Por que isso importa.** Antes de treinar, dá para prever quanto a loss deve valer. Se a loss
medida fica longe da prevista, há algo errado na inicialização. Esta seção deduz a previsão
passo a passo.

### 3.1 A conta

**Passo 1: a cross-entropy de uma posição (prévia de [10-loss.md](10-loss.md)).**

$$
\mathcal{L} = -\log p_y
$$

**Lendo a fórmula:**

- $y$: o ID do token correto (o próximo caractere de verdade).
- $p_y$: a probabilidade que o modelo deu a ele.
- $-\log$: transforma probabilidade alta em loss baixa. $p_y = 1$ dá $\mathcal{L} = 0$;
  $p_y \to 0$ dá $\mathcal{L} \to \infty$.
- **Exemplos:** no chute uniforme, $p_y = 1/97$ e $\mathcal{L} = \ln 97 = 4{,}575$. Com
  $z = (2, 1, 0)$ e alvo $y = 1$, $p_y = 0{,}245$ e $\mathcal{L} = -\ln 0{,}245 = 1{,}408$.
- A loss de um batch é a média dessa conta nas 4.096 posições. A unidade é o *nat* (por usar
  $\ln$).

**Passo 2: escrever a loss com logits.** Substituindo $\log p_y = z_y - \text{logsumexp}(z)$
(seção 2.3):

$$
\mathcal{L} = -\log p_y = \text{logsumexp}(z) - z_y
$$

**Lendo a fórmula:** a loss é o log-sum-exp de todos os logits (um valor um pouco acima do maior
logit) menos o logit do token correto. **Exemplo:** $2{,}408 - 1 = 1{,}408$, o mesmo valor do
passo 1.

**Passo 3: a hipótese.** Suponha logits aleatórios $z_j \sim \mathcal{N}(0, s^2)$, independentes do
alvo $y$. É o caso de um modelo recém-inicializado: ele não sabe nada, então o logit do token
correto é só mais um sorteio. Então $\mathbb{E}[z_y] = 0$.

- $z_j \sim \mathcal{N}(0, s^2)$: cada logit é sorteado de uma normal com média 0 e desvio $s$.
- $\mathbb{E}[z_y] = 0$: em média, o logit do correto vale 0, como todos os outros.

**Passo 4: a soma das exponenciais.** Somar $V$ valores sorteados da mesma distribuição dá
aproximadamente $V$ vezes a média deles:

$$
\sum_j e^{z_j} \approx V\, \mathbb{E}[e^{z}]
$$

**Lendo a fórmula:** $\mathbb{E}[e^z]$ é a média de $e^z$ sobre muitos sorteios de $z$. A
aproximação melhora com $V$ grande (lei dos grandes números).

**Passo 5: por que $\mathbb{E}[e^z] = e^{s^2/2}$ para uma normal.**

Seria tentador pensar: "$z$ tem média 0, então $e^z$ tem média $e^0 = 1$". **Não tem.** A
exponencial é assimétrica: subir $s$ ganha mais do que descer $s$ perde. Com $s = 0{,}227$:

- $e^{+0{,}227} = 1{,}255$ (ganhou 0,255);
- $e^{-0{,}227} = 0{,}797$ (perdeu 0,203);
- média dos dois: $1{,}026$, acima de 1.

Para uma normal, a conta exata dá $\mathbb{E}[e^z] = e^{s^2/2} = e^{0{,}0258} = 1{,}026$.

*Derivação (opcional).* A densidade de $\mathcal{N}(0, s^2)$ é
$\frac{1}{s\sqrt{2\pi}}\, e^{-z^2/(2s^2)}$. A esperança é a integral de $e^z$ vezes a densidade:

$$
\mathbb{E}[e^z] = \int \frac{1}{s\sqrt{2\pi}}\, e^{\,z - z^2/(2s^2)}\, dz,
\qquad
z - \frac{z^2}{2s^2} = -\frac{(z - s^2)^2}{2s^2} + \frac{s^2}{2}
$$

A segunda igualdade é "completar o quadrado" (expanda o lado direito para conferir). O fator
$e^{s^2/2}$ sai da integral, e o que sobra é a densidade de uma normal centrada em $s^2$, cuja
integral vale 1. Resultado: $\mathbb{E}[e^z] = e^{s^2/2}$.

**Passo 6: juntar tudo.**

$$
\text{logsumexp}(z) = \ln \sum_j e^{z_j} \approx \ln\big(V\, \mathbb{E}[e^{z}]\big) = \ln V + \frac{s^2}{2}
$$

usando $\mathbb{E}[e^z] = e^{s^2/2}$ para uma normal. Portanto

$$
\boxed{\;\mathcal{L}_{\text{inicial}} \approx \ln V + \frac{s^2}{2}\;}
$$

**Lendo a fórmula:**

- $\ln(V e^{s^2/2}) = \ln V + \ln e^{s^2/2} = \ln V + s^2/2$: produto vira soma, e $\ln e^a = a$.
- $\mathcal{L}_{\text{inicial}} = \text{logsumexp}(z) - \mathbb{E}[z_y] \approx \ln V + s^2/2 - 0$.
- $\ln V$: a loss do chute uniforme. Com $s = 0$ (todos os logits iguais), é exatamente ela.
- $s^2/2$: a penalidade por logits espalhados. Logits aleatórios, sem conhecimento, só pioram a
  loss.

**Com números** ($V = 97$, $s = 0{,}227$, a primeira linha da seção 3.2):

| passo | conta | valor |
|---|---|---:|
| $s^2 / 2$ | $0{,}227^2 / 2$ | 0,0258 |
| $\mathbb{E}[e^z] = e^{s^2/2}$ | $e^{0{,}0258}$ | 1,026 |
| $\sum_j e^{z_j} \approx V\, e^{s^2/2}$ | $97 \times 1{,}026$ | 99,5 |
| $\text{logsumexp}(z)$ | $\ln 99{,}5$ | 4,601 |
| $\mathcal{L}_{\text{inicial}}$ | $4{,}601 - 0$ | **4,601** |

O mesmo que $\ln 97 + 0{,}0258 = 4{,}575 + 0{,}026 = 4{,}601$. O medido foi 4,610.

Com logits pequenos, a loss inicial fica em $\ln 97 = 4{,}575$, o valor de um chute uniforme.
A aproximação só vale para $s$ pequeno: com logits grandes, a softmax satura.

**Saturação** é quando quase toda a probabilidade vai para um único token ($p \approx 1$ para
ele, $\approx 0$ para os demais). Nessa situação:

- a soma $\sum_j e^{z_j}$ é dominada pelo maior termo, e a média do passo 4 deixa de
  representá-la. O $\text{logsumexp}$ fica perto do maior logit, e a loss vira
  $\approx \max_j z_j - z_y$;
- com $s = 11{,}366$, a fórmula daria $4{,}575 + 11{,}366^2/2 \approx 69$, um valor sem sentido.
  Por isso o experimento imprime "—" quando $s \ge 1$ (linha 112);
- o maior de 97 sorteios de uma normal fica em média a ~2,5 desvios acima de 0 (estimado por
  simulação). Então $\max_j z_j - z_y \approx 2{,}5 \times 11{,}4 \approx 28$, perto dos 29 medidos.

### 3.2 A verificação

Seção 2 do experimento, com o mesmo modelo e quatro inicializações do LM head. Colunas: $s$
medido, a previsão $\ln V + s^2/2$, a loss medida e a probabilidade média dada ao caractere
**atual** $x_t$ (usada na seção 4.5).

```text
                                              s  ln V + s²/2    loss  P(caractere atual)
  LM head std 0,02, sem tying             0.227        4.601   4.610              0.0105
  weight tying (padrão)                   0.246        4.605   4.563              0.0283
  LM head com init padrão do nn.Linear    0.583        4.745   4.735              0.0117
  LM head std 1,0                        11.366            —  29.164              0.0133
```

- **Por que $s \approx 0{,}23$ com std 0,02.** O estado final tem norma ~11,3 (LayerNorm
  final), com coordenadas de variância ~1. Cada logit soma 128 produtos com pesos de desvio
  0,02: desvio $0{,}02 \cdot \sqrt{128} \approx 0{,}23$.
  - *Lendo:* somar 128 termos independentes soma as variâncias: $128 \cdot 0{,}02^2$. O desvio é a
    raiz: $0{,}02 \sqrt{128} = 0{,}226$.
- **A fórmula acerta:** 4,601 previsto e 4,610 medido; 4,745 e 4,735 com a inicialização
  padrão do `nn.Linear` (desvio $1/\sqrt{3 \cdot 128} \approx 0{,}051$ por peso).
  - *Lendo:* o `nn.Linear` sorteia pesos uniformes entre $\pm 1/\sqrt{128}$, e uma uniforme
    entre $\pm a$ tem desvio $a/\sqrt{3}$. Então $s \approx 0{,}051 \cdot 11{,}3 \approx 0{,}58$.
- **Isso explica a loss inicial vista em [08-gpt.md](08-gpt.md)** (~4,7): aquele mini-treino
  usava um `nn.Linear` provisório, com a inicialização padrão.
- **Com std 1,0, a loss inicial é 29.** O modelo começa *confiantemente errado*: logits com
  desvio 11 fazem a softmax dar quase toda a probabilidade a um token arbitrário. Além de uma
  loss enorme, uma softmax saturada tem gradientes ruins ([04-attention.md](04-attention.md), seção 2.3).

**Por que começar perto do uniforme.** "Não sei nada, então tudo é igualmente provável" é a
posição inicial mais honesta e a que produz a menor loss possível sem conhecimento. O treino
parte daí e só concentra probabilidade onde os dados justificam.

A linha do weight tying tem loss **abaixo** de $\ln 97$, e $P(\text{caractere atual})$ é quase
3× o uniforme. É o efeito da seção 4.5.

---

## 4. Weight tying

### 4.1 A ideia

**O que é.** **Weight tying** ("amarrar os pesos") é usar **uma única matriz** para dois papéis:
o token embedding (entrada) e o LM head (saída). A matriz vira um **parâmetro compartilhado**:
um só conjunto de números, usado em dois lugares do modelo, ajustado pelo treino uma vez só.

O token embedding $E$ ([03-embeddings.md](03-embeddings.md)) e o LM head $W$ têm o **mesmo shape**, `[V, d]`: uma linha por
token. O weight tying usa **a mesma matriz** para os dois papéis:

$$
W = E \quad\Rightarrow\quad z_i = h \cdot E_i
$$

**Lendo a fórmula:**

- $W = E$: a matriz do LM head **é** a do token embedding (não uma cópia).
- $\Rightarrow$: "logo".
- $z_i = h \cdot E_i$: o logit do token $i$ passa a ser o produto escalar entre o estado final e o
  **embedding** do token $i$. Com $h$ e $E_i$ na mesma direção, logit alto.

| uso | operação | papel da linha $E_i$ |
|---|---|---|
| entrada (embedding) | $h_0 = E_{x_t} + P_t$ | "o que é o token $i$" |
| saída (LM head) | $z_i = h \cdot E_i$ | "que estado final prevê o token $i$" |

- $h_0 = E_{x_t} + P_t$: na entrada, a linha do caractere atual $x_t$ é somada ao vetor da
  posição $t$ ([03-embeddings.md](03-embeddings.md)).
- $z_i = h \cdot E_i$: na saída, a mesma linha pontua o token $i$.

Com tying, **o próximo token é escolhido por similaridade** entre o estado final e o embedding
de cada token. A técnica foi proposta por Press & Wolf (2017) e Inan et al. (2017), e o GPT-2 a
usa.

### 4.2 Implementação

```python
self.lm_head = LMHead(config.d_model, vocab_size)
if config.tie_weights:
    self.lm_head.proj.weight = self.embeddings.token_embedding.weight
```

A atribuição não copia valores: faz os dois módulos apontarem para **o mesmo objeto
`Parameter`**.

**O que significa compartilhar um `Parameter`, passo a passo** ([`model/gpt.py`](../model/gpt.py)):

1. **O que é um `Parameter`.** Um `nn.Parameter` é um tensor que o PyTorch marca como
   "aprendível". Quando é atribuído como atributo de um módulo, o módulo o registra, e ele passa
   a aparecer em `model.parameters()`.
2. **Linha 33.** `LMHead(...)` cria um `nn.Linear` com o seu próprio `Parameter` `(97, 128)`,
   inicializado com desvio 0,02.
3. **Linha 37.** Em Python, atribuir um atributo **não copia** o objeto: o nome passa a apontar
   para o objeto da direita. Depois dela, `lm_head.proj.weight` e
   `embeddings.token_embedding.weight` são **dois nomes para o mesmo tensor**.
4. **O `Parameter` da linha 33 é descartado.** Ninguém mais aponta para ele. A matriz que fica
   é a do embedding, também inicializada com $\mathcal{N}(0, 0{,}02^2)$
   ([`model/embeddings.py`](../model/embeddings.py), linha 24).
5. **Analogia:** duas etiquetas coladas na mesma caixa. Mexer na caixa pela etiqueta "embedding"
   muda o que se vê pela etiqueta "LM head".

Consequências:

- `model.parameters()` não repete tensores compartilhados, então a matriz é contada e otimizada
  **uma vez**;
- no backward, os gradientes das duas utilizações são **somados** no mesmo tensor;
- `model.lm_head.proj.weight is model.embeddings.token_embedding.weight` é `True`
  (`test_tied_head_shares_the_token_embedding_matrix`).

Conferido com um script Python à parte (mesma config do `tiny.yaml`):

| | com tying | sem tying |
|---|---:|---:|
| tensores em `model.parameters()` | 52 | 53 |
| nomes em `named_parameters()` com essas matrizes | só `embeddings.token_embedding.weight` | os dois |
| chaves em `state_dict()` com essas matrizes | as duas (`embeddings.token_embedding.weight` e `lm_head.proj.weight`) | as duas |

O `state_dict` (usado nos checkpoints, [15-checkpoints.md](15-checkpoints.md)) lista os dois
nomes, mas com tying eles apontam para o mesmo tensor.

A opção fica na config: `tie_weights: true` no `tiny.yaml`.

### 4.3 Parâmetros

```text
  sem tying: 832,512 parâmetros | mesma matriz: False
  com tying: 820,096 parâmetros | mesma matriz: True
```

A economia é $V \cdot d = 12{.}416$ parâmetros, **1,5%** do Mini-GPT. No GPT-2 small, com
$V = 50{.}257$, são 38,6 milhões, **31%** do modelo. É uma das razões de o tying ser padrão em
modelos com vocabulários grandes.

**Lendo as contas:**

- $V \cdot d = 97 \times 128 = 12{.}416$: o tamanho da matriz que deixa de existir.
- $12{.}416 / 820{.}096 = 1{,}5\%$.
- GPT-2 small: $50{.}257 \times 768 = 38{.}597{.}376 \approx 38{,}6$ milhões, de ~124,4 milhões
  ($\approx 31\%$).

### 4.4 Gradiente para todas as linhas

**Dois termos.** Um gradiente é **esparso** quando só algumas linhas de uma matriz recebem valor
diferente de zero. É **denso** quando todas recebem.

Em [03-embeddings.md](03-embeddings.md) vimos que o gradiente de um embedding é **esparso**: só
as linhas dos tokens presentes no batch recebem sinal. O gradiente do LM head é **denso**. Com
softmax e cross-entropy ([10-loss.md](10-loss.md)),

$$
\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i = y]
\quad\Rightarrow\quad
\frac{\partial \mathcal{L}}{\partial w_i} = \sum_{\text{posições}} (p_i - \mathbb{1}[i = y])\, h
$$

**Lendo a fórmula:**

- $\frac{\partial \mathcal{L}}{\partial z_i}$: quanto a loss muda se o logit $z_i$ subir um
  pouquinho. O treino move $z_i$ no sentido **contrário** ao sinal.
- $p_i$: a probabilidade atual do token $i$.
- $\mathbb{1}[i = y]$: vale 1 só para o token correto.
- $p_i - \mathbb{1}[i = y]$:
  - para o **correto**, $p_y - 1$, negativo: o treino **sobe** o logit dele;
  - para **os outros**, $p_i$, positivo: o treino **desce** o logit deles, mais forte para quem
    tem mais probabilidade;
  - a soma sobre os $V$ tokens é $\sum_i p_i - 1 = 0$.
- De onde vem: $\mathcal{L} = \text{logsumexp}(z) - z_y$ (seção 3.1). A derivada de
  $\text{logsumexp}(z)$ em relação a $z_i$ é $e^{z_i}/\sum_j e^{z_j} = p_i$. A de $z_y$ é 1 se
  $i = y$ e 0 se não.
- $\frac{\partial \mathcal{L}}{\partial w_i}$: como $z_i = w_i \cdot h$, mudar $w_i$ muda $z_i$ na
  proporção de $h$ (regra da cadeia). Por isso o gradiente de $z_i$ é multiplicado por $h$.
- $\sum_{\text{posições}}$: a linha $w_i$ é usada nas 4.096 posições do batch, e as contribuições
  se somam, cada uma com o seu $h$, $p$ e $y$. Como `F.cross_entropy` tira a **média**, o
  gradiente real ainda é dividido por $B \cdot T = 4.096$.

**Exemplo com 3 tokens.** Retome $h = (2, 1)$, $z = (2, 1, 0)$ e $p = (0{,}665;\ 0{,}245;\ 0{,}090)$
das seções 1 e 2, com alvo $y = 1$:

| token $i$ | $p_i$ | $\mathbb{1}[i = y]$ | $\partial \mathcal{L} / \partial z_i$ | $\partial \mathcal{L} / \partial w_i = (p_i - \mathbb{1}[i = y])\, h$ |
|---|---:|---:|---:|---|
| 0 | 0,665 | 0 | +0,665 | $(1{,}330;\ 0{,}665)$ |
| 1 (correto) | 0,245 | 1 | −0,755 | $(-1{,}510;\ -0{,}755)$ |
| 2 | 0,090 | 0 | +0,090 | $(0{,}180;\ 0{,}090)$ |

- Soma da coluna $\partial \mathcal{L} / \partial z_i$: $0{,}665 - 0{,}755 + 0{,}090 = 0$.
- Um passo de descida do gradiente com learning rate $\eta$ faz
  $w_i \leftarrow w_i - \eta\, \partial \mathcal{L} / \partial w_i$:
  - $w_1$ ganha $+\eta \cdot 0{,}755 \cdot h$: **aproxima-se** de $h$, e $z_1$ sobe;
  - $w_0$, o favorito errado, perde $\eta \cdot 0{,}665 \cdot h$: **afasta-se** bastante;
  - $w_2$ perde só $\eta \cdot 0{,}090 \cdot h$: já tinha pouca probabilidade.
- Os valores foram conferidos com `loss.backward()` num script à parte.

Como $p_i > 0$ para todo token, **toda linha recebe gradiente em todo passo**: o token alvo é
puxado para perto de $h$, e todos os outros são empurrados para longe. Com tying, esse
gradiente denso cai na matriz do embedding:

```text
caracteres distintos na entrada deste batch: 64 de 97

  sem tying: ... linhas do token_embedding com gradiente: 64 de 97
               norma do gradiente na linha de '+': 0.00e+00
  com tying: ... linhas do token_embedding com gradiente: 97 de 97
               norma do gradiente na linha de '+': 3.61e-02
```

O `'+'`, que aparece em **1** dos 82 batches de uma época ([03-embeddings.md](03-embeddings.md)),
passa a ser ajustado em todos eles.

- **Sem tying:** o token embedding só recebe gradiente pelo uso na entrada (esparso): 64 linhas.
- **Com tying:** a mesma matriz também recebe o gradiente do LM head (denso): 97 linhas, somadas
  às 64 da entrada.

### 4.5 Um efeito colateral: o modelo começa "prevendo repetição"

Com tying, o logit do **caractere atual** $x_t$ é $h \cdot E_{x_t}$. Mas o próprio $E_{x_t}$
foi somado ao residual stream na entrada. Na inicialização, o stream é quase só o embedding:

- $\|E_{x_t}\|^2 \approx 128 \cdot 0{,}02^2 = 0{,}051$;
- a LayerNorm final divide pelo desvio por coordenada do stream,
  $\sigma \approx 0{,}556/\sqrt{128} \approx 0{,}049$ ([08-gpt.md](08-gpt.md));
- logo $h \cdot E_{x_t} \approx 0{,}051 / 0{,}049 \approx 1$, enquanto os outros logits ficam
  em $0 \pm 0{,}25$.

**A conta do "bônus", passo a passo:**

1. **O stream antes da LayerNorm final** contém $E_{x_t} + P_t$ mais as escritas dos blocos. Na
   inicialização, todos esses vetores são aleatórios e quase ortogonais entre si
   ([03-embeddings.md](03-embeddings.md), seção 5).
2. **A LayerNorm final** (com $\gamma = 1$ e $\beta = 0$ na inicialização) subtrai a média, ≈ 0, e
   divide pelo desvio por coordenada $\sigma$. Então $h \approx \text{stream} / \sigma$.
3. **O produto escalar com $E_{x_t}$:**
   $h \cdot E_{x_t} \approx (E_{x_t} \cdot E_{x_t} + P_t \cdot E_{x_t} + \dots) / \sigma$. Os termos
   cruzados são quase zero, e sobra $\lVert E_{x_t} \rVert^2 / \sigma$.
4. **Numerador:** $\lVert E_{x_t} \rVert^2$ soma 128 quadrados de pesos com desvio 0,02:
   $128 \times 0{,}0004 = 0{,}0512$.
5. **Denominador:** a norma do stream no fim dos 4 blocos é 0,556 ([08-gpt.md](08-gpt.md)), e a norma é
   $\sigma \sqrt{d}$. Logo $\sigma = 0{,}556 / 11{,}31 = 0{,}049$.
6. **Resultado:** $0{,}0512 / 0{,}049 \approx 1{,}04$. Os outros logits não têm esse termo:
   $h \cdot E_i \approx 0 \pm \lVert h \rVert \cdot 0{,}02 = 0 \pm 0{,}23$.

Medido com um script à parte, no mesmo batch e modelo da seção 3.2: o logit do caractere atual
tem média **1,030** com tying (0,064 sem tying), e os demais logits têm média ≈ 0 e desvio 0,22.

**Em probabilidade.** Um logit 1 acima dos outros multiplica a chance por $e^1 = 2{,}718$
(seção 2.2). Com os outros 96 logits ~ $\mathcal{N}(0;\ 0{,}246^2)$, a estimativa é
$e^1 / (96\, e^{0{,}246^2/2} + e^1) = 2{,}718 / (96 \times 1{,}031 + 2{,}718) \approx 0{,}027$.
O medido é 0,028.

O experimento mostra exatamente isso: logit $+1{,}5$ para o `'s'` atual, e
$P(\text{caractere atual}) = 0{,}028$ em média, contra 0,010 sem tying.

O `'s'` da seção 1 teve $+1{,}5$, e não $+1{,}04$: é o bônus somado ao ruído de $\pm 0{,}25$ que
todo logit tem.

**Por que a loss fica abaixo de $\ln 97$.** No *Dom Casmurro*, o próximo caractere repete o
atual em **2,16%** das posições ("ss", "\n\n", "ll", "--", ".."), o dobro do que o uniforme
supõe (1,03%). O viés acidental do tying acerta um pouco mais do que erra: 4,563 contra 4,610
sem tying.

### 4.6 O que o treino mostra

Seção 4 do experimento, mesma seed, 150 passos:

```text
                passo 0   passo 50  passo 100  passo 150
  sem tying       4.558      2.508      2.379      2.314
  com tying       4.558      2.745      2.455      2.349
```

**Neste cenário, o modelo sem tying aprende mais rápido.** A diferença é grande no passo 50
(2,51 contra 2,75) e pequena no 150 (2,31 contra 2,35). Uma explicação plausível: com tying,
cada vetor $E_i$ precisa servir a dois papéis que, com caracteres, podem entrar em conflito. O
vetor de `'u'` como **entrada** deveria representar "um u"; como **saída**, deveria se alinhar
com os estados que vêm depois de `'q'`. Sem tying, cada papel tem sua própria matriz.

**A decisão.** O `tiny.yaml` mantém `tie_weights: true`, fiel ao GPT-2, com três ressalvas
explícitas:

- aqui o benefício em parâmetros é pequeno (1,5%), enquanto com vocabulários grandes é decisivo
  (31%);
- 150 passos não dizem como fica um treino completo;
- a comparação deve ser refeita com o training loop ([11-training.md](11-training.md) e
  [14-evaluation.md](14-evaluation.md)). Mudar é uma linha na config.

---

## 5. O que o modelo prevê depois de 150 passos

Seção 5 do experimento, com o modelo com tying. Para cada prompt, os 5 caracteres mais
prováveis como **próximo** e as suas probabilidades:

```text
  'qu'               -> 'e' 0.528  'a' 0.104  'i' 0.100  ' ' 0.049  'o' 0.029
  'Capit'            -> 'a' 0.226  'o' 0.224  'i' 0.156  'e' 0.141  'r' 0.077
  'respond'          -> 'e' 0.290  'o' 0.256  'a' 0.226  'i' 0.126  'u' 0.016
  'Dom Casmurr'      -> 'a' 0.351  'e' 0.175  ' ' 0.119  'o' 0.091  'i' 0.082
  'olhos de ressac'  -> 'o' 0.271  'a' 0.241  'i' 0.099  'h' 0.087  'e' 0.085
```

- Depois de `'qu'`, o modelo dá **53%** para `'e'`: já aprendeu uma regra forte da língua.
- Nos outros casos, as apostas são vogais plausíveis, mas não as palavras certas: `'Capit'`
  não leva a `'ú'` (Capitú), e `'Dom Casmurr'` não leva a `'o'`. Com 150 passos, o modelo usa
  pouco contexto, e as previsões refletem estatísticas locais de letras.

Transformar essas distribuições em texto (escolher um caractere e repetir) é o assunto de
[13-generation.md](13-generation.md).

---

## 6. O tamanho do tensor de logits

**A conta.** O tensor de logits guarda $B \cdot T \cdot V$ números. Em float32, cada número ocupa
4 bytes, e 1 MiB $= 2^{20}$ bytes.

```text
  Mini-GPT     [B=32, T=128, V=97]              397,312 floats =      1.5 MiB (float32)
  GPT-2 small  [B=8, T=1024, V=50257]       411,705,344 floats =  1,570.5 MiB (float32)
```

**Lendo as contas:**

- Mini-GPT: $32 \times 128 \times 97 = 397{.}312$ números; $\times 4$ bytes $= 1{,}5$ MiB.
- GPT-2 small: $8 \times 1{.}024 \times 50{.}257 = 411{.}705{.}344$ números; $\times 4$ bytes
  $= 1.570{,}5$ MiB $\approx 1{,}5$ GiB (1 GiB $= 1.024$ MiB).

No Mini-GPT, os logits são pequenos. Com um vocabulário de 50 mil tokens, o tensor de logits
de um único batch ocupa 1,5 GiB, maior que as ativações de um bloco inteiro, e a softmax
correspondente ocupa o mesmo tanto no backward. É por isso que implementações grandes calculam
a cross-entropy em pedaços ou com kernels que não materializam todas as probabilidades.

---

## 7. Código

Visão condensada:

```python
class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.proj = nn.Linear(d_model, vocab_size, bias=False)   # weight: [V, d]
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h):              # [B, T, d]
        return self.proj(h)            # [B, T, V]
```

No `GPT`:

```python
def hidden_states(self, ids):          # [B, T] -> [B, T, d]
    h = self.embeddings(ids)
    for block in self.blocks:
        h = block(h)
    return self.ln_final(h)

def forward(self, ids):                # [B, T] -> [B, T, V]
    return self.lm_head(self.hidden_states(ids))
```

### 7.1 `LMHead.__init__` ([`model/lm_head.py`](../model/lm_head.py), linhas 13–17)

**Objetivo:** criar a matriz $W$ e inicializá-la com valores pequenos.

1. **Linha 14, `super().__init__()`:** inicializa o `nn.Module`. Sem isso, os submódulos e
   parâmetros não seriam registrados.
2. **Linha 16, `nn.Linear(d_model, vocab_size, bias=False)`:** cria `weight` com shape
   `[vocab_size, d_model]` = `(97, 128)`. O PyTorch guarda `[saída, entrada]`
   ([00-notacao.md](00-notacao.md), seção 5.5). `bias=False`: nenhum vetor é somado aos logits.
3. **Linha 17, `nn.init.normal_(self.proj.weight, std=0.02)`:** troca a inicialização padrão do
   `nn.Linear` (uniforme, desvio ≈ 0,051) por $\mathcal{N}(0, 0{,}02^2)$. O `_` no final do nome
   indica que o tensor é modificado no lugar. O motivo está na seção 3: logits pequenos, loss
   inicial $\approx \ln V$.

### 7.2 `LMHead.forward` (linhas 19–21)

**Objetivo:** aplicar $z = Wh$ a cada posição.

- `self.proj(h)` calcula `h @ W.T`: para cada vetor na última dimensão de `h`, os 97 produtos
  escalares.
- As dimensões da frente são mantidas: `[B, T, d]` → `[B, T, V]`, e um vetor solto `[d]` →
  `[V]` (é o caso de `test_logits_are_dot_products_with_each_row`).
- **Não há softmax aqui.** Quem usa os logits decide: `F.cross_entropy` ([10-loss.md](10-loss.md))
  espera logits e faz o `log_softmax` estável por dentro; a geração
  ([13-generation.md](13-generation.md)) ajusta os logits antes da softmax.

### 7.3 `GPT.hidden_states` ([`model/gpt.py`](../model/gpt.py), linhas 39–44)

**Objetivo:** calcular o estado final $h$ de cada posição, sem o LM head.

1. **Linha 41:** `self.embeddings(ids)`: IDs `[B, T]` viram vetores `[B, T, d]` ([03-embeddings.md](03-embeddings.md)).
2. **Linhas 42–43:** cada um dos $L = 4$ blocos lê e atualiza o residual stream ([07-transformer-block.md](07-transformer-block.md)).
3. **Linha 44:** `self.ln_final(h)` padroniza a escala de cada posição ([08-gpt.md](08-gpt.md)).

Existe separado para **inspeção**: o experimento (seção 1) e os testes de [08-gpt.md](08-gpt.md)
olham o estado final diretamente.

### 7.4 `GPT.forward` (linhas 46–48)

**Objetivo:** devolver logits `[B, T, V]`, a saída que a loss e a geração usam.

- `self.lm_head(self.hidden_states(ids))`: o corpo do modelo seguido do LM head.
- `model(ids)` chama `forward` por baixo (via `nn.Module.__call__`).

### 7.5 O tying em `GPT.__init__` (linhas 32–37)

1. **Linha 32:** cria `ln_final`.
2. **Linha 33:** cria o `LMHead`, com a sua própria matriz.
3. **Linha 34:** `if config.tie_weights:` lê a opção da config (`tie_weights: true` no
   `tiny.yaml`; o padrão em [`config.py`](../config.py) também é `True`).
4. **Linha 37:** aponta `lm_head.proj.weight` para a matriz do token embedding (seção 4.2). A
   ordem importa: a atribuição precisa vir depois de criar o `LMHead`, senão a linha 33 criaria
   uma matriz nova por cima.

### 7.6 Onde ficam `softmax` e `log_softmax`

Não estão no modelo. Aparecem em quem consome os logits:

| onde | código | o que faz |
|---|---|---|
| experimento, linha 70 | `torch.softmax(z, dim=-1)` | probabilidades de uma posição (seção 2.1) |
| experimento, linha 77 | `F.log_softmax(z, dim=-1)` | log-probabilidades estáveis (seção 2.3) |
| experimento, linhas 108–110 | `log_probs.gather(-1, y.unsqueeze(-1))` e `-(...).mean()` | pega $\log p_y$ de cada posição e tira a média com sinal trocado: a cross-entropy $\mathcal{L}$ |
| `test_softmax_gives_a_distribution_per_position` | `torch.softmax(..., dim=-1)` | cada posição vira uma distribuição válida |
| testes e [10-loss.md](10-loss.md) | `F.cross_entropy(logits.view(-1, V), y.view(-1))` | `log_softmax` + escolha do alvo + média, numa chamada |

`gather(-1, índices)` escolhe, em cada posição, a entrada da última dimensão indicada pelo
índice: aqui, a coluna do token correto $y$.

### 7.7 `num_parameters` (linhas 50–52)

**Objetivo:** contar os números aprendíveis do modelo.

- `p.numel()`: número de elementos do tensor `p` ($97 \times 128 = 12{.}416$ para a matriz de
  tokens).
- `self.parameters()` devolve cada objeto `Parameter` **uma vez**, mesmo que ele esteja
  registrado com dois nomes. Com tying, a matriz compartilhada entra uma vez só na soma.
- Resultado: 820.096 com tying, 832.512 sem. A diferença é $V \cdot d$
  (`test_tying_saves_vocab_size_times_d_model_parameters`).

| decisão | escolha | motivo |
|---|---|---|
| bias no LM head | não | como no GPT-2; com tying, o embedding não tem bias para compartilhar |
| inicialização | $\mathcal{N}(0, 0{,}02^2)$ | logits pequenos: loss inicial $\approx \ln V$ |
| weight tying | configurável, `true` no `tiny.yaml` | fiel ao GPT-2; a comparação será refeita com o treino completo |
| API | `forward` → logits; `hidden_states` → estados finais | o uso normal (loss, geração) precisa de logits; a inspeção, dos estados |

A saída do `train.py` agora termina em logits:

```text
modelo:     GPT com 820,096 parâmetros (com weight tying)
forward:    ids (32, 128) -> logits (32, 128, 97)
Loss e training loop ainda não implementados — nada para treinar.
```

---

## 8. O que cada teste garante

[`tests/test_lm_head.py`](../tests/test_lm_head.py):

| teste | propriedade |
|---|---|
| `test_head_output_shape` | `[B, T, d]` → `[B, T, V]` |
| `test_logits_are_dot_products_with_each_row` | $z_i = h \cdot w_i$ |
| `test_head_has_no_bias_and_small_init` | sem bias; pesos com desvio 0,02 |
| `test_model_returns_logits_over_the_vocabulary` | `model(ids)` tem shape `[B, T, V]`, com e sem tying |
| `test_softmax_gives_a_distribution_per_position` | cada posição vira uma distribuição válida |
| `test_logits_are_causal` | trocar o futuro não muda os logits do passado |
| `test_tied_head_shares_the_token_embedding_matrix` | com tying, é o mesmo `Parameter` |
| `test_untied_head_has_its_own_matrix` | sem tying, matrizes diferentes |
| `test_tied_logits_are_similarities_with_token_embeddings` | com tying, logits = $h E^\top$ |
| `test_tying_saves_vocab_size_times_d_model_parameters` | economia de $V \cdot d$ |
| `test_initial_loss_is_close_to_a_uniform_guess` | loss inicial $\approx \ln V$, com e sem tying |
| `test_tying_sends_gradient_to_every_embedding_row` | com tying, todas as linhas recebem gradiente; sem tying, só as presentes |

Os testes usam um modelo pequeno ($V = 20$, $d = 32$, 2 blocos). A tolerância da loss inicial é
de 0,15 em torno de $\ln 20 \approx 3{,}0$. No teste de gradiente, só os tokens 0 a 4 aparecem na
entrada, e mesmo assim, com tying, as 20 linhas recebem gradiente.

Os testes de [08-gpt.md](08-gpt.md) ([`tests/test_gpt.py`](../tests/test_gpt.py)) passaram a usar
`hidden_states` onde verificam o estado final, e `model(ids)` onde verificam os logits.

---

## Resumo

1. O LM head é uma `Linear(d, V)` sem bias: o logit do token $i$ é o produto escalar entre o
   estado final e o vetor $w_i$.
2. A softmax transforma os logits de cada posição numa distribuição; logits são
   log-probabilidades a menos de uma constante, e `log_softmax` calcula o log de forma estável.
3. Com logits de desvio $s$, a loss inicial é $\approx \ln V + s^2/2$. Pesos pequenos fazem o
   modelo começar perto do chute uniforme ($\ln 97 = 4{,}575$).
4. Weight tying reusa a matriz do embedding: economiza $V \cdot d$ parâmetros e dá gradiente a
   todas as linhas em todo passo. Como efeito colateral, o modelo começa favorecendo a repetição
   do caractere atual.
5. Neste cenário de 150 passos, o modelo sem tying aprendeu um pouco mais rápido. O projeto
   mantém o tying (como o GPT-2) e refaz a comparação com o treino completo.

---

## Para checar o entendimento

1. Se todos os logits de uma posição forem somados a 10, o que acontece com as probabilidades?
   E com a loss?
2. Deduza $\mathcal{L} \approx \ln V + s^2/2$. Qual seria a loss inicial com $V = 50{.}257$ e
   $s = 0{,}23$?
3. Por que `log(softmax(z))` pode dar $-\infty$ quando `log_softmax(z)` não dá?
4. Por que a atribuição `lm_head.proj.weight = token_embedding.weight` não duplica parâmetros no
   otimizador?
5. Um token que nunca aparece na **entrada** de um batch recebe gradiente sem tying? E com tying?
   De onde vem esse gradiente?
6. Explique por que, com tying, o logit do caractere atual é ~1 na inicialização. O que
   aconteceria com esse efeito se a LayerNorm final não existisse?
7. No GPT-2 small, quantos parâmetros o tying economiza, e que fração do modelo isso representa?
8. Dê um exemplo, com caracteres do português, de por que os papéis "vetor de entrada" e "vetor
   de saída" de um mesmo token poderiam entrar em conflito.
9. Com $z = (2, 1, 0)$ e alvo $y = 0$, calcule $\partial \mathcal{L} / \partial z_i$ para os três
   tokens. Qual logit o treino sobe? Qual desce mais?
10. Pela seção 2.2, se o logit de `'e'` está 2 acima do de `'a'`, quantas vezes `'e'` é mais
    provável que `'a'`?
11. Com a inicialização padrão do `nn.Linear` ($s = 0{,}583$), calcule $e^{s^2/2}$ e a loss inicial
    prevista. Confira com a tabela da seção 3.2.
