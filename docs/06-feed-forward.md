# Feed-Forward Network

> **O que o modelo faz com a informação que a attention trouxe?**

Código: [`model/feed_forward.py`](../model/feed_forward.py). Testes:
[`tests/test_feed_forward.py`](../tests/test_feed_forward.py). Experimento:
`uv run python -m experiments.e06_feed_forward`. Visão geral: [architecture.md](architecture.md).
Guia de notação: [00-notacao.md](00-notacao.md). Pré-requisitos:
[04-attention.md](04-attention.md) e [05-multi-head-attention.md](05-multi-head-attention.md).
Próximo documento: [07-transformer-block.md](07-transformer-block.md).

Notação: $d$ = `d_model`, $d_{ff}$ = dimensão interna ($4d$ = 512 no `tiny.yaml`),
$\sigma$ = função de ativação (GELU). A tabela completa está na seção "Notação deste documento".

---

## Para que serve

### O problema

A attention ([04-attention.md](04-attention.md) e [05-multi-head-attention.md](05-multi-head-attention.md)) deixa cada posição **olhar para as posições anteriores** e trazer
informação delas. Mas ela só **mistura** o que encontrou: faz médias ponderadas. Ela não
"pensa" sobre o resultado.

O feed-forward é a etapa que **processa** o que chegou em cada posição. Ele combina as
informações de forma não linear, detecta padrões e escreve algo novo no vetor.

### Uma analogia

Pense numa reunião de equipe:

- **attention = a conversa.** Cada pessoa ouve as outras e anota o que é relevante para ela.
- **feed-forward = o trabalho individual.** Depois da reunião, cada pessoa volta para a
  própria mesa e trabalha **sozinha** sobre as anotações. Todas seguem **o mesmo manual**.

O "manual" é uma lista de 512 regras do tipo **"se vir este padrão, acrescente esta
informação"**. Cada regra é um **neurônio** (seção 1.3).

### O que entra e o que sai

Valores do `tiny.yaml` ($B = 32$, $T = 128$, $d = 128$, $d_{ff} = 4d = 512$):

| etapa | shape | o que é |
|---|---|---|
| entrada `x` | `[32, 128, 128]` | um vetor de 128 números por posição, já com informação trazida pela attention |
| depois de `up_proj` | `[32, 128, 512]` | **camada escondida**: 512 números por posição, um por neurônio |
| depois da GELU | `[32, 128, 512]` | as mesmas 512 posições, agora "dobradas" pela não-linearidade |
| depois de `down_proj` | `[32, 128, 128]` | de volta à largura $d$ |
| saída (depois do dropout) | `[32, 128, 128]` | o que será **somado** ao residual stream (ver [07-transformer-block.md](07-transformer-block.md)) |

**Camada escondida** é a camada intermediária, que não é nem a entrada nem a saída. Aqui ela
tem $32 \times 128 \times 512 = 2.097.152$ números por batch.

O **residual stream** é o vetor de cada posição que atravessa o modelo inteiro. Cada
attention e cada feed-forward lê esse vetor e **soma** a sua contribuição a ele.

### Onde fica no GPT

```text
texto (Dom Casmurro)
  │  tokenizer: cada caractere vira um ID
  ▼
batches de IDs [32, 128]
  │  embeddings: vetor do token + vetor da posição
  ▼
h [32, 128, 128]
  │  ┌──── bloco Transformer, repetido L = 4 vezes ────────────────────┐
  │  │  h = h + attention(LayerNorm(h))                                │
  │  │  h = h + feed-forward(LayerNorm(h))    ← este documento         │
  │  └──────────────────────────────────────────────────────────────────┘
  ▼
LayerNorm final → LM head
  ▼
logits [32, 128, 97] → loss
```

### O que daria errado sem ele

- **Sem o feed-forward:** o bloco só teria attention. O conteúdo de cada vetor seria
  transformado apenas por médias ponderadas e projeções lineares. O modelo teria muito menos
  capacidade de combinar informações e perderia cerca de 2/3 dos parâmetros de cada bloco
  (seção 6.1), onde fica a maior parte do "conhecimento".
- **Com o feed-forward, mas sem a ativação:** as duas camadas lineares colapsariam numa só
  (seção 2.1). A rede não conseguiria resolver nem o XOR (seção 2.2).

---

## Notação deste documento

Símbolos gerais (soma, produto escalar, transposta, normal, derivadas) estão explicados com
exemplos em [00-notacao.md](00-notacao.md). Aqui está o sentido **neste** documento.

### Tamanhos

| símbolo | como se lê | o que significa | no Mini-GPT | no código |
|---|---|---|---|---|
| $B$ | "bê" | tamanho do batch | 32 | `batch_size` |
| $T$ | "tê" | tamanho do contexto (posições por sequência). **Não confundir** com $^\top$ (transposta) | 128 | `context_length`, `seq_len` |
| $d$ | "dê" | largura do residual stream | 128 | `d_model` |
| $d_{ff}$ | "dê éfe éfe" | número de neurônios da camada escondida | $4d = 512$ | `d_ff`, `config.model.d_ff` |
| $h$ | "agá" | número de heads da attention (só na seção 6.2) | 4 | `num_heads` |
| $d_h$ | "dê agá" | dimensão de cada head, $d / h$ | 32 | `head_dim` |
| $L$ | "ele" | número de blocos Transformer | 4 | `num_layers` |
| $p$ | "pê" | probabilidade de dropout (seção 8.2) | 0,1 | `dropout` |

### O feed-forward

| símbolo | como se lê | o que significa | no Mini-GPT | no código |
|---|---|---|---|---|
| $x$ | "xis" | vetor de **uma** posição que entra no feed-forward. Aqui não é a sequência de IDs (ver [02-dataset.md](02-dataset.md)) | $x \in \mathbb{R}^{128}$; no batch, `[32, 128, 128]` | `x` |
| $\mathbb{R}^d$ | "erre dê" | conjunto dos vetores com $d$ números reais | $\mathbb{R}^{128}$ | shape `(128,)` |
| $W_1$ | "dáblio um" | matriz da primeira camada (expande) | `[512, 128]` | `up_proj.weight` |
| $b_1$ | "bê um" | bias da primeira camada: um limiar por neurônio | `[512]` | `up_proj.bias` |
| $W_2$ | "dáblio dois" | matriz da segunda camada (contrai) | `[128, 512]` | `down_proj.weight` |
| $b_2$ | "bê dois" | bias da segunda camada: deslocamento fixo da saída | `[128]` | `down_proj.bias` |
| $\sigma$ | "sigma" | **função de ativação**, aplicada a cada número separadamente. **Aqui não é desvio padrão** (o sentido usual em [00-notacao.md](00-notacao.md)) | GELU | `activation`, `nn.GELU()` |
| $\text{FFN}(x)$ | "éfe éfe ene de xis" | saída do feed-forward para o vetor $x$ | $\mathbb{R}^{128}$ | `ff(x)` |
| $i$ | "i" | índice de um neurônio, de 1 a $d_{ff}$ (no código, de 0 a 511) | 1 … 512 | `i` |
| $k_i$ | "kê i" | **chave** do neurônio $i$ = linha $i$ de $W_1$: o padrão que ele procura. **Não é** a key $k_j$ da attention | `[128]` | `up_proj.weight[i]` |
| $b_i$ | "bê i" | limiar do neurônio $i$ = elemento $i$ de $b_1$ (um número) | escalar | `up_proj.bias[i]` |
| $v_i$ | "vê i" | **valor** do neurônio $i$ = coluna $i$ de $W_2$: o vetor que ele escreve. **Não é** $V$ (`vocab_size`) nem o value $v_j$ da attention | `[128]` | `down_proj.weight[:, i]` |
| $a_i$ | "a i" | **ativação** do neurônio $i$: $\sigma(k_i \cdot x + b_i)$, um número | escalar | `hidden[..., i]` |
| $k_i \cdot x + b_i$ | — | **pré-ativação** do neurônio $i$: o número antes da GELU | escalar | `up_proj(x)[..., i]` |
| $W$, $b$ | "dáblio", "bê" | na seção 2.1: matriz e bias da camada única equivalente, sem ativação. Na seção 7, $b$ é um bias hipotético da attention | `[128, 128]`, `[128]` | `W2 @ W1`, `W2 @ b1 + b2` |

### Operadores e funções

| símbolo | como se lê | o que significa | exemplo | no código |
|---|---|---|---|---|
| $a \cdot b$ (vetores) | "a escalar b" | produto escalar: multiplica elemento a elemento e soma | $(1, 1) \cdot (1, 2) = 3$ | `(a * b).sum()` |
| $Wx$ | "dáblio xis" | multiplicação de matriz por vetor | seção 1.2 | `x @ W.T` |
| $^\top$ | "transposta" | troca linhas por colunas | `[512, 128]` → `[128, 512]` | `.T` |
| $\odot$ | "elemento a elemento" | multiplica dois vetores posição por posição | $(1, 2) \odot (3, 4) = (3, 8)$ | `a * b` |
| $\sum_{i=1}^{n}$ | "soma para i de 1 a n" | soma de vários termos | $\sum_{i=1}^{3} i = 6$ | `sum(...)` |
| $\max(0, x)$ | "máximo entre zero e xis" | o maior dos dois números | $\max(0, -2) = 0$ | `torch.clamp(x, min=0)` |
| $\text{ReLU}(x)$ | "rélu" | $\max(0, x)$: zera negativos | $\text{ReLU}(-1) = 0$ | `F.relu` |
| $\text{GELU}(x)$ | "guélu" | $x \cdot \Phi(x)$: a ativação usada aqui | $\text{GELU}(1) = 0{,}841$ | `F.gelu`, `nn.GELU()` |
| $\text{GELU}'$, $\text{ReLU}'$ | "linha" | derivada: quanto a função muda quando $x$ muda um pouco | $\text{GELU}'(0) = 0{,}5$ | `torch.autograd.grad` |
| $\mathbb{1}[\cdot]$ | "indicadora" | 1 se a condição é verdadeira, 0 se não | $\mathbb{1}[2 > 0] = 1$ | `(x > 0).float()` |
| $Z \sim \mathcal{N}(0, 1)$ | "Z segue uma normal padrão" | número aleatório da curva de sino com média 0 e desvio 1 | — | `torch.randn(())` |
| $P(Z \le x)$ | "probabilidade de Z ser menor ou igual a xis" | chance de o sorteio de $Z$ cair abaixo de $x$ | $P(Z \le 0) = 0{,}5$ | — |
| $\Phi(x)$ | "fi maiúsculo de xis" | **CDF** da normal padrão: $P(Z \le x)$ | $\Phi(1) = 0{,}841$ | `0.5 * (1 + torch.erf(x / math.sqrt(2)))` |
| $\varphi(x)$ | "fi minúsculo de xis" | **densidade** da normal padrão: altura da curva de sino em $x$ | $\varphi(0) = 0{,}399$ | `torch.exp(-x**2 / 2) / math.sqrt(2 * math.pi)` |
| $\text{erf}(z)$ | "função erro" | função da estatística ligada à normal; usada para calcular $\Phi$ | $\text{erf}(0{,}7071) = 0{,}6827$ | `torch.erf`, `math.erf` |
| $\int_a^b f(t)\,dt$ | "integral de a até b" | área sob a curva de $f$ entre $a$ e $b$; $t$ é só a variável que percorre o intervalo | — | — |
| $e$ | "e" | número de Euler, ≈ 2,718 | $e^{-0{,}5} = 0{,}607$ | `math.e`, `torch.exp` |
| $\pi$ | "pi" | ≈ 3,14159 | $\sqrt{2\pi} = 2{,}507$ | `math.pi` |
| $\tanh$ | "tangente hiperbólica" | função em forma de S entre −1 e 1 (aproximação da GELU) | — | `F.gelu(x, approximate="tanh")` |
| $m \sim \text{Bernoulli}(q)$ | "m segue uma Bernoulli de q" | sorteio que dá 1 com probabilidade $q$ e 0 caso contrário | moeda: $q = 0{,}5$ | `torch.bernoulli` |
| $r_j$ | "erre j" | sorteio do dropout para a coordenada $j$: 1 = mantém, 0 = zera | — | interno ao `nn.Dropout` |
| $\mathbb{E}[\cdot]$ | "esperança" | média de um valor aleatório | $\mathbb{E}[\text{dado}] = 3{,}5$ | — |
| $\text{sigmoid}(z)$ | "sigmoide" | $1/(1 + e^{-z})$: transforma um logit em probabilidade (XOR, seção 2.2) | $\text{sigmoid}(0) = 0{,}5$ | `torch.sigmoid` |
| $\ln$ | "logaritmo natural" | inverso da exponencial | $\ln 2 = 0{,}693$ | `math.log` |
| $\partial$ | "d parcial" | derivada em relação a uma variável (Jacobiana, seção 4.1) | — | `.backward()` |
| $O(\cdot)$ | "ó de" | ordem de crescimento do custo | $O(T^2)$: dobrar $T$ quadruplica | — |
| $\approx$, $\in$ | "aproximadamente", "pertence a" | — | $\pi \approx 3{,}14$ | — |

### Símbolos emprestados de outros documentos

| símbolo | como se lê | o que significa | no Mini-GPT | no código |
|---|---|---|---|---|
| $q_i$, $k_j$, $v_j$ | "quê i", "kê j", "vê j" | query da posição $i$, key e value da posição $j$ (attention, ver [04-attention.md](04-attention.md)) | `[32]` por head | `q`, `k`, `v` em `attention.py` |
| $w_{ij}$ | "dáblio i j" | peso de attention: quanto a posição $i$ olha para $j$ | somam 1 por linha | `weights` |
| $o_i$ | "ó i" | saída da attention na posição $i$ | `[128]` | saída de `MultiHeadAttention` |
| $W_O$ | "dáblio ó" | projeção de saída da attention | `[128, 128]` | `out_proj.weight` |
| $\hat{x}$ | "xis chapéu" | vetor normalizado pela LayerNorm (média 0, variância 1) | `[128]` | — |
| $\gamma$, $\beta$ | "gama", "beta" | escala e deslocamento aprendidos da LayerNorm (ver [07-transformer-block.md](07-transformer-block.md)) | `[128]` cada | `ln_2` |

---

## 0. Onde estamos

A multi-head attention ([04-attention.md](04-attention.md) e [05-multi-head-attention.md](05-multi-head-attention.md)) **move informação entre posições**. Mas olhe com
cuidado para o que ela faz com o conteúdo. A fórmula abaixo calcula a saída da posição $i$
(simplificando para uma head):

$$
o_i = \Big(\sum_{j \le i} w_{ij}\, v_j\Big) W_O
$$

**Lendo a fórmula:**

- $v_j$: o value da posição $j$ (o "conteúdo" que ela oferece).
- $w_{ij}$: o peso que a posição $i$ dá à posição $j$; vem da softmax sobre $q_i \cdot k_j$.
- $\sum_{j \le i}$: soma sobre a própria posição e as anteriores (máscara causal).
- $W_O$: uma matriz fixa, aplicada ao resultado.

Dados os pesos $w_{ij}$, a saída é uma **combinação linear** dos values seguida de outra
projeção linear. A não-linearidade da attention está só em **escolher os pesos** (a softmax
sobre $q \cdot k$). A transformação do conteúdo em si é linear.

**Linear** quer dizer: só somas de termos do tipo "número × vetor", mais multiplicações por
matrizes. **Não-linearidade** é qualquer operação que foge disso, como zerar negativos ou
multiplicar um valor por uma função dele mesmo.

Falta uma etapa que **processe** o que chegou: combine as informações de forma não linear,
detecte padrões e produza algo novo. Esse é o papel do feed-forward. Uma forma comum de
resumir a divisão de trabalho (Elhage et al., 2021):

| componente | papel | mistura posições? | não-linear no conteúdo? |
|---|---|---|---|
| attention | **comunicação**: traz informação de outras posições | sim | não |
| feed-forward | **computação**: processa a informação de cada posição | não | sim |

---

## 1. A arquitetura

### 1.1 A fórmula

A fórmula abaixo descreve tudo o que o feed-forward calcula para o vetor $x \in \mathbb{R}^d$
de **uma** posição: expandir, aplicar a ativação, contrair.

$$
\text{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2
$$

**Lendo a fórmula** (de dentro para fora):

1. $W_1 x$: multiplica a matriz $W_1$ (`[512, 128]`) pelo vetor $x$ (128 números). O
   resultado tem **512** números.
2. $+ b_1$: soma um número diferente em cada uma das 512 posições (o limiar de cada neurônio).
3. $\sigma(\cdot)$: aplica a GELU em cada um dos 512 números, **separadamente**.
4. $W_2\,(\cdot)$: multiplica pela matriz $W_2$ (`[128, 512]`). Volta a ter **128** números.
5. $+ b_2$: soma um deslocamento fixo de 128 números.

| termo | shape | nome no código | papel |
|---|---|---|---|
| $W_1$ | $d_{ff} \times d$ | `up_proj.weight` | expande: $d \to d_{ff}$ |
| $b_1$ | $d_{ff}$ | `up_proj.bias` | limiar de cada neurônio |
| $\sigma$ | — | `activation` (GELU) | não-linearidade, elemento a elemento |
| $W_2$ | $d \times d_{ff}$ | `down_proj.weight` | contrai: $d_{ff} \to d$ |
| $b_2$ | $d$ | `down_proj.bias` | deslocamento da saída |

**Função de ativação** é a função não linear aplicada depois de uma camada linear. Ela decide
quanto cada neurônio "acende".

No código, a fórmula é escrita com vetores como linhas: $W_1 x$ vira `x @ W1.T`. É a mesma
conta ([00-notacao.md](00-notacao.md), seção 5.5).

### 1.2 Um feed-forward minúsculo, calculado à mão

Para ver cada passo com números, use $d = 2$ e $d_{ff} = 3$:

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & -1 \end{bmatrix}, \quad
b_1 = \begin{bmatrix} 0 \\ 0 \\ 0{,}5 \end{bmatrix}, \quad
W_2 = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \end{bmatrix}, \quad
b_2 = \begin{bmatrix} 0 \\ 0{,}5 \end{bmatrix}
$$

**Passo 1 — expandir ($W_1 x + b_1$).** Cada linha de $W_1$ faz um produto escalar com $x$:

| neurônio | linha de $W_1$ | produto escalar com $x = (1, 2)$ | $+ b_1$ | pré-ativação |
|---|---|---|---|---|
| 1 | $(1, 0)$ | $1 \cdot 1 + 0 \cdot 2 = 1$ | $+ 0$ | $1$ |
| 2 | $(0, 1)$ | $0 \cdot 1 + 1 \cdot 2 = 2$ | $+ 0$ | $2$ |
| 3 | $(1, -1)$ | $1 \cdot 1 - 1 \cdot 2 = -1$ | $+ 0{,}5$ | $-0{,}5$ |

**Passo 2 — ativar ($\sigma$ = GELU).** Valores da tabela da seção 3.2:

$$
\text{GELU}(1) = 0{,}8413, \quad \text{GELU}(2) = 1{,}9545, \quad \text{GELU}(-0{,}5) = -0{,}1543
$$

**Passo 3 — contrair ($W_2 \cdot$ ativações $+ b_2$).** Cada linha de $W_2$ faz um produto
escalar com as 3 ativações:

- coordenada 1: $1 \cdot 0{,}8413 + 0 \cdot 1{,}9545 + 2 \cdot (-0{,}1543) + 0 = 0{,}5328$
- coordenada 2: $0 \cdot 0{,}8413 + 1 \cdot 1{,}9545 - 1 \cdot (-0{,}1543) + 0{,}5 = 2{,}6088$

$$
\text{FFN}(x) = (0{,}5328;\ 2{,}6088)
$$

Conferido com o próprio `FeedForward(2, 3, 0.0)`, copiando esses pesos (cálculo feito para
este documento, não faz parte do experimento). Com ReLU no lugar da GELU, o neurônio 3 seria
zerado e a saída seria $(1;\ 2{,}5)$.

### 1.3 O que é um "neurônio" aqui

Um **neurônio** é **uma coordenada da camada escondida**. Concretamente, o neurônio $i$ é
formado por:

| peça | o que é | tamanho no `tiny.yaml` | no código |
|---|---|---|---|
| chave $k_i$ | linha $i$ de $W_1$: o padrão que ele procura | 128 números | `up_proj.weight[i]` |
| limiar $b_i$ | elemento $i$ de $b_1$ | 1 número | `up_proj.bias[i]` |
| ativação | GELU aplicada a $k_i \cdot x + b_i$ | — | `activation` |
| valor $v_i$ | coluna $i$ de $W_2$: o vetor que ele escreve na saída | 128 números | `down_proj.weight[:, i]` |

O funcionamento de um neurônio, em três passos:

1. **mede** o alinhamento entre a entrada e sua chave: $k_i \cdot x + b_i$ (um número);
2. **decide** quanto acender: $a_i = \text{GELU}(k_i \cdot x + b_i)$;
3. **escreve** o vetor $v_i$ na saída, multiplicado por $a_i$.

No exemplo da seção 1.2, o neurônio 3 tem chave $(1, -1)$, limiar $0{,}5$ e valor $(2, -1)$:
ele acende quando a primeira coordenada de $x$ supera a segunda em mais de 0,5. Para
$x = (1, 2)$ ficou quase desligado ($a_3 = -0{,}1543$).

Cada neurônio tem $128 + 1 + 128 = 257$ parâmetros. São 512 neurônios, mais $b_2$, que é
compartilhado: $512 \times 257 + 128 = 131.712$, o mesmo total da seção 6.1.

### 1.4 Os shapes no `tiny.yaml`

```text
x [B, T, 128] ──up_proj──► [B, T, 512] ──GELU──► [B, T, 512] ──down_proj──► [B, T, 128] ──dropout──►
```

Cada uma das 512 coordenadas intermediárias é um **neurônio**. O formato "expande, aplica a
não-linearidade, contrai" é o mesmo do Transformer original (Vaswani et al., 2017) e do
GPT-2.

---

## 2. Por que a não-linearidade é indispensável

### 2.1 Camadas lineares empilhadas colapsam

A conta abaixo mostra o que sobra do feed-forward se tirarmos $\sigma$: ele vira uma única
camada linear.

$$
W_2 (W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{W}\, x + \underbrace{(W_2 b_1 + b_2)}_{b}
$$

**Lendo a fórmula, passo a passo:**

1. **Distribuir** $W_2$ sobre a soma: $W_2 (W_1 x + b_1) = W_2 W_1 x + W_2 b_1$.
2. **Agrupar:** a multiplicação de matrizes é associativa, então $W_2 W_1 x = (W_2 W_1)\, x$.
3. **Dar nomes:** $W = W_2 W_1$ é uma matriz fixa; $b = W_2 b_1 + b_2$ é um vetor fixo.
4. **Shapes:** $W_2 W_1$ é `[128, 512] @ [512, 128]` → `[128, 128]`; $W_2 b_1$ é
   `[128, 512] @ [512]` → `[128]`.

Duas camadas lineares são **exatamente** uma camada linear, com $W = W_2 W_1 \in \mathbb{R}^{d \times d}$.
Expandir para $d_{ff}$ dimensões e contrair de volta não acrescenta nada; o mesmo vale para
100 camadas lineares seguidas.

**Exemplo com o feed-forward minúsculo (seção 1.2), sem GELU:**

$$
W = W_2 W_1 = \begin{bmatrix} 3 & -2 \\ -1 & 2 \end{bmatrix}, \qquad
b = W_2 b_1 + b_2 = \begin{bmatrix} 1 \\ -0{,}5 \end{bmatrix} + \begin{bmatrix} 0 \\ 0{,}5 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

- Por dois caminhos: pré-ativações $(1, 2, -0{,}5)$ sem GELU; $W_2$ vezes isso, mais $b_2$, dá
  $(1 - 1;\ 2 + 0{,}5 + 0{,}5) = (0;\ 3)$.
- Por um caminho: $Wx + b = (3 - 4 + 1;\ -1 + 4 + 0) = (0;\ 3)$. Mesmo resultado.

O experimento confirma com $d = 8$, $d_{ff} = 32$:

```text
down_proj(up_proj(x)) sem GELU:
  igual a uma única camada x·(W2·W1)ᵀ + (W2·b1 + b2): True
  W2·W1 tem shape (8, 8): expandir para 32 dims não acrescentou nada
```

Uma função linear só consegue rotacionar, escalar, projetar e deslocar. Ela **não consegue
dobrar o espaço**: tudo que é separável depois dela já era separável por um hiperplano antes.

**Hiperplano** é a "fronteira reta" que divide o espaço em dois lados: o conjunto dos pontos
com $k \cdot x + b = 0$. Em 2 dimensões é uma reta; em 3, um plano; em 128, um hiperplano.

### 2.2 O exemplo clássico: XOR

XOR devolve 1 quando exatamente uma das entradas é 1. Os pontos `01` e `10` (saída 1) e
`00` e `11` (saída 0) **não são separáveis por uma reta**.

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```text
 x2
  1 │  01 → 1        11 → 0
    │
    │
  0 │  00 → 0        10 → 1
    └──────────────────────── x1
       0              1
```

Os pontos com saída 1 estão numa diagonal; os com saída 0, na outra. As diagonais se cruzam,
então nenhuma reta deixa `01` e `10` de um lado e `00` e `11` do outro.

**Por que nenhuma função linear resolve.** Uma função linear de duas entradas é
$f(x_1, x_2) = w_1 x_1 + w_2 x_2 + c$. Some os valores nos cantos:

$$
f(0,0) + f(1,1) = c + (w_1 + w_2 + c) = (w_2 + c) + (w_1 + c) = f(0,1) + f(1,0)
$$

**Lendo a fórmula:** $w_1$, $w_2$ e $c$ são os parâmetros da função. A soma nos pontos de
saída 0 é **sempre igual** à soma nos pontos de saída 1. Para acertar, precisaríamos de
$f(0,1) > 0$ e $f(1,0) > 0$ (lado "1") e de $f(0,0) < 0$ e $f(1,1) < 0$ (lado "0"). Aí o lado
direito seria positivo e o esquerdo negativo, e eles não poderiam ser iguais.

O experimento treina o mesmo `FeedForward` (com $d = 2$, $d_{ff} = 16$) com e sem ativação. A
primeira coordenada da saída é um **logit** $z$, e a probabilidade de saída 1 é
$\text{sigmoid}(z) = 1/(1 + e^{-z})$:

```text
  sem ativação  loss 0.693 | P(saída = 1): 00→0.50  01→0.50  10→0.50  11→0.50
  com GELU      loss 0.000 | P(saída = 1): 00→0.00  01→1.00  10→1.00  11→0.00
  (responder 0.5 para tudo dá loss ln 2 = 0.693)
```

Sem ativação, o melhor que o modelo consegue é desistir e responder 0,5 para tudo. Com
GELU, acerta os quatro casos.

**De onde vem o 0,693.** A loss usada é a *binary cross-entropy*, a média sobre os 4 pontos de
$-\ln(\text{probabilidade dada à resposta certa})$. Respondendo 0,5 em todos, cada termo vale
$-\ln 0{,}5 = \ln 2 = 0{,}693$, e a média também.

**Uma solução feita à mão, com dois neurônios.** Com ReLU (para a aritmética ficar limpa):

$$
\text{XOR}(x_1, x_2) = 1 \cdot \text{ReLU}(x_1 + x_2) - 2 \cdot \text{ReLU}(x_1 + x_2 - 1)
$$

**Lendo a fórmula:** o neurônio 1 tem chave $(1, 1)$, limiar $0$ e valor $1$; o neurônio 2 tem
chave $(1, 1)$, limiar $-1$ e valor $-2$. Na tabela, $n_1$ e $n_2$ são as saídas (ativações)
dos neurônios 1 e 2.

| entrada | $x_1 + x_2$ | neurônio 1 | neurônio 2 | $1 \cdot n_1 - 2 \cdot n_2$ |
|---|---|---|---|---|
| 00 | 0 | 0 | 0 | 0 |
| 01 | 1 | 1 | 0 | 1 |
| 10 | 1 | 1 | 0 | 1 |
| 11 | 2 | 2 | 1 | 0 |

O neurônio 2 só "acende" em `11` e cancela o excesso. A dobra da ReLU é o que permite isso.
O treino com GELU do experimento encontra a sua própria solução, com 16 neurônios.

### 2.3 O resultado geral

O **teorema da aproximação universal** (Cybenko, 1989; Hornik et al., 1989) garante que uma
rede com **uma** camada escondida e ativação não linear aproxima qualquer função contínua
num domínio limitado, com precisão arbitrária, desde que tenha neurônios suficientes. É
exatamente a forma do feed-forward. O teorema não diz quantos neurônios são necessários nem
como encontrá-los; isso fica com o treino e com o $d_{ff}$ escolhido.

Em palavras simples: cada neurônio contribui com uma "dobra" num lugar diferente, e somando
dobras suficientes dá para imitar qualquer curva razoável.

---

## 3. GELU

### 3.1 Definição

A GELU (*Gaussian Error Linear Unit*) é a ativação do Mini-GPT. A fórmula abaixo diz como ela
calcula a saída: multiplica a entrada por uma "porcentagem" que depende da própria entrada.

$$
\text{GELU}(x) = x \cdot \Phi(x), \qquad \Phi(x) = P(Z \le x) = \tfrac{1}{2}\left[1 + \text{erf}\!\left(\tfrac{x}{\sqrt{2}}\right)\right], \quad Z \sim \mathcal{N}(0, 1)
$$

**Lendo a fórmula:**

- $x$: aqui é **um número** (uma pré-ativação), não o vetor inteiro. A GELU age em cada
  coordenada separadamente.
- $Z \sim \mathcal{N}(0, 1)$: um número sorteado da normal padrão (média 0, desvio 1).
- $\Phi(x) = P(Z \le x)$: a probabilidade de esse sorteio cair abaixo de $x$. Fica entre 0 e 1.
- $\text{erf}$: a **função erro**, uma função pronta das bibliotecas (`torch.erf`) que permite
  calcular $\Phi$. A parte $\tfrac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$ é só a forma de escrever
  $\Phi$ com ela.
- $x \cdot \Phi(x)$: a entrada vezes a sua "porcentagem".

$\Phi$ é a **CDF** (*cumulative distribution function*, função de distribuição acumulada) da
normal padrão: a área sob a curva de sino à esquerda de $x$. Equivalentemente:

$$
\Phi(x) = \int_{-\infty}^{x} \varphi(t)\, dt, \qquad \text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2}\, dt
$$

**Lendo a fórmula:** $\varphi$ é a altura da curva de sino (seção 3.2); $\int_{-\infty}^{x}$ é
a área dessa curva desde o extremo esquerdo até $x$; $t$ é só a variável que percorre o
intervalo. Não é preciso calcular integrais à mão: `torch.erf` faz isso.

**Exemplos** (confira na tabela da seção 3.2):

| $x$ | $\text{erf}(x/\sqrt{2})$ | $\Phi(x)$ | $\text{GELU}(x) = x \cdot \Phi(x)$ |
|---|---|---|---|
| $1$ | $\text{erf}(0{,}7071) = 0{,}6827$ | $\tfrac{1}{2}(1 + 0{,}6827) = 0{,}8413$ | $1 \cdot 0{,}8413 = 0{,}8413$ |
| $0$ | $0$ | $0{,}5$ | $0 \cdot 0{,}5 = 0$ |
| $-1$ | $-0{,}6827$ | $\tfrac{1}{2}(1 - 0{,}6827) = 0{,}1587$ | $-1 \cdot 0{,}1587 = -0{,}1587$ |

A GELU multiplica a entrada por "quanto ela é grande, em probabilidade": valores muito
positivos passam quase intactos ($\Phi \approx 1$), valores muito negativos são quase zerados
($\Phi \approx 0$).

**Interpretação probabilística** (Hendrycks & Gimpel, 2016): imagine multiplicar $x$ por uma
máscara aleatória $m \sim \text{Bernoulli}(\Phi(x))$, que mantém $x$ com probabilidade maior
quanto maior ele for. O valor esperado é $\mathbb{E}[x \cdot m] = x\,\Phi(x)$. A GELU é um
"dropout que depende do valor", tomado em média.

$$
\mathbb{E}[x \cdot m] = x \cdot 1 \cdot \Phi(x) + x \cdot 0 \cdot \big(1 - \Phi(x)\big) = x\,\Phi(x)
$$

**Lendo a fórmula:** $m$ vale 1 ("mantém") com probabilidade $\Phi(x)$ e 0 ("zera") com
probabilidade $1 - \Phi(x)$. A esperança é a soma de "valor × probabilidade" dos dois casos.
Exemplo: com $x = 1$, mantém em 84,13% das vezes; a média é $0{,}8413 = \text{GELU}(1)$.

### 3.2 GELU × ReLU

A ReLU (*Rectified Linear Unit*) é a ativação mais simples: zera os negativos e deixa os
positivos passarem. A fórmula e a sua derivada:

$$
\text{ReLU}(x) = \max(0, x), \qquad \text{ReLU}'(x) = \mathbb{1}[x > 0]
$$

**Lendo a fórmula:** $\max(0, x)$ escolhe o maior entre 0 e $x$. A derivada é 1 onde $x > 0$
(a função é uma reta de inclinação 1) e 0 onde $x < 0$ (a função é constante). Exemplos:
$\text{ReLU}(2) = 2$, $\text{ReLU}(-3) = 0$; $\text{ReLU}'(2) = 1$, $\text{ReLU}'(-3) = 0$. Em
$x = 0$ a derivada não existe (quina); o PyTorch usa 0.

Seção 1 do experimento:

```text
     x |   ReLU    GELU |   Φ(x) |  ReLU′   GELU′
 -3.00 |  0.000  -0.004 |  0.001 |   0.00  -0.012
 -2.00 |  0.000  -0.046 |  0.023 |   0.00  -0.085
 -1.00 |  0.000  -0.159 |  0.159 |   0.00  -0.083
 -0.75 |  0.000  -0.170 |  0.227 |   0.00  +0.001
 -0.50 |  0.000  -0.154 |  0.309 |   0.00  +0.133
  0.00 |  0.000  +0.000 |  0.500 |   0.00  +0.500
  0.50 |  0.500  +0.346 |  0.691 |   1.00  +0.867
  1.00 |  1.000  +0.841 |  0.841 |   1.00  +1.083
  2.00 |  2.000  +1.954 |  0.977 |   1.00  +1.085
  3.00 |  3.000  +2.996 |  0.999 |   1.00  +1.012

mínimo da GELU: -0.1700 em x = -0.752  (a ReLU nunca é negativa)
```

A derivada da GELU mede quanto a saída muda quando a pré-ativação muda um pouco. É ela que o
backpropagation usa para mandar gradiente através do neurônio. Pela regra do produto, é

$$
\text{GELU}'(x) = \Phi(x) + x\,\varphi(x), \qquad \varphi(x) = \tfrac{1}{\sqrt{2\pi}} e^{-x^2/2}
$$

**Lendo a fórmula, passo a passo:**

1. A GELU é um produto de duas funções: $f(x) = x$ e $g(x) = \Phi(x)$.
2. **Regra do produto:** $(f \cdot g)' = f' \cdot g + f \cdot g'$.
3. $f'(x) = 1$ (a derivada de $x$ é 1).
4. $g'(x) = \varphi(x)$: a derivada da CDF é a **densidade** $\varphi$, a altura da curva de sino.
5. Juntando: $1 \cdot \Phi(x) + x \cdot \varphi(x)$.

Sobre $\varphi$: $\tfrac{1}{\sqrt{2\pi}} \approx 0{,}3989$ é a altura do sino no centro; o
fator $e^{-x^2/2}$ faz a curva cair rapidamente quando $x$ se afasta de 0. Ela é simétrica:
$\varphi(-x) = \varphi(x)$.

Confira na tabela: em $x = 1$, $0{,}841 + 1 \cdot 0{,}242 = 1{,}083$.

Com quatro casas decimais (calculado com Python):

- $x = 1$: $\varphi(1) = 0{,}3989 \cdot e^{-0{,}5} = 0{,}3989 \cdot 0{,}6065 = 0{,}2420$, então
  $\text{GELU}'(1) = 0{,}8413 + 0{,}2420 = 1{,}0833$.
- $x = -1$: $\Phi(-1) = 0{,}1587$ e $\varphi(-1) = 0{,}2420$, então
  $\text{GELU}'(-1) = 0{,}1587 - 1 \cdot 0{,}2420 = -0{,}0833$ (tabela: $-0.083$).
- $x = 0$: $\text{GELU}'(0) = 0{,}5 + 0 = 0{,}5$ (tabela: $+0.500$).

| propriedade | ReLU | GELU |
|---|---|---|
| suave | não (quina em 0) | sim |
| monotônica | sim | **não**: cai até $-0{,}170$ em $x \approx -0{,}752$, depois sobe para 0 |
| gradiente para $x < 0$ | **exatamente 0** | pequeno, mas diferente de zero (negativo abaixo de $-0{,}752$) |
| para $x$ grande | $x$ | $\approx x$ |

**Suave** = sem quinas (a derivada existe e varia continuamente). **Monotônica** = nunca muda
de direção (só sobe, ou só desce).

**Por que o gradiente em $x < 0$ importa.** Com ReLU, um neurônio cuja pré-ativação fica
negativa para todas as entradas recebe gradiente zero e **nunca mais se recupera**: é um
"neurônio morto". Com GELU sempre há algum sinal de gradiente, e o neurônio pode voltar a
ser útil.

- **Neurônio morto:** como $\text{ReLU}' = 0$ para $x < 0$, nem a chave $k_i$ nem o limiar
  $b_i$ recebem gradiente. O treino não tem como mexer neles. (Ele só "revive" se outras
  camadas mudarem as entradas que chegam a ele.)
- **Com GELU:** a derivada só é exatamente zero num único ponto ($x \approx -0{,}752$). Para $x$
  muito negativo ela fica minúscula ($-0{,}012$ em $x = -3$), mas não some.

**Na inicialização**, os pesos são pequenos e simétricos em torno de zero. No batch real da
seção 4 do experimento, **50,2%** das pré-ativações são negativas: metade dos neurônios está
"no lado de baixo" para cada posição.

**Aproximação.** O GPT-2 original usava uma aproximação da GELU com $\tanh$, disponível no
PyTorch como `F.gelu(x, approximate="tanh")`. O erro máximo em relação à fórmula exata é
$4{,}7 \times 10^{-4}$. Usamos a versão exata, o padrão do `nn.GELU()`.

### 3.3 Alternativas modernas

Modelos como o LLaMA usam **SwiGLU** (Shazeer, 2020): uma
variante com três matrizes em que uma projeção funciona como "portão" multiplicativo da
outra. Fica como possível evolução futura do projeto.

Uma forma de escrever (sem biases, como no LLaMA):

$$
\text{FFN}_{\text{SwiGLU}}(x) = W_2\,\big(\text{Swish}(W_1 x) \odot W_3 x\big), \qquad \text{Swish}(z) = z \cdot \text{sigmoid}(z)
$$

**Lendo a fórmula:** $W_1 x$ e $W_3 x$ são duas expansões diferentes da mesma entrada;
$\text{Swish}(W_1 x)$ faz o papel de **portão** (quanto deixar passar); $\odot$ multiplica
portão e conteúdo coordenada a coordenada; $W_2$ contrai de volta para $d$. Como são três
matrizes em vez de duas, costuma-se usar um $d_{ff}$ menor (cerca de $\tfrac{2}{3} \cdot 4d$)
para manter o número de parâmetros.

---

## 4. Uma posição de cada vez

### 4.1 Mesmo peso para todas as posições

`nn.Linear` atua só na **última** dimensão. A fórmula abaixo diz que o feed-forward de um
tensor inteiro é só o feed-forward de cada vetor, um por um. Para `x: [B, T, d]`:

$$
\text{FFN}(x)_{b,t} = \text{FFN}(x_{b,t}) \quad \text{para toda posição } (b, t)
$$

**Lendo a fórmula:** à esquerda, aplica-se o FFN ao tensor `[B, T, d]` e pega-se o vetor da
sequência $b$, posição $t$. À direita, pega-se primeiro o vetor $x_{b,t}$ (128 números) e
aplica-se o FFN só nele. Os dois lados são iguais. Aqui $b$ é um índice de sequência no batch,
não um bias.

O feed-forward é a **mesma** rede aplicada a cada um dos $B \cdot T$ vetores,
independentemente. Três consequências, todas no experimento e nos testes:

```text
mudando só a entrada da posição 5, saídas alteradas nas posições: [5]
mesmo vetor nas posições 3 e 9 -> mesma saída: True
ff([B, T, d]) == ff([B·T, d]) remontado: True
```

- **Nenhuma informação atravessa posições.** A Jacobiana é diagonal por blocos:
  $\partial\, \text{FFN}(x)_t / \partial x_s = 0$ para $s \ne t$, inclusive para o passado
  (`test_gradient_only_reaches_the_same_position`).
- **Não há noção de posição.** O mesmo vetor gera a mesma saída em qualquer lugar; a posição
  só entra se já estiver codificada no vetor, via embeddings ou attention.
- **`[B, T, d]` é só uma pilha de $B \cdot T$ vetores.** Achatar para `[B·T, d]` dá o mesmo
  resultado.

**Lendo a Jacobiana:** $\partial\, \text{FFN}(x)_t / \partial x_s$ é "quanto a saída da
posição $t$ muda quando a entrada da posição $s$ muda um pouco". Organizando essas derivadas
numa grande matriz, posição contra posição, só os blocos da diagonal ($s = t$, cada um
$128 \times 128$) podem ser diferentes de zero. Por isso "diagonal **por blocos**".

No `tiny.yaml`, são $32 \times 128 = 4.096$ vetores processados independentemente em cada
feed-forward.

### 4.2 Causalidade de graça

Como o feed-forward nunca mistura posições, ele não pode quebrar a causalidade. É o mesmo
argumento de [04-attention.md](04-attention.md) (seção 3.1): basta a attention respeitar a
máscara para o GPT inteiro ser causal.

### 4.3 Comparação com a attention

| | attention | feed-forward |
|---|---|---|
| a saída da posição $t$ depende de | $x_0, \dots, x_t$ | só $x_t$ |
| Jacobiana entre posições | triangular inferior | diagonal por blocos |
| parâmetros compartilhados entre posições | sim | sim |
| custo cresce com $T$ | $O(T^2)$ | $O(T)$ |

$O(T^2)$ quer dizer que o custo total cresce com o quadrado de $T$ (dobrar o contexto
quadruplica o custo); $O(T)$, que cresce na mesma proporção (dobrar o contexto dobra o custo).

---

## 5. Interpretação: uma soma de neurônios

### 5.1 Chaves e valores

A mesma fórmula da seção 1.1 pode ser reescrita **neurônio por neurônio**. Para isso,
escreva $W_1$ por linhas e $W_2$ por colunas:

$$
W_1 = \begin{bmatrix} k_1^\top \\ \vdots \\ k_{d_{ff}}^\top \end{bmatrix}, \qquad
W_2 = \begin{bmatrix} v_1 & \cdots & v_{d_{ff}} \end{bmatrix}
$$

**Lendo a fórmula:** $W_1$ é uma pilha de $d_{ff}$ linhas; a linha $i$ é o vetor $k_i$ (o
$^\top$ indica que ele está deitado, como linha). $W_2$ é uma fileira de $d_{ff}$ colunas; a
coluna $i$ é o vetor $v_i$. No `tiny.yaml`: 512 chaves de 128 números e 512 valores de 128
números.

Então

$$
\boxed{\;\text{FFN}(x) = b_2 + \sum_{i=1}^{d_{ff}} \sigma(k_i \cdot x + b_i)\; v_i\;}
$$

**Lendo a fórmula, passo a passo:**

1. $W_1 x$ multiplica cada **linha** por $x$. A coordenada $i$ do resultado é $k_i \cdot x$.
2. Somando $b_1$, a coordenada $i$ vira $k_i \cdot x + b_i$.
3. Aplicando $\sigma$ em cada coordenada: $a_i = \sigma(k_i \cdot x + b_i)$.
4. Multiplicar uma matriz por um vetor de coeficientes é **somar as colunas** da matriz,
   cada uma multiplicada pelo seu coeficiente: $W_2\, a = \sum_i a_i\, v_i$.
5. Somando $b_2$, chega-se à fórmula do quadro.

**Exemplo com o feed-forward minúsculo (seção 1.2).** As colunas de $W_2$ são
$v_1 = (1, 0)$, $v_2 = (0, 1)$, $v_3 = (2, -1)$:

| neurônio | $a_i$ | $a_i \cdot v_i$ |
|---|---|---|
| 1 | $0{,}8413$ | $(0{,}8413;\ 0)$ |
| 2 | $1{,}9545$ | $(0;\ 1{,}9545)$ |
| 3 | $-0{,}1543$ | $(-0{,}3085;\ 0{,}1543)$ |
| $b_2$ | — | $(0;\ 0{,}5)$ |
| **soma** | | $(0{,}5328;\ 2{,}6088)$ |

É o mesmo resultado da seção 1.2, agora visto como uma soma de contribuições.

Cada neurônio $i$ faz duas coisas:

1. **Detecta um padrão:** $k_i \cdot x + b_i$ mede quanto $x$ se alinha com a **chave**
   $k_i$, e $\sigma$ decide quanto o neurônio "acende".
2. **Escreve um vetor:** adiciona o **valor** $v_i \in \mathbb{R}^d$ à saída, escalado pela
   ativação.

Com $d_{ff} = 512$, cada bloco tem 512 pares "se vir este padrão, escreva este vetor". Geva
et al. (2021) mostraram que, em modelos treinados, muitas chaves respondem a padrões
interpretáveis do texto e os valores empurram a previsão para tokens relacionados, daí o
nome **memória chave-valor**.

**Memória chave-valor** é uma estrutura que guarda pares (chave, valor): quando a consulta
combina com uma chave, devolve o valor correspondente. Um dicionário do Python é uma memória
chave-valor "dura" (casa ou não casa). O feed-forward é uma versão "suave": cada chave casa um
pouco, e os valores são somados em proporção.

Seção 3 do experimento:

```text
ff(x) == b2 + Σ_i GELU(k_i·x + b_i) · v_i: True

os 5 neurônios com maior |ativação| para este x:
  neurônio  9: k·x + b = +2.324 -> GELU = +2.300  (escreve +2.300 × v_9)
  neurônio  3: k·x + b = +1.690 -> GELU = +1.613  (escreve +1.613 × v_3)
  ...
ativos (> 0,1): 15 de 32; quase desligados (|a| < 0,05): 5
```

Leitura: para este $x$ (com $d = 8$, $d_{ff} = 32$), o neurônio 9 tem pré-ativação $+2{,}324$,
a GELU a transforma em $+2{,}300$ (quase igual, porque $\Phi(2{,}324) \approx 0{,}99$), e ele
escreve $2{,}300 \times v_9$ na saída. Cerca de metade dos neurônios (15 de 32) contribui de
fato.

### 5.2 Paralelo com as heads

Em [05-multi-head-attention.md](05-multi-head-attention.md) vimos que a multi-head attention é uma **soma** das contribuições de cada head. O
feed-forward também é uma soma, das contribuições de cada neurônio. Nos dois casos, cada
unidade lê o residual stream e escreve nele de forma aditiva; a diferença é **o que** cada
unidade lê:

| unidade | lê | escreve |
|---|---|---|
| head de attention | outras posições, escolhidas por $q \cdot k$ | $\text{head}_i\, W_O^{(i)}$ |
| neurônio do feed-forward | só a posição atual, através da chave $k_i$ | $\sigma(k_i \cdot x + b_i)\, v_i$ |

---

## 6. Por que expandir para $4d$

### 6.1 Parâmetros

A conta abaixo soma quantos números aprendidos o feed-forward tem, peça por peça, e depois
substitui $d_{ff} = 4d$.

$$
\underbrace{d \cdot d_{ff} + d_{ff}}_{W_1,\; b_1} + \underbrace{d_{ff} \cdot d + d}_{W_2,\; b_2} = 2\, d\, d_{ff} + d_{ff} + d \;\overset{d_{ff} = 4d}{=}\; 8d^2 + 5d
$$

**Lendo a fórmula, passo a passo** (com $d = 128$, $d_{ff} = 512$):

| peça | conta | valor |
|---|---|---|
| $W_1$ | $d_{ff} \cdot d = 512 \cdot 128$ | 65.536 |
| $b_1$ | $d_{ff}$ | 512 |
| $W_2$ | $d \cdot d_{ff} = 128 \cdot 512$ | 65.536 |
| $b_2$ | $d$ | 128 |
| **total** | $2\,d\,d_{ff} + d_{ff} + d$ | **131.712** |

- $\overset{d_{ff} = 4d}{=}$ significa "igual, substituindo $d_{ff}$ por $4d$".
- $2\,d\,d_{ff} = 2 \cdot d \cdot 4d = 8d^2$ e $d_{ff} + d = 4d + d = 5d$.
- Conferindo: $8 \cdot 128^2 + 5 \cdot 128 = 131.072 + 640 = 131.712$.

Seção 5 do experimento:

```text
attention (Q, K, V, W_O, sem bias): 4·d² = 65,536 parâmetros

        d_ff | parâmetros | × attention
   1·d = 128 |     33,024 |        0.50
   2·d = 256 |     65,920 |        1.01
   4·d = 512 |    131,712 |        2.01
  8·d = 1024 |    263,296 |        4.02
```

A attention tem 4 matrizes $d \times d$ sem bias: $4 \cdot 128^2 = 65.536$. A coluna
"× attention" é a razão: $131.712 / 65.536 \approx 2{,}01$.

Com $d_{ff} = 4d$, o feed-forward tem **o dobro** dos parâmetros da attention: cerca de 2/3
dos parâmetros de cada bloco. Se os neurônios são "memórias", é aqui que fica a maior parte
do conhecimento armazenado pelo modelo.

Conferido com um `TransformerBlock` do `tiny.yaml` (cálculo feito para este documento): o
bloco tem 197.760 parâmetros = 65.536 (attention) + 131.712 (feed-forward) + 512 (duas
LayerNorms, $2 \times (128 + 128)$). O feed-forward é 66,6% do bloco.

**Por que $4d$ e não $d$?** $d_{ff}$ é o número de detectores de padrão. A largura $d$ do
residual stream limita o que cada posição **transporta**, mas nada impede que cada posição
seja **analisada** por mais detectores do que isso. O fator 4 é convenção: 512 → 2.048 no
Transformer original, 768 → 3.072 no GPT-2. Mudar esse fator é um experimento válido.

### 6.2 Custo por token

A conta abaixo estima o **trabalho** de cada componente por token, contando as multiplicações
feitas nos produtos de matrizes durante o forward. É uma aproximação dos **FLOPs**
(*floating point operations*, operações com números reais): ignora somas, biases, softmax,
GELU e LayerNorm, que são baratas em comparação.

Multiplicações no forward, por token:

```text
multiplicações por token no forward (T = 128):
  feed-forward: 2 · d · 4d               = 131,072
  attention:    4d² (projeções) + 2·T·d (scores e soma ponderada) = 98,304
```

**Lendo as contas, passo a passo:**

- **Feed-forward.** Multiplicar $W_1$ (`[512, 128]`) por $x$ exige $512 \times 128 = 4d^2$
  multiplicações (cada linha faz 128). $W_2$ exige outras $4d^2$. Total: $8d^2 = 131.072$.
- **Attention, projeções.** $W_Q$, $W_K$, $W_V$ e $W_O$ são `[128, 128]`: $4 \times d^2 = 65.536$.
- **Attention, scores.** A query do token é comparada com $T$ keys. Cada comparação, numa head,
  custa $d_h$ multiplicações; com $h$ heads, $T \cdot h \cdot d_h = T \cdot d = 16.384$.
- **Attention, soma ponderada.** Somar $T$ values de $d_h$ números por head, em $h$ heads:
  $T \cdot d = 16.384$.
- Total da attention: $65.536 + 16.384 + 16.384 = 98.304$.

O custo do feed-forward não depende de $T$; o da attention cresce com $T$. Os dois se
igualam quando $4d^2 + 2Td = 8d^2$, ou seja, $T = 2d$. **Com contextos curtos ($T < 2d$) o
feed-forward domina a computação; com contextos longos, a attention.** No `tiny.yaml`,
$T = 128 < 256$.

**Resolvendo o cruzamento, passo a passo:**

1. Igualar os custos: $4d^2 + 2Td = 8d^2$.
2. Tirar $4d^2$ dos dois lados: $2Td = 4d^2$.
3. Dividir os dois lados por $2d$: $T = 2d$.
4. No `tiny.yaml`: $T = 2 \cdot 128 = 256$.

**Exemplo:** se o contexto fosse $T = 512$, a attention faria
$65.536 + 2 \cdot 512 \cdot 128 = 196.608$ multiplicações por token, contra 131.072 do
feed-forward.

---

## 7. Bias: por que aqui sim e na attention não

Em [04-attention.md](04-attention.md) decidimos não usar bias em Q, K, V e $W_O$ ([`model/attention.py`](../model/attention.py),
`nn.Linear(..., bias=False)`). No feed-forward usamos (`nn.Linear` tem `bias=True` por padrão).
A decisão tem justificativa matemática, verificada numericamente:

**Bias** é o vetor $b$ somado depois da multiplicação por uma matriz, em $Wx + b$. Ele desloca
o resultado sem depender da entrada.

```text
bias em K não muda os pesos: True
bias em Q muda os pesos:    True
bias em V = somar b_v na saída: True
```

Observação: nenhum experimento atual (`e04` a `e06`) imprime essas três linhas. As três
afirmações foram conferidas de novo com `attention_weights` de `model/attention.py` (queries,
keys, values e bias aleatórios, máscara causal) e continuam valendo. A seção 7.1 refaz as
contas à mão.

- **Bias em K é redundante.** $q_i \cdot (k_j + b) = q_i \cdot k_j + q_i \cdot b$. O termo
  $q_i \cdot b$ é o mesmo para todo $j$ na linha $i$, e somar uma constante a uma linha não
  muda a softmax (ver [04-attention.md](04-attention.md), seção 2.5).
- **Bias em V vira um deslocamento fixo.** $\sum_j w_{ij}(v_j + b) = \sum_j w_{ij} v_j + b$,
  porque os pesos somam 1. Depois de $W_O$, é equivalente a um bias na saída.
- **Bias em Q muda os pesos:** $(q_i + b) \cdot k_j = q_i \cdot k_j + b \cdot k_j$ dá uma
  preferência fixa por keys alinhadas com $b$. Tem efeito real, mas é omitido por
  simplicidade, como nos modelos recentes.
- **No feed-forward, o bias define o limiar de cada neurônio.** $\sigma(k_i \cdot x + b_i)$
  acende quando $k_i \cdot x > -b_i$. Sem bias, todos os 512 hiperplanos de decisão passariam
  pela origem, e nenhum neurônio poderia estar "ligado por padrão". A LayerNorm antes do
  feed-forward (ver [07-transformer-block.md](07-transformer-block.md)) tem um deslocamento
  $\beta$ que compensa em parte, mas só oferece
  offsets da forma $k_i \cdot \beta$, amarrados à própria chave, e não limiares livres.

### 7.1 As contas, uma por uma, com números

Neste trecho, $b$ é um bias **hipotético** que somaríamos às keys, queries ou values da
attention. $q_i$, $k_j$, $v_j$ são os da attention, não os do feed-forward.

**Bias em K.** Uma query $q = (1, 2)$ e duas keys $k_1 = (1, 0)$, $k_2 = (0, 1)$. Bias
$b = (1, 1)$ somado às keys:

$$
q \cdot (k_j + b) = q \cdot k_j + \underbrace{q \cdot b}_{\text{mesmo para todo } j}
$$

**Lendo a fórmula:** o produto escalar se distribui sobre a soma. O segundo termo não tem
$j$: é a mesma constante em todos os scores daquela query.

| | score com $k_1$ | score com $k_2$ | softmax |
|---|---|---|---|
| sem bias | $q \cdot k_1 = 1$ | $q \cdot k_2 = 2$ | $(0{,}269;\ 0{,}731)$ |
| com bias ($q \cdot b = 3$) | $1 + 3 = 4$ | $2 + 3 = 5$ | $(0{,}269;\ 0{,}731)$ |

Os pesos não mudam. A divisão por $\sqrt{d_k}$ também não altera isso: divide todos os scores
pelo mesmo número.

**Bias em Q.** Mesmo exemplo, agora com $b = (1, 0)$ somado à query:

$$
(q + b) \cdot k_j = q \cdot k_j + \underbrace{b \cdot k_j}_{\text{depende de } j}
$$

**Lendo a fórmula:** o termo extra muda de key para key. Aqui $b \cdot k_1 = 1$ e
$b \cdot k_2 = 0$, então os scores viram $(1 + 1;\ 2 + 0) = (2;\ 2)$ e a softmax vira
$(0{,}5;\ 0{,}5)$. Os pesos mudaram.

**Bias em V.**

$$
\sum_j w_{ij}\,(v_j + b) = \sum_j w_{ij}\, v_j + \Big(\sum_j w_{ij}\Big)\, b = \sum_j w_{ij}\, v_j + b
$$

**Lendo a fórmula:** distribuir $w_{ij}$ sobre a soma; o termo com $b$ fica
$(\sum_j w_{ij})\, b$; como os pesos de uma linha somam 1, sobra só $b$. Depois de $W_O$, vira
o vetor fixo $b\,W_O$ somado à saída: exatamente o que um bias de saída faria.

**Bias no feed-forward: o limiar.** O neurônio $i$ acende quando

$$
k_i \cdot x + b_i > 0 \quad \Leftrightarrow \quad k_i \cdot x > -b_i
$$

**Lendo a fórmula:** $\Leftrightarrow$ = "é equivalente a". O neurônio compara o alinhamento
$k_i \cdot x$ com o limiar $-b_i$. A fronteira $k_i \cdot x + b_i = 0$ é um hiperplano; se
$b_i = 0$, o ponto $x = 0$ está sempre nela, ou seja, o hiperplano passa pela origem.

Exemplo: $k_i = (1, 1)$ e $b_i = -1$ acendem só quando $x_1 + x_2 > 1$. Com $x = (1, 0)$:
$1 - 1 = 0$, não acende. Com $x = (1, 1)$: $2 - 1 = 1$, acende. É o neurônio 2 da solução do
XOR (seção 2.2): sem bias, ele não existiria.

**Por que a LayerNorm não substitui.** A LayerNorm entrega $\gamma \odot \hat{x} + \beta$ ao
feed-forward. Sem $b_i$, a pré-ativação seria

$$
k_i \cdot (\gamma \odot \hat{x} + \beta) = k_i \cdot (\gamma \odot \hat{x}) + k_i \cdot \beta
$$

**Lendo a fórmula:** o deslocamento do neurônio $i$ seria $k_i \cdot \beta$. Ele depende da
chave $k_i$, que também define o padrão procurado. E $\beta$ tem só 128 números para gerar os
512 deslocamentos. Com $b_1$, cada neurônio tem o seu limiar livre.

O LLaMA remove todos os biases e funciona bem. Em escala, a diferença é pequena; aqui
seguimos o GPT-2 no feed-forward porque o papel do bias é fácil de explicar.

---

## 8. Inicialização e dropout

- **Pesos** $\mathcal{N}(0, 0{,}02^2)$ e **biases zero**, como no GPT-2. O GPT-2 ainda reduz a
  escala de `down_proj` (e de $W_O$) por $1/\sqrt{2L}$, porque essas são as matrizes que
  escrevem no residual stream; no Mini-GPT isso é feito pelo `TransformerBlock`
  ([07-transformer-block.md](07-transformer-block.md)).
- **Dropout na saída**, depois de `down_proj`: zera aleatoriamente coordenadas do que vai ser
  somado ao residual stream (o `resid_pdrop` do GPT-2). Em `eval()`, é desligado.

### 8.1 Inicialização

**Objetivo:** começar com pesos pequenos e aleatórios, para que os neurônios sejam diferentes
entre si (se fossem iguais, receberiam gradientes iguais e continuariam iguais) e as saídas não
explodam.

**Funcionamento** ([`model/feed_forward.py`](../model/feed_forward.py), linhas 22–24):

```python
for proj in (self.up_proj, self.down_proj):
    nn.init.normal_(proj.weight, std=0.02)
    nn.init.zeros_(proj.bias)
```

- $\mathcal{N}(0, 0{,}02^2)$: cada peso é sorteado de uma normal com média 0 e desvio padrão
  0,02 (variância $0{,}02^2$). Cerca de 95% ficam entre −0,04 e +0,04.
- Biases zero: no início, nenhum neurônio está "ligado por padrão"; o treino ajusta os limiares.

**Quanto vale uma pré-ativação no início?** Depois da LayerNorm, cada vetor tem soma dos
quadrados igual a $d$, ou seja, tamanho $\sqrt{128} \approx 11{,}3$. O produto $k_i \cdot x$ com
pesos de desvio 0,02 tem desvio $0{,}02 \times 11{,}3 \approx 0{,}226$ (calculado para este
documento). As pré-ativações começam perto de zero, metade de cada lado, o que bate com os
50,2% negativos do experimento.

**Reescala no bloco.** O `TransformerBlock` ([`model/transformer_block.py`](../model/transformer_block.py),
linhas 29–33) sorteia de novo **só o peso** de `down_proj`, com um desvio menor:

$$
\text{desvio de } W_2 = \frac{0{,}02}{\sqrt{2L}} = \frac{0{,}02}{\sqrt{8}} \approx 0{,}00707
$$

**Lendo a fórmula:** $L = 4$ blocos; cada bloco soma **duas** contribuições ao residual stream
(attention e feed-forward), então há $2L = 8$ contribuições. Somas de muitas parcelas
aleatórias têm variância que cresce com o número de parcelas; dividir o desvio de cada uma por
$\sqrt{2L}$ mantém a variância da soma estável. `up_proj` continua com 0,02, e o bias de
`down_proj` continua zero. Detalhes em [07-transformer-block.md](07-transformer-block.md).

### 8.2 Dropout

**Objetivo:** regularizar, isto é, dificultar que o modelo decore o treino. Zerar coordenadas
ao acaso obriga a rede a não depender de nenhuma coordenada específica.

**Funcionamento** (linha 30, `self.dropout(self.down_proj(hidden))`). Em `train()`, cada
coordenada $j$ da saída $y$ vira

$$
\text{saída}_j = \frac{r_j \cdot y_j}{1 - p}, \qquad r_j \sim \text{Bernoulli}(1 - p)
$$

**Lendo a fórmula:** $p = 0{,}1$ é a probabilidade de zerar; $r_j$ é 1 (mantém) com
probabilidade $1 - p = 0{,}9$ e 0 (zera) com probabilidade $0{,}1$. Dividir por $1 - p$
compensa os zeros: em média, $\mathbb{E}[r_j / (1 - p)] = 0{,}9 / 0{,}9 = 1$, e a saída tem o
mesmo valor esperado que na inferência (*inverted dropout*, como em [03-embeddings.md](03-embeddings.md)).

**Exemplo:** $y = (0{,}5;\ -0{,}2;\ 0{,}8;\ 0{,}1)$ e o sorteio zera a segunda coordenada. A
saída é $(0{,}5/0{,}9;\ 0;\ 0{,}8/0{,}9;\ 0{,}1/0{,}9) = (0{,}556;\ 0;\ 0{,}889;\ 0{,}111)$.

Em `eval()`, a saída é $y$, sem mudança (`test_dropout_only_in_train_mode`).

---

## 9. Da matemática ao código

Sem abstração:

```python
hidden = F.gelu(x @ W1.T + b1)     # [B, T, d_ff]
out = hidden @ W2.T + b2           # [B, T, d]
```

**Lendo o código:** `x @ W1.T + b1` é $W_1 x + b_1$ com vetores como linhas; `F.gelu` é
$\sigma$; `hidden @ W2.T + b2` é $W_2\,(\cdot) + b_2$. É exatamente o que
`test_matches_manual_formula` confere.

O módulo:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        self.up_proj = nn.Linear(d_model, d_ff)      # W1, b1
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(d_ff, d_model)    # W2, b2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                             # [B, T, d]
        hidden = self.activation(self.up_proj(x))     # [B, T, d_ff]
        return self.dropout(self.down_proj(hidden))   # [B, T, d]
```

```python
from model import FeedForward

ff = FeedForward(d_model=128, d_ff=config.model.d_ff, dropout=0.1)   # d_ff = 512
out = ff(h)       # h: [B, T, 128]  ->  out: [B, T, 128]
```

Os nomes `up_proj` e `down_proj` descrevem a direção da projeção (expande e contrai), como
em implementações recentes. No GPT-2 original, as mesmas matrizes se chamam `c_fc` e
`c_proj`. `ModelConfig.d_ff` devolve $4 \cdot d$.

### 9.1 Cada parte, com as linhas de [`model/feed_forward.py`](../model/feed_forward.py)

| parte | linha | objetivo | o que faz |
|---|---|---|---|
| `up_proj` | 16 | expandir: dar a cada posição $d_{ff}$ detectores | `nn.Linear(128, 512)`: guarda $W_1$ `[512, 128]` e $b_1$ `[512]`; calcula `x @ W1.T + b1` |
| `activation` | 17 | não-linearidade: "dobrar" o espaço | `nn.GELU()`: $x\,\Phi(x)$ em cada um dos 512 números; não tem parâmetros |
| `down_proj` | 19 | contrair: voltar à largura do residual stream | `nn.Linear(512, 128)`: guarda $W_2$ `[128, 512]` e $b_2$ `[128]` |
| `dropout` | 20 | regularizar o que será somado ao residual stream | `nn.Dropout(0.1)`: ativo só em `train()` |
| inicialização | 22–24 | pesos pequenos e distintos | pesos $\mathcal{N}(0, 0{,}02^2)$, biases zero (seção 8.1) |
| `forward` | 26–30 | aplicar tudo, posição por posição | `hidden = activation(up_proj(x))`; `return dropout(down_proj(hidden))` |

**O `forward`, passo a passo, com shapes do `tiny.yaml`:**

1. `x`: `[32, 128, 128]`.
2. `self.up_proj(x)`: `[32, 128, 512]`. O `nn.Linear` age só na última dimensão; as duas
   primeiras são tratadas como "lote" (seção 4.1).
3. `self.activation(...)`: `[32, 128, 512]`, a variável `hidden`.
4. `self.down_proj(hidden)`: `[32, 128, 128]`.
5. `self.dropout(...)`: `[32, 128, 128]`, devolvido.

**`ModelConfig.d_ff`** ([`config.py`](../config.py), linhas 38–41) é uma propriedade, não uma
chave do YAML:

```python
@property
def d_ff(self) -> int:
    # Dimensão interna do feed-forward, como no GPT: 4 × d_model.
    return 4 * self.d_model
```

$$
d_{ff} = 4 \cdot d = 4 \cdot 128 = 512
$$

**Lendo a fórmula:** o fator 4 é fixo no código (seção 6.1). Mudar `d_model` no YAML muda
$d_{ff}$ junto. Quem usa: o `GPT` passa `config.d_ff` para cada `TransformerBlock`
([`model/gpt.py`](../model/gpt.py), linha 27), que cria o `FeedForward`
([`model/transformer_block.py`](../model/transformer_block.py), linha 27).

---

## 10. O que cada teste garante

| teste | propriedade |
|---|---|
| `test_output_shape` | `[B, T, d]` → `[B, T, d]` |
| `test_hidden_layer_expands_to_d_ff` | a camada escondida tem $d_{ff}$ neurônios |
| `test_matches_manual_formula` | módulo = $W_2\,\text{GELU}(W_1 x + b_1) + b_2$ |
| `test_output_is_a_sum_of_neurons` | módulo = $b_2 + \sum_i \sigma(k_i \cdot x + b_i)\, v_i$ |
| `test_without_activation_two_layers_collapse_into_one` | sem ativação, vira $x W^\top + b$ com $W = W_2 W_1$ |
| `test_gelu_matches_formula` | `F.gelu` = $x\,\Phi(x)$ |
| `test_gelu_shape` | GELU(0) = 0, $\approx x$ e $\approx 0$ nos extremos, mínimo $-0{,}170$ em $-0{,}752$ |
| `test_each_position_is_processed_independently` | mudar uma posição só muda a saída dela |
| `test_same_vector_gives_same_output_at_any_position` | mesmos pesos em todas as posições |
| `test_permuting_positions_permutes_output` | permutar posições só permuta a saída |
| `test_gradient_only_reaches_the_same_position` | Jacobiana diagonal por blocos |
| `test_parameter_count` | $2\,d\,d_{ff} + d_{ff} + d$ |
| `test_initialization` | pesos com desvio 0,02, biases zero |
| `test_gradients_reach_all_parameters` | todos os parâmetros recebem gradiente |
| `test_dropout_only_in_train_mode` | dropout ativo só em `train()` |

Os testes usam $d = 16$ e $d_{ff} = 64$. Nos testes de fórmula, os biases são sorteados
(fixture `ff_random_biases`), porque na inicialização eles são zero e não apareceriam nas
contas.

---

## Resumo

1. A attention move informação entre posições, mas transforma o conteúdo de forma linear;
   o feed-forward é a etapa **não linear**, aplicada a cada posição separadamente.
2. Sem a ativação, as duas camadas colapsariam numa só; com ela, a rede aproxima funções
   arbitrárias. No XOR: loss 0,693 sem ativação, 0,000 com GELU.
3. GELU $= x\,\Phi(x)$: suave, não monotônica e com gradiente para entradas negativas, o que
   evita neurônios mortos.
4. O feed-forward é uma soma de $d_{ff}$ neurônios, cada um um par "chave detecta um padrão,
   valor é escrito no residual stream".
5. Com $d_{ff} = 4d$, ele concentra 2/3 dos parâmetros do bloco e domina a computação em
   contextos curtos ($T < 2d$).
6. O bias $b_1$ dá a cada neurônio um limiar próprio; na attention, bias em K seria
   redundante.
7. Pesos começam em $\mathcal{N}(0, 0{,}02^2)$ e biases em zero; no bloco, `down_proj` é
   reduzido para $0{,}02/\sqrt{2L}$. O dropout age só no treino, na saída.

---

## Para checar o entendimento

1. Mostre que três camadas lineares seguidas, sem ativação, também equivalem a uma. Qual é a
   matriz resultante?
2. Por que o XOR não pode ser resolvido por uma única camada linear? Desenhe os quatro pontos.
3. Para qual valor de $x$ a GELU atinge o mínimo? Escreva a equação que esse ponto satisfaz,
   usando a derivada.
4. Um neurônio com ReLU cuja pré-ativação é negativa para todo o dataset recebe gradiente?
   E com GELU?
5. Se você trocar a ordem das posições da entrada, o que acontece com a saída do
   feed-forward? E com a saída da attention causal?
6. Quantos parâmetros tem o feed-forward com $d = 768$ e $d_{ff} = 3072$ (GPT-2 small)?
7. A partir de que tamanho de contexto a attention passa a fazer mais multiplicações por
   token que o feed-forward no GPT-2 small?
8. Por que um bias em K não tem efeito nenhum nos pesos da attention, mas um bias em $W_1$
   muda o comportamento do feed-forward?
9. Refaça o feed-forward minúsculo da seção 1.2 com $x = (2, 1)$ usando ReLU. Quais neurônios
   acendem, e qual é a saída?
10. Na solução do XOR com dois neurônios (seção 2.2), o que acontece se o limiar do neurônio 2
    for 0 em vez de −1?
11. Onde, no código, está o "valor" $v_7$ do neurônio 7? E a sua chave $k_7$?
