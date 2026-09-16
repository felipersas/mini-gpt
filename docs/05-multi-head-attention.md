# Multi-Head Attention

> **Por que existem várias heads?**

Código: [`model/attention.py`](../model/attention.py) (`split_heads`, `merge_heads`,
`LoopMultiHeadAttention`, `MultiHeadAttention`). Testes:
[`tests/test_multi_head_attention.py`](../tests/test_multi_head_attention.py). Experimento:
`uv run python -m experiments.e05_multi_head_attention`. Visão geral:
[architecture.md](architecture.md). Pré-requisito: [04-attention.md](04-attention.md). Guia de
notação: [00-notacao.md](00-notacao.md).

Notação rápida: $h$ = `num_heads`, $d_h = d / h$ = `head_dim`. No `tiny.yaml`: $d = 128$, $h = 4$,
$d_h = 32$. A tabela completa está na seção [Notação deste documento](#notação-deste-documento).

---

## Para que serve

### O problema, em linguagem simples

Em [04-attention.md](04-attention.md) construímos **uma head** de attention. Uma **head** ("cabeça") é um conjunto
completo de query, key e value: ela decide para onde cada posição olha e o que traz de lá.

Uma head tem uma limitação: em cada posição, ela faz **uma única média ponderada** das
posições anteriores. Se a posição precisa de duas informações vindas de dois lugares
diferentes, as duas acabam **somadas no mesmo vetor**, e não dá mais para separá-las.

A multi-head attention resolve isso rodando **várias heads em paralelo**. Cada uma faz sua
própria busca e escreve o resultado **no seu próprio pedaço** do vetor de saída.

Ela resolve também um problema de formato. Uma head sozinha devolve vetores de $d_h = 32$
números, mas o residual stream (a "esteira" de vetores que atravessa o modelo, ver
[07-transformer-block.md](07-transformer-block.md)) tem largura $d = 128$. A saída da attention
precisa voltar a ter 128 números para ser somada a ele.

### Uma analogia

Pense numa equipe de 4 pesquisadores lendo o mesmo texto, com uma planilha de 4 colunas:

- cada pesquisador tem **uma pergunta diferente** ("qual é o caractere anterior?", "onde
  começou a palavra?"...);
- cada um escreve a resposta **só na sua coluna**, então as respostas não se misturam;
- no fim, um **editor** lê as 4 colunas e escreve um relatório único, combinando tudo.

Os pesquisadores são as heads. As colunas são os blocos de $d_h$ números. O editor é a
projeção de saída $W_O$.

Com um pesquisador só, ele teria que responder às duas perguntas **na mesma célula**,
fazendo uma média das duas respostas.

### O que entra e o que sai

| | shape | com o `tiny.yaml` | o que é |
|---|---|---|---|
| entrada | `[B, T, d]` | `[32, 128, 128]` | um vetor de 128 números por posição, vindo do residual stream (depois da LayerNorm) |
| dentro: cada head | `[B, T, d_h]` | `[32, 128, 32]` | a saída de uma das 4 heads |
| dentro: pesos de attention | `[B, h, T, T]` | `[32, 4, 128, 128]` | uma matriz $128 \times 128$ por head e por sequência |
| saída | `[B, T, d]` | `[32, 128, 128]` | o mesmo shape da entrada, pronto para somar ao residual stream |

Parâmetros de uma `MultiHeadAttention` no `tiny.yaml`: $4d^2 = 4 \times 128^2 = 65.536$ (seção
2.1).

### Onde isto fica no GPT

```text
texto (corpus/dom_casmurro.txt)
  │  tokenizer: caracteres → IDs
  ▼
batches: x [32, 128] IDs  e  alvos y [32, 128]
  │  embeddings: token + posição
  ▼
[32, 128, 128]
  │  ┌──── bloco Transformer, repetido L = 4 vezes ─────────────────────────┐
  │  │  LayerNorm → multi-head attention [32, 128, 128]  ← este documento  │
  │  │  LayerNorm → feed-forward [32, 128, 128]                            │
  │  └─────────────────────────────────────────────────────────────────────┘
  ▼
[32, 128, 128]
  │  LayerNorm final + LM head
  ▼
logits [32, 128, 97]: um score por caractere do vocabulário
  │  cross-entropy com os alvos y
  ▼
loss (um número)
```

### O que daria errado com uma head só

- **Informações se misturam.** Se uma posição quer saber o caractere de $t-2$ **e** o de
  $t-1$, uma head só consegue dar peso 0,5 para cada um e somar. O resultado diz *quais*
  caracteres apareceram, mas não *em que ordem* (seção 1).
- **Um padrão de atenção por camada.** A camada inteira teria uma única matriz $T \times T$
  por sequência. Rastrear "caractere anterior" e "início da palavra" ao mesmo tempo exigiria
  camadas extras.
- **Shape errado.** Uma head com $d_h = 32$ não devolve os 128 números do residual stream.
  Seria preciso uma head com $d_h = d = 128$, que tem o mesmo número de parâmetros das 4 heads
  (seção 2.1), mas continua com uma distribuição de pesos por posição.

---

## Notação deste documento

Os símbolos gerais ($\sum$, produto de matrizes, transposta, softmax, $\mathbb{R}^{m \times n}$)
estão explicados com exemplos em [00-notacao.md](00-notacao.md). Aqui está tudo o que este
documento usa.

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $B$ | "bê" | tamanho do batch | 32 | `batch`, `batch_size` |
| $T$ | "tê" | posições por sequência | 128 | `seq_len` |
| $t$ | "tê" minúsculo | índice de posição | 0 a 127 | `t` |
| $d$ | "dê" | largura dos vetores do modelo (residual stream) | 128 | `d_model` |
| $h$ | "agá" | **número de heads**. Em outros documentos ([07](07-transformer-block.md)–[09](09-lm-head.md)), $h$ pode ser o **estado oculto**; aqui isso só acontece na fórmula do bloco da seção 2.2, onde está avisado | 4 | `num_heads` |
| $d_h$ ($d_k$, $d_v$) | "dê agá" | dimensão de cada head: $d / h$. $d_k$ e $d_v$ são os nomes usados em [04-attention.md](04-attention.md) (dimensão de key e de value); aqui os três são iguais | 32 | `head_dim` |
| $i$ | "i" | índice da head. Nas fórmulas vai de 1 a $h$; no código, de 0 a $h - 1$ | 1 a 4 (fórmula), 0 a 3 (código) | `i` |
| $X$ | "xis" | a sequência de entrada, uma linha por posição | $T \times d$ = $128 \times 128$ (com batch: `[32, 128, 128]`) | `x` |
| $W_Q^i, W_K^i, W_V^i$ | "dáblio quê da head i" | matrizes de projeção de query, key e value **da head $i$**. O $i$ em cima é um **índice** ("da head $i$"), **não um expoente**: não é "$W_Q$ elevado a $i$" | $d \times d_h$ = $128 \times 32$ | `loop.heads[i].q_proj.weight` guarda a transposta, shape `(32, 128)` |
| $W_Q, W_K, W_V$ | "dáblio quê" | as projeções de **todas** as heads empilhadas numa matriz só (seção 4.1) | $d \times d$ = $128 \times 128$ | `q_proj.weight` (transposta), shape `(128, 128)` |
| $Q_i, K_i, V_i$ | "quê i" | queries, keys e values da head $i$: $Q_i = X W_Q^i$ | $T \times d_h$ = $128 \times 32$ | `q[:, i]` depois de `split_heads` |
| $\text{Attention}(Q, K, V)$ | "attention de Q, K, V" | a fórmula de [04-attention.md](04-attention.md): $\text{softmax}(QK^\top / \sqrt{d_k} + M)\, V$ | | `attention_weights(q, k, mask) @ v` |
| $M$ | "eme" | máscara causal: 0 no passado, $-\infty$ no futuro | $T \times T$ | `causal_mask(seq_len)` (booleana) |
| $\text{head}_i$ | "head i" | a saída da head $i$ | $T \times d_h$ = $128 \times 32$ | `(weights @ v)[:, i]`, ou `head(x)` na lista |
| $\text{Concat}(\dots)$ | "concatenação" | colocar vetores **lado a lado**: $\text{Concat}((1, 2), (3, -1)) = (1, 2, 3, -1)$ | $T \times h d_h = T \times d$ | `torch.cat(..., dim=-1)`, `merge_heads` |
| $W_O$ | "dáblio ó" | projeção de saída (O de *output*): mistura as heads | $d \times d$ = $128 \times 128$ | `out_proj.weight` (transposta) |
| $W_O^{(i)}$ | "bloco i de dáblio ó" | as $d_h$ **linhas** de $W_O$ que multiplicam o bloco da head $i$. Os parênteses indicam "pedaço $i$", **não expoente** | $d_h \times d$ = $32 \times 128$ | `out_proj.weight[:, i*d_h:(i+1)*d_h].T` |
| $\text{MultiHead}(X)$ | "multi-head de X" | a saída da camada | $T \times d$ | `mha(x)` |
| $v_{\texttt{a}}, v_{\texttt{t}}$ | "vê a", "vê tê" | values que carregam os caracteres `'a'` e `'t'` (seção 1) | one-hot de 4 números | `W_v` do experimento |
| $\sum_{i=1}^{h}$ | "soma para i de 1 até h" | somar um termo por head | 4 termos | `sum(...)` |
| $A^\top$ | "A transposta" | troca linhas por colunas | `[m, n]` → `[n, m]` | `A.T`, `transpose(-2, -1)` |
| $\mathbb{R}^{m \times n}$ | "erre m por n" | matriz de $m$ linhas e $n$ colunas | | shape `(m, n)` |
| posto | "posto" (em inglês, *rank*) | quantas direções realmente diferentes as linhas de uma matriz têm (seção 2.1) | $\text{posto}(Q_i K_i^\top) \le 32$ | `torch.linalg.matrix_rank` |
| $\min(a, b)$ | "mínimo" | o menor dos dois | | `min(a, b)` |
| $d \bmod h$ | "d módulo h" | resto da divisão de $d$ por $h$ | $128 \bmod 4 = 0$ | `d_model % num_heads` |
| stride $s_e$ | "stride do eixo e" | quantas casas andar na memória para avançar 1 no eixo $e$ de um tensor (seção 4.4) | `x.stride()` = `(24, 8, 1)` no exemplo | `tensor.stride()` |
| $n_e$ | "ene e" | índice usado no eixo $e$ (seção 4.4) | | `x[n_0, n_1, n_2]` |
| $p$ | "pê" | probabilidade de dropout | 0,1 | `dropout` |
| $m_n$ | "eme ene" | sorteio do dropout: 1 (mantém) ou 0 (zera) | | máscara interna do `nn.Dropout` |
| $L$ | "ele" | número de blocos Transformer | 4 | `num_layers` |
| $\text{LN}$ | "layer norm" | LayerNorm (ver [07-transformer-block.md](07-transformer-block.md)) | | `ln_1` |
| $\mathcal{N}(0, \sigma^2)$ | "normal de média 0 e variância sigma ao quadrado" | sorteio da inicialização dos pesos | $\sigma = 0{,}02$ | `nn.init.normal_(w, std=0.02)` |
| $\sqrt{\cdot}$ | "raiz" | raiz quadrada | $\sqrt{8} \approx 2{,}83$ | `math.sqrt` |
| $\approx$ | "aproximadamente" | igualdade arredondada | | |
| $a\times$ (em tempos) | "a vezes" | razão entre dois tempos: 2,7× = 2,7 vezes mais rápido | | |
| MiB | "mebibyte" | $2^{20} = 1.048.576$ bytes | | |

---

## 1. O limite de uma head

### 1.1 Uma distribuição por posição

Em cada posição, a softmax produz **uma** distribuição de pesos, e a saída é **uma** média
ponderada dos values. Se a posição precisa trazer duas informações de dois lugares
diferentes, a head tem que dividir o peso entre eles. Aí as duas informações são **somadas
no mesmo vetor**.

### 1.2 A média destrói informação

Seção 1 do experimento. Queremos que a última posição saiba **quais** caracteres estavam em
$t-2$ e $t-1$, **e em que ordem**. As heads são construídas à mão, como em
[04-attention.md](04-attention.md): a query procura posições, e o value carrega o caractere
(one-hot).

**Como o experimento monta isso:**

- vocabulário `['a', 'g', 'o', 't']` (os caracteres de `"gato"`, em ordem alfabética);
- cada posição é `[one-hot do caractere | one-hot da posição]`: $d = 4 + 4 = 8$;
- $d_k = 4$, e a query tem força $8\sqrt{d_k} = 16$, então o score escalado do alvo vale 8;
- a head "1 head" procura **ao mesmo tempo** $t-1$ e $t-2$; cada head da versão "2 heads"
  procura **uma** das duas posições;
- o value de cada posição é o one-hot do seu caractere, por exemplo $v_{\texttt{a}} = (1, 0, 0, 0)$
  e $v_{\texttt{t}} = (0, 0, 0, 1)$.

```text
'gato': na última posição, t−2 = 'a' e t−1 = 't'
  1 head procurando t−2 e t−1: [a=0.50 t=0.50]
  2 heads, concatenadas:        [a=1.00] | [t=1.00]

'gtao': na última posição, t−2 = 't' e t−1 = 'a'
  1 head procurando t−2 e t−1: [a=0.50 t=0.50]
  2 heads, concatenadas:        [t=1.00] | [a=1.00]

saída igual para 'gato' e 'gtao'?  1 head: True   2 heads: False
```

**De onde vêm 0,50 e 1,00** (conferido com Python). Na última posição, a head única tem
score escalado 8 nas duas posições procuradas e 0 nas outras duas:

$$
w = \frac{e^8}{e^8 + e^8 + e^0 + e^0} = \frac{2981}{2 \cdot 2981 + 2} \approx 0{,}4998 \approx 0{,}50
$$

Cada head da versão com duas tem score 8 numa posição só:
$e^8 / (e^8 + 3) \approx 0{,}9990 \approx 1{,}00$.

A fórmula abaixo calcula a saída da head única na última posição. Ela mostra por que a ordem
se perde.

$$
0{,}5\, v_{\texttt{a}} + 0{,}5\, v_{\texttt{t}} = 0{,}5\, v_{\texttt{t}} + 0{,}5\, v_{\texttt{a}}
$$

**Lendo a fórmula:**

- $v_{\texttt{a}}$ e $v_{\texttt{t}}$: os values das posições com `'a'` e `'t'`;
- $0{,}5$: o peso de attention de cada uma das duas posições;
- o lado esquerdo é `"gato"` (`'a'` em $t-2$, `'t'` em $t-1$); o direito é `"gtao"` (a ordem
  inversa);
- os dois lados são iguais porque a soma é **comutativa** (a ordem das parcelas não muda o
  resultado).

**Exemplo numérico:** $0{,}5 \cdot (1, 0, 0, 0) + 0{,}5 \cdot (0, 0, 0, 1) = (0{,}5;\ 0;\ 0;\ 0{,}5)$
para as duas palavras. É o `[a=0.50 t=0.50]` da saída.

O resultado diz *quais* caracteres apareceram, mas não *onde*. `"gato"` e `"gtao"` ficam
**indistinguíveis** nesta posição.

Com **duas heads**, uma procura $t-2$ e a outra $t-1$. Cada uma escreve no **seu próprio
bloco** de coordenadas, e a concatenação mantém os blocos separados. A ordem é preservada:

| palavra | head de $t-2$ | head de $t-1$ | concatenação (8 números) |
|---|---|---|---|
| `"gato"` | $(1, 0, 0, 0)$ = `a` | $(0, 0, 0, 1)$ = `t` | $(1, 0, 0, 0,\ 0, 0, 0, 1)$ |
| `"gtao"` | $(0, 0, 0, 1)$ = `t` | $(1, 0, 0, 0)$ = `a` | $(0, 0, 0, 1,\ 1, 0, 0, 0)$ |

### 1.3 A ideia

Várias heads em paralelo, cada uma com:

- suas próprias matrizes $W_Q^i, W_K^i, W_V^i$ (lembrete: $i$ é "da head $i$", não expoente);
- seu próprio padrão de atenção (uma matriz $T \times T$ por head);
- seu próprio bloco de $d_h$ coordenadas na saída.

Assim, numa única camada, uma head pode rastrear o caractere anterior, outra o início da
palavra, outra a última pontuação, e as informações chegam separadas.

O diagrama do caminho completo, com os números do `tiny.yaml`:

```text
                 ┌── head 1 ──► [B, T, 32] ──┐
                 ├── head 2 ──► [B, T, 32] ──┤
[B, T, 128] ─────┤                           ├──► concat [B, T, 128] ──► W_O ──► [B, T, 128]
                 ├── head 3 ──► [B, T, 32] ──┤
                 └── head 4 ──► [B, T, 32] ──┘
```

As 4 heads recebem **a mesma entrada** inteira de 128 números. O que é dividido é a
**saída**: cada head produz 32 números, e $4 \times 32 = 128$.

---

## 2. A fórmula

A fórmula abaixo diz, de uma vez, tudo o que a camada calcula: cada head faz a attention de
[04-attention.md](04-attention.md) com as suas próprias projeções, as saídas são postas lado a
lado, e $W_O$ mistura o resultado.

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O,
\qquad \text{head}_i = \text{Attention}\big(XW_Q^i,\; XW_K^i,\; XW_V^i\big)
$$

**Lendo a fórmula:**

- $X$: a entrada, uma linha de $d$ números por posição;
- $XW_Q^i$: as queries **da head $i$** ($Q_i$). Multiplicar $X$ ($T \times d$) por $W_Q^i$
  ($d \times d_h$) dá $T \times d_h$: cada posição vira uma pergunta de $d_h$ números. O mesmo
  vale para $XW_K^i$ (keys) e $XW_V^i$ (values);
- $\text{Attention}(\cdot)$: a mesma fórmula, $\text{softmax}(Q_i K_i^\top / \sqrt{d_h} + M)\, V_i$,
  com a máscara causal $M$;
- $\text{head}_i$: o resultado da head $i$, $T \times d_h$;
- $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$: em cada posição, junta os $h$ vetores
  de $d_h$ números num vetor de $h \cdot d_h = d$ números, na ordem das heads;
- $W_O$: uma matriz $d \times d$ que mistura esses $d$ números;
- a segunda parte (depois da vírgula) é a **definição** de $\text{head}_i$ usada na primeira.

| termo | shape | no `tiny.yaml` |
|---|---|---|
| $X$ | $T \times d$ | $128 \times 128$ |
| $W_Q^i, W_K^i, W_V^i$ | $d \times d_h$ | $128 \times 32$ |
| $\text{head}_i$ | $T \times d_h$ | $128 \times 32$ |
| $\text{Concat}(\dots)$ | $T \times h d_h = T \times d$ | $128 \times 128$ |
| $W_O$ | $d \times d$ | $128 \times 128$ |
| saída | $T \times d$ | $128 \times 128$ |

A fórmula está escrita para **uma** sequência. No código, tudo ganha uma dimensão $B$ na
frente, e as contas são as mesmas para cada sequência do batch.

### 2.1 Por que $d_h = d / h$

**Orçamento de parâmetros.** Queremos saber quantos números aprendíveis as heads têm, para
comparar com uma head grande. Cada head tem três matrizes $d \times d_h$ ($W_Q^i$, $W_K^i$,
$W_V^i$), ou seja, $3 \cdot d \cdot d_h$ parâmetros. Com $h$ heads:

$$
h \cdot 3 \cdot d \cdot \frac{d}{h} = 3d^2
$$

**Lendo a fórmula:**

- $3 \cdot d \cdot d_h$: parâmetros de uma head (3 matrizes, cada uma com $d \cdot d_h$
  números);
- $\frac{d}{h}$: é o $d_h$, escrito em função de $d$ e $h$;
- $h \cdot$ (na frente): são $h$ heads;
- o $h$ da frente cancela com o $h$ do denominador, e sobra $3 \cdot d \cdot d = 3d^2$.

**Passo a passo com o `tiny.yaml`:**

| conta | valor |
|---|---|
| uma matriz de uma head: $d \cdot d_h = 128 \cdot 32$ | 4.096 |
| uma head: $3 \cdot 4.096$ | 12.288 |
| 4 heads: $4 \cdot 12.288$ | 49.152 |
| uma head com $d_k = d$: $3 \cdot 128^2$ | 49.152 |
| $W_O$: $128^2$ | 16.384 |
| total: $49.152 + 16.384 = 4 \cdot 128^2$ | 65.536 |

É exatamente o mesmo que **uma** head com $d_k = d$. **Multi-head não adiciona parâmetros:
divide o mesmo orçamento em subespaços.** Um **subespaço**, aqui, é o espaço menor de $d_h =
32$ eixos para onde cada head projeta os vetores de 128 números. Somando $W_O$ ($d^2$), o total
é $4d^2$:

```text
lista de heads  65,536  (4 · d_model² = 65,536)
tensor único    65,536  (4 · d_model² = 65,536)
```

| modelo | $d$ | $h$ | $d_h$ |
|---|---:|---:|---:|
| Transformer original (Vaswani et al., 2017) | 512 | 8 | 64 |
| GPT-2 small | 768 | 12 | 64 |
| Mini-GPT (`tiny.yaml`) | 128 | 4 | 32 |

**E se $d$ não for divisível por $h$?** Não existe $d_h$ inteiro. `_check_heads` (linhas 95–97
de `attention.py`) testa `d_model % num_heads != 0` e levanta `ValueError` com uma mensagem
clara. Exemplo do teste `test_d_model_must_be_divisible_by_num_heads`: $d = 10$, $h = 3$ dá
$10 \bmod 3 = 1$; com $d_h = \lfloor 10/3 \rfloor = 3$, as heads cobririam só $3 \cdot 3 = 9$
das 10 features. A `ModelConfig` (`config.py`) faz a mesma checagem ao carregar o YAML.

**O trade-off.** Mais heads permitem mais padrões independentes, mas cada head fica menor.

Antes de continuar, uma definição. O **posto** (em inglês, *rank*) de uma matriz é o número de
linhas **linearmente independentes**: linhas que não podem ser obtidas somando múltiplos das
outras. Intuitivamente, é quantas "direções realmente diferentes" a matriz contém.

- Exemplo: $\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ tem posto 1, porque a segunda linha
  é $2\times$ a primeira.
- Exemplo: $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ tem posto 2.
- Uma matriz com $n$ colunas tem posto no máximo $n$: não cabem mais de $n$ direções
  diferentes em $n$ números.

A fórmula abaixo limita o posto da matriz de scores de uma head. Ela explica por que heads
pequenas demais perdem expressividade.

$$
\text{posto}\big(Q_i K_i^\top\big) \le \min\big(\text{posto}(Q_i),\ \text{posto}(K_i)\big) \le d_h
$$

**Lendo a fórmula:**

- $Q_i \in \mathbb{R}^{T \times d_h}$ e $K_i \in \mathbb{R}^{T \times d_h}$: queries e keys da
  head $i$, com $d_h$ colunas cada;
- $Q_i K_i^\top \in \mathbb{R}^{T \times T}$: a matriz de scores (antes da escala e da
  máscara);
- o posto de um produto nunca passa do posto de nenhum dos fatores;
- $Q_i$ e $K_i$ têm só $d_h$ colunas, então o posto de cada um é no máximo $d_h$.

**Exemplo numérico:** com $d_h = 32$ e $T = 128$, a matriz de scores tem $128 \times 128$
entradas, mas posto no máximo 32. Conferido com Python: para $Q_i$ e $K_i$ aleatórios
`[128, 32]`, `torch.linalg.matrix_rank(Q @ K.T)` dá 32.

Ou seja, a matriz de scores de uma head, $Q_i K_i^\top \in \mathbb{R}^{T \times T}$, tem **posto
no máximo $d_h$**: com $d_h = 32$ e $T = 128$, ela não consegue representar um padrão
$128 \times 128$ arbitrário. Heads pequenas demais limitam o que cada uma consegue expressar
(Bhojanapalli et al., 2020, chamam isso de *low-rank bottleneck*: um "gargalo de posto
baixo", em que o tamanho $d_h$ restringe os padrões possíveis). O número de heads é um
hiperparâmetro do modelo, ajustável na config antes do treino.

**Custo de memória.** Cada head tem sua matriz de pesos: `[B, h, T, T]`. Com $B = 32$,
$T = 128$ e $h = 4$, são $4 \times 2$ MiB = 8 MiB por camada. A conta:

| conta | valor |
|---|---|
| números: $32 \cdot 4 \cdot 128 \cdot 128$ | 2.097.152 |
| bytes (float32 = 4 bytes por número) | 8.388.608 |
| em MiB ($\div\ 1.048.576$) | 8 |
| por head ($8 \div 4$) | 2 MiB |

### 2.2 A projeção de saída $W_O$

**Objetivo.** Sem $W_O$, a head $i$ ficaria presa às coordenadas $[i \cdot d_h,\ (i+1) \cdot d_h)$
do residual stream, uma associação arbitrária (contando as heads a partir de 0, como no
código). $W_O$ mistura as heads: cada coordenada de saída é uma combinação linear de **todas**
as heads.

Uma forma mais precisa de ver: divida $W_O$ em $h$ blocos de linhas,
$W_O^{(i)} \in \mathbb{R}^{d_h \times d}$ (as linhas que multiplicam o bloco da head $i$). A
identidade abaixo reescreve "concatenar e depois projetar" como "projetar cada head e somar".
Ela mostra que cada head contribui separadamente para a saída.

$$
\text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O = \sum_{i=1}^{h} \text{head}_i\, W_O^{(i)}
$$

**Lendo a fórmula:**

- lado esquerdo: o vetor concatenado ($d$ números por posição) multiplicado pela matriz inteira
  $W_O$ ($d \times d$);
- $W_O^{(i)}$: o pedaço de $W_O$ formado pelas linhas $i \cdot d_h$ até $(i+1) \cdot d_h - 1$
  (contando de 0). O $(i)$ é índice, não expoente;
- $\text{head}_i\, W_O^{(i)}$: $T \times d_h$ vezes $d_h \times d$ dá $T \times d$. É a
  contribuição da head $i$, já no tamanho do residual stream;
- $\sum_{i=1}^{h}$: soma as $h$ contribuições.

**Por que é verdade.** Um vetor linha $c$ multiplicado por uma matriz $W$ dá uma **combinação
das linhas de $W$**: o 1º número de $c$ multiplica a linha 1, o 2º multiplica a linha 2, e
assim por diante. Os números da head $i$ ocupam as posições do bloco $i$ do vetor concatenado,
então multiplicam só as linhas do bloco $i$, que são exatamente $W_O^{(i)}$.

**Exemplo com matrizes minúsculas** (conferido com Python). Uma posição ($T = 1$), $h = 2$,
$d_h = 2$, $d = 4$:

$$
\text{head}_1 = (1,\ 2), \qquad \text{head}_2 = (3,\ -1), \qquad
W_O = \begin{bmatrix} 1 & 0 & 2 & 1 \\ 0 & 1 & 1 & 0 \\ 2 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix}
$$

Os blocos: $W_O^{(1)}$ são as linhas 1 e 2; $W_O^{(2)}$ são as linhas 3 e 4.

| caminho | conta | resultado |
|---|---|---|
| esquerdo | $\text{Concat} = (1, 2, 3, -1)$; $1 \cdot (1,0,2,1) + 2 \cdot (0,1,1,0) + 3 \cdot (2,1,0,1) - 1 \cdot (1,1,1,0)$ | $(6,\ 4,\ 3,\ 4)$ |
| head 1 | $\text{head}_1 W_O^{(1)} = 1 \cdot (1,0,2,1) + 2 \cdot (0,1,1,0)$ | $(1,\ 2,\ 4,\ 1)$ |
| head 2 | $\text{head}_2 W_O^{(2)} = 3 \cdot (2,1,0,1) - 1 \cdot (1,1,1,0)$ | $(5,\ 2,\ -1,\ 3)$ |
| direito | $(1, 2, 4, 1) + (5, 2, -1, 3)$ | $(6,\ 4,\ 3,\ 4)$ |

Os dois caminhos dão $(6, 4, 3, 4)$.

**A saída é a soma das contribuições de cada head**, cada uma projetada para $d$ dimensões
pela sua própria matriz. Numa mesma camada, as heads não conversam entre si. Cada uma lê o
residual stream, calcula e **escreve de forma aditiva**. A combinação entre heads acontece
nas camadas seguintes, que leem o que foi escrito.

**No código.** `out_proj = nn.Linear(d_model, d_model, bias=False)` guarda `weight` com shape
`(128, 128)` e calcula `x @ weight.T`. Então $W_O$ = `out_proj.weight.T`, e as **linhas** de
$W_O$ são as **colunas** de `out_proj.weight`: $W_O^{(i)}$ = `out_proj.weight[:, i*d_h:(i+1)*d_h].T`.

**No bloco Transformer**, $W_O$ é a matriz que escreve de volta no residual stream:
$h \leftarrow h + \text{MultiHead}(\text{LN}(h))$. **Atenção:** nesta fórmula, e só nela, $h$ é
o **estado oculto** (o residual stream), não o número de heads; $\text{LN}$ é a LayerNorm; e
$\leftarrow$ significa "passa a valer". No código: `x = x + self.attention(self.ln_1(x))`.

O GPT-2 inicializa essa projeção com escala reduzida para estabilizar redes profundas. No
Mini-GPT isso já está implementado no bloco Transformer: o `TransformerBlock` re-inicializa
`attention.out_proj.weight` com desvio $0{,}02/\sqrt{2L}$ (`model/transformer_block.py`,
linhas 29–32). Com $L = 4$: $0{,}02/\sqrt{8} \approx 0{,}00707$. A explicação da escala está em
[07-transformer-block.md](07-transformer-block.md), e os efeitos no modelo completo, em
[08-gpt.md](08-gpt.md). Dentro da `MultiHeadAttention` isolada, `out_proj` começa com desvio
0,02, como as demais.

---

## 3. Implementação didática: uma lista de heads

**Objetivo.** Traduzir a fórmula da seção 2 ao pé da letra, reaproveitando a
`CausalSelfAttention` de [04-attention.md](04-attention.md). É a versão mais fácil de ler, e
serve de referência para a versão eficiente.

```python
class LoopMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        _check_heads(d_model, num_heads)
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList(CausalSelfAttention(d_model, head_dim, dropout)
                                   for _ in range(num_heads))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                  # [B, T, d]
        heads = [head(x) for head in self.heads]           # h × [B, T, d_h]
        concatenated = torch.cat(heads, dim=-1)            # [B, T, d]
        return self.dropout(self.out_proj(concatenated))   # [B, T, d]
```

**`__init__` passo a passo** (linhas 107–116 de `attention.py`):

1. `_check_heads`: garante que $d$ é divisível por $h$ (seção 2.1).
2. `head_dim = d_model // num_heads`: $128 // 4 = 32$.
3. `nn.ModuleList(...)`: cria $h$ heads **independentes**, cada uma com suas próprias
   `q_proj`, `k_proj` e `v_proj` de shape `(32, 128)`. A `ModuleList` registra as heads como
   submódulos. Com uma lista Python comum, os parâmetros delas não apareceriam em
   `parameters()`, e o otimizador não os treinaria.
4. `out_proj`: a matriz $W_O$, `(128, 128)`, sem bias.
5. `self.dropout`: o dropout aplicado na saída (seção 4.6).
6. `nn.init.normal_(self.out_proj.weight, std=0.02)`: inicialização $\mathcal{N}(0, 0{,}02^2)$.
   As heads já se inicializam sozinhas dentro de `CausalSelfAttention`.

**`forward` passo a passo** (linhas 118–122):

1. `[head(x) for head in self.heads]`: cada head recebe o **mesmo** `x` `[B, T, 128]` e devolve
   `[B, T, 32]`. Dentro de cada head acontece a attention inteira, inclusive o dropout nos pesos.
2. `torch.cat(heads, dim=-1)`: concatena na última dimensão. É o $\text{Concat}$ da fórmula:
   $4 \times 32 = 128$.
3. `self.out_proj(...)`: aplica $W_O$.
4. `self.dropout(...)`: dropout na saída.

É fácil de ler e corresponde exatamente ao diagrama. O custo é fazer $h$ vezes as mesmas
operações pequenas: 3 projeções e 2 multiplicações de matrizes por head.

Essa versão **não** é usada no modelo. Ela fica no código como **referência**: o teste
`test_fused_matches_loop` garante que a versão eficiente calcula a mesma coisa.

---

## 4. Implementação eficiente: todas as heads num tensor

**Objetivo.** Calcular exatamente o mesmo que a lista, mas com todas as heads dentro dos
mesmos tensores, para que cada operação seja **uma** chamada em vez de $h$. A versão usada na
prática aplica três truques.

### 4.1 Truque 1: empilhar as projeções

A head $i$ da lista tem `q_proj.weight` de shape `[d_h, d]`. Pela convenção do PyTorch
(`nn.Linear` calcula `x @ weight.T`), essa matriz guardada é $(W_Q^i)^\top$.

Empilhando as $h$ matrizes guardadas **verticalmente** (uma embaixo da outra), obtemos a
`q_proj.weight` da versão eficiente:

$$
\underbrace{\begin{bmatrix} (W_Q^1)^\top \\ (W_Q^2)^\top \\ \vdots \\ (W_Q^h)^\top \end{bmatrix}}_{[h \cdot d_h,\; d] \;=\; [d,\; d]}
$$

**Lendo a fórmula:**

- cada $(W_Q^i)^\top$ é um bloco de $d_h$ linhas e $d$ colunas ($32 \times 128$);
- empilhar $h$ blocos dá $h \cdot d_h = d$ linhas: $4 \times 32 = 128$;
- o subscrito embaixo da chave, $[h \cdot d_h,\ d] = [d,\ d]$, é o shape do resultado;
- na notação das fórmulas (sem transposta), o mesmo empilhamento aparece **lado a lado**:
  $W_Q = \begin{bmatrix} W_Q^1 & W_Q^2 & \cdots & W_Q^h \end{bmatrix} \in \mathbb{R}^{d \times d}$;
- cuidado com a orientação: como $W_Q^i \in \mathbb{R}^{d \times d_h}$, o que se empilha
  **verticalmente** são as matrizes guardadas pelo PyTorch, $(W_Q^i)^\top$.

Uma única `nn.Linear(d, d)` calcula as queries de **todas** as heads de uma vez. As
primeiras $d_h$ features da saída são a query da head 1, as próximas $d_h$ são da head 2, e
assim por diante. O mesmo vale para K e V.

**Por quê.** A linha $r$ de `q_proj.weight` produz a feature $r$ da saída (é o produto escalar
dessa linha com `x`). As linhas 0 a 31 produzem as features 0 a 31, que são justamente o bloco
da head 0. No `tiny.yaml`:

| head (código) | linhas de `q_proj.weight` | features da saída |
|---|---|---|
| 0 | 0 a 31 | 0 a 31 |
| 1 | 32 a 63 | 32 a 63 |
| 2 | 64 a 95 | 64 a 95 |
| 3 | 96 a 127 | 96 a 127 |

O teste `test_head_uses_only_its_block_of_weights` altera as linhas
`[d_h : 2 d_h]` de `q_proj.weight` e verifica que **só a head 1** muda. (No teste, $d = 16$,
$h = 4$ e $d_h = 4$: soma 1,0 às linhas 4 a 7 e confere que a lista de "mudou?" é
`[False, True, False, False]`.)

É esse mapeamento que `copy_heads` usa para transferir pesos da versão com lista:

```python
stacked = torch.cat([head.q_proj.weight for head in loop.heads], dim=0)  # [d, d]
fused.q_proj.weight.copy_(stacked)
```

**`copy_heads` passo a passo** (definida em `tests/test_multi_head_attention.py` e, igual, no
experimento):

1. Para cada nome em `("q_proj", "k_proj", "v_proj")`, pega a `weight` de cada head da lista:
   $h$ tensores `[d_h, d]`.
2. `torch.cat(..., dim=0)` empilha na dimensão 0 (linhas): `[h · d_h, d]` = `[d, d]`.
3. `.copy_(stacked)` escreve esses valores **dentro** do parâmetro já existente da versão
   eficiente (o `_` final indica operação *in-place*, que altera o próprio tensor).
4. `out_proj` não precisa empilhar: tem o mesmo shape `(d, d)` nas duas versões, então é
   copiada direto.
5. Tudo roda dentro de `torch.no_grad()` (nos testes, explícito; no experimento, o `main`
   inteiro roda assim). Sem isso, o PyTorch recusaria alterar in-place um parâmetro que exige
   gradiente.

Depois de `copy_heads`, as duas versões têm **os mesmos números** organizados de formas
diferentes, e podem ser comparadas (seção 5).

### 4.2 Truque 2: `view` separa as heads

**Objetivo de `split_heads`** (linhas 77–85): transformar `[B, T, d]` em `[B, h, T, d_h]`,
dando a cada head o seu bloco de features. Ela reinterpreta a última dimensão $d$ como $h$
blocos de $d_h$.

Na seção 2 do experimento, com $T = 3$, $d = 8$ e $h = 2$ (logo $d_h = 4$), `x` contém os
números 0 a 23, então cada valor é também a sua posição na memória:

```text
x (1, 3, 8): cada linha é uma posição, cada coluna uma feature
  t=0: [0, 1, 2, 3, 4, 5, 6, 7]
  t=1: [8, 9, 10, 11, 12, 13, 14, 15]
  t=2: [16, 17, 18, 19, 20, 21, 22, 23]

  head 0 (features 0..3): [[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]]
  head 1 (features 4..7): [[4, 5, 6, 7], [12, 13, 14, 15], [20, 21, 22, 23]]
```

**Passo 1** (linha 83): `x.view(batch, seq_len, num_heads, d_model // num_heads)`, aqui
`(1, 3, 2, 4)`. Cada vetor de 8 features passa a ser lido como 2 blocos de 4:

```text
t=0: [0, 1, 2, 3, 4, 5, 6, 7]  ──view──►  [[0, 1, 2, 3], [4, 5, 6, 7]]
                                            └─ head 0 ─┘  └─ head 1 ─┘
```

`x.view(B, T, h, d_h)` não copia nada: é o mesmo bloco de memória com outro shape. `view`
("visão") é isso: uma nova forma de **olhar** para os mesmos números.

### 4.3 Truque 3: `transpose` coloca as heads como dimensão de lote

**Passo 2 de `split_heads`** (linha 85): `x.transpose(1, 2)` troca as dimensões 1 e 2.

```text
[B, T, h, d_h]  ──transpose(1, 2)──►  [B, h, T, d_h]
```

**Por que trocar.** Em `[B, T, h, d_h]`, para cada posição temos as heads. Em
`[B, h, T, d_h]`, para cada head temos as posições. O segundo é o formato que a attention
precisa: para cada head, uma sequência de $T$ vetores.

As funções de [04-attention.md](04-attention.md) operam nas **duas últimas dimensões** (`[..., T, d_k]`), e o `@` do
PyTorch trata todas as dimensões anteriores como lote. Com as heads antes de $T$, **uma única
chamada** calcula a attention de todas as heads, de todas as sequências do batch:

- `q @ k.transpose(-2, -1)`: `[B, h, T, d_h] @ [B, h, d_h, T]` → `[B, h, T, T]`;
- a máscara `[T, T]` é expandida por broadcasting para todos os $B$ e $h$;
- `weights @ v`: `[B, h, T, T] @ [B, h, T, d_h]` → `[B, h, T, d_h]`.

Nenhuma linha de código de attention precisou mudar em relação a [04-attention.md](04-attention.md).

### 4.4 Voltando: `merge_heads`, strides e `contiguous()`

**Objetivo de `merge_heads`** (linhas 88–92): fazer o $\text{Concat}$, ou seja,
`[B, h, T, d_h]` → `[B, T, d]`. Desfazemos o caminho: `[B, h, T, d_h]` → `transpose` →
`[B, T, h, d_h]` → `view` → `[B, T, d]`. O `view` exige um cuidado.

**Como um tensor é guardado.** Um tensor do PyTorch é um bloco **1-D** de memória
(*storage*: os números guardados em fila, um atrás do outro) mais duas listas: `shape` e
`stride`. O **stride** diz quantas posições do storage andar para avançar 1 em cada dimensão.

A memória de `x` do exemplo, com seus 24 números:

```text
posição no storage:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
valor:               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                     └──────── t=0 ───────┘ └──────── t=1 ───────┘ └──────── t=2 ───────┘
                     └ head 0 ─┘└ head 1 ─┘ └ head 0 ─┘└ head 1 ─┘ └ head 0 ─┘└ head 1 ─┘
```

A fórmula abaixo diz onde, na fila de memória, está o elemento de um tensor com índices
$(n_0, n_1, \dots)$. É assim que o PyTorch encontra cada número sem copiar nada.

$$
\text{posição no storage} = \sum_{e} n_e \cdot s_e
$$

**Lendo a fórmula:**

- $e$: um eixo (dimensão) do tensor: 0, 1, 2, …;
- $n_e$: o índice usado nesse eixo (em `x[0, 2, 5]`, $n_0 = 0$, $n_1 = 2$, $n_2 = 5$);
- $s_e$: o stride desse eixo;
- $\sum_e$: soma um termo por eixo. (Existe ainda um deslocamento inicial, o
  `storage_offset`, que é 0 em todos os tensores deste exemplo.)

**Exemplos numéricos:**

- `x[0, 2, 5]`, com stride `(24, 8, 1)`: $0 \cdot 24 + 2 \cdot 8 + 5 \cdot 1 = 21$. De fato,
  a feature 5 de $t = 2$ vale 21.
- `split[0, 1, 2, 3]` (head 1, $t = 2$, feature 3 do bloco), com stride `(24, 4, 8, 1)`:
  $0 \cdot 24 + 1 \cdot 4 + 2 \cdot 8 + 3 \cdot 1 = 23$. De fato, o bloco da head 1 em $t = 2$
  é `[20, 21, 22, 23]`.

O que o experimento mostra:

```text
          x: shape (1, 3, 8)      stride (24, 8, 1)       contíguo: True
       view: shape (1, 3, 2, 4)   stride (24, 8, 4, 1)    contíguo: True
  transpose: shape (1, 2, 3, 4)   stride (24, 4, 8, 1)    contíguo: False
```

Um tensor é **contíguo** quando percorrer seus elementos na ordem do shape (o último índice
muda mais rápido, como num odômetro) corresponde a andar pelo storage **em sequência**, de 1
em 1, sem pulos.

- `x`: stride `(24, 8, 1)`. Avançar uma feature anda 1 casa; avançar uma posição $t$ anda 8
  (as 8 features de uma posição); avançar um batch anda 24.
- `view` escolheu novos strides compatíveis com a ordem da memória: continua contíguo.
  Avançar 1 dentro do bloco anda 1; avançar uma head anda 4 (o tamanho do bloco); avançar $t$
  anda 8.
- `transpose` **só troca os strides** `(8, 4) → (4, 8)`. É instantâneo e não copia nada, mas
  agora andar pela última dimensão não corresponde mais a andar em sequência na memória.

Percorrendo o resultado do `transpose` na ordem do seu shape (head, depois $t$, depois
feature), as posições visitadas no storage são:

```text
head 0: t=0 → 0 1 2 3 | t=1 → 8 9 10 11 | t=2 → 16 17 18 19
head 1: t=0 → 4 5 6 7 | t=1 → 12 13 14 15 | t=2 → 20 21 22 23
        (pula de 3 para 8, de 19 volta para 4...: não contíguo)
```

A saída da attention (`weights @ v`) é um tensor **novo**, contíguo no layout
`[B, h, T, d_h]`. Depois do `transpose` de volta, não existe um conjunto de strides que
descreva `[B, T, d]` sobre aquela memória, e o `view` falha.

**Em números** (conferido com Python). No experimento, `attention_output = split.contiguous()`
imita essa saída nova:

```text
storage da saída da attention [1, 2, 3, 4], stride (24, 12, 4, 1):
  0 1 2 3  8 9 10 11  16 17 18 19  |  4 5 6 7  12 13 14 15  20 21 22 23
  └───────── head 0 ─────────────┘    └───────── head 1 ──────────────┘

depois de transpose(1, 2): shape (1, 3, 2, 4), stride (24, 4, 12, 1)
para ler t=0 como 8 features seguidas, as posições no storage seriam:
  0 1 2 3 12 13 14 15   (passos 1, 1, 1, 9, 1, 1, 1)
```

Um `view` para `(1, 3, 8)` precisaria de **um único** stride para a última dimensão, mas os
passos são 1 e às vezes 9. Nenhum número serve, e o PyTorch levanta erro:

```text
view após transpose, sem contiguous(): RuntimeError: view size is not compatible with input tensor's size and str...
merge_heads (com contiguous) reconstrói x: True
```

`.contiguous()` copia os dados para um bloco novo, na ordem do shape atual, e aí o `view`
funciona. No exemplo, a cópia fica com storage `0 1 2 … 23` e stride `(24, 8, 4, 1)`, e o
`view` para `(1, 3, 8)` volta a ser possível. `reshape` faria a cópia automaticamente quando
necessário. O código usa `contiguous().view()` para deixar a cópia explícita.

Se o tensor já for contíguo, `.contiguous()` não copia nada e devolve o próprio tensor.

**`merge_heads` passo a passo** (linha 92):

1. `x.transpose(1, 2)`: `[B, h, T, d_h]` → `[B, T, h, d_h]`, só trocando strides.
2. `.contiguous()`: copia para memória na ordem `[B, T, h, d_h]`.
3. `.view(batch, seq_len, num_heads * head_dim)`: junta os $h$ blocos em $d$ features. É o
   $\text{Concat}$.

### 4.5 O forward completo, etapa por etapa

**`MultiHeadAttention.__init__`** (linhas 131–143):

1. `_check_heads(d_model, num_heads)`: $d$ divisível por $h$.
2. `self.num_heads = num_heads`: guardado para o `forward`. O `head_dim` não é guardado:
   `split_heads` o calcula como `d_model // num_heads`.
3. `q_proj`, `k_proj`, `v_proj`: três `nn.Linear(d, d, bias=False)`, cada uma com as $h$
   heads empilhadas (seção 4.1).
4. `out_proj`: $W_O$, `nn.Linear(d, d, bias=False)`.
5. `self.dropout`: um único `nn.Dropout(dropout)`, usado em dois lugares (seção 4.6).
6. As quatro matrizes começam com $\mathcal{N}(0, 0{,}02^2)$. Total: $4 \cdot 128^2 = 65.536$
   parâmetros.

**`MultiHeadAttention.forward`** (linhas 145–159). Seção 4 do experimento, com um batch real:

| etapa | shape | significado |
|---|---|---|
| `x` | `(32, 128, 128)` | `[B, T, d]` |
| `q_proj(x)` | `(32, 128, 128)` | queries de todas as heads, concatenadas |
| `split_heads` | `(32, 4, 128, 32)` | `[B, h, T, d_h]` |
| `pesos` | `(32, 4, 128, 128)` | uma matriz $T \times T$ por head |
| `pesos @ v` | `(32, 4, 128, 32)` | saída de cada head |
| `merge_heads` | `(32, 128, 128)` | heads concatenadas |
| `out_proj` | `(32, 128, 128)` | mistura das heads, `[B, T, d]` |

(O experimento percorre as etapas sem dropout: o módulo foi criado com `dropout=0.0`.)

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x):                                     # [B, T, d]
        seq_len = x.shape[1]
        q = split_heads(self.q_proj(x), self.num_heads)       # [B, h, T, d_h]
        k = split_heads(self.k_proj(x), self.num_heads)       # [B, h, T, d_h]
        v = split_heads(self.v_proj(x), self.num_heads)       # [B, h, T, d_h]
        mask = causal_mask(seq_len, device=x.device)          # [T, T]
        weights = attention_weights(q, k, mask)               # [B, h, T, T]
        weights = self.dropout(weights)                       # dropout nos pesos
        heads = weights @ v                                   # [B, h, T, d_h]
        out = self.out_proj(merge_heads(heads))               # [B, T, d]
        return self.dropout(out)                              # dropout na saída
```

Passo a passo:

1. **Projeções** (linhas 148–150): `q_proj(x)` dá `[32, 128, 128]`, com as queries das 4 heads
   lado a lado; `split_heads` separa em `[32, 4, 128, 32]`. O mesmo para K e V.
2. **Máscara** (linha 152): `causal_mask(128)` é `[128, 128]`, criada no mesmo dispositivo
   (CPU ou GPU) de `x`. Serve para todas as sequências e heads, por broadcasting.
3. **Pesos** (linha 153): `attention_weights` faz escala por $\sqrt{d_h} = \sqrt{32}$, máscara e
   softmax: `[32, 4, 128, 128]`. Note que a escala usa $d_h$, não $d$: é `q.shape[-1]`.
4. **Dropout nos pesos** (linha 154).
5. **Soma ponderada** (linha 155): `weights @ v` dá `[32, 4, 128, 32]`, a saída das 4 heads.
6. **Concat + $W_O$** (linha 157): `merge_heads` junta em `[32, 128, 128]` e `out_proj` mistura.
7. **Dropout na saída** (linha 159).

### 4.6 Dropout: nos pesos e na saída

**Dropout** é uma técnica de regularização (para o modelo não decorar os dados de treino).
Durante o treino, cada número é zerado com probabilidade $p$, e os que sobram são
aumentados para compensar. A fórmula calcula o valor de um elemento depois do dropout.

$$
\text{dropout}(a)_n = \frac{m_n\, a_n}{1 - p}, \qquad m_n = \begin{cases} 1 & \text{com probabilidade } 1 - p \\ 0 & \text{com probabilidade } p \end{cases}
$$

**Lendo a fórmula:**

- $a_n$: o elemento $n$ do tensor de entrada;
- $m_n$: um sorteio independente para cada elemento: 1 mantém, 0 zera;
- $p$: a probabilidade de zerar; no `tiny.yaml`, $p = 0{,}1$;
- $\frac{1}{1-p}$: o fator de compensação, $1/0{,}9 \approx 1{,}111$. Com ele, a **média** do
  valor fica igual à de antes: $(1 - p) \cdot \frac{a_n}{1-p} = a_n$;
- em `eval()`, o dropout é desligado: devolve a entrada sem mudança.

**Exemplo numérico:** uma linha de pesos $(0{,}5;\ 0{,}5)$, com sorteio $m = (1, 0)$ e
$p = 0{,}1$, vira $(0{,}5/0{,}9;\ 0) \approx (0{,}556;\ 0)$. A linha deixa de somar 1 durante o
treino, como já visto em [04-attention.md](04-attention.md).

A `MultiHeadAttention` aplica o mesmo `self.dropout` em **dois lugares**:

| onde | linha | efeito | nome no GPT-2 |
|---|---|---|---|
| nos pesos `[B, h, T, T]` | 154 | corta ligações aleatórias entre posições, em cada head | `attn_pdrop` |
| na saída `[B, T, d]` | 159 | zera partes do que será somado ao residual stream | `resid_pdrop` |

O dropout na saída foi acrescentado no bloco Transformer, simétrico ao `FeedForward` (ver
[07-transformer-block.md](07-transformer-block.md)). A `LoopMultiHeadAttention` tem os mesmos
dois pontos: cada `CausalSelfAttention` aplica dropout nos seus pesos, e o módulo aplica na
saída. Como os sorteios são aleatórios, as duas versões só dão números iguais com
`dropout=0.0` ou em `eval()`, que é como os testes e o experimento as comparam.

---

## 5. As duas versões calculam a mesma coisa

Seção 3 do experimento, com pesos copiados via `copy_heads`:

```text
entrada (32, 128, 128), d_model = 128, num_heads = 4
  max |lista − tensor único|:    0.00e+00
  max |tensor único − PyTorch|:  2.61e-08
```

**Como ler:** `max |a − b|` é a maior diferença absoluta entre as duas saídas, olhando todos os
$32 \times 128 \times 128$ números. `2.61e-08` significa $2{,}61 \times 10^{-8} = 0{,}0000000261$.

- **Lista × tensor único:** diferença zero neste batch. O teste confirma para
  $h \in \{1, 2, 4, 8\}$.
- **Tensor único × `nn.MultiheadAttention`:** a implementação oficial do PyTorch, com
  `bias=False` e os mesmos pesos (`in_proj_weight` = Q, K e V empilhados), dá o mesmo resultado
  a menos de arredondamento de ponto flutuante.

`in_proj_weight` tem shape `[3d, d]`: são `q_proj.weight`, `k_proj.weight` e `v_proj.weight`
empilhados um embaixo do outro, com `torch.cat` na dimensão 0.

**Atenção à convenção de máscara.** Em `nn.MultiheadAttention`, `True` em `attn_mask`
significa **proibido**, o inverso da nossa `causal_mask`. Por isso o teste passa
`attn_mask=~causal_mask(T)` (o `~` inverte True e False).

---

## 6. Desempenho: quando o tensor único ganha

É comum ouvir que a versão com tensor único é "mais rápida". As medições deste projeto
mostram que a história é mais interessante.

Três variantes, com os mesmos pesos:

- **lista**: `LoopMultiHeadAttention`;
- **único**: `MultiHeadAttention`, com a attention escrita à mão (como em [04-attention.md](04-attention.md));
- **único + SDPA**: o mesmo layout `[B, h, T, d_h]`, trocando só as linhas da attention por
  `F.scaled_dot_product_attention`, o kernel otimizado do PyTorch.

Dois termos:

- **SDPA** é a sigla de *scaled dot-product attention*, e aqui designa a função
  `F.scaled_dot_product_attention` do PyTorch;
- um **kernel** é uma rotina de baixo nível (C++, CUDA, Metal), escrita e otimizada para um
  tipo de hardware, que executa uma operação inteira de uma vez.

Uma execução num Mac (ms por forward; a seção 5 do experimento repete a parte de CPU, e os
números variam entre execuções):

| dispositivo | entrada | $h$ | lista | único | único + SDPA |
|---|---|---:|---:|---:|---:|
| CPU | `[1, 16, 128]` | 4 | 0,197 | 0,072 | 0,038 |
| CPU | `[1, 16, 128]` | 16 | 0,737 | 0,085 | 0,044 |
| CPU | `[32, 128, 128]` | 4 | 2,443 | 3,071 | 1,414 |
| CPU | `[32, 128, 128]` | 16 | 9,328 | 10,226 | 4,670 |
| GPU (MPS) | `[1, 16, 128]` | 4 | 0,389 | 0,133 | 0,095 |
| GPU (MPS) | `[1, 16, 128]` | 16 | 1,475 | 0,120 | 0,104 |
| GPU (MPS) | `[32, 128, 128]` | 4 | 1,786 | 2,009 | 0,728 |
| GPU (MPS) | `[32, 128, 128]` | 16 | 7,931 | 7,937 | 7,401 |

**Como calcular o ganho:** divida o tempo da lista pelo tempo da outra versão. Na primeira
linha: $0{,}197 / 0{,}072 \approx 2{,}7$, ou seja, o tensor único é 2,7× mais rápido. Um valor
menor que 1 significa **mais lento**: $2{,}443 / 3{,}071 \approx 0{,}8$.

| dispositivo | entrada | $h$ | lista ÷ único | lista ÷ (único + SDPA) |
|---|---|---:|---:|---:|
| CPU | `[1, 16, 128]` | 4 | 2,7× | 5,2× |
| CPU | `[1, 16, 128]` | 16 | 8,7× | 16,8× |
| CPU | `[32, 128, 128]` | 4 | 0,8× | 1,7× |
| CPU | `[32, 128, 128]` | 16 | 0,9× | 2,0× |
| GPU (MPS) | `[1, 16, 128]` | 4 | 2,9× | 4,1× |
| GPU (MPS) | `[1, 16, 128]` | 16 | 12,3× | 14,2× |
| GPU (MPS) | `[32, 128, 128]` | 4 | 0,9× | 2,5× |
| GPU (MPS) | `[32, 128, 128]` | 16 | 1,0× | 1,1× |

(Razões calculadas com Python a partir da tabela de tempos acima.)

**1. As versões fazem a mesma quantidade de aritmética.** $h$ projeções `128 → 32` somam o
mesmo número de multiplicações que uma projeção `128 → 128`, e o mesmo vale para as matrizes
de attention. Qualquer diferença de tempo vem de **como** as contas são despachadas, não de
quantas são.

Exemplo: 4 projeções `128 → 32` fazem $4 \times 32 \times 128 = 16.384$ multiplicações; uma
projeção `128 → 128` faz $128 \times 128 = 16.384$.

**2. Com entradas pequenas, o tensor único ganha muito.** Com `[1, 16, 128]`, cada operação é
barata, e o **custo fixo de cada chamada** domina. Esse custo fixo (em inglês, *overhead*) é o
trabalho que se paga a cada operação, **independente do tamanho do tensor**: Python, despacho
do PyTorch, alocação de memória e, na GPU, o lançamento do kernel. A lista faz $\approx 5h$
chamadas; o tensor único faz um número fixo, independente de $h$. Resultado: 2,7× a 8,7× mais
rápido na CPU e até 12× na GPU, com ganho que cresce com $h$. É o regime da geração token a
token (ver [13-generation.md](13-generation.md)), com batch 1.

**3. Com entradas grandes, não há ganho.** Com `[32, 128, 128]`, cada operação já é pesada,
e o custo fixo por chamada fica desprezível. O tempo é dominado pelas operações sobre
`[B, h, T, T]` (escala, máscara, softmax), que as duas versões pagam igualmente. O tensor
único ainda paga a cópia do `contiguous()` e fica até um pouco mais lento (0,8× a 1,0×).

Uma primeira medição neste projeto mostrou "2,1× mais rápido" com 16 heads. Repetida, virou
0,9×. **Medições de tempo são ruidosas: repita antes de concluir.**

**4. O que realmente importa é o layout.** `[B, h, T, d_h]` é o formato de entrada dos kernels
otimizados. Trocando só a attention por `F.scaled_dot_product_attention`, o mesmo módulo fica
1,7× a 2,5× mais rápido que a lista com entradas grandes (a exceção nesta execução é a GPU com
16 heads, só 1,1×). Esses kernels fundem escala, máscara, softmax e multiplicação sem
**materializar** os tensores intermediários (sem nunca guardar na memória a matriz
`[B, h, T, T]` inteira), e em GPUs compatíveis usam **FlashAttention**: um algoritmo que calcula
o mesmo resultado processando a attention em blocos que cabem na memória rápida da GPU (ver
[04-attention.md](04-attention.md), seção 6). A lista de heads não consegue aproveitar isso de
forma natural.

**Por que o modelo não usa o kernel por padrão?** Porque o objetivo deste documento é entender
a conta, e as três linhas escritas à mão mostram exatamente o que acontece. O teste
`test_matches_pytorch_implementation` (em [04-attention.md](04-attention.md)) garante que os
resultados são iguais, então a troca pode ser feita com segurança — é o que
[17-performance.md](17-performance.md) faz, tornando o kernel otimizado uma opção de config.

---

## 7. Na inicialização

A seção 4 do experimento mostra a linha 5 (6 posições visíveis, uniforme = 0,167) de cada head.
A linha 5 é a posição $t = 5$: pela máscara causal, ela enxerga as posições 0 a 5, e peso
uniforme seria $1/6 \approx 0{,}167$.

```text
  head 0: 0.161 0.185 0.162 0.149 0.182 0.161
  head 1: 0.181 0.173 0.156 0.160 0.177 0.153
  head 2: 0.168 0.161 0.166 0.163 0.169 0.172
  head 3: 0.161 0.166 0.167 0.164 0.172 0.170
```

Todas estão perto do uniforme (ver [04-attention.md](04-attention.md), seção 5), mas **cada uma
de um jeito ligeiramente diferente**. Por exemplo, a head 0 dá o maior peso (0,185) à posição 1,
e a head 1 dá o maior peso (0,181) à posição 0.

Essas pequenas diferenças aleatórias são essenciais. Se todas as heads, e os blocos
correspondentes de $W_O$, começassem **idênticos**, receberiam gradientes idênticos e
continuariam idênticas para sempre: seriam $h$ cópias da mesma head.

**Por quê.** Heads idênticas calculam a mesma saída e ocupam posições equivalentes na fórmula.
O gradiente de cada uma, portanto, é o mesmo, e o otimizador aplica a mesma atualização a
todas. Depois do passo, continuam iguais, e o argumento se repete a cada passo.

A inicialização aleatória **quebra essa simetria** (isto é, garante que as heads comecem
diferentes, para que o gradiente possa levá-las a caminhos diferentes), e o treino amplifica as
diferenças até cada head se especializar. Visualizar essas especializações depois do treino é
uma observação opcional interessante — ver [14-evaluation.md](14-evaluation.md).

---

## 8. Decisões de projeto

| decisão | escolha | motivo |
|---|---|---|
| versão usada no modelo | tensor único | menos chamadas (ganho com entradas pequenas) e layout aceito pelos kernels otimizados |
| versão com lista | mantida no código | referência legível e oráculo dos testes |
| $d_h$ | $d / h$ | mesmo orçamento de parâmetros de uma head com $d_k = d$ |
| bias em $W_O$ | não | coerente com Q, K e V (ver [04-attention.md](04-attention.md)) |
| inicialização de $W_O$ | $\mathcal{N}(0, 0{,}02^2)$ dentro do módulo; no `TransformerBlock`, $\mathcal{N}(0, (0{,}02/\sqrt{2L})^2)$ | por padrão, igual às demais; o bloco Transformer aplica a escala por profundidade do GPT-2 ($\approx 0{,}00707$ com $L = 4$); ver [07](07-transformer-block.md) e [08](08-gpt.md) |
| dropout | nos pesos `[B, h, T, T]` e, também na saída (bloco Transformer) | como no GPT-2 (`attn_pdrop` e `resid_pdrop`); ver [07](07-transformer-block.md) |
| `contiguous().view()` | em vez de `reshape` | deixa a cópia de memória explícita |

---

## 9. O que cada teste garante

Os testes usam $d = 16$ e $h = 4$ (logo $d_h = 4$), para rodar rápido.

| teste | propriedade |
|---|---|
| `test_split_heads_shape` | `[B, T, d]` → `[B, h, T, d_h]` |
| `test_each_head_gets_a_contiguous_block_of_features` | a head $i$ recebe as features $[i d_h, (i+1) d_h)$ |
| `test_merge_undoes_split` | `merge_heads(split_heads(x)) == x` |
| `test_merge_works_on_a_fresh_attention_output` | `merge_heads` funciona com tensores não contíguos após o `transpose` |
| `test_head_uses_only_its_block_of_weights` | as linhas $[i d_h, (i+1) d_h)$ de `q_proj.weight` são a head $i$ |
| `test_output_shape` | `[B, T, d]` → `[B, T, d]` |
| `test_d_model_must_be_divisible_by_num_heads` | erro claro se $d \bmod h \neq 0$ |
| `test_fused_matches_loop` | tensor único = lista de heads, para $h \in \{1, 2, 4, 8\}$ |
| `test_matches_pytorch_multihead_attention` | tensor único = `nn.MultiheadAttention` |
| `test_future_tokens_do_not_affect_past` | causalidade preservada |
| `test_gradient_only_reaches_current_and_past_inputs` | Jacobiana triangular inferior |
| `test_gradients_reach_all_projections` | Q, K, V e $W_O$ recebem gradiente |
| `test_both_versions_have_4_d_model_squared_parameters` | $4d^2$ parâmetros nas duas versões |
| `test_deterministic_in_eval_mode` | reprodutibilidade |

Notas sobre alguns testes:

- **`[i d_h, (i+1) d_h)`**: intervalo que inclui $i d_h$ e exclui $(i+1) d_h$. Com $d_h = 4$,
  a head 1 recebe as features 4, 5, 6 e 7.
- **`test_future_tokens_do_not_affect_past`**: troca as posições 6 a 9 de uma sequência de 10
  e confere que as saídas das posições 0 a 5 não mudam.
- **`test_gradient_only_reaches_current_and_past_inputs`**: faz o backward a partir da saída da
  posição 2 e confere que o gradiente é positivo para as entradas 0, 1 e 2 e **exatamente zero**
  para 3, 4 e 5. A **Jacobiana** (matriz de derivadas saída × entrada) é triangular inferior.
- **`test_deterministic_in_eval_mode`**: cria o módulo duas vezes com a mesma seed e
  `dropout=0.1`, em `eval()`, e confere saídas iguais.

---

## Resumo

1. Uma head produz **uma média ponderada** por posição; informações de lugares diferentes
   se misturam no mesmo vetor, e a ordem se perde.
2. Multi-head roda $h$ heads com $d_h = d/h$: cada uma com seu padrão de atenção e seu
   próprio bloco de coordenadas, **sem parâmetros extras** ($h \cdot 3 d d_h = 3d^2$, mais
   $d^2$ de $W_O$: $4d^2 = 65.536$ no `tiny.yaml`).
3. A concatenação mantém as heads separadas, e $W_O$ soma a contribuição de cada uma no
   espaço de $d$ dimensões: $\text{Concat}(\dots) W_O = \sum_i \text{head}_i W_O^{(i)}$.
4. Na prática, as heads vivem num único tensor: projeções empilhadas, `view` para separar,
   `transpose` para tratá-las como lote, `contiguous().view()` para juntar.
5. `transpose` só troca strides e deixa o tensor não contíguo; por isso `merge_heads` precisa
   de `contiguous()` antes do `view`.
6. Dropout aparece duas vezes: nos pesos `[B, h, T, T]` e na saída. O `TransformerBlock`
   inicializa `out_proj` com desvio $0{,}02/\sqrt{2L}$.
7. As duas implementações dão o mesmo resultado. O tensor único ganha com entradas pequenas
   (menos chamadas), empata com entradas grandes, e é o layout que permite usar kernels
   otimizados como `F.scaled_dot_product_attention`.

---

## Para checar o entendimento

1. Com $d = 128$, compare $h = 1$ e $h = 128$ ($d_h = 1$): quantos parâmetros cada uma tem? O
   que cada head consegue expressar no segundo caso?
2. Por que uma head não distingue `"gato"` de `"gtao"` na última posição? Que informação os
   values precisariam carregar para distinguir, e qual seria o custo?
3. Mostre que $\text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O = \sum_i \text{head}_i W_O^{(i)}$.
   O que isso diz sobre a interação entre heads dentro de uma mesma camada?
4. O que aconteceria se `attention_weights` recebesse `[B, T, h, d_h]`, sem o `transpose`?
   Que shape teriam os "pesos", e o que eles significariam?
5. Por que `view` falha depois do `transpose` e `reshape` não? Qual dos dois pode copiar
   memória?
6. Quais linhas de `k_proj.weight` pertencem à head 3 no `tiny.yaml`?
7. Qual o posto máximo de $Q_i K_i^\top$ com $d_h = 32$ e $T = 128$? Por que isso não
   impede a softmax de produzir uma matriz de pesos de posto maior?
8. Se todas as heads fossem inicializadas com os mesmos pesos, por que elas nunca se
   diferenciariam durante o treino?
9. No exemplo de 24 números, qual posição do storage guarda `x[0, 1, 6]`? E `split[0, 0, 2, 1]`
   (stride `(24, 4, 8, 1)`)? Confira com os valores impressos pelo experimento.
10. Com $h = 2$ e $d_h = 2$ no exemplo minúsculo da seção 2.2, troque $\text{head}_2$ por
    $(0, 0)$. Quais linhas de $W_O$ deixam de influenciar a saída?
11. Por que as duas versões (lista e tensor único) só dão resultados iguais com `dropout=0.0`
    ou em `eval()`?
