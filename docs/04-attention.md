# Scaled Dot-Product Attention

> **Como o modelo decide quais tokens são relevantes?**
> **Por que o modelo não pode olhar para o futuro?**

Código: [`model/attention.py`](../model/attention.py). Testes:
[`tests/test_attention.py`](../tests/test_attention.py). Experimento:
`uv run python -m experiments.e04_attention`. Visão geral: [architecture.md](architecture.md).
Guia de notação: [00-notacao.md](00-notacao.md).

---

## Para que serve

### O problema

Depois dos embeddings ([03-embeddings.md](03-embeddings.md)), cada posição tem um vetor que sabe duas
coisas: **qual é o seu caractere** e **em que posição ele está**. Nenhuma posição sabe nada
sobre as outras.

Mas prever o próximo caractere exige contexto. Depois de `"o gat"`, a resposta `'o'` só é
provável porque antes vieram `'g'`, `'a'` e `'t'`. O vetor da posição de `'t'`, sozinho,
não tem essa informação.

A attention é o mecanismo que **move informação entre posições**. É a única operação do
GPT que faz isso: feed-forward e LayerNorm atuam em cada posição isoladamente.

```text
[B, T, d]  ──►  attention  ──►  [B, T, d_v]
cada posição só                 cada posição com uma mistura
conhece a si mesma              de informação das posições anteriores
```

### Uma analogia

Pense num leitor que lê um texto letra por letra e **não pode espiar adiante**. Ao chegar
em cada letra, ele volta os olhos para o que já leu e decide em quais trechos prestar mais
atenção. Depois, resume o que viu nesses trechos numa anotação ao lado da letra atual.

- "decidir onde prestar atenção" = calcular os **pesos de attention**;
- "resumir o que viu" = fazer uma **média ponderada** do conteúdo das posições anteriores;
- "não espiar adiante" = a **causal mask** (máscara causal).

### O que entra e o que sai

Valores do [`configs/tiny.yaml`](../configs/tiny.yaml): $B = 32$, $T = 128$, $d = 128$,
$h = 4$ heads, logo $d_k = d_v = d / h = 32$.

| tensor | o que é | shape | no `tiny.yaml` |
|---|---|---|---|
| entrada `x` | um vetor por posição (vindo dos embeddings ou do bloco anterior) | `[B, T, d]` | `[32, 128, 128]` |
| `q`, `k` | queries e keys (o que cada posição procura / como se anuncia) | `[B, T, d_k]` | `[32, 128, 32]` |
| `v` | values (o que cada posição entrega) | `[B, T, d_v]` | `[32, 128, 32]` |
| pesos | quanto cada posição olha para cada outra | `[B, T, T]` | `[32, 128, 128]` |
| saída | um vetor por posição, agora com contexto | `[B, T, d_v]` | `[32, 128, 32]` |

Este documento implementa **uma head**. Uma *head* ("cabeça") é um cálculo completo de
attention, com suas próprias matrizes $W_Q$, $W_K$ e $W_V$.
[05-multi-head-attention.md](05-multi-head-attention.md) roda $h = 4$ heads em paralelo e
junta as saídas de volta em `[32, 128, 128]`.

### Onde fica no pipeline

```text
texto (corpus)
  │  tokenizer                                        ids: [N]
  ▼
batches                                              x, y: [B, T] = [32, 128]
  │
  ▼
embeddings                                           [B, T, d] = [32, 128, 128]
  │
  ▼  ┌─ bloco Transformer (× L = 4) ───────────────────────────┐
     │  LayerNorm → attention  ← este documento             │   [32, 128, 128]
     │  LayerNorm → feed-forward                             │   [32, 128, 128]
     └──────────────────────────────────────────────────────┘
  │
  ▼  LayerNorm final                                  [32, 128, 128]
  ▼  LM head                                          logits: [B, T, V] = [32, 128, 97]
  ▼  loss (ainda não implementada)                    um número
```

### O que daria errado sem esta parte

- **Sem attention:** a previsão da posição $t$ dependeria só do caractere $x_t$ e da posição
  $t$. O modelo viraria uma tabela "caractere atual → próximo caractere" (um modelo de
  bigramas). Ele nunca saberia que `"o gat"` pede `'o'`: só veria o `'t'`.
- **Com attention, mas sem a máscara causal:** cada posição enxergaria também os caracteres
  seguintes, inclusive a resposta que deveria prever. A loss cairia para perto de zero no
  treino, mas o modelo não aprenderia linguagem. Na geração, em que o futuro não existe, ele
  falharia (seção 3).

---

## Notação deste documento

Os símbolos gerais (soma, produto escalar, transposta, softmax, derivadas) estão explicados
com exemplos em [00-notacao.md](00-notacao.md). Aqui está **tudo** o que aparece neste
documento, com o sentido usado aqui.

### Tamanhos

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $B$ | "bê" | tamanho do batch: quantas sequências juntas | 32 | `batch_size`, `x.shape[0]` |
| $T$ | "tê" | número de posições (tokens) por sequência | 128; 4 no exemplo `"gato"` | `seq_len`, `context_length` |
| $d$ | "dê" | largura do vetor de entrada de cada posição | 128; 6 no exemplo `"gato"` | `d_model` |
| $d_k$ | "dê kê" | dimensão de cada query e de cada key ($k$ de *key*) | 32; 3 no exemplo `"gato"` | `head_dim`, `q.shape[-1]` |
| $d_v$ | "dê vê" | dimensão de cada value e da saída ($v$ de *value*) | 32; 3 no exemplo `"gato"` | `head_dim` |
| $h$ | "agá" | número de heads ([05-multi-head-attention.md](05-multi-head-attention.md)) | 4 | `num_heads` |
| $n$ | "ene" | quantas posições há numa linha da softmax (genérico) | $i + 1$ na linha $i$; 16 na seção 2.3 | — |

Neste documento, $d_k = d_v$ = `head_dim`. As fórmulas usam os dois nomes porque eles **podem**
ser diferentes (na seção 4, $d_k = 6$ e $d_v = 5$).

### Tensores e índices

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $X$ | "xis" | entrada: uma linha por posição | `[T, d]`; com batch `[32, 128, 128]` | `x`, `X` |
| $x_i$ | "xis i" | vetor da posição $i$ (linha $i$ de $X$) | `[d]` | `x[:, i]` |
| $Q$ | "quê" | matriz de **queries** (perguntas), uma linha por posição | `[T, d_k]` = `[128, 32]` | `q` |
| $K$ | "kê" | matriz de **keys** (etiquetas), uma linha por posição | `[T, d_k]` = `[128, 32]` | `k` |
| $V$ | "vê" | matriz de **values** (conteúdos), uma linha por posição. **Aqui $V$ não é `vocab_size`**: neste documento $V$ sempre significa values | `[T, d_v]` = `[128, 32]` | `v` |
| $q_i$ | "quê i" | query da posição $i$ (linha $i$ de $Q$) | `[d_k]` | `q[..., i, :]` |
| $k_j$ | "kê j" | key da posição $j$ (linha $j$ de $K$) | `[d_k]` | `k[..., j, :]` |
| $v_j$ | "vê j" | value da posição $j$ (linha $j$ de $V$) | `[d_v]` | `v[..., j, :]` |
| $W_Q$, $W_K$ | "dáblio quê", "dáblio kê" | matrizes aprendidas que geram queries e keys | `[d, d_k]` = `[128, 32]` | `q_proj.weight.T`, `k_proj.weight.T` (guardadas como `[32, 128]`) |
| $W_V$ | "dáblio vê" | matriz aprendida que gera values | `[d, d_v]` = `[128, 32]` | `v_proj.weight.T` |
| $S$ | "ésse" | matriz de **scores** $QK^\top$ (afinidades, antes da escala) | `[T, T]` | `scores` na linha 24 |
| $S_{ij}$ | "ésse i j" | score da posição $i$ pela posição $j$ | um número | `scores[..., i, j]` |
| $\tilde S$ | "ésse til" | scores já escalados e mascarados, prontos para a softmax | `[T, T]` | `scores` depois da linha 29 |
| $M$ | "eme" | máscara causal aditiva: 0 onde pode olhar, $-\infty$ onde não pode | `[T, T]` | `causal_mask(T)` (versão booleana) + `masked_fill` |
| $w_{ij}$ | "dáblio i j" | **peso de attention**: fração da atenção da posição $i$ dada à posição $j$ | entre 0 e 1 | `weights[..., i, j]` |
| $W$ (sem índice) | "dáblio" | matriz de todos os pesos $w_{ij}$. Não confundir com $W_Q$, $W_K$, $W_V$ | `[T, T]`; com batch `[32, 128, 128]` | `weights` |
| $o_i$ | "ó i" | saída da attention na posição $i$ | `[d_v]` | `out[..., i, :]` |
| $O$ | "ó" | matriz de saídas, uma linha por posição. Não confundir com o $O(\cdot)$ de custo | `[T, d_v]`; com batch `[32, 128, 32]` | valor devolvido |
| $i$ | "i" | posição que **pergunta** (linha das matrizes $T \times T$) | 0 a $T - 1$ | `i` |
| $j$, $j'$ | "j", "j linha" | posição **consultada** (coluna); $j'$ é só o índice da soma no denominador da softmax | 0 a $i$ | `j` |
| $m$ | "eme" | índice de uma coordenada dentro de um vetor | 1 a $d_k$ | — |
| $\theta_{ij}$ | "teta i j" | ângulo entre $q_i$ e $k_j$ | cerca de 65° no exemplo da seção 2.2 | — |
| $s_j$, $p_i$ | "ésse j", "pê i" | na seção 2.3: entradas (scores) e saídas (probabilidades) de uma softmax genérica | — | — |
| $J$ | "jota" | **Jacobiana**: matriz de todas as derivadas de saídas em relação a entradas | `[n, n]` na seção 2.3 | — |
| $P$ | "pê" | na seção 3.4: **matriz de permutação** (reordena posições). Aqui não é o positional embedding de [03-embeddings.md](03-embeddings.md) | `[T, T]` | — |
| $I$ | "i" maiúsculo | matriz identidade (1 na diagonal, 0 fora) | `[T, T]` | `torch.eye(T)` |
| $\beta$ | "beta" | na seção 4: intensidade da query na head construída à mão. Não é o $\beta$ da LayerNorm | $8\sqrt{6} \approx 19{,}6$ | `strength` |
| $e_p$ | "e pê" | vetor **one-hot**: todo zero, com um 1 na coordenada $p$. Não confundir com o número $e$ | — | `torch.eye(n)[p]`, `F.one_hot` |
| $p$ | "pê" | posição (seções 1.3 e 4) **ou** probabilidade de dropout (seção 7.3), conforme o contexto | dropout = 0,1 | `p`, `dropout` |
| $c$ | "cê" | um caractere (seções 1.3 e 4) **ou** uma constante somada aos scores (seção 2.5) | — | — |
| $H$ | "agá" maiúsculo | entropia de uma linha de pesos (seção 5) | entre 0 e $\ln n$ | `entropy` |

### Operadores e símbolos

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $\text{Attention}(Q, K, V)$ | "attention de Q, K, V" | a operação inteira deste documento | `[T, d_v]` | `scaled_dot_product_attention` |
| $a \cdot b$ | "a escalar b" | **produto escalar**: multiplica elemento a elemento e soma; dá um número | $q_i \cdot k_j$ | `torch.dot(a, b)`, `(a * b).sum()` |
| $AB$ | "A vezes B" | produto de matrizes | `[T, d] @ [d, d_k]` → `[T, d_k]` | `A @ B` |
| $A^\top$ | "A transposta" | troca linhas por colunas | `[T, d_k]` → `[d_k, T]` | `A.T`, `k.transpose(-2, -1)` |
| $\sqrt{d_k}$ | "raiz de dê kê" | raiz quadrada, o fator de escala | $\sqrt{32} \approx 5{,}66$; $\sqrt{3} \approx 1{,}732$ | `math.sqrt(d_k)` |
| $\lVert q \rVert$ | "norma de q" | comprimento do vetor | 0,862 para $q_t$ na seção 2.2 | `q.norm(dim=-1)` |
| $\cos\theta$ | "cosseno de teta" | quanto dois vetores apontam na mesma direção (−1 a 1) | 0,42 na seção 2.2 | `F.cosine_similarity` |
| $-\infty$ | "menos infinito" | valor que a softmax transforma em peso 0 | — | `float("-inf")` |
| $e^x$, $\exp(x)$ | "e elevado a x", "exponencial de x" | exponencial; $e \approx 2{,}718$; $e^{-\infty} = 0$ | $e^8 \approx 2981$ | `torch.exp(x)` |
| $\ln x$ | "logaritmo natural de x" | inverso da exponencial | $\ln 128 \approx 4{,}85$ | `torch.log(x)` |
| $\text{softmax}$ | "softmax" | transforma uma lista de números em probabilidades (positivas, somando 1) | por linha, `[T, T]` | `torch.softmax(scores, dim=-1)` |
| $\sum_{j \le i}$ | "soma para j até i" | soma sobre as posições visíveis $j = 0, 1, \dots, i$ | — | `.sum(dim=-1)` |
| $\in \mathbb{R}^{T \times d}$ | "pertence a erre T por d" | "é uma matriz de números reais com $T$ linhas e $d$ colunas" | — | `.shape == (T, d)` |
| $\mathbb{E}[\cdot]$ | "esperança de" | média teórica de uma variável aleatória | — | `.mean()` (estimativa) |
| $\text{Var}(\cdot)$ | "variância de" | quanto os valores se espalham em torno da média | $\text{Var}(q \cdot k) = d_k$ | `.var()` |
| $\mathcal{N}(0,\ 0{,}02^2)$ | "normal com média 0 e desvio 0,02" | distribuição usada para inicializar pesos | desvio 0,02 | `nn.init.normal_(w, std=0.02)` |
| $\delta_{ij}$ | "delta i j" | **delta de Kronecker**: 1 se $i = j$, 0 se não | — | `torch.eye(n)[i, j]` |
| $\partial y / \partial x$ | "derivada parcial de y em relação a x" | quanto $y$ muda quando só $x$ muda um pouco | — | `y.backward()`, depois `x.grad` |
| $[\,j = i-1\,]$ | "indicadora de j igual a i menos 1" | 1 se a condição é verdadeira, 0 se não (o mesmo que $\mathbb{1}[\cdot]$) | — | `(j == i - 1)` |
| $O(T^2)$ | "ó de tê ao quadrado" | o custo cresce proporcionalmente a $T^2$ | dobrar $T$ quadruplica o custo | — |
| $\tfrac{1}{1-p}$ | "um sobre um menos p" | fator que o dropout aplica aos valores que sobrevivem | $1/0{,}9 \approx 1{,}111$ | dentro de `nn.Dropout` |
| $\approx$, $\Rightarrow$, $\Longleftrightarrow$ | "aproximadamente", "implica", "equivale a" | — | — | — |

---

## 1. Intuição

### 1.1 Uma busca "suave"

Pense num dicionário Python: `valor = tabela[chave]`. A busca é **exata**: ou a chave bate
e você recebe aquele valor, ou não recebe nada.

A attention faz uma busca **suave**:

1. cada posição formula uma **pergunta** (query);
2. a pergunta é comparada com a **etiqueta** (key) de todas as posições visíveis;
3. a resposta é uma **mistura dos conteúdos** (values), com mais peso para as etiquetas
   que combinaram melhor com a pergunta.

Por ser uma mistura, e não uma escolha exata, tudo é diferenciável. *Diferenciável* quer
dizer que dá para calcular derivadas: uma pequena mudança nos parâmetros produz uma pequena
mudança na saída. O gradiente consegue ajustar como as perguntas e etiquetas são formadas,
e é assim que o modelo **aprende** o que procurar.

### 1.2 Três papéis para o mesmo vetor

Cada posição gera três vetores diferentes a partir do mesmo $x_i$:

| vetor | nome | papel | analogia |
|---|---|---|---|
| $q_i = x_i W_Q$ | query | o que a posição $i$ **procura** | o assunto que você pesquisa na biblioteca |
| $k_j = x_j W_K$ | key | como a posição $j$ **se anuncia** | a etiqueta na lombada do livro |
| $v_j = x_j W_V$ | value | o que a posição $j$ **entrega** se for escolhida | o conteúdo do livro |

$W_Q$, $W_K$ e $W_V$ são matrizes aprendidas. O nome **self**-attention vem de Q, K e V
saírem da mesma sequência. Em modelos encoder-decoder existe a *cross*-attention, em que
as queries vêm de uma sequência e keys/values de outra.

### 1.3 Por que separar key e value

O que torna uma posição **encontrável** não precisa ser o que ela **transmite**. Na seção 4
construímos uma head em que:

- a **key** diz "estou na posição $p$";
- a **query** diz "procuro a posição $p - 1$";
- o **value** diz "meu caractere é $c$".

A busca acontece por posição, mas o que se copia é o caractere. Com um único vetor para
os dois papéis, isso seria impossível.

---

## 2. A fórmula, passo a passo

A fórmula abaixo resume todo o mecanismo: ela diz como transformar queries, keys e values na
saída, respeitando a máscara causal.

$$
\boxed{\;\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V\;}
$$

**Lendo a fórmula:**

- $Q$, $K$, $V$: queries, keys e values, uma linha por posição;
- $QK^\top$: a matriz $T \times T$ de **scores**. O elemento $(i, j)$ é o produto escalar
  $q_i \cdot k_j$, a afinidade da posição $i$ pela posição $j$;
- $\sqrt{d_k}$: divide todos os scores pela raiz da dimensão das keys, para mantê-los numa
  escala estável;
- $+\,M$: soma a máscara causal, que põe $-\infty$ nas posições do futuro;
- $\text{softmax}(\cdot)$: aplicada **em cada linha**, transforma os scores em pesos
  positivos que somam 1;
- $(\cdot)\,V$: multiplica a matriz de pesos pelos values. Cada linha da saída é uma média
  ponderada dos values.

Cada etapa tem um objetivo e um lugar no código:

| etapa | objetivo | onde no código ([`model/attention.py`](../model/attention.py)) | seção |
|---|---|---|---|
| projeções | dar a cada posição uma pergunta, uma etiqueta e um conteúdo | `q_proj`, `k_proj`, `v_proj` (linhas 66–68) | 2.1 |
| scores | medir a afinidade entre cada par de posições | `q @ k.transpose(-2, -1)` (linha 24) | 2.2 |
| escala | evitar que a softmax sature quando $d_k$ é grande | `scores / math.sqrt(d_k)` (linha 26) | 2.3 |
| máscara | proibir olhar para o futuro | `masked_fill(~mask, -inf)` (linha 29) | 2.4 |
| softmax | transformar afinidades em proporções que somam 1 | `torch.softmax(scores, dim=-1)` (linha 30) | 2.5 |
| dropout | regularizar, cortando ligações ao acaso no treino | `self.dropout(weights)` (linha 73) | 7.3 |
| soma ponderada | ler o conteúdo das posições escolhidas | `weights @ v` (linhas 43 e 74) | 2.6 |

Vamos percorrer cada operação com os números da seção 1 do experimento: a palavra
`"gato"`, $d = 6$, $d_k = d_v = 3$. A entrada $X$ tem um vetor aleatório por caractere,
fazendo o papel dos embeddings:

```text
X: entrada  (4, 6)
    'g'  -1.126  -1.152  -0.251  -0.434  +0.849  +0.692
    'a'  -0.316  -2.115  +0.468  -0.158  +1.444  +0.266
    't'  +0.166  +0.874  -0.143  -0.112  +0.932  +1.259
    'o'  +2.005  +0.054  +0.618  -0.413  -0.841  -2.316
```

### 2.1 Projeções: Q, K e V

**Objetivo.** Um mesmo vetor $x_i$ precisa cumprir três papéis diferentes (seção 1.2). Três
matrizes aprendidas, uma por papel, deixam o modelo escolher o que cada papel enfatiza.

**O que faz.** Multiplica a entrada por três matrizes. Para uma sequência
$X \in \mathbb{R}^{T \times d}$ (uma linha por posição):

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V, \qquad W_Q, W_K \in \mathbb{R}^{d \times d_k},\; W_V \in \mathbb{R}^{d \times d_v}
$$

**Lendo a fórmula:**

- $X$: a entrada, `[T, d]`;
- $W_Q$, $W_K$: matrizes `[d, d_k]`; $W_V$: matriz `[d, d_v]`. São os **parâmetros** desta
  etapa, ajustados no treino;
- $XW_Q$: produto de matrizes `[T, d] @ [d, d_k]` → `[T, d_k]`. A linha $i$ do resultado é
  $q_i = x_i W_Q$: a mesma matriz é aplicada a cada posição, separadamente;
- $\in \mathbb{R}^{d \times d_k}$: "é uma matriz de números reais com $d$ linhas e $d_k$
  colunas".

No exemplo: `[4, 6] @ [6, 3]` → `[4, 3]`. No `tiny.yaml`: `[32, 128, 128] @ [128, 32]` →
`[32, 128, 32]`.

| tensor | shape | significado da linha $i$ |
|---|---|---|
| $X$ | $T \times d$ | vetor da posição $i$ |
| $Q$ | $T \times d_k$ | pergunta da posição $i$ |
| $K$ | $T \times d_k$ | etiqueta da posição $i$ |
| $V$ | $T \times d_v$ | conteúdo da posição $i$ |

No experimento:

```text
Q = X W_q: o que cada posição procura  (4, 3)
    'g'  +0.002  -0.393  -0.317
    'a'  -0.193  -0.817  -1.132
    't'  -0.172  -0.712  +0.454
    'o'  -0.955  +0.875  -0.129

K = X W_k: como cada posição se anuncia  (4, 3)
    'g'  -1.361  -0.632  -0.277
    'a'  -0.556  -0.229  +0.472
    't'  +0.356  +0.593  +0.343
    'o'  +1.635  +1.195  +1.372
```

```text
V = X W_v: o que cada posição entrega  (4, 3)
    'g'  +0.334  +1.098  -0.686
    'a'  +0.369  +1.144  -0.078
    't'  -0.958  -0.410  -0.912
    'o'  +0.248  -0.753  +1.783
```

**Convenção do PyTorch.** `nn.Linear(d, d_k)` guarda `weight` com shape `[d_k, d]` e
calcula `x @ weight.T`. Logo, o $W_Q$ da fórmula é `q_proj.weight.T`. O resultado é o mesmo;
só a orientação da matriz guardada muda.

### 2.2 Scores: o produto escalar mede afinidade

**Objetivo.** Dar um número a cada par (quem pergunta, quem é consultado) dizendo o quanto
eles "combinam".

**O que faz.** Calcula o produto escalar de cada query com cada key, todos de uma vez, com
um único produto de matrizes:

$$
S = QK^\top \in \mathbb{R}^{T \times T}, \qquad S_{ij} = q_i \cdot k_j = \|q_i\|\,\|k_j\|\cos\theta_{ij}
$$

**Lendo a fórmula:**

- $K^\top$: a transposta de $K$, `[T, d_k]` → `[d_k, T]`;
- $QK^\top$: `[T, d_k] @ [d_k, T]` → `[T, T]`. Pela regra do produto de matrizes, o elemento
  $(i, j)$ é a linha $i$ de $Q$ "escalar" a coluna $j$ de $K^\top$, que é justamente $k_j$;
- $S_{ij}$: um número, a afinidade da posição $i$ pela posição $j$;
- $q_i \cdot k_j = \sum_{m=1}^{d_k} q_{i,m}\, k_{j,m}$: multiplica coordenada a coordenada e soma;
- $\|q_i\|$, $\|k_j\|$: os comprimentos dos dois vetores;
- $\cos\theta_{ij}$: o cosseno do ângulo $\theta_{ij}$ entre eles (1 se apontam na mesma
  direção, 0 se são perpendiculares, −1 se opostos).

A segunda igualdade mostra que o score mistura **direção** (o cosseno) e **tamanho** (as
normas). O produto escalar é grande quando a pergunta de $i$ e a etiqueta de $j$ apontam na
mesma direção, perto de zero quando são ortogonais e negativo quando são opostas.

**Leitura da matriz:** a linha $i$ é quem pergunta, a coluna $j$ é quem é consultado.

Conta à mão do score de `'t'` (linha) com `'g'` (coluna):

$$
S_{t,g} = (-0{,}172)(-1{,}361) + (-0{,}712)(-0{,}632) + (0{,}454)(-0{,}277) = 0{,}234 + 0{,}450 - 0{,}126 \approx 0{,}559
$$

(Com os valores arredondados a 3 casas a soma dá 0,558; o experimento usa os valores
completos e imprime 0,559.)

A mesma conta pelo lado geométrico, com os valores arredondados:
$\|q_t\| = \sqrt{0{,}172^2 + 0{,}712^2 + 0{,}454^2} \approx 0{,}862$,
$\|k_g\| \approx 1{,}526$, e $\cos\theta = 0{,}558 / (0{,}862 \cdot 1{,}526) \approx 0{,}42$
(um ângulo de cerca de 65°). Os vetores apontam "mais ou menos" para o mesmo lado.

```text
scores = Q Kᵀ: linha i, coluna j = afinidade de i por j  (4, 4)
    'g'  +0.334  -0.061  -0.341  -0.902
    'a'  +1.092  -0.240  -0.940  -2.844
    't'  +0.559  +0.473  -0.328  -0.509
    'o'  +0.783  +0.270  +0.135  -0.693
```

**A matriz não é simétrica.** $S_{a,g} = 1{,}092$, mas $S_{g,a} = -0{,}061$. Como
$W_Q \neq W_K$, "o quanto `'a'` procura `'g'`" é diferente de "o quanto `'g'` procura `'a'`".

Repare que a matriz ainda tem valores **acima da diagonal** (por exemplo, `'g'` com `'o'`).
Eles são calculados, mas a máscara (seção 2.4) vai descartá-los.

### 2.3 A escala $1/\sqrt{d_k}$

**Objetivo.** Manter os scores numa faixa de valores que não dependa de $d_k$, para que a
softmax continue sensível e o modelo continue aprendendo.

**O que faz.** Divide todos os scores por $\sqrt{d_k}$ (linha 26). No exemplo, $\sqrt{3}
\approx 1{,}732$, e o score de `'t'` com `'g'` vira $0{,}559 / 1{,}732 \approx 0{,}323$. No
`tiny.yaml`, $\sqrt{32} \approx 5{,}66$.

```text
scores / √d_k   (√3 = 1.732)  (4, 4)
    'g'  +0.193  -0.035  -0.197  -0.521
    'a'  +0.631  -0.138  -0.543  -1.642
    't'  +0.323  +0.273  -0.189  -0.294
    'o'  +0.452  +0.156  +0.078  -0.400
```

#### O problema: a variância de $q \cdot k$ cresce com $d_k$

Suponha que as entradas de $q$ e $k$ sejam independentes, com média 0 e variância 1. A
conta abaixo mostra quanto os scores se espalham nessa situação.

$$
q \cdot k = \sum_{m=1}^{d_k} q_m k_m, \qquad \mathbb{E}[q_m k_m] = 0, \qquad \text{Var}(q_m k_m) = \mathbb{E}[q_m^2]\,\mathbb{E}[k_m^2] = 1
$$

$$
\Rightarrow \text{Var}(q \cdot k) = d_k, \qquad \text{desvio padrão} = \sqrt{d_k}
$$

**Lendo a fórmula:**

- $q_m$, $k_m$: a coordenada $m$ de $q$ e de $k$;
- $\mathbb{E}[\cdot]$: a média teórica (esperança);
- $\text{Var}(\cdot)$: a variância, o quanto os valores se espalham;
- $\Rightarrow$: "logo".

**Passo a passo, sem pular etapas:**

1. **Definição.** $q \cdot k$ é uma soma de $d_k$ termos $q_m k_m$.
2. **Média de cada termo.** Para variáveis independentes, a média do produto é o produto das
   médias: $\mathbb{E}[q_m k_m] = \mathbb{E}[q_m]\,\mathbb{E}[k_m] = 0 \cdot 0 = 0$.
3. **Variância de cada termo.** Por definição, $\text{Var}(Z) = \mathbb{E}[Z^2] - (\mathbb{E}[Z])^2$.
   Com $Z = q_m k_m$ e média 0: $\text{Var}(q_m k_m) = \mathbb{E}[q_m^2 k_m^2]$. Pela
   independência, isso é $\mathbb{E}[q_m^2]\,\mathbb{E}[k_m^2]$.
4. **Quanto vale $\mathbb{E}[q_m^2]$.** Pela mesma definição, $\mathbb{E}[q_m^2] = \text{Var}(q_m) + (\mathbb{E}[q_m])^2 = 1 + 0 = 1$.
   Igual para $k_m$. Logo $\text{Var}(q_m k_m) = 1 \cdot 1 = 1$.
5. **Variância da soma.** Os $d_k$ termos são independentes entre si, e a variância de uma
   soma de termos independentes é a soma das variâncias: $\text{Var}(q \cdot k) = \underbrace{1 + 1 + \dots + 1}_{d_k \text{ vezes}} = d_k$.
6. **Desvio padrão.** É a raiz da variância: $\sqrt{d_k}$.

Com $d_k = 32$ (`tiny.yaml`), os scores teriam desvio $\approx 5{,}66$: diferenças de 10 ou
mais entre scores de uma mesma linha seriam comuns.

**A correção.** Multiplicar uma variável por uma constante $c$ multiplica a variância por
$c^2$: $\text{Var}(cZ) = c^2\,\text{Var}(Z)$. Com $c = 1/\sqrt{d_k}$:

$$
\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k}\,\text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1
$$

**Lendo a fórmula:** o fator $\frac{1}{\sqrt{d_k}}$, elevado ao quadrado, vira
$\frac{1}{d_k}$, que cancela exatamente o $d_k$ da variância.

Dividir por $\sqrt{d_k}$ traz a variância de volta a 1, **qualquer que seja $d_k$**. O teste
`test_scaling_keeps_score_variance_near_one` confere isso com $d_k = 256$.

#### Por que isso importa: saturação da softmax

Scores com desvio grande deixam a diferença entre o maior e os demais enorme, e a softmax
vira praticamente um one-hot (um 1 e o resto 0). Chamamos isso de **saturação**: a saída
fica "presa" num extremo, e pequenas mudanças nos scores quase não mudam os pesos.

Exemplo à mão. Com scores $(1,\ 0,\ -1)$, a softmax dá $(0{,}665;\ 0{,}245;\ 0{,}090)$. Se
os mesmos scores forem multiplicados por 8 (o desvio de $q \cdot k$ quando $d_k = 64$),
viram $(8,\ 0,\ -8)$, e a softmax dá $(0{,}99966;\ 0{,}00034;\ 0{,}0000001)$: saturada.

Uma softmax saturada não aprende. Para ver por quê, calculamos a derivada de cada saída
$p_i$ da softmax em relação a cada entrada $s_j$. O resultado é

$$
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)
$$

**Lendo a fórmula:**

- $s_j$: o score $j$ (entrada da softmax); $p_i$: o peso $i$ (saída da softmax);
- $\partial p_i / \partial s_j$: quanto o peso $i$ muda quando só o score $j$ muda um pouco;
- $\delta_{ij}$: delta de Kronecker, 1 se $i = j$ e 0 se não;
- caso $i = j$: a derivada vale $p_i(1 - p_i)$ (aumentar o próprio score aumenta o próprio peso);
- caso $i \neq j$: vale $-p_i p_j$ (aumentar o score de outro diminui o meu peso).

**De onde vem, passo a passo.** Escreva $p_i = e^{s_i} / Z$, com $Z = \sum_{j'} e^{s_{j'}}$.

1. Derivada do denominador: só o termo $e^{s_j}$ de $Z$ depende de $s_j$, então
   $\partial Z / \partial s_j = e^{s_j}$.
2. Derivada do numerador: $\partial e^{s_i} / \partial s_j = e^{s_i}$ se $i = j$ e 0 se
   não, ou seja, $\delta_{ij}\, e^{s_i}$.
3. Regra do quociente, $\left(\frac{f}{g}\right)' = \frac{f' g - f g'}{g^2}$:
   $$
   \frac{\partial p_i}{\partial s_j} = \frac{\delta_{ij}\, e^{s_i} \cdot Z - e^{s_i} \cdot e^{s_j}}{Z^2}
   $$
4. Separe as duas frações: $\delta_{ij}\,\frac{e^{s_i}}{Z} - \frac{e^{s_i}}{Z} \cdot \frac{e^{s_j}}{Z} = \delta_{ij}\, p_i - p_i\, p_j = p_i(\delta_{ij} - p_j)$.

**Por que ela vai a zero quando algum $p_i \to 1$.** Numa softmax saturada, um peso vale 1 e
os outros valem 0. Em cada termo $p_i(\delta_{ij} - p_j)$, ou $p_i = 0$, ou $p_i = 1$ e
então $(\delta_{ij} - p_j)$ vale $1 - 1 = 0$ (se $j = i$) ou $0 - 0 = 0$ (se $j \neq i$).
Todas as derivadas são zero: o gradiente não passa, e $W_Q$ e $W_K$ param de aprender.

**A Jacobiana.** Juntando todas as derivadas numa matriz $J$, com $J_{ij} = \partial p_i / \partial s_j$,
temos a **Jacobiana** da softmax. Exemplo com os pesos da linha de `'t'` (seção 2.5),
$p = (0{,}392;\ 0{,}373;\ 0{,}235)$ (calculado em Python):

$$
J = \begin{bmatrix} 0{,}238 & -0{,}146 & -0{,}092 \\ -0{,}146 & 0{,}234 & -0{,}088 \\ -0{,}092 & -0{,}088 & 0{,}180 \end{bmatrix}
$$

Na diagonal, $0{,}392 \cdot (1 - 0{,}392) = 0{,}238$; fora dela, por exemplo,
$-0{,}392 \cdot 0{,}373 = -0{,}146$.

O traço dessa Jacobiana, $\sum_i p_i(1-p_i) = 1 - \sum_i p_i^2$,
resume quanto a softmax ainda "responde" a mudanças nos scores. Vale 0 quando saturada e
$1 - 1/n$ quando uniforme.

**Lendo a fórmula:**

- **traço** = soma dos elementos da diagonal de uma matriz;
- $\sum_i p_i(1 - p_i) = \sum_i p_i - \sum_i p_i^2$, e $\sum_i p_i = 1$ (os pesos somam 1);
- $n$: número de posições na linha;
- **saturada** ($p = (1, 0, \dots, 0)$): $\sum_i p_i^2 = 1$, traço $= 0$;
- **uniforme** ($p_i = 1/n$ para todo $i$): $\sum_i p_i^2 = n \cdot \frac{1}{n^2} = \frac{1}{n}$,
  traço $= 1 - 1/n$. Com $n = 16$, $0{,}9375$.

Exemplos: na linha de `'t'`, $1 - (0{,}392^2 + 0{,}373^2 + 0{,}235^2) \approx 0{,}652$. Nos
scores $(1, 0, -1)$ acima, $0{,}489$; nos scores multiplicados por 8, $0{,}00067$.

Seção 2 do experimento (16 posições por linha; uniforme = 0,0625). Cada linha tem
$q, k \sim \mathcal{N}(0, 1)$ (entradas com média 0 e variância 1):

| $d_k$ | var$(q \cdot k)$ | var$(q \cdot k / \sqrt{d_k})$ | maior peso, sem escala | maior peso, com escala | $1 - \sum p^2$, sem | $1 - \sum p^2$, com |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4,0 | 1,002 | 0,422 | 0,234 | 0,712 | 0,869 |
| 16 | 15,9 | 0,993 | 0,685 | 0,244 | 0,429 | 0,867 |
| 64 | 64,6 | 1,009 | 0,844 | 0,247 | 0,219 | 0,867 |
| 256 | 254,0 | 0,992 | 0,920 | 0,245 | 0,113 | 0,868 |
| 1024 | 1030,8 | 1,007 | 0,964 | 0,250 | 0,052 | 0,866 |

**Como ler a tabela:**

- a coluna var$(q \cdot k)$ confirma a derivação: a variância é $\approx d_k$;
- a coluna seguinte confirma a correção: depois da escala, $\approx 1$;
- "maior peso" é a média, sobre as linhas, do maior $w_{ij}$ de cada linha (1 = one-hot);
- $1 - \sum p^2$ é o traço da Jacobiana (1 − 1/16 = 0,9375 seria o uniforme). Com escala ele
  fica em 0,87, e não em 0,94, porque scores com variância 1 ainda não são todos iguais.

**Sem a escala,** quanto maior a head, mais a softmax satura: com $d_k = 1024$, o maior
peso é 0,96 e a sensibilidade cai para 0,05. **Com a escala,** o regime é o mesmo para
qualquer $d_k$. É por isso que o artigo original chama a operação de *scaled*
dot-product attention (Vaswani et al., 2017).

Uma forma equivalente de ver: dividir os scores por $\sqrt{d_k}$ é aplicar uma
**temperatura** $\sqrt{d_k}$ à softmax, a mesma ideia de temperatura da geração
([13-generation.md](13-generation.md)).
Temperatura é um número pelo qual se dividem os scores antes da softmax: maior que 1 deixa
os pesos mais espalhados; menor que 1, mais concentrados.

### 2.4 A causal mask

**Objetivo.** Garantir que a posição $i$ só use informação das posições $0, \dots, i$ (seção 3
explica por que isso é indispensável).

**O que faz.** A máscara é uma matriz $M \in \mathbb{R}^{T \times T}$ somada aos scores
escalados. Ela deixa o passado e o presente como estão e empurra o futuro para $-\infty$:

$$
M_{ij} = \begin{cases} 0 & \text{se } j \le i \quad \text{(passado ou presente: permitido)} \\ -\infty & \text{se } j > i \quad \text{(futuro: proibido)} \end{cases}
$$

**Lendo a fórmula:**

- $M_{ij}$: o elemento da linha $i$, coluna $j$ da máscara;
- $i$: a posição que pergunta; $j$: a posição consultada;
- $j \le i$: $j$ é a própria posição ou uma anterior. Somar 0 não muda o score;
- $j > i$: $j$ está no futuro de $i$. Somar $-\infty$ transforma o score em $-\infty$;
- a chave "{" lista os dois casos possíveis.

Para $T = 4$, a máscara aditiva e a versão booleana que o código usa são:

$$
M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}
\qquad
\texttt{causal\_mask(4)} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}
$$

(1 = `True` = pode olhar.) É uma matriz **triangular inferior**: só tem valores permitidos
na diagonal e abaixo dela. É exatamente o que o teste `test_causal_mask_is_lower_triangular`
verifica.

```text
máscara causal: -inf onde j > i  (4, 4)
    'g'  +0.193    -inf    -inf    -inf
    'a'  +0.631  -0.138    -inf    -inf
    't'  +0.323  +0.273  -0.189    -inf
    'o'  +0.452  +0.156  +0.078  -0.400
```

**Por que $-\infty$ antes da softmax, e não zerar pesos depois?** Como $e^{-\infty} = 0$,
a softmax dá peso zero ao futuro **e** renormaliza automaticamente sobre as posições
visíveis. Zerar pesos depois da softmax deixaria linhas somando menos que 1, e seria
preciso renormalizar à mão.

Exemplo com a linha de `'a'` (calculado em Python a partir dos scores escalados acima):

| abordagem | pesos de `'a'` sobre (`'g'`, `'a'`, `'t'`, `'o'`) | soma |
|---|---|---:|
| softmax sem máscara | (0,533; 0,247; 0,165; 0,055) | 1 |
| zerar o futuro depois | (0,533; 0,247; 0; 0) | 0,780 |
| zerar e renormalizar à mão | (0,683; 0,317; 0; 0) | 1 |
| $-\infty$ antes da softmax | (0,683; 0,317; 0; 0) | 1 |

A primeira linha mostra o vazamento: sem máscara, `'a'` daria 22% da atenção a `'t'` e
`'o'`, que ainda não existem do ponto de vista dela.

**Implementação.** Em vez de uma matriz de $-\infty$, o código usa uma máscara booleana
(`torch.tril`, True no triângulo inferior) e
`scores.masked_fill(~mask, float("-inf"))`. O efeito é o mesmo.

- `torch.tril` ("triangle lower") zera tudo acima da diagonal de uma matriz de uns;
- `~mask` inverte a máscara: `True` passa a marcar o **futuro**;
- `masked_fill(cond, valor)` escreve `valor` onde `cond` é `True` e mantém o resto.

**Cuidado com linhas totalmente mascaradas.** Se todas as entradas de uma linha fossem
$-\infty$, a softmax calcularia $0/0$ = NaN ("not a number", resultado inválido). A causal
mask nunca faz isso, porque a diagonal ($j = i$) é sempre permitida. Máscaras de padding
(que escondem posições de preenchimento em sequências de tamanhos diferentes), que não
usamos, precisam desse cuidado.

### 2.5 Softmax por linha

**Objetivo.** Transformar afinidades (números quaisquer, positivos ou negativos) em
**proporções**: quanto da atenção da posição $i$ vai para cada posição $j$.

**O que faz.** Aplica a softmax em cada linha de $\tilde S$ (linha 30, `dim=-1` = ao longo
da última dimensão, as colunas $j$):

$$
w_{ij} = \frac{\exp(\tilde S_{ij})}{\sum_{j' \le i} \exp(\tilde S_{ij'})}, \qquad \tilde S = \frac{S}{\sqrt{d_k}} + M
$$

**Lendo a fórmula:**

- $w_{ij}$: o **peso de attention** da posição $i$ sobre a posição $j$;
- $\tilde S$: os scores escalados e mascarados;
- $\exp(\tilde S_{ij}) = e^{\tilde S_{ij}}$: sempre positivo; scores maiores dão
  exponenciais muito maiores;
- $\sum_{j' \le i}$: soma sobre todas as posições visíveis da linha $i$. O índice se chama
  $j'$ só para não confundir com o $j$ do numerador;
- dividir pela soma faz a linha somar 1. Somar sobre todos os $j'$, inclusive o futuro, daria
  o mesmo resultado, porque $e^{-\infty} = 0$.

Propriedades:

- $w_{ij} \ge 0$ e $\sum_j w_{ij} = 1$: cada linha é uma distribuição de probabilidade
  sobre as posições visíveis;
- só **diferenças** entre scores de uma linha importam: somar a mesma constante a toda a
  linha não muda os pesos.

**Por que só diferenças importam.** Somando uma constante $c$ a todos os scores da linha,
cada exponencial vira $e^{\tilde S_{ij} + c} = e^{c}\, e^{\tilde S_{ij}}$. O fator $e^c$
aparece no numerador e em todos os termos do denominador, e se cancela. Exemplo: scores
$(1, 0, -1)$ e $(6, 5, 4)$ dão a mesma softmax, $(0{,}665;\ 0{,}245;\ 0{,}090)$.

Conta à mão da linha de `'t'`, com scores escalados $0{,}323$, $0{,}273$ e $-0{,}189$:

$$
e^{0{,}323} = 1{,}381,\quad e^{0{,}273} = 1{,}314,\quad e^{-0{,}189} = 0{,}828,\quad \text{soma} = 3{,}523
$$

$$
w = \left(\tfrac{1{,}381}{3{,}523},\; \tfrac{1{,}314}{3{,}523},\; \tfrac{0{,}828}{3{,}523}\right) = (0{,}392,\; 0{,}373,\; 0{,}235)
$$

O quarto score da linha é $-\infty$ (futuro): $e^{-\infty} = 0$, então ele não entra na soma
e seu peso é 0.

```text
pesos = softmax de cada linha  (4, 4)
    'g'  +1.000  +0.000  +0.000  +0.000
    'a'  +0.683  +0.317  +0.000  +0.000
    't'  +0.392  +0.373  +0.235  +0.000
    'o'  +0.350  +0.260  +0.241  +0.149
  soma de cada linha: [1.0, 1.0, 1.0, 1.0]
```

A primeira linha é sempre `[1, 0, 0, 0]`: a primeira posição não tem escolha, só enxerga a
si mesma.

### 2.6 Soma ponderada dos values

**Objetivo.** Até aqui o modelo só decidiu **para onde** olhar. Esta etapa **lê** o
conteúdo dessas posições e o entrega à posição $i$.

**O que faz.** Multiplica a matriz de pesos pelos values (linha 43 ou 74):

$$
o_i = \sum_{j \le i} w_{ij}\, v_j \qquad \Longleftrightarrow \qquad O = W V
$$

**Lendo a fórmula:**

- $o_i$: a saída da posição $i$, um vetor de $d_v$ números;
- $\sum_{j \le i}$: soma sobre as posições visíveis;
- $w_{ij}\, v_j$: o value da posição $j$ multiplicado pelo peso que $i$ deu a $j$ (um número
  vezes um vetor: cada coordenada é multiplicada);
- $\Longleftrightarrow$: as duas formas dizem a mesma coisa;
- $O = WV$: a forma matricial. $W$ é a matriz de pesos `[T, T]` e $V$ é `[T, d_v]`, então
  $O$ é `[T, d_v]`. A linha $i$ de $WV$ é exatamente $\sum_j w_{ij} v_j$, e os $w_{ij}$ do
  futuro são 0.

A saída de cada posição é uma **média ponderada** dos conteúdos visíveis:

```text
saída['t'] = 0.392·V['g'] + 0.373·V['a'] + 0.235·V['t']
```

Conta à mão da primeira coordenada, com os values da seção 2.1:
$0{,}392 \cdot 0{,}334 + 0{,}373 \cdot 0{,}369 + 0{,}235 \cdot (-0{,}958) = 0{,}131 + 0{,}138 - 0{,}225 \approx 0{,}043$.
As outras duas dão $0{,}761$ e $-0{,}512$. Compare com a linha `'t'` da saída do experimento:

```text
saída = pesos @ V  (4, 3)
    'g'  +0.334  +1.098  -0.686
    'a'  +0.345  +1.113  -0.493
    't'  +0.043  +0.761  -0.512
    'o'  +0.019  +0.471  -0.214
```

Repare que a linha `'g'` é idêntica a `V['g']`: a primeira posição só copia o próprio value.

Consequências importantes:

- **A attention só mistura.** Como os pesos são não negativos e somam 1, $o_i$ é uma
  combinação convexa dos values: fica "entre" eles. Uma **combinação convexa** é uma soma
  de vetores com coeficientes $\ge 0$ que somam 1; o resultado nunca sai da "região" delimitada
  pelos vetores originais. Toda a não-linearidade está em *como*
  os pesos são escolhidos. O processamento não linear do conteúdo fica com o feed-forward
  ([06-feed-forward.md](06-feed-forward.md)).
- **Casos extremos** (ambos com teste):
  - posição 0: $o_0 = v_0$;
  - scores todos iguais (por exemplo, $q = 0$): pesos uniformes, e $o_i$ é a média simples
    $\frac{1}{i+1}\sum_{j \le i} v_j$.

**Lendo a última fórmula:** a linha $i$ enxerga $i + 1$ posições (de 0 a $i$). Com pesos
iguais, cada um vale $\frac{1}{i+1}$, e a soma ponderada vira a média aritmética dos values
visíveis.

### 2.7 Com batch: os shapes

No modelo real há um batch de $B$ sequências. Todas as operações acima continuam iguais; só
ganham uma dimensão a mais à esquerda.

| passo | operação | shape |
|---|---|---|
| entrada | `x` | `[B, T, d]` |
| projeções | `q_proj(x)`, `k_proj(x)`, `v_proj(x)` | `[B, T, d_k]`, `[B, T, d_k]`, `[B, T, d_v]` |
| scores | `q @ k.transpose(-2, -1)` | `[B, T, T]` |
| escala | `/ math.sqrt(d_k)` | `[B, T, T]` |
| máscara | `masked_fill(~mask, -inf)`, `mask: [T, T]` | `[B, T, T]` (broadcasting) |
| pesos | `softmax(dim=-1)` | `[B, T, T]` |
| saída | `weights @ v` | `[B, T, d_v]` |

O operador `@` com tensores 3-D faz uma multiplicação de matrizes **por elemento do
batch**. `transpose(-2, -1)` troca as duas últimas dimensões: `[B, T, d_k] → [B, d_k, T]`.

**Broadcasting** é a regra do PyTorch que "estica" um tensor com menos dimensões para combinar
com outro ([00-notacao.md](00-notacao.md), seção 8): a mesma máscara `[T, T]` vale para
todas as $B$ sequências, sem ser copiada.

Com o `tiny.yaml`: `[32, 128, 128]` → projeções `[32, 128, 32]` → scores, pesos
`[32, 128, 128]` → saída `[32, 128, 32]`.

---

## 3. A causal mask em profundidade

### 3.1 Por que o modelo não pode olhar para o futuro

Em [02-dataset.md](02-dataset.md), cada janela tem `y[t] = x[t+1]`, e o modelo resolve as $T$
previsões **em paralelo**. Se a posição $t$ pudesse ver $x_{t+1}$, estaria vendo a resposta.
Bastaria copiar: a loss cairia para perto de zero sem o modelo aprender nada de linguagem.

Formalmente, exigimos que **a saída da posição $i$ dependa só de $x_0, \dots, x_i$**. Em
termos de derivadas:

$$
\frac{\partial o_i}{\partial x_j} = 0 \quad \text{para todo } j > i
$$

**Lendo a fórmula:**

- $o_i$: a saída da posição $i$;
- $x_j$: a entrada da posição $j$;
- $\partial o_i / \partial x_j$: quanto $o_i$ muda quando só $x_j$ muda um pouco. Como
  $o_i$ e $x_j$ são vetores, isso é uma matriz `[d_v, d]` de derivadas;
- $= 0$ para $j > i$: mexer em qualquer posição futura não altera nada na saída de $i$.

**Por que a attention com máscara garante isso, passo a passo:**

1. A saída é $o_i = \sum_{j \le i} w_{ij}\, v_j$. Só entram values de posições $j \le i$.
2. Cada value $v_j = x_j W_V$ depende só de $x_j$.
3. Cada peso $w_{ij} = \exp(q_i \cdot k_j / \sqrt{d_k}) \,/\, \sum_{j' \le i} \exp(q_i \cdot k_{j'} / \sqrt{d_k})$
   depende de $q_i$ (que vem de $x_i$) e das keys $k_{j'}$ com $j' \le i$ (que vêm de
   $x_0, \dots, x_i$).
4. Logo, $o_i$ é uma função **apenas** de $x_0, \dots, x_i$. Mudar $x_j$ com $j > i$ não
   muda nenhum ingrediente de $o_i$, e a derivada é zero.

A Jacobiana da attention é triangular inferior. Aqui a **Jacobiana** é a tabela com todas as
derivadas $\partial o_i / \partial x_j$: linha $i$ = saída, coluna $j$ = entrada. Os zeros
acima da diagonal ($j > i$) a tornam triangular inferior. O teste
`test_gradient_only_reaches_current_and_past_inputs` verifica isso diretamente: o
gradiente de $o_2$ chega a $x_0, x_1, x_2$ e é **exatamente zero** para $x_3, x_4, x_5$.

**Isso vale para o GPT inteiro.** Embeddings, LayerNorm e feed-forward operam posição por
posição, e as conexões residuais somam vetores da mesma posição. Uma composição de
operações causais é causal. Basta a attention respeitar a máscara para o modelo todo
respeitar.

**O argumento, passo a passo.** Chame uma operação $f$ de **causal** se a saída $f(x)_i$
depende só de $x_0, \dots, x_i$.

1. **Operações por posição são causais.** Em LayerNorm e feed-forward, a saída $i$ depende
   só de $x_i$, um caso particular de "só de $x_0, \dots, x_i$".
2. **A composição de causais é causal.** Se $f$ e $g$ são causais, $g(f(x))_i$ depende de
   $f(x)_0, \dots, f(x)_i$. Cada $f(x)_m$ com $m \le i$ depende de $x_0, \dots, x_m$, que
   está contido em $x_0, \dots, x_i$. Logo, $g(f(x))_i$ só depende de $x_0, \dots, x_i$.
3. **A conexão residual é causal.** $h_i + f(h)_i$ soma dois termos que só dependem de
   $h_0, \dots, h_i$.
4. **Conclusão.** O GPT é uma composição de embeddings, attentions causais, feed-forwards,
   LayerNorms, somas residuais e LM head, todos causais. Logo, é causal.

### 3.2 A demonstração

Seção 3 do experimento: as sequências `A = "o gato dorme"` e `B = "o gato corre"` diferem
nas posições 7 e 10. A tabela mostra a maior diferença entre as saídas de A e B em cada
posição:

```text
   t    A   B   com máscara   sem máscara
   0  'o' 'o'      0.00e+00      2.24e-01
   1  ' ' ' '      0.00e+00      3.44e-01
   ...
   6  ' ' ' '      0.00e+00      2.74e-01
   7  'd' 'c'      4.79e-01      7.55e-01   <- entrada diferente
   8  'o' 'o'      1.84e-01      1.67e-01
   9  'r' 'r'      2.78e-01      3.50e-01
  10  'm' 'r'      4.85e-01      4.26e-01   <- entrada diferente
  11  'e' 'e'      3.44e-01      3.44e-01
```

(As linhas omitidas, posições 2 a 5, também dão exatamente `0.00e+00` com máscara.)

"Maior diferença" é $\max_m |o^A_{i,m} - o^B_{i,m}|$: compara as duas saídas coordenada a
coordenada e fica com a maior diferença em valor absoluto.

Como ler:

- **Com máscara, posições 0 a 6: diferença exatamente 0.** Não é "muito pequena": as
  contas dessas posições são literalmente as mesmas nas duas sequências.
- **Posição 8 (`'o'` nas duas):** o caractere é igual, mas a saída muda, porque ela lê o
  passado, que agora é diferente.
- **Sem máscara, todas as posições mudam,** até a posição 0, que passaria a "saber" como
  a frase termina.
- **Posição 11 tem o mesmo valor nas duas colunas.** A última posição enxerga a sequência
  inteira com ou sem máscara.

O experimento usa pesos de projeção com escala maior que a da inicialização padrão, para os
efeitos ficarem visíveis (seção 5 explica por quê).

### 3.3 Treino e geração ficam iguais

Na geração ([13-generation.md](13-generation.md)), o futuro não existe: o modelo produz um
caractere de cada vez. A máscara faz o treino acontecer **nas mesmas condições** da geração.
Sem ela, o modelo aprenderia a depender de uma informação que nunca terá.

Há uma consequência de eficiência, discutida em [17-performance.md](17-performance.md) (seção
"O que fica para depois"): como a saída das posições passadas **não muda** quando um novo token
é adicionado, as keys e values já calculados podem ser reaproveitados. Essa é a ideia do **KV
cache**: guardar na memória as keys e values das posições já processadas, para que cada novo
token só precise calcular a própria query, key e value, em vez de refazer a sequência inteira.

Removendo a máscara e treinando, isso fica visível na prática: a loss despenca para perto de
zero, porque cada posição passa a poder copiar a resposta certa em vez de aprender o padrão.

### 3.4 Máscara, posição e permutações

[03-embeddings.md](03-embeddings.md) (seção 4.1) afirma que, sem informação de posição, a
attention **não distingue ordem**. Aqui está a derivação completa, com a notação deste
documento.

**Permutação.** Uma matriz de permutação $P \in \mathbb{R}^{T \times T}$ tem exatamente um 1
em cada linha e em cada coluna, e zeros no resto. Multiplicar $PX$ reordena as linhas de
$X$. Exemplo com $T = 4$:

$$
P = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \end{bmatrix}
\quad\Rightarrow\quad
PX = \begin{bmatrix} x_2 \\ x_0 \\ x_3 \\ x_1 \end{bmatrix}
$$

Ela tem uma propriedade útil: $P^\top P = I$ (aplicar a permutação inversa desfaz a
reordenação).

**Sem máscara, permutar a entrada só permuta a saída:**

$$
\text{Attention}(PX) = P\,\text{Attention}(X)
$$

**Lendo a fórmula:** reordenar os tokens de entrada e depois aplicar attention dá o mesmo
que aplicar attention e depois reordenar as saídas. Isso se chama **equivariância a
permutações**.

**Passo a passo:**

1. **Projeções.** $(PX)W_Q = P(XW_Q) = PQ$. Do mesmo modo, as keys viram $PK$ e os values,
   $PV$. As projeções atuam em cada linha separadamente, então só acompanham a reordenação.
2. **Scores.** $(PQ)(PK)^\top = PQK^\top P^\top = PSP^\top$, usando $(AB)^\top = B^\top A^\top$.
3. **O que $PSP^\top$ faz.** Multiplicar por $P$ à esquerda reordena as linhas; por $P^\top$
   à direita, reordena as colunas **do mesmo jeito**. O score entre dois tokens continua o
   mesmo; só muda de lugar na matriz.
4. **Softmax por linha.** Reordenar os elementos de uma linha só reordena os pesos dessa
   linha, e reordenar as linhas só troca qual linha é qual. Logo
   $\text{softmax}(PSP^\top / \sqrt{d_k}) = P\,W\,P^\top$.
5. **Soma ponderada.** $(PWP^\top)(PV) = PW(P^\top P)V = PWIV = P(WV) = PO$.

Conclusão: sem posição nos vetores e sem máscara, "amor" e "roma" produziriam os mesmos
quatro vetores de saída, só em outra ordem.

**A máscara quebra essa simetria.** No passo 4, a conta com máscara seria
$\text{softmax}(PSP^\top / \sqrt{d_k} + M)$. Para a igualdade continuar valendo, a máscara
também precisaria ser reordenada ($PMP^\top$), mas $M$ é fixa: a posição $i$ sempre enxerga
exatamente as $i + 1$ primeiras. Conferido em Python com $T = 4$, $d = 6$ e $d_k = 3$: a
igualdade $\text{Attention}(PX) = P\,\text{Attention}(X)$ vale sem máscara e falha com
máscara. Como diz o documento de embeddings, isso permite inferir posição implicitamente, mas dar a posição
explicitamente (positional embedding) é o padrão.

---

## 4. Uma head construída à mão

A seção 4 do experimento monta $W_Q$, $W_K$ e $W_V$ **manualmente**, sem treino, para
responder: *que tipo de regra essas três matrizes conseguem expressar?*

**Entrada.** Para `"o gato"` (vocabulário `[' ', 'a', 'g', 'o', 't']`, 6 posições), cada
posição é a concatenação de dois one-hots:

$$
x_p = [\,\underbrace{\text{one-hot do caractere}}_{5\text{ dims}}\; | \;\underbrace{\text{one-hot da posição}}_{6\text{ dims}}\,] \in \mathbb{R}^{11}
$$

**Lendo a fórmula:**

- $x_p$: o vetor de entrada da posição $p$;
- **one-hot**: vetor todo zero com um único 1. O one-hot do caractere tem o 1 na coordenada
  do caractere no vocabulário (5 opções); o da posição, na coordenada da posição (6 opções);
- $[\,a \mid b\,]$: concatenação, colocar $b$ logo depois de $a$;
- $\underbrace{\cdot}_{5\text{ dims}}$: indica o tamanho do trecho;
- $\in \mathbb{R}^{11}$: o resultado tem $5 + 6 = 11$ números. Então $d = 11$.

Exemplo: a posição 2 tem `'g'` (índice 2 no vocabulário), então
$x_2 = [0, 0, 1, 0, 0 \mid 0, 0, 1, 0, 0, 0]$. É o papel que token embedding e positional
embedding cumprem no modelo real.

**Matrizes.** A key e a query vivem em $d_k = 6$ dimensões (uma por posição). O value vive
em $d_v = 5$ dimensões (uma por caractere). $e_p$ é o one-hot com 1 na coordenada $p$.

| matriz | regra | efeito |
|---|---|---|
| $W_K$ | posição $p$ → $e_p$ | key: "estou na posição $p$" |
| $W_Q$ | posição $p$ → $\beta\, e_{p-1}$ | query: "procuro a posição $p-1$" |
| $W_V$ | caractere $c$ → $e_c$ | value: "carrego o caractere $c$" |

Então $S_{ij} = q_i \cdot k_j = \beta \cdot [\,j = i-1\,]$. Com $\beta = 8\sqrt{d_k}$, o score
escalado do alvo vale 8, e o peso fica $e^8 / (e^8 + i) \approx 0{,}998$.

**Lendo as fórmulas:**

- $q_i = \beta\, e_{i-1}$ e $k_j = e_j$;
- $e_a \cdot e_b$ vale 1 se $a = b$ (os dois uns coincidem) e 0 se não. Logo
  $q_i \cdot k_j = \beta\,(e_{i-1} \cdot e_j) = \beta \cdot [\,j = i-1\,]$;
- $[\,j = i-1\,]$: indicadora, 1 se $j = i - 1$ e 0 se não;
- $\beta$ ("beta"): a intensidade da query. Com $d_k = 6$, $\beta = 8\sqrt{6} \approx 19{,}6$;
- **escalado:** $\beta / \sqrt{d_k} = 8\sqrt{6}/\sqrt{6} = 8$;
- **o peso:** a linha $i$ enxerga $i + 1$ posições. O alvo tem score escalado 8, e sua
  exponencial é $e^8 \approx 2981$. As outras $i$ posições têm score 0, e $e^0 = 1$ cada. A
  soma é $e^8 + i$, e o peso do alvo é $e^8 / (e^8 + i)$;
- $\approx 0{,}998$ é o pior caso, $i = 5$: $2981 / 2986 \approx 0{,}998$. Com $i = 1$, dá
  $0{,}9997$ (impresso como 1.000).

**Resultado:**

```text
   t  char   olha para   peso   copia
   0   'o'   t=0 'o'    1.000   'o'
   1   ' '   t=0 'o'    1.000   'o'
   2   'g'   t=1 ' '    0.999   ' '
   3   'a'   t=2 'g'    0.999   'g'
   4   't'   t=3 'a'    0.999   'a'
   5   'o'   t=4 't'    0.998   't'
```

Cada posição passa a carregar **o caractere anterior**. Na posição 0 a query é zero, os
scores são iguais, e ela só enxerga a si mesma.

Como ler a coluna "copia": a saída $o_i \approx v_{i-1} = e_{c_{i-1}}$ é quase um one-hot do
caractere anterior, e o experimento mostra a coordenada de maior valor (`argmax`).

**Por que isso importa:**

- **A regra foi escrita como matrizes, não como código.** Treinar é o gradiente
  encontrar matrizes como essas, sozinho, porque elas ajudam a prever o próximo
  caractere.
- **Heads assim aparecem em modelos treinados.** São as *previous token heads*, peça das
  *induction heads* (Olsson et al., 2022), um dos mecanismos mais estudados em
  Transformers.
- **A head só funciona porque a entrada tem informação de posição.** Sem ela, não haveria
  como "procurar a posição $p-1$". É a razão prática para os positional embeddings
  ([03-embeddings.md](03-embeddings.md)).
- **$d_k$ e $d_v$ podem ser diferentes.** Aqui a busca usa 6 dimensões e o conteúdo, 5.

---

## 5. Um módulo recém-criado: atenção quase uniforme

Seção 5 do experimento: um batch real `[32, 128]` passa pelos embeddings e por uma head
recém-inicializada.

```text
x (32, 128) -> embeddings (32, 128, 128) -> attention (32, 128, 32)

entropia / entropia máxima (1 = uniforme); peso uniforme na última linha = 0.0078
  embeddings (norma ~0,3)                        entropia 1.0000 | maior peso na última linha 0.0078
  embeddings normalizados (como após LayerNorm)  entropia 0.9996 | maior peso na última linha 0.0091
```

**O que é a entropia desta tabela.** A **entropia** de uma linha de pesos mede o quanto a
atenção está espalhada:

$$
H(w_i) = -\sum_{j \le i} w_{ij} \ln w_{ij}
$$

**Lendo a fórmula:**

- $H$: a entropia da linha $i$;
- $w_{ij} \ln w_{ij}$: como $0 < w_{ij} \le 1$, o log é $\le 0$; o sinal de menos deixa $H \ge 0$;
- $H = 0$ quando um peso vale 1 (toda a atenção num lugar só);
- $H$ é máxima, $\ln n$, quando os $n = i + 1$ pesos são iguais (uniforme).

O experimento divide $H$ por $\ln n$ e tira a média sobre as linhas e o batch (sem a linha
0, em que $\ln 1 = 0$). O resultado vai de 0 (concentrado) a 1 (uniforme). "Peso uniforme na
última linha" é $1/128 \approx 0{,}0078$, porque a última posição enxerga 128 posições.

**A attention começa praticamente uniforme**: cada posição faz quase uma média simples do
seu passado. Estimando as escalas:

- as entradas dos embeddings têm desvio $\approx 0{,}02\sqrt{2} \approx 0{,}028$ (token +
  posição), e os pesos das projeções, $0{,}02$;
- cada entrada de $q$ soma $d = 128$ termos: desvio $\approx 0{,}02 \cdot 0{,}028 \cdot \sqrt{128} \approx 0{,}006$;
- $q \cdot k / \sqrt{d_k}$ fica na ordem de $10^{-5}$: todos os scores são praticamente
  iguais.

**As regras usadas nessas estimativas** (para variáveis independentes com média 0):

1. **Soma:** a variância de uma soma é a soma das variâncias. Somar $n$ termos de desvio
   $\sigma$ dá desvio $\sigma\sqrt{n}$.
2. **Produto:** $\text{Var}(ab) = \text{Var}(a)\,\text{Var}(b)$, ou seja, os desvios se
   multiplicam.

**Aplicando, passo a passo:**

1. **Entrada.** Token embedding e positional embedding têm desvio 0,02 cada. A soma tem
   desvio $0{,}02\sqrt{2} \approx 0{,}028$. Um vetor de 128 números assim tem norma
   $\approx 0{,}028 \cdot \sqrt{128} \approx 0{,}32$ (o "norma ~0,3" da saída).
2. **Uma entrada de $q$.** É $\sum_{m=1}^{128} (W_Q)_{m} \, x_m$: 128 termos, cada um com
   desvio $0{,}02 \cdot 0{,}028$. Desvio total: $0{,}02 \cdot 0{,}028 \cdot \sqrt{128} \approx 0{,}0063$.
   O mesmo vale para $k$.
3. **Score.** $q \cdot k$ soma $d_k = 32$ termos, cada um com desvio $0{,}0063^2$: desvio
   $0{,}0063^2 \cdot \sqrt{32}$. Dividindo por $\sqrt{32}$, sobra $0{,}0063^2 \approx 4 \cdot 10^{-5}$.

Mesmo com a entrada normalizada (desvio 1 por dimensão, como depois da LayerNorm de
[07-transformer-block.md](07-transformer-block.md)), $q$ tem desvio $\approx 0{,}02\sqrt{128} \approx 0{,}23$, e os scores escalados ficam
com desvio $\approx 0{,}05$. Ainda é quase plano.

(Pela mesma conta do passo 3: desvio do score escalado $= 0{,}23^2 \approx 0{,}05$.) Scores
que variam $\pm 0{,}05$ dão exponenciais entre $e^{-0{,}05} \approx 0{,}95$ e $e^{0{,}05} \approx 1{,}05$:
todos os pesos ficam perto de $1/n$.

Isso é esperado e saudável: o modelo começa sem preferências, e **o treino cria os
padrões**. Pequenas diferenças aleatórias da inicialização quebram a simetria entre as
heads, e o gradiente amplifica as que ajudam a reduzir a loss.

Foi também por isso que a seção 3 do experimento usou pesos com escala maior: com a
inicialização padrão, todas as posições fariam quase a mesma média, e a demonstração
ficaria pouco legível.

---

## 6. Custo: $O(T^2)$

**Lendo o título:** $O(T^2)$ ("ó de tê ao quadrado") quer dizer que o custo cresce
proporcionalmente a $T^2$. Dobrar $T$ multiplica o custo por 4.

A matriz de pesos tem uma entrada por par de posições:

| $T$ | `[B=32, T, T]` | memória (float32) |
|---:|---:|---:|
| 128 | 524.288 | 2,0 MiB |
| 512 | 8.388.608 | 32,0 MiB |
| 2.048 | 134.217.728 | 512,0 MiB |
| 8.192 | 2.147.483.648 | 8.192,0 MiB |

**A conta da primeira linha:** $32 \cdot 128 \cdot 128 = 524.288$ números. Cada número
`float32` ocupa 4 bytes: $524.288 \cdot 4 = 2.097.152$ bytes. Um MiB é $2^{20} = 1.048.576$
bytes, então são 2,0 MiB.

Isso é **por head e por camada**. Tempo e memória crescem com $T^2$: dobrar o contexto
quadruplica o custo. Com $T = 128$ não há problema, mas é o gargalo dos modelos de
contexto longo.

No `tiny.yaml`, com $h = 4$ heads e $L = 4$ camadas, são 16 matrizes de pesos desse
tamanho: cerca de $16 \cdot 2 = 32$ MiB só para elas, que ficam guardadas para o backward.

O **FlashAttention** (Dao et al., 2022) calcula exatamente o mesmo resultado sem nunca
materializar a matriz $T \times T$ inteira: processa em blocos que cabem na memória rápida
da GPU. `F.scaled_dot_product_attention` do PyTorch usa kernels desse tipo quando
disponíveis. (Um **kernel** é uma rotina otimizada que roda direto na GPU ou CPU.) O teste
`test_matches_pytorch_implementation` confirma que a nossa
implementação produz os mesmos números, o que abre caminho para trocá-la por ela
([17-performance.md](17-performance.md)).

---

## 7. Da matemática ao código

### 7.1 Sem abstração

```python
Q, K, V = X @ W_q, X @ W_k, X @ W_v                    # [T, d_k], [T, d_k], [T, d_v]
scores = Q @ K.T / math.sqrt(d_k)                      # [T, T]
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # True onde j <= i
scores = scores.masked_fill(~mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)                # [T, T], linhas somam 1
output = weights @ V                                   # [T, d_v]
```

É exatamente o que a seção 1 do experimento executa e imprime.

| linha | fórmula | seção |
|---|---|---|
| 1 | $Q = XW_Q$, $K = XW_K$, $V = XW_V$ | 2.1 |
| 2 | $S / \sqrt{d_k}$ | 2.2 e 2.3 |
| 3–4 | $+\,M$ | 2.4 |
| 5 | $w_{ij} = \text{softmax}$ por linha | 2.5 |
| 6 | $O = WV$ | 2.6 |

### 7.2 As funções

| função | faz | por que existe separada |
|---|---|---|
| `causal_mask(T)` | `torch.tril` booleano `[T, T]` | reutilizada pelo módulo, pelos testes e pelos experimentos |
| `attention_weights(q, k, mask)` | escala, máscara e softmax | separa "**para onde** olhar" de "**o que** ler"; permite inspecionar os pesos e aplicar dropout entre as duas etapas |
| `scaled_dot_product_attention(q, k, v, mask)` | `attention_weights(...) @ v` | a fórmula completa; devolve saída **e** pesos |

As funções aceitam dimensões extras à esquerda (`[..., T, d_k]`), então funcionam com ou
sem batch. Em [05-multi-head-attention.md](05-multi-head-attention.md), também vão
funcionar com uma dimensão de heads.

#### `causal_mask(seq_len, device=None)` (linhas 9–11)

- **Objetivo:** produzir a máscara da seção 2.4 num só lugar.
- **Funcionalidade:** cria uma matriz `[seq_len, seq_len]` de `True` e aplica `torch.tril`,
  que mantém a diagonal e o triângulo inferior e põe `False` acima. `True` = "a posição $i$
  pode ver a posição $j$".
- **`device`:** cria a máscara direto na mesma CPU/GPU dos dados, para não precisar copiar.

#### `attention_weights(q, k, mask=None)` (linhas 14–30)

- **Objetivo:** calcular $\text{softmax}(QK^\top/\sqrt{d_k} + M)$, a parte "para onde olhar".
- **Funcionalidade, linha a linha:**
  1. linha 22: lê $d_k$ da última dimensão de `q`;
  2. linha 24: scores $S = QK^\top$, shape `[..., T, T]`;
  3. linha 26: divide por $\sqrt{d_k}$;
  4. linhas 27–29: se houver máscara, escreve $-\infty$ onde `mask` é `False`;
  5. linha 30: softmax na última dimensão. Cada linha passa a somar 1.
- **Sem máscara** (`mask=None`), é a attention bidirecional: a seção 3 do experimento usa
  isso para comparar.

#### `scaled_dot_product_attention(q, k, v, mask=None)` (linhas 33–43)

- **Objetivo:** a fórmula completa, $\text{Attention}(Q, K, V)$.
- **Funcionalidade:** chama `attention_weights` (linha 41) e faz a soma ponderada
  `weights @ v` (linha 43). Devolve a tupla `(saída, pesos)`: a saída `[..., T, d_v]` e os
  pesos `[..., T, T]`, úteis para inspecionar e para os testes.

### 7.3 O módulo `CausalSelfAttention`

```python
from model import CausalSelfAttention

attn = CausalSelfAttention(d_model=128, head_dim=32, dropout=0.1)
out = attn(h)     # h: [B, T, 128]  ->  out: [B, T, 32]
```

Um `nn.Module` é a forma do PyTorch de juntar **parâmetros** (pesos treináveis) e o cálculo
que os usa (`forward`). O módulo é **uma head** completa: as funções da seção 7.2 mais as
projeções e o dropout.

#### `__init__(d_model, head_dim, dropout)` (linhas 52–61)

- **Objetivo:** criar os parâmetros da head.
- **Funcionalidade:**
  1. linhas 55–57: três `nn.Linear(d_model, head_dim, bias=False)`, uma para cada papel. Cada
     uma guarda `weight: [head_dim, d_model]` = `[32, 128]`;
  2. linha 58: um `nn.Dropout(dropout)`, que não tem parâmetros;
  3. linhas 60–61: inicializa os três pesos com $\mathcal{N}(0,\ 0{,}02^2)$, isto é, números
     aleatórios em torno de 0 com desvio padrão 0,02.

#### `forward(x)` (linhas 63–74)

- **Objetivo:** aplicar a head a um batch.
- **Funcionalidade, com os shapes do `tiny.yaml`:**

| linha | código | shape | etapa |
|---|---|---|---|
| 65 | `seq_len = x.shape[1]` | $T = 128$ | lê o tamanho da sequência |
| 66–68 | `q_proj(x)`, `k_proj(x)`, `v_proj(x)` | `[32, 128, 32]` cada | projeções (2.1) |
| 70 | `causal_mask(seq_len, device=x.device)` | `[128, 128]` | máscara (2.4) |
| 71 | `attention_weights(q, k, mask)` | `[32, 128, 128]` | scores, escala, máscara, softmax (2.2–2.5) |
| 73 | `self.dropout(weights)` | `[32, 128, 128]` | dropout nos pesos (abaixo) |
| 74 | `weights @ v` | `[32, 128, 32]` | soma ponderada (2.6) |

#### O dropout nos pesos

**Dropout** é uma técnica de regularização (reduz o *overfitting*, decorar o treino em vez
de generalizar). Durante o treino, `nn.Dropout(p)` zera cada número de forma independente
com probabilidade $p$ e multiplica os que sobrevivem por $\tfrac{1}{1-p}$.

**Lendo o fator $\tfrac{1}{1-p}$:** cada número $w$ sobrevive com probabilidade $1 - p$ e
vira $w/(1-p)$, ou é zerado com probabilidade $p$. Em média,
$(1 - p) \cdot \tfrac{w}{1-p} + p \cdot 0 = w$: o valor esperado não muda.

- **Onde:** nos pesos de attention, antes de `@ v`. Zerar $w_{ij}$ corta a ligação "posição
  $i$ lê a posição $j$" naquele passo de treino.
- **Exemplo** (calculado à mão, $p = 0{,}1$ como no `tiny.yaml`): a linha de `'t'`,
  $(0{,}392;\ 0{,}373;\ 0{,}235;\ 0)$, vira $(0{,}436;\ 0{,}414;\ 0{,}261;\ 0)$ se nada for
  cortado (soma 1,111), ou $(0{,}436;\ 0;\ 0{,}261;\ 0)$ se o peso de `'a'` for cortado
  (soma 0,697).
- **Em `eval()`** o dropout vira a identidade: nada é zerado nem escalado, e as linhas voltam
  a somar exatamente 1. O teste `test_deterministic_in_eval_mode` confere que a saída fica
  determinística.

#### Decisões de projeto

| decisão | escolha | motivo |
|---|---|---|
| bias nas projeções | **não** | é exatamente $Q = XW_Q$, sem termo aditivo; padrão em modelos recentes (LLaMA). O GPT-2 usa bias |
| inicialização | $\mathcal{N}(0, 0{,}02^2)$ | mesma convenção dos embeddings (GPT-2) |
| máscara | recalculada a cada forward | custo desprezível com $T = 128$; o nanoGPT a guarda num buffer |
| dropout | nos **pesos**, antes de `@ v` | como o GPT-2 (`attn_pdrop`): corta ligações aleatórias entre posições durante o treino. Com o fator $1/(1-p)$, as linhas deixam de somar exatamente 1 no treino; em `eval()` o dropout é desligado |
| saída | `head_dim`, não `d_model` | uma head sozinha não precisa ter a largura do residual stream; [05-multi-head-attention.md](05-multi-head-attention.md) junta $h$ heads e projeta de volta para $d$ |

(Um **buffer** é um tensor guardado no módulo que não é treinado. O **residual stream** é o
vetor de largura $d$ que atravessa todos os blocos, ver [architecture.md](architecture.md).)

Parâmetros: $3 \cdot d \cdot d_k$. Com $d = 128$ e $d_k = 32$, são 12.288 por head.

**Lendo a fórmula:** são três matrizes ($W_Q$, $W_K$, $W_V$), cada uma com $d \cdot d_k$
números, e nenhum bias: $3 \cdot 128 \cdot 32 = 12.288$.

### 7.4 Quem usa estas funções depois

- **`MultiHeadAttention`** ([05-multi-head-attention.md](05-multi-head-attention.md)): faz as
  projeções para as $h$ heads de uma vez, separa em `[B, h, T, d_h]` e chama `causal_mask` e
  `attention_weights`. A dimensão de heads é tratada como mais uma dimensão de lote. Desde
  [07-transformer-block.md](07-transformer-block.md), aplica dropout também na saída (depois
  da projeção de saída), além dos pesos.
- **`LoopMultiHeadAttention`:** a versão didática, uma lista de
  `CausalSelfAttention`, uma por head. Serve de referência nos testes.
- O GPT completo ([08-gpt.md](08-gpt.md), com LM head em [09-lm-head.md](09-lm-head.md)) usa a
  `MultiHeadAttention` dentro de cada bloco.

---

## 8. O que cada teste garante

| teste | propriedade |
|---|---|
| `test_causal_mask_is_lower_triangular` | a máscara é triangular inferior, como na seção 2.4 |
| `test_matches_explicit_loops` | a versão matricial é igual a produtos escalares e somas feitos um a um |
| `test_matches_pytorch_implementation` | mesmo resultado que `F.scaled_dot_product_attention(is_causal=True)` |
| `test_weights_rows_sum_to_one` | cada linha é uma distribuição |
| `test_masked_positions_get_zero_weight` | peso exatamente 0 no futuro |
| `test_first_position_attends_only_to_itself` | $w_{00} = 1$ e $o_0 = v_0$ |
| `test_equal_scores_average_visible_values` | scores iguais dão a média do prefixo |
| `test_scaling_keeps_score_variance_near_one` | $\text{Var}(q\cdot k) \approx d_k$ e $\approx 1$ após a escala |
| `test_no_nan_with_large_scores` | estabilidade numérica com scores enormes |
| `test_future_tokens_do_not_affect_past` | trocar o futuro não muda o passado |
| `test_past_token_affects_every_later_position` | o passado de fato chega a todas as posições seguintes (o teste anterior não passa por vacuidade) |
| `test_gradient_only_reaches_current_and_past_inputs` | Jacobiana triangular inferior |
| `test_module_matches_function` | o módulo é a função aplicada às projeções |
| `test_gradients_reach_all_projections` | $W_Q$, $W_K$ e $W_V$ recebem gradiente |
| `test_deterministic_in_eval_mode` | mesma seed e `eval()` dão a mesma saída |
| `test_parameter_count` | $3 \cdot d \cdot d_k$ parâmetros |
| `test_module_output_shape` | `[B, T, d]` vira `[B, T, d_k]` (no teste, `[3, 10, 16]` → `[3, 10, 8]`) |

Os testes do módulo usam `d_model = 16` e `head_dim = 8`, e `dropout = 0.0` (exceto o de
`eval()`), para que os resultados sejam exatos.

---

## Resumo

1. Cada posição gera uma **query** (o que procura), uma **key** (como se anuncia) e um
   **value** (o que entrega), por três projeções lineares aprendidas.
2. O **produto escalar** entre query e keys dá a afinidade com cada posição; dividir por
   $\sqrt{d_k}$ mantém a variância em 1 e evita que a softmax sature.
3. A **causal mask** soma $-\infty$ ao futuro, e a **softmax** transforma cada linha numa
   distribuição sobre as posições visíveis.
4. A saída é a **média ponderada dos values**: a attention move informação entre
   posições, mas só por mistura.
5. Como só a attention mistura posições, **mascará-la torna o GPT inteiro causal**, e o
   treino em paralelo fica equivalente à geração token a token.
6. No módulo, o **dropout** corta pesos ao acaso durante o treino; em `eval()` ele é
   desligado.
7. O custo é $O(T^2)$ em tempo e memória. Recém-inicializada, a attention é quase uniforme;
   o treino é que cria padrões como a *previous token head*.

---

## Para checar o entendimento

1. Na linha de `'a'`, os scores escalados visíveis são $+0{,}631$ e $-0{,}138$. Calcule os
   pesos à mão e confira com $(0{,}683;\ 0{,}317)$.
2. Por que $S = QK^\top$ não é simétrica? O que mudaria se $W_Q = W_K$?
3. O que acontece com os pesos de uma linha se você somar 5 a todos os seus scores? E se
   multiplicar todos por 2?
4. A posição 0 sempre produz $o_0 = v_0$. O que isso implica para a previsão do segundo
   caractere de uma janela?
5. Mostre que, se a attention é causal e feed-forward e LayerNorm atuam por posição, o GPT
   inteiro é causal.
6. Na tabela da seção 3.2, por que a posição 11 tem a mesma diferença com e sem máscara?
7. Pela tabela da seção 2.3, o que você espera que aconteça com o treino de uma head com
   $d_k = 1024$ sem a divisão por $\sqrt{d_k}$?
8. Na head construída à mão, que mudança em $W_Q$ faria cada posição copiar o caractere de
   **duas** posições atrás?
9. Com dropout $p = 0{,}1$, uma linha de pesos pode somar 0,697 no treino. Por que isso não
   acontece em `eval()`? E por que, em média, o valor de cada peso não muda?
10. Usando a seção 3.4, explique por que uma attention sem máscara e sem positional
    embedding não distingue `"amor"` de `"roma"`.
11. Com $d = 128$ e $d_k = 32$, quantos parâmetros tem uma `CausalSelfAttention`? E quantos
    números tem a matriz de pesos de um batch do `tiny.yaml`?
