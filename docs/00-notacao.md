# Guia de notação e leitura das fórmulas

> **Como ler a matemática dos documentos do Mini-GPT?**

Os documentos deste projeto usam fórmulas para dizer com precisão o que o código calcula. Este
guia explica **uma vez** cada letra, símbolo e operação que aparece neles, sempre com um
exemplo numérico e o equivalente em PyTorch. Cada documento também tem a sua própria
tabela "Notação deste documento", com os símbolos específicos daquela parte.

Se uma fórmula parecer opaca, volte aqui: quase sempre ela é só uma forma curta de escrever
algo que também cabe numa frase e em uma ou duas linhas de código.

---

## 1. Três formas de dizer a mesma coisa

Todo conceito do projeto aparece em três "linguagens":

| linguagem | exemplo (embeddings, `03-embeddings.md`) |
|---|---|
| **palavras** | "o vetor de entrada de cada posição é o vetor do caractere somado ao vetor da posição" |
| **fórmula** | $h_t = E_{x_t} + P_t$ |
| **código** | `tok + pos` |

A fórmula é a ponte: mais precisa que as palavras e mais curta que o código. Para ler uma
fórmula, identifique **o que cada letra representa** (tabela de notação) e **qual operação liga
as letras** (seções 5 a 7 deste guia).

---

## 2. Números, vetores, matrizes e tensores

| objeto | o que é | notação | exemplo | PyTorch |
|---|---|---|---|---|
| **escalar** | um único número | $a$ | $a = 4{,}57$ | `torch.tensor(4.57)` |
| **vetor** | uma lista de números | $x \in \mathbb{R}^d$ | $x = (2, 4, 4)$, com $d = 3$ | `torch.tensor([2., 4., 4.])`, shape `(3,)` |
| **matriz** | uma tabela de números (linhas × colunas) | $W \in \mathbb{R}^{m \times n}$ | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, com $m = n = 2$ | shape `(2, 2)` |
| **tensor** | generalização com qualquer número de eixos | shape `[B, T, d]` | um batch de sequências de vetores | shape `(32, 128, 128)` |

**Como ler $x \in \mathbb{R}^d$:** "$x$ pertence a $\mathbb{R}^d$", ou seja, "$x$ é um vetor com
$d$ números reais". $\mathbb{R}$ é o conjunto dos números reais (qualquer número com vírgula).

**Como ler $W \in \mathbb{R}^{m \times n}$:** "$W$ é uma matriz de $m$ linhas e $n$ colunas".

**Índices (os números pequenos embaixo):**

| notação | significado | exemplo com $x = (2, 4, 7)$ e $W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
|---|---|---|
| $x_i$ | o elemento $i$ do vetor | $x_2 = 4$ (contando a partir de 1) |
| $W_{ij}$ | o elemento na linha $i$, coluna $j$ | $W_{21} = 3$ |
| $W_i$ | a linha $i$ inteira (um vetor) | $W_2 = (3, 4)$ |

No código, os índices começam em **0**: `x[1]` é o segundo elemento. Nos documentos, índices de
posição ($t = 0, 1, 2, \dots$) e de tokens (IDs) também começam em 0, como no código.

**Shapes com colchetes.** `[B, T, d]` descreve os eixos de um tensor, lido "B por T por d". Um
tensor `[32, 128, 128]` tem 32 sequências, cada uma com 128 posições, cada posição com um vetor
de 128 números. No total são $32 \times 128 \times 128 = 524{.}288$ números.

---

## 3. Letras que representam tamanhos

Estas letras são **hiperparâmetros** (valores escolhidos antes do treino, na config) ou tamanhos
derivados dos dados. Aparecem em quase todos os documentos.

| letra | como se lê | significado | no `tiny.yaml` | no código |
|---|---|---|---|---|
| $B$ | "bê" | tamanho do batch: quantas sequências processadas juntas | 32 | `batch_size` |
| $T$ | "tê" | tamanho do contexto: quantos tokens por sequência | 128 | `context_length`, `seq_len` |
| $V$ | "vê" | tamanho do vocabulário: quantos tokens diferentes existem | 97 | `vocab_size` |
| $d$ | "dê" | largura de cada vetor do modelo | 128 | `d_model` |
| $h$ | "agá" | número de heads de attention | 4 | `num_heads` |
| $d_h$, $d_k$, $d_v$ | "dê agá", "dê kê", "dê vê" | dimensão de cada head ($d / h$); $k$ de *key*, $v$ de *value* | 32 | `head_dim` |
| $d_{ff}$ | "dê éfe éfe" | dimensão interna do feed-forward | 512 | `d_ff` |
| $L$ | "ele" | número de blocos Transformer empilhados | 4 | `num_layers` |
| $N$ | "ene" | número de tokens do corpus (ou de um trecho dele) | 337.307 no treino | `len(ids)` |

**Atenção a letras reaproveitadas.** Existem mais conceitos que letras. Em alguns documentos,
$h$ significa **estado oculto** (um vetor), e não o número de heads; $s$ é o **stride** em
[02-dataset.md](02-dataset.md) e o **desvio dos logits** em [09-lm-head.md](09-lm-head.md). A
tabela de notação de cada documento sempre diz o sentido usado ali.

---

## 4. Letras que representam dados e parâmetros

**Parâmetros** são os números que o modelo aprende no treino (pesos). **Dados** são as entradas
e os valores intermediários calculados.

| símbolo | significado | aparece em |
|---|---|---|
| $x$, $x_t$ | a sequência de IDs de entrada; $x_t$ é o ID na posição $t$ | `02-dataset.md`, `03-embeddings.md` |
| $y$, $y_t$ | os alvos: o próximo token de cada posição ($y_t = x_{t+1}$) | `02-dataset.md`, `09-lm-head.md`, `10-loss.md` |
| $t$ | índice de **posição** na sequência (0, 1, 2, …) | todos |
| $i$, $j$, $k$, $m$ | índices genéricos ("para cada elemento $i$…") | todos |
| $E$ | matriz de **token embedding**, `[V, d]`: uma linha por token | `03-embeddings.md`, `09-lm-head.md` |
| $P$ | matriz de **positional embedding**, `[T, d]`: uma linha por posição | `03-embeddings.md` |
| $h$, $h_\ell$ | **estado oculto** / residual stream; $h_\ell$ é o valor depois do bloco $\ell$ | `07-transformer-block.md` a `09-lm-head.md` |
| $Q$, $K$, $V$ | queries, keys e values da attention | `04-attention.md`, `05-multi-head-attention.md` |
| $W_Q$, $W_K$, $W_V$, $W_O$ | matrizes de projeção da attention (O de *output*) | `04-attention.md`, `05-multi-head-attention.md` |
| $S$ | matriz de scores da attention | `04-attention.md` |
| $M$ | máscara causal | `04-attention.md` |
| $w_{ij}$ | peso de attention: quanto a posição $i$ olha para a posição $j$ | `04-attention.md` |
| $o_i$ | saída da attention na posição $i$ | `04-attention.md`, `05-multi-head-attention.md` |
| $W_1, b_1, W_2, b_2$ | pesos e biases das duas camadas do feed-forward | `06-feed-forward.md` |
| $k_i$, $v_i$ | no feed-forward: chave e valor do neurônio $i$ | `06-feed-forward.md` |
| $\gamma$, $\beta$ | escala e deslocamento aprendidos da LayerNorm | `07-transformer-block.md` |
| $W$ | matriz do LM head, `[V, d]` | `09-lm-head.md` |
| $z$, $z_i$ | **logits**: um score por token do vocabulário | `09-lm-head.md` |
| $p$, $p_i$ | probabilidades (saída da softmax) | `09-lm-head.md`, `10-loss.md` |
| $\mathcal{L}$ | **loss** (erro que o treino tenta diminuir) | `09-lm-head.md`, `10-loss.md` |
| $\theta$ | todos os parâmetros do modelo, vistos juntos | `03-embeddings.md` (uso prático no AdamW) |
| $I$ | matriz identidade: 1 na diagonal, 0 fora (multiplicar por ela não muda nada) | `07-transformer-block.md` |
| $J$ | matriz Jacobiana (seção 7) | `07-transformer-block.md` |

### Letras gregas

| letra | nome | uso no projeto |
|---|---|---|
| $\alpha$ | alfa | (reservado; pouco usado) |
| $\beta$ | beta | deslocamento da LayerNorm |
| $\gamma$ | gama | escala da LayerNorm |
| $\delta$ | delta | $\delta_{ij}$ = 1 se $i = j$, senão 0 |
| $\epsilon$ | épsilon | número minúsculo para evitar divisão por zero ($10^{-5}$) |
| $\eta$ | eta | learning rate (tamanho do passo do otimizador) |
| $\theta$ | teta | parâmetros do modelo; também ângulo entre vetores |
| $\lambda$ | lambda | weight decay |
| $\mu$ | mi | média |
| $\sigma$ | sigma | desvio padrão; $\sigma^2$ é a variância; em [06-feed-forward.md](06-feed-forward.md), também a função de ativação |
| $\varphi$, $\Phi$ | fi (minúsculo, maiúsculo) | densidade e distribuição acumulada da normal ([06-feed-forward.md](06-feed-forward.md)) |
| $\pi$ | pi | 3,14159…; aparece em fórmulas da normal |
| $\ell$ | "ele" cursivo (não é grega) | índice de camada: bloco $\ell$ |
| $\Sigma$ | sigma maiúsculo | soma (seção 5) |
| $\Pi$ | pi maiúsculo | produto (seção 5) |

---

## 5. Operações

### 5.1 Soma e produto de uma sequência

$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n
\qquad\qquad
\prod_{i=1}^{n} x_i = x_1 \cdot x_2 \cdot \ldots \cdot x_n
$$

- **Lendo:** "soma de $x_i$ para $i$ indo de 1 até $n$".
- **Exemplo:** com $x = (2, 4, 7)$: $\sum_i x_i = 13$ e $\prod_i x_i = 56$.
- **PyTorch:** `x.sum()` e `x.prod()`.
- **Variações:** $\sum_{j \le i}$ significa "somar sobre todos os $j$ menores ou iguais a $i$".

### 5.2 Produto escalar (dot product)

Multiplica dois vetores do mesmo tamanho **elemento a elemento** e soma tudo. O resultado é **um
número**.

$$
a \cdot b = \sum_{i=1}^{d} a_i b_i
$$

- **Exemplo:** $(1, 2, 3) \cdot (4, 5, 6) = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32$.
- **Intuição:** mede o quanto dois vetores "apontam na mesma direção". É grande e positivo se
  estão alinhados, perto de zero se são perpendiculares e negativo se apontam em sentidos
  opostos.
- **PyTorch:** `torch.dot(a, b)` ou `(a * b).sum()`.
- **Onde aparece:** scores da attention ($q \cdot k$), logits ($h \cdot w_i$).

### 5.3 Multiplicação de matrizes

$C = AB$ só existe se o número de **colunas** de $A$ for igual ao número de **linhas** de $B$:

$$
A \in \mathbb{R}^{m \times n},\; B \in \mathbb{R}^{n \times p} \;\Rightarrow\; AB \in \mathbb{R}^{m \times p},
\qquad (AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

- **Lendo:** o elemento $(i, j)$ do resultado é o **produto escalar** da linha $i$ de $A$ com a
  coluna $j$ de $B$.
- **Exemplo:**
  $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1{\cdot}5 + 2{\cdot}7 & 1{\cdot}6 + 2{\cdot}8 \\ 3{\cdot}5 + 4{\cdot}7 & 3{\cdot}6 + 4{\cdot}8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$
- **Regra dos shapes:** `[m, n] @ [n, p] → [m, p]`. As dimensões "de dentro" precisam ser iguais
  e desaparecem.
- **PyTorch:** `A @ B`. Com tensores de 3 ou mais eixos, `@` multiplica as **duas últimas**
  dimensões e trata as anteriores como lote: `[B, T, d] @ [d, V] → [B, T, V]`.
- **Ordem importa:** em geral $AB \neq BA$.

### 5.4 Transposta

$A^\top$ troca linhas por colunas: $(A^\top)_{ij} = A_{ji}$. Uma matriz `[m, n]` vira `[n, m]`.

- **Exemplo:** $\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$
- **PyTorch:** `A.T` (matrizes) ou `A.transpose(-2, -1)` (troca as duas últimas dimensões de um
  tensor).

### 5.5 Vetor linha, vetor coluna e o `nn.Linear`

Livros de álgebra costumam escrever vetores como **colunas** e a transformação como $Wx$. O
PyTorch guarda vetores como **linhas** e escreve $xW^\top$. É a mesma conta:

| forma | quando aparece |
|---|---|
| $y = Wx + b$ | fórmulas "por vetor" (`06-feed-forward.md`, `09-lm-head.md`) |
| $Y = XW^\top + b$ | fórmulas com vários vetores empilhados como linhas de $X$ |
| `y = x @ W.T + b` | o que `nn.Linear` calcula |

`nn.Linear(entrada, saída)` guarda `weight` com shape `[saída, entrada]`. Por isso
`nn.Linear(128, 512).weight` tem shape `(512, 128)`.

### 5.6 Norma (tamanho de um vetor)

$$
\|x\| = \sqrt{\sum_{i=1}^{d} x_i^2}
$$

- **Exemplo:** $\|(3, 4)\| = \sqrt{9 + 16} = 5$.
- **Intuição:** o comprimento da seta que representa o vetor.
- **PyTorch:** `x.norm()`; por vetor dentro de um tensor, `x.norm(dim=-1)`.

### 5.7 Similaridade de cosseno

$$
\cos\theta = \frac{a \cdot b}{\|a\|\,\|b\|}
$$

- **Lendo:** o produto escalar dividido pelos tamanhos. Mede **só a direção**, ignorando o
  tamanho.
- **Valores:** 1 = mesma direção; 0 = perpendiculares; −1 = direções opostas.
- **Exemplo:** $a = (1, 0)$, $b = (1, 1)$: $\cos\theta = 1 / (1 \cdot \sqrt{2}) \approx 0{,}707$
  (ângulo de 45°).
- **PyTorch:** `F.cosine_similarity(a, b, dim=0)`.

### 5.8 Exponencial e logaritmo

| notação | significado | exemplo | PyTorch |
|---|---|---|---|
| $e$ | número de Euler, ≈ 2,718 | | `math.e` |
| $e^x$, $\exp(x)$ | exponencial: sempre positiva, cresce muito rápido | $e^2 \approx 7{,}389$ | `torch.exp(x)` |
| $\ln x$, $\log x$ | logaritmo natural: o inverso da exponencial | $\ln 97 \approx 4{,}575$ | `torch.log(x)` |

Propriedades usadas nos documentos:

- $\ln(e^x) = x$ e $e^{\ln x} = x$;
- $\ln(ab) = \ln a + \ln b$ (produto vira soma);
- $\ln 1 = 0$, e $\ln x < 0$ para $0 < x < 1$ (probabilidades têm log negativo);
- $e^{-\infty} = 0$ (usado na máscara causal).

Nos documentos, $\log$ sem base significa sempre o logaritmo **natural**.

### 5.9 Softmax

Transforma uma lista de números quaisquer em **probabilidades**: todos positivos e somando 1.

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

- **Lendo:** exponencia cada número (tornando todos positivos) e divide pela soma das
  exponenciais (fazendo o total dar 1).
- **Exemplo:** $z = (2, 1, 0)$: $e^2 = 7{,}389$, $e^1 = 2{,}718$, $e^0 = 1$, soma $= 11{,}107$.
  $\text{softmax}(z) = (0{,}665;\ 0{,}245;\ 0{,}090)$.
- **Propriedades:** o maior número recebe a maior probabilidade; somar a mesma constante a todos
  não muda o resultado; números muito maiores que os outros levam a probabilidades perto de 1
  (saturação).
- **PyTorch:** `torch.softmax(z, dim=-1)`.

### 5.10 Outras notações

| notação | significado | exemplo |
|---|---|---|
| $\sqrt{x}$ | raiz quadrada | $\sqrt{128} \approx 11{,}31$ |
| $\max$, $\min$ | maior e menor valor | $\max(2, 7, 4) = 7$ |
| $\arg\max_i x_i$ | o **índice** do maior valor | $\arg\max (2, 7, 4) = 2$ (posição do 7, contando de 1) |
| $\lfloor x \rfloor$ | parte inteira (arredonda para baixo) | $\lfloor 7/2 \rfloor = 3$ |
| $\mathbb{1}[\text{condição}]$ | indicadora: 1 se a condição é verdadeira, 0 se não | $\mathbb{1}[3 = 3] = 1$ |
| $[a, b)$ | intervalo que inclui $a$ e exclui $b$ | features $[0, 32)$ = 0 a 31 |
| $x_{<t}$, $x_{\le t}$ | todos os elementos antes de $t$ (ou até $t$, inclusive) | $x_{<3} = (x_0, x_1, x_2)$ |
| $f: A \to B$ ou "`[B, T]` → `[B, T, d]`" | uma função que leva algo de um formato a outro | |
| $\approx$ | aproximadamente igual | $\pi \approx 3{,}14$ |
| $\propto$ | proporcional a | $y \propto 1/x$: dobrar $x$ divide $y$ por 2 |
| $\Rightarrow$ | "implica", "logo" | |
| $\Leftrightarrow$ | "é equivalente a" | |
| $\leftarrow$ | atribuição: "passa a valer" (como `=` no código) | $x \leftarrow x + 1$ |
| $\equiv$ | "definido como" ou "idêntico a" | |

---

## 6. Estatística e probabilidade

### 6.1 Média, variância e desvio padrão

Para um vetor $x = (x_1, \dots, x_d)$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2
\qquad
\sigma = \sqrt{\sigma^2}
$$

- **$\mu$ (média):** o valor "central".
- **$\sigma^2$ (variância):** a média dos quadrados das distâncias até a média. Mede o quanto os
  valores se espalham.
- **$\sigma$ (desvio padrão):** a raiz da variância, na mesma unidade dos dados.
- **Exemplo** (LayerNorm, [07-transformer-block.md](07-transformer-block.md)): $x = (2, 4, 4, 4, 5, 7)$ tem $\mu = 4{,}333$,
  $\sigma^2 = 2{,}222$ e $\sigma = 1{,}491$.
- **PyTorch:** `x.mean()`, `x.var()`, `x.std()`. Cuidado: `x.var()` e `x.std()` dividem por
  $d - 1$ por padrão; a LayerNorm divide por $d$.

`Var(·)` é outra forma de escrever variância: $\text{Var}(X) = \sigma^2$.

### 6.2 Esperança

$\mathbb{E}[X]$ ("esperança de $X$", ou valor esperado) é a média que se obteria repetindo um
experimento aleatório infinitas vezes. Exemplo: para um dado de 6 faces,
$\mathbb{E}[X] = (1 + 2 + 3 + 4 + 5 + 6)/6 = 3{,}5$.

### 6.3 Distribuição normal

$\mathcal{N}(\mu, \sigma^2)$ é a "curva de sino" com média $\mu$ e variância $\sigma^2$. Os
documentos usam para a **inicialização dos pesos**:

- $\mathcal{N}(0,\ 0{,}02^2)$: números aleatórios em torno de 0, com desvio padrão 0,02. Cerca de
  68% ficam entre −0,02 e +0,02, e 95% entre −0,04 e +0,04;
- $\mathcal{N}(0, 1)$: a **normal padrão**, com média 0 e desvio 1;
- **PyTorch:** `nn.init.normal_(peso, std=0.02)` ou `torch.randn(...) * 0.02`.

$Z \sim \mathcal{N}(0, 1)$ se lê "$Z$ segue uma normal padrão". $\Phi(x) = P(Z \le x)$ é a
probabilidade de $Z$ ser menor ou igual a $x$ (usada na GELU, [06-feed-forward.md](06-feed-forward.md)).

### 6.4 Probabilidade condicional

| notação | como se lê | exemplo |
|---|---|---|
| $P(A)$ | "probabilidade de A" | $P(\text{cara}) = 0{,}5$ |
| $P(A \mid B)$ | "probabilidade de A **dado** B" (sabendo que B aconteceu) | $P(x_t = \texttt{'u'} \mid x_{t-1} = \texttt{'q'})$: chance do próximo ser `'u'` depois de um `'q'` |
| $P(x_t \mid x_{<t})$ | probabilidade do token $t$ dado tudo que veio antes | o que o GPT estima |

---

## 7. Derivadas e gradientes

O treino ajusta os parâmetros para diminuir a loss. Para saber **em que direção** mexer cada
parâmetro, usa-se a derivada.

### 7.1 Derivada

$\dfrac{df}{dx}$ ou $f'(x)$ mede **quanto $f$ muda quando $x$ muda um pouquinho**.

- **Exemplo:** $f(x) = x^2$ tem $f'(x) = 2x$. Em $x = 3$, $f'(3) = 6$. De fato,
  $f(3{,}01) = 9{,}0601$: $x$ subiu 0,01 e $f$ subiu ~0,06, ou seja, $6 \times 0{,}01$.
- **Sinal:** derivada positiva = aumentar $x$ aumenta $f$; negativa = aumentar $x$ diminui $f$.

### 7.2 Derivada parcial e gradiente

Quando $f$ depende de várias variáveis, $\dfrac{\partial f}{\partial x_i}$ (lê-se "d parcial de
$f$ em relação a $x_i$") é a derivada em relação a **uma** delas, mantendo as outras fixas. O
símbolo $\partial$ é um "d" curvo.

O **gradiente** $\nabla f$, ou $\dfrac{\partial \mathcal{L}}{\partial \theta}$, é o vetor com
todas as derivadas parciais. Ele aponta na direção em que $f$ **mais cresce**.

- **Exemplo:** $f(a, b) = a^2 + 3b$ tem $\partial f/\partial a = 2a$ e $\partial f/\partial b = 3$.
- **PyTorch:** depois de `loss.backward()`, cada parâmetro `p` tem o seu gradiente em `p.grad`,
  com o mesmo shape de `p`.

### 7.3 Regra da cadeia

Se $y$ depende de $u$, e $u$ depende de $x$, as derivadas se **multiplicam**:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

- **Exemplo:** $y = u^2$ com $u = 2x + 1$. $dy/du = 2u$ e $du/dx = 2$. Em $x = 1$: $u = 3$ e
  $dy/dx = 2 \cdot 3 \cdot 2 = 12$.
- **Por que importa:** uma rede é uma longa cadeia de funções. O backpropagation é a regra da
  cadeia aplicada da loss até cada parâmetro. O `loss.backward()` faz isso automaticamente.

### 7.4 Jacobiana

Quando a entrada **e** a saída são vetores, as derivadas formam uma matriz, a **Jacobiana**
$J$: a linha $i$, coluna $j$ diz quanto a saída $i$ muda quando a entrada $j$ muda. Em
[04-attention.md](04-attention.md), "a Jacobiana é triangular inferior" significa: a saída da posição $i$ não depende das entradas
das posições $j > i$ (o futuro), então essas derivadas são zero.

### 7.5 Descida do gradiente

A regra básica de treino:

$$
\theta \leftarrow \theta - \eta\, \frac{\partial \mathcal{L}}{\partial \theta}
$$

- **Lendo:** cada parâmetro $\theta$ dá um passo **contra** o gradiente (onde a loss diminui), de
  tamanho proporcional à learning rate $\eta$.
- **Exemplo:** loss $\mathcal{L} = \theta^2$, com $\theta = 3$ e $\eta = 0{,}1$. O gradiente é
  $2\theta = 6$, e o novo valor é $\theta = 3 - 0{,}1 \cdot 6 = 2{,}4$ (a loss cai de 9 para 5,76).
- **PyTorch:** `optimizer.step()`. AdamW ([11-training.md](11-training.md)) é uma versão mais sofisticada dessa regra.

---

## 8. Da fórmula ao PyTorch

| matemática | PyTorch | observação |
|---|---|---|
| $x \in \mathbb{R}^{B \times T \times d}$ | `x.shape == (B, T, d)` | |
| $x_t$ | `x[t]` | índices começam em 0 |
| $a \cdot b$ | `(a * b).sum()` ou `torch.dot(a, b)` | |
| $AB$ | `A @ B` | |
| $A^\top$ | `A.T`, `A.transpose(-2, -1)` | |
| $a \odot b$ (elemento a elemento) | `a * b` | |
| $\sum_i x_i$ | `x.sum()` | `x.sum(dim=-1)` soma ao longo da última dimensão |
| $\mu$ de cada vetor | `x.mean(dim=-1, keepdim=True)` | `keepdim` mantém a dimensão com tamanho 1 para o broadcasting |
| $\|x\|$ | `x.norm(dim=-1)` | |
| $\text{softmax}(z)$ | `torch.softmax(z, dim=-1)` | |
| $\log \text{softmax}(z)$ | `F.log_softmax(z, dim=-1)` | numericamente estável |
| $e^x$, $\ln x$ | `torch.exp(x)`, `torch.log(x)` | |
| $\mathcal{N}(0, \sigma^2)$ | `torch.randn(...) * sigma` | |
| $\mathbb{1}[i = y]$ | `F.one_hot(y, V)` | vetor com 1 na posição $y$ |
| $\dfrac{\partial \mathcal{L}}{\partial \theta}$ | `loss.backward()`, depois `theta.grad` | |

### O que é `dim=-1`

Muitas operações precisam saber **ao longo de qual eixo** agir. `dim=-1` significa "a última
dimensão". Num tensor `[B, T, d]`, `x.mean(dim=-1)` calcula uma média **por vetor** (sobre as
$d$ features) e devolve `[B, T]`.

### O que é broadcasting

Quando dois tensores de shapes diferentes são combinados, o PyTorch "estica" automaticamente as
dimensões de tamanho 1 (ou ausentes à esquerda). Exemplo de [03-embeddings.md](03-embeddings.md): `tok` tem shape
`[B, T, d]` e `pos` tem `[T, d]`; `tok + pos` soma o mesmo vetor de posição em todas as $B$
sequências, sem copiar `pos` $B$ vezes.

---

## 9. Convenções dos documentos

- **Números:** no texto, vírgula decimal e ponto de milhar (4,575; 820.096). Nas saídas de
  experimento, copiadas do terminal, ponto decimal e vírgula de milhar (4.575; 820,096).
- **Blocos `text`** são saídas reais dos experimentos, sem edição.
- **`NN-nome.md`** (ex.: `04-attention.md`) é como um documento cita outro: o número é só a ordem
  de leitura sugerida, do tokenizer até performance.
- **Shapes** usam colchetes (`[B, T, d]`); nas saídas do PyTorch aparecem com parênteses
  (`(32, 128, 128)`).
- **Termos técnicos** ficam em inglês quando é assim que aparecem no código e na literatura
  (attention, embedding, batch, logits).
