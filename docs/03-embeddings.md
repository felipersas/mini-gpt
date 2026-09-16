# Embeddings

> **Como um ID vira um vetor?**

Código: [`model/embeddings.py`](../model/embeddings.py). Testes:
[`tests/test_embeddings.py`](../tests/test_embeddings.py). Experimento:
`uv run python -m experiments.e03_embeddings`. Notação geral: [00-notacao.md](00-notacao.md).
Visão geral: [architecture.md](architecture.md).

---

## Para que serve

### O problema

Depois de [02-dataset.md](02-dataset.md), cada batch é um tensor `x` de shape `[32, 128]` cheio de
IDs inteiros. Isso ainda não serve para a rede, por dois motivos:

1. **Um ID é só um rótulo** (seção 1). A rede precisa de números que possam expressar
   semelhanças entre tokens e que possam ser **aprendidos**.
2. **As camadas seguintes não sabem a posição de cada token** (seção 4). Sem essa
   informação, a ordem das letras se perde.

Este documento resolve os dois problemas. Ele troca cada ID por um vetor de $d = 128$ números reais
(o **embedding** do token) e soma um segundo vetor, que representa a posição.

Termos usados daqui em diante:

- **Vetor:** uma lista de números. Aqui, 128 números reais (números com casas decimais).
- **Embedding:** o vetor que representa um token (ou uma posição). A palavra vem de "embutir":
  cada token é colocado como um ponto num espaço de 128 dimensões.
- **Parâmetro:** um número que o treino ajusta. As duas tabelas de embedding são parâmetros.
- **Inicialização:** os valores dos parâmetros antes do treino. Aqui, sorteados (seção 6).
- **Gradiente:** para cada parâmetro, quanto a loss muda se ele mudar um pouco. O otimizador
  usa isso para decidir como ajustá-lo (seção 3).
- **Loss:** o número que mede o erro das previsões; o treino tenta diminuí-lo.

### Uma analogia

Pense numa ficha de personagem com 128 atributos. Cada caractere tem a sua ficha. No começo,
os valores são sorteados. O treino vai corrigindo as fichas até que caracteres que se
comportam de forma parecida no texto tenham fichas parecidas.

A posição também tem a sua ficha ("estou na posição 3"). As duas fichas são somadas, atributo
por atributo, e o resultado é o que entra na rede.

### O que entra e o que sai

| | shape | tipo | exemplo |
|---|---|---|---|
| entrada `ids` | `[B, T]` = `[32, 128]` | `int64` (inteiros) | `"casa"` → `[[55, 53, 71, 53]]`, shape `[1, 4]` |
| saída | `[B, T, d]` = `[32, 128, 128]` | `float32` (reais de 32 bits) | shape `[1, 4, 128]`: um vetor de 128 números por caractere |

Parâmetros deste componente: `token_embedding.weight` com shape `[97, 128]` e
`position_embedding.weight` com shape `[128, 128]`, somando 28.800 números (seção 8).

### Onde este documento entra

```text
texto: "Uma noite destas, vindo da cidade..."
  │
  ▼  tokenizer (01-tokenizer.md)               str → IDs [374.785]
  ▼  dataset e batches (02-dataset.md)         IDs → x, y: [32, 128]
  ▼  embeddings                                [32, 128] → [32, 128, 128]       ← este documento
  ▼  blocos Transformer × 4 (04 a 08)          [32, 128, 128] → [32, 128, 128]
  ▼  LM head (09-lm-head.md)                   [32, 128, 128] → logits [32, 128, 97]
  ▼  loss (10-loss.md, ainda não implementada) logits e y → um número
```

### O que daria errado sem esta parte

- **Usar o ID como número** criaria relações falsas: `'c'` (55) pareceria "maior" que `'a'` (53).
- **Sem parâmetros na entrada,** o modelo não teria como aprender o que cada caractere
  significa.
- **Sem posição,** "amor" e "roma" produziriam os mesmos vetores, só em outra ordem
  (seção 4.1).
- **Com a inicialização padrão do PyTorch,** os vetores seriam ~50 vezes maiores, e a loss
  inicial ficaria muito acima de $\ln 97$ (seção 6).

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $B$ | "bê" | tamanho do batch: quantas sequências por vez | 32 | `ids.shape[0]` |
| $T$ | "tê" | `context_length`: número máximo de posições. A sequência real pode ser menor (`seq_len`) | 128 | `context_length`, `seq_len` |
| $V$ | "vê" | tamanho do vocabulário. **Exceção:** na fórmula da seção 4.1, $V$ é a matriz de *values* da attention | 97 | `vocab_size` |
| $d$ | "dê" | tamanho de cada vetor (`d_model`) | 128 | `d_model` |
| $b$ | "bê" minúsculo | qual sequência dentro do batch | de 0 a 31 | primeiro índice de `ids[b, t]` |
| $t$ | "tê" minúsculo | posição dentro da sequência | de 0 a 127 | `positions` |
| $i$, $j$ | "i", "jota" | IDs de tokens, usados como números de linha | `'a'` = 53 | `ids` |
| $k$ | "cá" | posição de um número dentro de um vetor | de 0 a $d - 1$ | último índice |
| $x_{b,t}$ | "xis bê tê" | o ID na sequência $b$, posição $t$ | | `ids[b, t]` |
| $\mathbf{e}_i$ | "e i", em negrito | vetor **one-hot** do token $i$: 1 na posição $i$, 0 no resto | shape `[97]` | `F.one_hot(i, V)` |
| $\{0,1\}^V$ | "zero-um à vê" | vetores de $V$ entradas, cada uma 0 ou 1 | | |
| $\mathbb{R}^{V \times d}$ | "erre vê por dê" | matrizes de números reais com $V$ linhas e $d$ colunas | | shape `(V, d)` |
| $\mathbb{R}^d$ | "erre dê" | vetores de $d$ números reais | | shape `(d,)` |
| $E$ | "é" maiúsculo | matriz de **token embedding**: uma linha por token | `[97, 128]` | `token_embedding.weight` |
| $E_i$ | "é i" | linha $i$ de $E$: o vetor do token $i$ | `[128]` | `token_embedding.weight[i]` |
| $\text{emb}(i)$ | "emb de i" | o embedding do token $i$ | `[128]` | `token_embedding(ids)` |
| $P$ | "pê" maiúsculo | matriz de **positional embedding**: uma linha por posição | `[128, 128]` | `position_embedding.weight` |
| $P_t$ | "pê tê" | linha $t$ de $P$: o vetor da posição $t$ | `[128]` | `position_embedding.weight[t]` |
| $^\top$ | "transposta" | troca linhas por colunas; transforma um vetor coluna em vetor linha | | `.T` |
| $O(\cdot)$ | "ó de" | ordem de grandeza do custo: como o número de operações cresce | $O(d)$: ~128 operações | |
| $\mathcal{L}$ | "ele caligráfico" | a loss. Não confundir com $L$ (número de blocos) | um número | `loss` |
| $\partial \mathcal{L} / \partial E_i$ | "d parcial de ele em relação a é i" | gradiente da loss em relação à linha $i$ | `[128]` | `token_embedding.weight.grad[i]` |
| $u_{b,t}$ | "u bê tê" | o vetor copiado da tabela para a posição $(b, t)$: $u_{b,t} = E_{x_{b,t}}$ | `[128]` | `tok[b, t]` |
| $\theta$ | "teta" | um parâmetro qualquer | | `p` |
| $\eta$ | "eta" | learning rate: tamanho do passo do otimizador | 0,0003 | `learning_rate` |
| $\lambda$ | "lambda" | weight decay: quanto os parâmetros encolhem a cada passo | 0,01 | `weight_decay` |
| $m$, $v$ | "eme", "vê" minúsculo | momentos do Adam: médias móveis do gradiente e do gradiente ao quadrado | mesmo shape de cada parâmetro | estado interno do otimizador |
| $X$ | "xis" maiúsculo | matriz com um vetor de token por linha, sem posição | `[T, d]` | |
| $R$ | "erre" | matriz de permutação: reordena as linhas de $X$ | `[T, T]` | |
| $I$ | "i" maiúsculo | matriz identidade: multiplicar por ela não muda nada | `[T, T]` | `torch.eye(T)` |
| $W_Q$, $W_K$, $W_V$ | "dáblio quê", "dáblio cá", "dáblio vê" | matrizes de projeção da attention (`04-attention.md`) | | |
| $Q$, $K$, $V$ | "quê", "cá", "vê" | queries, keys e values: $Q = XW_Q$, $K = XW_K$, $V = XW_V$ (só na seção 4.1) | | |
| $d_k$ | "dê cá" | dimensão das keys | 32 por head no Mini-GPT | `head_dim` |
| $\text{softmax}$ | "softmax" | transforma números em probabilidades, linha por linha ([00-notacao.md](00-notacao.md), seção 5.9) | | `torch.softmax` |
| $\text{Attn}(X)$ | "atenção de xis" | a saída da self-attention para a entrada $X$ | `[T, d]` | `04-attention.md` |
| $h^{(0)}_{b,t}$ | "agá zero, bê tê" | o vetor que entra no primeiro bloco Transformer (depois de zero blocos) | `[128]` | saída de `Embeddings.forward` |
| $a \cdot b$ | "produto escalar de a e b" | multiplica elemento a elemento e soma | um número | `(a * b).sum()` |
| $\lVert a \rVert$ | "norma de a" | tamanho (comprimento) do vetor | 0,226 na inicialização | `a.norm()` |
| $\cos(a, b)$ | "cosseno entre a e b" | similaridade de direção: 1 = mesma direção, 0 = perpendiculares | 0,435 (seção 5) | `F.cosine_similarity` |
| $\sigma$ | "sigma" | desvio padrão da inicialização | 0,02 | `std=0.02` |
| $\mathcal{N}(\mu, \sigma^2)$ | "normal de mi, sigma ao quadrado" | distribuição normal (curva de sino) usada para sortear os pesos | $\mathcal{N}(0,\, 0{,}02^2)$ | `nn.init.normal_` |
| $\mathbb{E}[\cdot]$ | "valor esperado de" | a média que se obteria repetindo o sorteio muitas vezes | | |
| $p$ | "pê" minúsculo | probabilidade de **dropout**. Diferente do guia geral, onde $p$ é uma probabilidade da softmax | 0,1 | `dropout` |
| $a_k$ | "a cá" | o número $k$ do vetor que entra no dropout (`tok + pos`) | | `(tok + pos)[..., k]` |
| $\kappa_k$ | "capa cá" | a máscara do dropout: 1 mantém o número $k$, 0 zera | 0 ou 1 | sorteado dentro de `nn.Dropout` |
| $\sqrt{\cdot}$ | "raiz quadrada de" | | $\sqrt{128} \approx 11{,}31$ | `math.sqrt` |
| $\leftarrow$ | "passa a valer" | atribuição, como `=` no código | | `=` |
| $\approx$ | "aproximadamente" | | | |

---

## 1. Por que não usar o ID diretamente

Um ID é um **rótulo**, não uma quantidade. No vocabulário do corpus, `'a'` tem ID 53 e
`'c'` tem ID 55, mas isso não significa que `'c'` seja "maior" que `'a'`, nem que `'a'`
esteja mais perto de `'c'` do que de `'x'`. A ordem vem de `sorted()` e não diz nada
sobre linguagem. Entregar o ID como número a uma rede criaria relações falsas.

A primeira alternativa é o **one-hot**: o token $i$ vira o vetor $\mathbf{e}_i \in \{0,1\}^V$,
com um único 1 na posição $i$. Isso elimina a ordem falsa, mas cria outros problemas.

A definição do one-hot, entrada por entrada:

$$
\mathbf{e}_i \in \{0,1\}^V,
\qquad
(\mathbf{e}_i)_j =
\begin{cases}
1 & \text{se } j = i \\
0 & \text{se } j \neq i
\end{cases}
$$

**Lendo a fórmula:**

- $\mathbf{e}_i$: o one-hot do token $i$. O negrito o diferencia de um número.
- $\{0,1\}^V$: o conjunto dos vetores com $V$ entradas, cada uma igual a 0 ou 1.
- $(\mathbf{e}_i)_j$: a entrada de número $j$ desse vetor.
- A chave `{` separa os casos: vale 1 só na posição igual ao ID, e 0 em todas as outras.

**Exemplo:** com $V = 4$, o token 2 vira $\mathbf{e}_2 = (0, 0, 1, 0)$. No Mini-GPT, `'a'`
(ID 53) vira um vetor de 97 números com um único 1, na posição 53.

Os problemas do one-hot:

- todos os pares de tokens ficam à mesma distância ($\sqrt{2}$), então não há como
  expressar que dois tokens são parecidos;
- a dimensão cresce com o vocabulário (com BPE, $V \approx 50$ mil);
- o vetor não carrega nada aprendido.

A distância $\sqrt{2}$ vem da norma da diferença entre dois one-hots:

$$
\lVert \mathbf{e}_i - \mathbf{e}_j \rVert = \sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1{,}414
\qquad (i \neq j)
$$

**Lendo a fórmula:**

- $\mathbf{e}_i - \mathbf{e}_j$: a diferença tem um $+1$ na posição $i$, um $-1$ na posição $j$
  e zeros no resto.
- $\lVert \cdot \rVert$: a norma, raiz da soma dos quadrados. Os zeros não contribuem.
- $(i \neq j)$: vale para qualquer par de tokens diferentes.

**Exemplo:** $\mathbf{e}_0 - \mathbf{e}_1 = (1, -1, 0, 0)$, com norma $\sqrt{2}$. A distância
entre `'a'` e `'b'` é a mesma que entre `'a'` e `'»'`.

---

## 2. Embedding é uma tabela de vetores aprendidos

Um embedding é uma matriz de parâmetros $E \in \mathbb{R}^{V \times d}$: uma linha por
token. O vetor do token $i$ é simplesmente a linha $i$.

A fórmula diz que buscar a linha $i$ é o mesmo que multiplicar o one-hot pela tabela:

$$
\text{emb}(i) = E_{i} = \mathbf{e}_i^\top E
$$

**Lendo a fórmula:**

- $\text{emb}(i)$: o embedding do token $i$.
- $E$: a tabela, com $V$ linhas (uma por token) e $d$ colunas.
- $E_i$: a linha $i$ da tabela, um vetor de $d$ números.
- $\mathbf{e}_i^\top$: o one-hot "deitado" como vetor linha, com shape `[1, V]`. O $^\top$
  (transposta) troca coluna por linha.
- $\mathbf{e}_i^\top E$: multiplicação de matrizes `[1, V] @ [V, d]` → `[1, d]`. Cada linha de
  $E$ é multiplicada pela entrada correspondente do one-hot. Só a linha $i$ é multiplicada por
  1; as outras são multiplicadas por 0 e somem.

**Exemplo** (feito à mão): com $V = 3$, $d = 2$ e
$E = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$, o token 1 tem
$\mathbf{e}_1 = (0, 1, 0)$ e
$\mathbf{e}_1^\top E = 0 \cdot (1, 2) + 1 \cdot (3, 4) + 0 \cdot (5, 6) = (3, 4)$, que é a linha 1
(a segunda, contando de 0).

A segunda igualdade mostra que **o embedding é uma camada linear sem bias aplicada ao
one-hot**. A diferença é só de custo: indexar uma linha custa $O(d)$, e multiplicar o
one-hot pela matriz custa $O(Vd)$. O experimento confirma a equivalência numericamente:

```text
IDs de 'casa': [55, 53, 71, 53]
one-hot: (1, 4, 97) -> one_hot @ weight: (1, 4, 128)
embedding(ids) == weight[ids]:      True
embedding(ids) == one_hot @ weight: True
```

**O que é $O(\cdot)$.** A notação "ó grande" descreve como o custo cresce, sem contar
constantes. $O(d)$ significa "proporcional a $d$"; $O(Vd)$, "proporcional a $V$ vezes $d$".

**Exemplo:** indexar copia $d = 128$ números. Multiplicar o one-hot faz
$V \cdot d = 97 \times 128 = 12.416$ multiplicações por token, quase todas por zero. Com os
tamanhos do GPT-2 ($V = 50.257$, $d = 768$), seriam cerca de 38,6 milhões por token.

Em PyTorch, `nn.Embedding(V, d)` guarda `weight: [V, d]` e o forward é `weight[ids]`.
Para `ids: [B, T]`, o resultado é `[B, T, d]`: cada ID é trocado pelo seu vetor.

**Exemplo:** `ids = [[55, 53, 71, 53]]` (shape `[1, 4]`) vira $[E_{55}, E_{53}, E_{71}, E_{53}]$
(shape `[1, 4, 128]`). As posições 1 e 3 recebem **o mesmo** vetor, porque são o mesmo `'a'`.
Os primeiros 8 dos 128 números do vetor de `'c'`, recém-inicializado:

```text
vetor de 'c' (8 de 128 dims): +0.016 +0.003 -0.010 -0.005 -0.019 +0.011 +0.015 -0.010
```

Na inicialização os vetores são aleatórios e não significam nada. O treino ajusta cada
linha para que a rede consiga prever o próximo token. Tokens que aparecem em contextos
parecidos tendem a ficar com vetores parecidos: é a hipótese distribucional, a mesma
ideia por trás do word2vec. Com caracteres, é razoável esperar vogais próximas entre si,
maiúsculas perto das minúsculas correspondentes e pontuação num grupo próprio. Isso vai
poder ser verificado depois do treino.

- **Hipótese distribucional:** o significado de um token se revela pelos contextos em que ele
  aparece.
- **word2vec:** um método clássico (2013) que aprende vetores de palavras a partir dos
  contextos em que elas aparecem.

---

## 3. O gradiente de um embedding é esparso

**O que esta seção responde:** quando o treino corrige a tabela $E$, quais linhas mudam?

Para ajustar um parâmetro, o treino calcula o **gradiente**: quanto a loss $\mathcal{L}$ muda
quando o parâmetro muda um pouco. Ele é calculado de trás para frente pela **regra da
cadeia** ([00-notacao.md](00-notacao.md), seção 7). Um gradiente **esparso** é um gradiente em
que a maioria dos valores é zero.

No forward, a saída na posição $(b, t)$ é $u_{b,t} = E_{x_{b,t}}$. Pela regra da cadeia,
o gradiente da linha $i$ soma as contribuições de todas as posições onde o token $i$
apareceu:

$$
\frac{\partial \mathcal{L}}{\partial E_i} = \sum_{(b,t)\,:\,x_{b,t} = i} \frac{\partial \mathcal{L}}{\partial u_{b,t}}
$$

**Lendo a fórmula:**

- $u_{b,t} = E_{x_{b,t}}$: o vetor que a tabela copiou para a posição $(b, t)$. No código, é
  `tok[b, t]`. (Usamos $u$, e não $y$, porque $y$ é o alvo em [02-dataset.md](02-dataset.md).)
- $\frac{\partial \mathcal{L}}{\partial u_{b,t}}$: o gradiente que chega das camadas de cima
  para essa posição, um vetor de $d$ números.
- $\sum_{(b,t)\,:\,x_{b,t} = i}$: soma **só** sobre as posições cujo ID é $i$. Posições com
  outros tokens não entram.
- $\frac{\partial \mathcal{L}}{\partial E_i}$: o gradiente da linha $i$ da tabela, também com $d$
  números.

**Exemplo** (teste `test_gradient_accumulates_over_repeated_tokens`, com $d = 16$): a entrada
é `[[2, 5, 5]]` e a loss é a soma de todas as saídas. Nesse caso, cada posição recebe o
gradiente $(1, 1, \dots, 1)$. A linha 2 aparece uma vez e recebe $(1, \dots, 1)$. A linha 5
aparece duas vezes e recebe $(1, \dots, 1) + (1, \dots, 1) = (2, \dots, 2)$. A linha 0 não
aparece e recebe zeros (valores conferidos rodando o teste em Python).

Duas consequências, ambas cobertas por testes:

1. **Tokens ausentes no batch recebem gradiente zero.**
2. **Tokens repetidos acumulam gradiente**: um token que aparece duas vezes recebe a soma
   das duas contribuições.

No corpus real (um batch de `[32, 128]`):

```text
caracteres distintos no batch: 64 de 97
linhas de token_embedding com gradiente: 64
sem gradiente: '<PAD>' '<UNK>' '<BOS>' '<EOS>' '$' '(' ')' '+' '0' '1' ... 'W' 'Z' 'k' '«' '»' ...
linhas de position_embedding com gradiente: 128 de 128
```

Ao longo de uma época (82 batches), cada caractere aparece em:

| caractere | batches com o caractere |
|---|---:|
| `' '`, `'a'`, `'x'` | 82 |
| `'?'` | 79 |
| `'É'` | 38 |
| `'W'` | 3 |
| `'+'` | 1 |

O vetor de `'+'` recebe sinal de treino em **um** batch por época. É a contrapartida,
no modelo, da cauda longa observada no corpus (`02-dataset.md`).

**Isso vale para o módulo `Embeddings` sozinho.** No GPT completo, com weight tying
(`tie_weights: true`), a mesma matriz $E$ também é usada pelo LM head, e o gradiente do LM head
chega a todas as linhas em todo passo. Ver [09-lm-head.md](09-lm-head.md), seção 4.4.

**Detalhe sobre o AdamW ([11-training.md](11-training.md)).** Gradiente zero não significa que o
parâmetro fica parado:

- o *weight decay* desacoplado encolhe todos os parâmetros a cada passo, com ou sem
  gradiente (fórmula abaixo);
- os momentos do Adam ($m$, $v$) persistem entre passos, então uma linha continua se
  movendo por alguns passos depois do último gradiente.

Termos: o **AdamW** é o otimizador, a regra que usa os gradientes para atualizar os
parâmetros. **Weight decay** é um encolhimento dos parâmetros em direção a zero, que ajuda a
evitar valores exagerados. Os **momentos** $m$ e $v$ são médias móveis que o Adam guarda do
gradiente e do gradiente ao quadrado; por isso, o passo depende também dos gradientes
antigos.

A parte do weight decay, isolada:

$$
\theta \leftarrow \theta - \eta \lambda \theta
$$

**Lendo a fórmula:**

- $\theta$: um parâmetro, por exemplo um dos 128 números da linha de `'W'`.
- $\leftarrow$: "passa a valer".
- $\eta$: learning rate (0,0003 no `tiny.yaml`).
- $\lambda$: weight decay (0,01 no `tiny.yaml`).
- $\eta \lambda \theta$: uma fração minúscula do próprio $\theta$. Não há gradiente nessa conta,
  por isso ela acontece mesmo quando o token não apareceu no batch.

**Exemplo** (valores do `tiny.yaml`): $\eta \lambda = 0{,}0003 \times 0{,}01 = 0{,}000003$. A cada
passo, $\theta$ é multiplicado por $0{,}999997$. Depois de 820 passos (10 épocas de 82 batches),
o fator acumulado é $0{,}999997^{820} \approx 0{,}9975$: a linha encolhe cerca de 0,25% mesmo sem
receber gradiente nenhum.

O PyTorch oferece `nn.Embedding(sparse=True)` com `SparseAdam` para evitar isso. Não
usamos: com $V = 97$, não há ganho que justifique a complexidade.

---

## 4. Informação de posição

### 4.1 Por que é necessária

Sem posição, a self-attention ([04-attention.md](04-attention.md)) **não distingue ordem**. Seja
$X \in \mathbb{R}^{T \times d}$ a matriz com um token por linha e $R$ uma matriz de
permutação. Como $Q$, $K$ e $V$ são projeções lineares aplicadas linha a linha, e a
softmax é aplicada por linha, vale a conta abaixo.

Uma **matriz de permutação** é a identidade com as linhas trocadas de lugar. Multiplicar $RX$
reordena as linhas de $X$. Por exemplo, com $T = 3$,
$R = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ troca a primeira linha
com a segunda. Desfazer a troca e refazê-la não muda nada: $R^\top R = I$.

A conta mostra o que acontece com a saída da attention quando a ordem dos tokens de entrada é
trocada:

$$
\text{Attn}(RX) = \text{softmax}\!\left(\frac{RQ\,(RK)^\top}{\sqrt{d_k}}\right) RV
= R\,\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) R^\top R\, V
= R\,\text{Attn}(X)
$$

**Lendo a fórmula:**

- $X$: os vetores dos tokens, um por linha, **sem** informação de posição.
- $RX$: os mesmos vetores, em outra ordem.
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$: as queries, keys e values da attention (`04-attention.md`).
  **Nesta fórmula, e só nela, $V$ é a matriz de values, não o tamanho do vocabulário.** Como cada linha
  é projetada separadamente, trocar a ordem das linhas de $X$ troca a ordem das linhas de
  $Q$, $K$ e $V$: viram $RQ$, $RK$ e $RV$.
- $RQ\,(RK)^\top = R\,QK^\top R^\top$: a tabela de scores fica com linhas **e** colunas
  reordenadas.
- $\sqrt{d_k}$: a divisão que mantém os scores numa escala razoável (`04-attention.md`).
- $\text{softmax}$ aplicada por linha: reordenar linhas e colunas antes da softmax dá o mesmo
  que reordenar depois. Por isso o $R$ e o $R^\top$ "saem" da softmax.
- $R^\top R = I$: a identidade some da conta.
- $R\,\text{Attn}(X)$: a saída é a mesma de antes, só com as linhas reordenadas.

Permutar a entrada só permuta a saída. Feed-forward e LayerNorm também operam posição a
posição, então **um Transformer sem posição é equivariante a permutações**: "amor" e
"roma" produziriam os mesmos quatro vetores, só em outra ordem.

**Equivariante a permutações** significa: trocar a ordem da entrada troca a ordem da saída do
mesmo jeito, e nada mais muda.

**Exemplo:** "roma" é "amor" de trás para frente, então $R$ inverte a ordem das 4 linhas. O
vetor de saída do `'a'` é o mesmo nos dois casos: em "amor" ele está na linha 0, e em "roma",
na linha 3. O modelo não teria como saber que as palavras são diferentes.

A causal mask quebra parte dessa simetria, porque a posição $t$ enxerga exatamente $t+1$
tokens, e isso permite ao modelo inferir posição implicitamente. Haviv et al. (2022)
mostram que decoders sem positional encoding ainda aprendem alguma noção de posição. Mesmo
assim, dar a posição explicitamente é o padrão e facilita o aprendizado.

### 4.2 Alternativas

| método | como funciona | usado em |
|---|---|---|
| sinusoidal fixo | vetor determinístico de senos e cossenos por posição, somado ao token | Transformer original (Vaswani et al., 2017) |
| **absoluto aprendido** | **tabela $P \in \mathbb{R}^{T \times d}$ treinada como qualquer parâmetro** | **GPT-1, GPT-2, BERT** |
| RoPE | rotaciona $Q$ e $K$ por um ângulo proporcional à posição, dentro da attention | RoFormer (Su et al., 2021), LLaMA |
| ALiBi | soma aos attention scores uma penalidade proporcional à distância | Press et al. (2022) |

Usamos **absoluto aprendido**, como no GPT-2. É o mais simples: uma segunda tabela de
embedding, indexada por `0, 1, ..., T-1`. Tem duas limitações:

- **não extrapola**: não existe linha para a posição $T$, e sequências maiores que
  `context_length` geram `ValueError`;
- **cada posição é aprendida separadamente**: nada obriga as posições 40 e 41 a terem
  vetores parecidos, e isso precisa emergir do treino.

**Extrapolar** é funcionar em posições que não existiam no treino. A tabela $P$ tem linhas de
0 a 127; a posição 128 simplesmente não tem vetor.

Diferente dos tokens, **toda posição recebe gradiente em todo batch** (128 de 128 no
experimento), porque toda janela tem exatamente $T$ posições.

RoPE é uma possível evolução futura do projeto, fora do escopo implementado aqui.

---

## 5. Soma, não concatenação

A entrada do primeiro bloco é a soma do vetor do token com o vetor da posição:

$$
h^{(0)}_{b,t} = E_{x_{b,t}} + P_t \quad \in \mathbb{R}^d
$$

**Lendo a fórmula:**

- $h^{(0)}_{b,t}$: o vetor da sequência $b$, posição $t$, antes de qualquer bloco. O $(0)$
  quer dizer "depois de zero blocos"; depois do bloco $\ell$ ele se chama $h^{(\ell)}$
  ([architecture.md](architecture.md)).
- $x_{b,t}$: o ID naquela posição.
- $E_{x_{b,t}}$: a linha da tabela de tokens escolhida por esse ID.
- $P_t$: a linha $t$ da tabela de posições. Não depende do token nem da sequência $b$.
- $+$: soma elemento a elemento: os 128 números de um com os 128 do outro.
- $\in \mathbb{R}^d$: o resultado continua sendo um vetor de $d = 128$ números.

**Exemplo** (feito à mão, com $d = 2$): se $E_{53} = (0{,}10;\ 0{,}20)$ e
$P_1 = (0{,}30;\ -0{,}10)$, então $h = (0{,}40;\ 0{,}10)$. No experimento, em `"casa"`, os dois
`'a'` viram $E_{53} + P_1$ e $E_{53} + P_3$: mesmo token, posições diferentes.

Concatenar exigiria dividir $d$ em partes fixas para token e posição. Somar mantém a
dimensão e deixa a rede decidir como usar o espaço. Isso funciona porque, em alta
dimensão, **vetores aleatórios são quase ortogonais**: o cosseno entre dois vetores
gaussianos independentes em $\mathbb{R}^d$ tem média 0 e desvio $\approx 1/\sqrt{d}$.

- **Concatenar:** colocar um vetor ao lado do outro. Por exemplo, 64 números do token seguidos
  de 64 números da posição.
- **Ortogonais:** vetores com produto escalar zero (perpendiculares). "Quase ortogonais" quer
  dizer cosseno perto de zero.
- **Gaussiano:** sorteado de uma distribuição normal.

O tamanho típico do cosseno entre dois vetores sorteados depende só da dimensão:

$$
\text{desvio do cosseno entre vetores aleatórios} \approx \frac{1}{\sqrt{d}}
$$

**Lendo a fórmula:** quanto maior $d$, mais os cossenos se concentram perto de zero, e mais
"espaço" existe para token e posição não se atrapalharem.

**Exemplo:** com $d = 4$, o desvio seria $1/\sqrt{4} = 0{,}5$, e cossenos longe de zero seriam
comuns. Com $d = 128$, é $1/\sqrt{128} \approx 1/11{,}31 \approx 0{,}088$. O experimento mede
0,090:

```text
cosseno entre tokens distintos: média -0.000, desvio 0.090
  referência para vetores aleatórios: desvio ≈ 1/√d = 0.088
```

Token e posição começam em direções praticamente independentes, e as camadas lineares
seguintes conseguem separá-los.

**Quanto a posição muda um vetor?** Considere o mesmo token, de ID $i$, em duas posições, com
vetores de posição $P_1$ e $P_3$. Supondo os três vetores aproximadamente ortogonais e de
mesma norma, a conta abaixo estima o cosseno entre as duas somas:

$$
\cos(E_i + P_1,\; E_i + P_3) = \frac{\lVert E_i \rVert^2 + E_i \cdot P_3 + P_1 \cdot E_i + P_1 \cdot P_3}{\lVert E_i + P_1 \rVert\,\lVert E_i + P_3 \rVert}
\approx \frac{\lVert E_i \rVert^2}{2\lVert E_i \rVert^2} = 0{,}5
$$

**Lendo a fórmula:**

- $\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert}$: produto escalar dividido
  pelos tamanhos ([00-notacao.md](00-notacao.md), seção 5.7).
- Numerador: o produto escalar $(E_i + P_1) \cdot (E_i + P_3)$ se abre em quatro parcelas.
  - $\lVert E_i \rVert^2 = E_i \cdot E_i$: o token com ele mesmo, a parte em comum.
  - $E_i \cdot P_3$, $P_1 \cdot E_i$ e $P_1 \cdot P_3$: termos cruzados, $\approx 0$ porque os
    vetores são quase ortogonais.
- Denominador: cada soma tem norma $\approx \sqrt{2}\,\lVert E_i \rVert$ (dois vetores
  perpendiculares de mesmo tamanho, como os catetos de um triângulo). O produto das duas dá
  $2\lVert E_i \rVert^2$.
- Resultado: metade de cada vetor é "token" (igual nos dois) e metade é "posição" (diferente).

**Exemplo** (feito à mão, com $d = 3$ e vetores exatamente ortogonais): $E_i = (1, 0, 0)$,
$P_1 = (0, 1, 0)$, $P_3 = (0, 0, 1)$. As somas são $(1, 1, 0)$ e $(1, 0, 1)$, com produto
escalar 1 e normas $\sqrt{2}$. Então $\cos = 1 / (\sqrt{2} \cdot \sqrt{2}) = 0{,}5$ exatamente.

O experimento mede o `'a'` de "casa" nas posições 1 e 3:

```text
só token:        cos = 1.000
token + posição: cos = 0.435
```

O valor fica perto de 0,5, com o desvio vindo do ruído dos termos cruzados. Na
inicialização, token e posição pesam igualmente. O treino ajusta as normas de cada tabela
e pode mudar essa proporção.

---

## 6. Inicialização

As duas tabelas são inicializadas com $\mathcal{N}(0,\, 0{,}02^2)$, a convenção do GPT-2.
A norma esperada de cada vetor é $\sigma\sqrt{d} = 0{,}02 \cdot \sqrt{128} \approx 0{,}226$,
que é o valor medido.

$\mathcal{N}(0,\, 0{,}02^2)$ quer dizer: cada número é sorteado de uma curva de sino centrada em
0, com desvio padrão $\sigma = 0{,}02$. Cerca de 95% dos valores ficam entre −0,04 e +0,04.

De onde vem $\sigma\sqrt{d}$: a norma soma os quadrados dos $d$ números do vetor.

$$
\lVert E_i \rVert = \sqrt{\sum_{k=0}^{d-1} E_{i,k}^2} \approx \sqrt{d\,\sigma^2} = \sigma\sqrt{d}
$$

**Lendo a fórmula:**

- $E_{i,k}$: o número $k$ da linha $i$.
- $E_{i,k}^2$: o quadrado. Para um número sorteado com média 0 e desvio $\sigma$, o quadrado
  vale $\sigma^2$ em média.
- $\sum_{k=0}^{d-1}$: somando $d$ quadrados, o total fica perto de $d\,\sigma^2$.
- $\sqrt{d\,\sigma^2} = \sigma\sqrt{d}$: a raiz dá a norma.

**Exemplo:** $0{,}02 \times \sqrt{128} = 0{,}02 \times 11{,}31 \approx 0{,}226$. Com o padrão do
PyTorch ($\sigma = 1$), a norma seria $\sqrt{128} \approx 11{,}3$, 50 vezes maior. O experimento
confirma o valor pequeno:

```text
norma média de um vetor de token: 0.226
  referência: 0.02·√d = 0.226
```

O padrão do `nn.Embedding` é $\mathcal{N}(0, 1)$, com norma $\approx 11{,}3$. O embedding
define a escala inicial do *residual stream* ([architecture.md](architecture.md)), o
vetor que atravessa todos os blocos. Com escala pequena, as contribuições dos blocos não
ficam ofuscadas pela entrada no começo do treino. Há também uma razão ligada ao
[09-lm-head.md](09-lm-head.md): o LM head compartilha pesos com o token embedding (*weight
tying*, `tie_weights: true` no
`tiny.yaml`, como no GPT-2). Vetores grandes gerariam logits grandes, e a loss inicial ficaria
muito acima de $\ln V \approx 4{,}57$.

- **Residual stream:** o vetor de cada posição que passa por todos os blocos; cada bloco soma
  algo a ele.
- **Logits:** os scores, um por token do vocabulário, que o LM head produz no fim (`09-lm-head.md`).
- **Weight tying:** usar a mesma matriz $E$ na entrada (embedding) e na saída (LM head).

---

## 7. Dropout

Durante o treino, `nn.Dropout(p)` zera cada elemento de `tok + pos` com probabilidade
$p$ (0,1 no `tiny.yaml`) e multiplica os restantes por $1/(1-p)$. O fator mantém o valor
esperado igual ao da inferência (*inverted dropout*). Em `model.eval()` o dropout vira a
identidade. O GPT-2 aplica dropout exatamente neste ponto, depois da soma.

- **Dropout:** zerar valores aleatórios durante o treino, para que o modelo não dependa demais
  de nenhum valor isolado. É uma forma de combater o **overfitting** (quando o modelo decora o
  treino e piora em texto novo).
- **Inferência:** usar o modelo já treinado para prever, sem aprender.
- **`model.train()` e `model.eval()`:** os dois modos de um módulo do PyTorch. O dropout só atua
  no modo de treino; no modo de avaliação, vira a identidade (devolve a entrada sem mudar).

A regra aplicada a cada número do vetor:

$$
\text{dropout}(a)_k = \frac{\kappa_k}{1 - p}\, a_k,
\qquad
\kappa_k =
\begin{cases}
0 & \text{com probabilidade } p \\
1 & \text{com probabilidade } 1 - p
\end{cases}
$$

**Lendo a fórmula:**

- $a$: o vetor que entra no dropout, `tok + pos`; $a_k$ é o seu número $k$.
- $\kappa_k$ (capa, de *keep*, "manter"): uma moeda viciada sorteada para cada número. Dá 0
  com probabilidade $p$ e 1 com probabilidade $1 - p$.
- $\frac{1}{1 - p}$: o fator que aumenta os números que sobreviveram, para compensar os que
  foram zerados.
- $p$: a probabilidade de zerar (0,1).

**Exemplo** (feito à mão): $a = (0{,}2;\ -0{,}4;\ 0{,}1;\ 0{,}3)$, $p = 0{,}1$, e o sorteio deu
$\kappa = (1, 0, 1, 1)$. O fator é $1 / 0{,}9 \approx 1{,}111$. O resultado é
$(0{,}222;\ 0;\ 0{,}111;\ 0{,}333)$.

Por que o fator mantém o valor esperado:

$$
\mathbb{E}\big[\text{dropout}(a)_k\big] = (1 - p) \cdot \frac{a_k}{1 - p} + p \cdot 0 = a_k
$$

**Lendo a fórmula:**

- $\mathbb{E}[\cdot]$: a média sobre todos os sorteios possíveis de $\kappa_k$.
- $(1 - p) \cdot \frac{a_k}{1-p}$: com probabilidade $1 - p$, o número é mantido e aumentado.
- $p \cdot 0$: com probabilidade $p$, ele vira zero.
- $= a_k$: em média, o número sai igual ao que entrou. Por isso, na inferência, basta não fazer
  nada.

**Exemplo:** com $a_k = 0{,}2$ e $p = 0{,}1$: $0{,}9 \times 0{,}222 + 0{,}1 \times 0 = 0{,}2$.

---

## 8. Shapes e parâmetros

```text
ids                     [B, T]      int64
token_embedding(ids)    [B, T, d]   float32
position_embedding(0..T-1)  [T, d]  float32   ── broadcasting soma em todo o batch
dropout(tok + pos)      [B, T, d]   float32
```

| tabela | shape | parâmetros |
|---|---|---:|
| `token_embedding` | `[97, 128]` | 12.416 |
| `position_embedding` | `[128, 128]` | 16.384 |
| **total** | | **28.800** |

O total é o número de linhas vezes colunas de cada tabela:

$$
\text{parâmetros} = V d + T d = 97 \cdot 128 + 128 \cdot 128 = 12.416 + 16.384 = 28.800
$$

**Lendo a fórmula:**

- $V d$: a tabela de tokens tem $V$ linhas de $d$ números.
- $T d$: a tabela de posições tem $T$ linhas de $d$ números.
- O dropout não tem parâmetros.

A tabela de posições é **maior** que a de tokens, porque $T = 128 > V = 97$. Com
caracteres, "onde" custa mais parâmetros que "o quê". No GPT-2 é o contrário: o token
embedding tem $50{.}257 \times 768 \approx 38{,}6$ milhões de parâmetros, e o de posição,
$1{.}024 \times 768 \approx 0{,}79$ milhão.

---

## 9. Implementação manual e o módulo `Embeddings`

Sem abstração, tudo cabe em três linhas:

```python
E = torch.randn(V, d) * 0.02          # [V, d]
P = torch.randn(T, d) * 0.02          # [T, d]
h = E[ids] + P[: ids.shape[1]]        # [B, T, d]
```

Linha por linha:

1. `torch.randn(V, d)` sorteia uma matriz $\mathcal{N}(0, 1)$; multiplicar por 0,02 dá
   $\mathcal{N}(0,\, 0{,}02^2)$.
2. O mesmo para a tabela de posições, com $T$ linhas.
3. `E[ids]` usa o tensor de IDs `[B, T]` para escolher linhas: o resultado é `[B, T, d]`.
   `P[: ids.shape[1]]` pega as primeiras `seq_len` linhas de posição. A soma usa broadcasting
   (seção 9.3).

O módulo faz o mesmo, com `nn.Embedding` (para os pesos serem parâmetros treináveis),
validação de tamanho e dropout.

### 9.1 `Embeddings.__init__`: criar as tabelas

**Objetivo:** criar os parâmetros com os tamanhos certos e a escala certa.

Passo a passo (linhas 14 a 25 de `embeddings.py`):

1. Guarda `context_length`, usado depois para validar a entrada.
2. `self.token_embedding = nn.Embedding(vocab_size, d_model)`: tabela $E$, `[V, d]`.
3. `self.position_embedding = nn.Embedding(context_length, d_model)`: tabela $P$, `[T, d]`.
   É o mesmo tipo de camada; só muda o que o índice significa (posição em vez de token).
4. `self.dropout = nn.Dropout(dropout)`: guarda o dropout com probabilidade $p$.
5. `nn.init.normal_(..., std=0.02)` nas duas tabelas: troca a inicialização padrão
   $\mathcal{N}(0, 1)$ por $\mathcal{N}(0,\, 0{,}02^2)$ (seção 6).

### 9.2 `Embeddings.forward`: de IDs a vetores

**Objetivo:** receber `ids: [B, seq_len]` e devolver `[B, seq_len, d]`.

Passo a passo (linhas 27 a 38):

1. `seq_len = ids.shape[1]`: o comprimento real das sequências. Pode ser menor que $T$, o que
   vai ser útil na geração, que começa com um prompt curto.
2. Se `seq_len > context_length`, lança `ValueError`: não existe linha de posição para
   além de $T - 1$.
3. `positions = torch.arange(seq_len, device=ids.device)`: o vetor `[0, 1, ..., seq_len - 1]`,
   criado no mesmo dispositivo (CPU ou GPU) que os IDs.
4. `tok = self.token_embedding(ids)`: `[B, seq_len, d]`. Um ID maior ou igual a $V$ gera
   `IndexError`.
5. `pos = self.position_embedding(positions)`: `[seq_len, d]`.
6. `tok + pos`: `[B, seq_len, d]`, por broadcasting.
7. `self.dropout(...)`: mesmo shape; só zera valores no modo de treino.

**Exemplo** (experimento): o batch real passa pelo módulo assim:

```text
x (32, 128) torch.int64 -> embeddings (32, 128, 128) torch.float32
```

E, para `"casa"`: `ids` `[1, 4]` → `positions = [0, 1, 2, 3]` → `tok` `[1, 4, 128]` e
`pos` `[4, 128]` → saída `[1, 4, 128]`.

### 9.3 Broadcasting

**Broadcasting** é a regra do PyTorch para combinar tensores de shapes diferentes. Quando
falta uma dimensão à esquerda, o tensor menor é tratado como se fosse repetido ao longo dela,
sem copiar a memória.

- `tok` tem shape `[B, seq_len, d]`, e `pos` tem `[seq_len, d]`.
- O PyTorch trata `pos` como `[1, seq_len, d]` e o "estica" ao longo das $B$ sequências.
- Resultado: **o mesmo vetor de posição $P_t$ é somado em todas as sequências do batch**.

**Exemplo:** com $B = 2$, as entradas `[[7, 1], [7, 2]]` têm o token 7 na posição 0 das duas
sequências. As duas saídas na posição 0 são iguais, $E_7 + P_0$ (teste
`test_same_token_and_position_match_across_batch`).

### 9.4 O que os testes garantem

| teste | propriedade |
|---|---|
| `test_output_shape` | `[B, seq_len]` → `[B, seq_len, d]` para `seq_len` de 1 até $T$ |
| `test_sequence_longer_than_context_raises` | `seq_len = T + 1` gera `ValueError` |
| `test_id_outside_vocab_raises` | ID igual a $V$ gera `IndexError` |
| `test_lookup_equals_one_hot_times_matrix` | $E_i = \mathbf{e}_i^\top E$ (seção 2) |
| `test_output_is_token_plus_position` | saída $= E_{x_{b,t}} + P_t$ (seção 5) |
| `test_same_token_differs_by_position` | o mesmo token em posições diferentes dá vetores diferentes |
| `test_same_token_and_position_match_across_batch` | broadcasting: mesma posição, mesmo vetor em todo o batch |
| `test_gradient_reaches_only_used_rows` | gradiente esparso (seção 3) |
| `test_gradient_accumulates_over_repeated_tokens` | tokens repetidos somam gradiente (seção 3) |
| `test_parameter_count` | parâmetros $= Vd + Td$ (seção 8) |
| `test_init_std_is_small` | desvio padrão dos pesos ≈ 0,02 (seção 6) |
| `test_dropout_only_in_train_mode` | dropout só atua em `train()` (seção 7) |

### 9.5 Interface

```python
from model import Embeddings

emb = Embeddings(vocab_size=tok.vocab_size, context_length=128, d_model=128, dropout=0.1)
h = emb(x)        # x: [B, T] int64  ->  h: [B, T, 128] float32
```

---

## Resumo

- Um ID é um rótulo. O one-hot remove a ordem falsa, mas deixa todos os tokens à mesma
  distância e não aprende nada.
- O embedding é uma tabela $E$ `[V, d]`: o vetor do token $i$ é a linha $E_i$, o mesmo que
  $\mathbf{e}_i^\top E$, só que muito mais barato.
- O gradiente de $E$ é esparso: só as linhas dos tokens presentes no batch recebem sinal, e
  repetições somam. Com AdamW, mesmo linhas sem gradiente se movem um pouco.
- Sem posição, a attention é equivariante a permutações. O Mini-GPT usa uma segunda tabela
  aprendida $P$ `[T, d]`, como o GPT-2, que não extrapola além de $T$.
- A entrada do primeiro bloco é $h^{(0)} = E_{x} + P_t$. Somar funciona porque vetores
  aleatórios em 128 dimensões são quase ortogonais.
- Inicialização $\mathcal{N}(0,\, 0{,}02^2)$ dá vetores de norma ~0,226, pequenos para o
  residual stream e para os logits com weight tying.
- Dropout ($p = 0{,}1$) depois da soma, só no treino. Total deste componente: 28.800 parâmetros.

## Para checar o entendimento

1. Mostre que `nn.Embedding` e `nn.Linear(V, d, bias=False)` aplicado a um one-hot
   calculam a mesma coisa. Qual a relação entre os dois pesos?
2. Num batch em que `'W'` não aparece, a linha de `'W'` muda depois de `optimizer.step()`
   com AdamW? E com SGD sem momentum e sem weight decay?
3. Sem positional embedding e sem causal mask, por que a previsão na última posição seria
   idêntica para "pato" e "tapo"? Com causal mask e dois ou mais blocos, isso continua
   valendo?
4. O que acontece se você chamar o módulo com uma sequência de 129 tokens? Como RoPE
   muda isso?
5. Se as tabelas fossem inicializadas com $\mathcal{N}(0, 1)$, qual seria o cosseno
   esperado entre o mesmo token em duas posições? Depende de $\sigma$?
6. Por que o `forward` usa `torch.arange(seq_len)` e não `torch.arange(context_length)`?
