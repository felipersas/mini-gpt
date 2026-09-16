# Transformer Block

> **Como attention e feed-forward trabalham juntas?**

Código: [`model/layer_norm.py`](../model/layer_norm.py),
[`model/transformer_block.py`](../model/transformer_block.py). Testes:
[`tests/test_layer_norm.py`](../tests/test_layer_norm.py),
[`tests/test_transformer_block.py`](../tests/test_transformer_block.py). Experimento:
`uv run python -m experiments.e07_transformer_block` (~30 s, por causa do mini-treino).
Visão geral: [architecture.md](architecture.md). Guia de notação: [00-notacao.md](00-notacao.md).

---

## Para que serve

### O problema

Até aqui existem duas peças prontas, mas soltas:

- a **multi-head attention** ([04](04-attention.md), [05](05-multi-head-attention.md)) traz
  informação de outras posições;
- o **feed-forward** ([06](06-feed-forward.md)) processa a informação de cada posição.

Um modelo útil aplica essas peças **muitas vezes, uma depois da outra**. Empilhar de forma
ingênua (a saída de uma vira a entrada da próxima) dá errado por dois motivos:

1. **O sinal de erro se perde no caminho.** O treino corrige cada camada usando o
   **gradiente**: a derivada da loss em relação àquela camada ([00-notacao.md](00-notacao.md),
   seção 7). Para chegar às camadas iniciais, o gradiente atravessa todas as camadas
   seguintes, e cada uma o transforma. Depois de muitas transformações, ele chega
   embaralhado ou minúsculo.
2. **A escala das entradas muda o tempo todo.** O vetor que cada camada recebe pode ser
   pequeno numa posição e grande em outra, pequeno no começo do treino e grande no fim. Uma
   camada que precisa funcionar bem com qualquer escala aprende mais devagar.

Este documento resolve os dois problemas com duas peças de "cola" e junta tudo numa unidade:

| peça | problema que resolve | ideia em uma frase |
|---|---|---|
| **conexão residual** | sinal de erro se perdendo | cada subcamada **soma** o seu resultado à entrada, em vez de substituí-la |
| **LayerNorm** | escala variável | antes de cada subcamada, cada vetor é padronizado para média 0 e variância 1 |
| **Transformer Block** | organizar as peças | attention e feed-forward, cada uma com a sua LayerNorm e a sua conexão residual, numa unidade que pode ser empilhada |

Termos que aparecem o tempo todo:

- **Subcamada:** cada uma das duas partes de um bloco (a attention ou o feed-forward).
- **Conexão residual** (ou *skip connection*, "conexão de atalho"): em vez de
  $x \leftarrow F(x)$, fazer $x \leftarrow x + F(x)$. A entrada "pula" a subcamada e é somada
  à saída dela.
- **Residual stream** ("fluxo residual"): o vetor de cada posição que atravessa o modelo
  inteiro, dos embeddings até o fim. Os blocos não o substituem; só somam coisas a ele.
- **Normalização:** reescalar números para uma faixa padrão. A **LayerNorm** normaliza cada
  vetor usando as suas próprias $d$ features.
- **Pre-norm:** a LayerNorm fica **antes** da subcamada, no ramo que lê o stream. É o que o
  projeto usa. **Post-norm:** a LayerNorm fica **depois** da soma, no próprio stream (seção 4).

### Uma analogia

Pense no residual stream como um **quadro de avisos** que passa por várias salas:

- em cada sala (subcamada), alguém **lê** o quadro, pensa e **cola um post-it** novo;
- ninguém apaga o que já estava lá: o quadro final é o conteúdo original **mais** todos os
  post-its (seção 3.1);
- a LayerNorm é como **ajustar o brilho da foto do quadro** antes de ler: não importa se o
  quadro está claro ou escuro, quem lê sempre recebe uma imagem com o mesmo contraste. O
  quadro em si não é alterado.

O atalho também vale no caminho de volta: uma correção feita no quadro final chega
**diretamente** a qualquer sala anterior, sem precisar ser reinterpretada por cada sala do
meio (seção 3.2).

### O que entra e o que sai

Valores do [`configs/tiny.yaml`](../configs/tiny.yaml):

| | shape | no Mini-GPT |
|---|---|---|
| **entrada** `x` | `[B, T, d]` | `[32, 128, 128]`: 32 sequências × 128 posições × vetor de 128 números. No primeiro bloco, é a saída dos embeddings (norma média 0,315 por posição, seção 3.5) |
| **saída** | `[B, T, d]` | `[32, 128, 128]`: mesmo shape, com as duas escritas somadas (norma média 0,387 num bloco recém-inicializado) |
| **parâmetros** | | 197.760 por bloco; 791.040 nos $L = 4$ blocos |
| **hiperparâmetros** | | `d_model` = 128, `num_heads` = 4, `d_ff` = 512, `dropout` = 0,1, `num_layers` = 4 (usado só na inicialização: desvio $0{,}02/\sqrt{8} \approx 0{,}0071$) |

Como o shape de saída é igual ao de entrada, a saída de um bloco pode ser a entrada do
próximo. É isso que permite empilhar $L$ blocos.

### Onde fica no pipeline

```text
texto (Dom Casmurro)
  │  tokenizer (01-tokenizer.md)                          [N]
  ▼
batches de IDs (02-dataset.md)                            x, y: [32, 128]
  │
  ▼
embeddings (03-embeddings.md)                             x_0: [32, 128, 128]
  │
  ▼
┌─ bloco Transformer, repetido L = 4 vezes ──────────┐
│  x = x + MHA(LN_1(x))      (attention)             │  [32, 128, 128]   ← este documento
│  x = x + FFN(LN_2(x))      (feed-forward)          │  [32, 128, 128]   ← este documento
└────────────────────────────────────────────────────┘
  │
  ▼
LayerNorm final (08-gpt.md; a classe nasce aqui)          [32, 128, 128]
  │
  ▼
LM head (09-lm-head.md)
  │
  ▼
logits                                                    [32, 128, 97]
  │
  ▼
loss: cross-entropy com y (10-loss.md)                    escalar
```

### O que daria errado sem cada peça

- **Sem conexão residual:** o gradiente precisa atravessar todas as subcamadas, uma por uma.
  Na seção 3.3, com 16 blocos, ele chega às camadas iniciais com cosseno ≈ 0,01 em relação
  ao erro da saída: é ruído. No mini-treino da seção 3.4, a loss **estaciona em ~3,1**,
  enquanto as variantes com residual chegam a ~2,4.
- **Sem LayerNorm:** cada subcamada receberia o stream "cru", cuja escala muda com a posição,
  a profundidade e o treino. A attention é sensível a isso: os scores $q \cdot k$ vêm de duas
  projeções lineares da entrada, então **dobrar a entrada quadruplica os scores**. Com
  entradas pequenas a softmax fica quase uniforme; com entradas grandes, satura
  ([04-attention.md](04-attention.md)). A LayerNorm entrega sempre a mesma escala (norma
  $\approx \sqrt{d} = 11{,}31$), e o comportamento fica previsível.

---

## Notação deste documento

Símbolos gerais (soma, produto, norma, derivada, Jacobiana, variância) estão explicados com
exemplos em [00-notacao.md](00-notacao.md). Aqui está o sentido **neste documento**.

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $B$, $T$, $d$ | "bê", "tê", "dê" | batch, contexto, largura do vetor | 32, 128, 128 | `batch`, `seq_len`, `d_model` |
| $h$ | "agá" | número de heads (só em shapes) | 4 | `num_heads` |
| $d_{ff}$ | "dê éfe éfe" | dimensão interna do feed-forward | 512 | `d_ff` |
| $L$ | "ele" | número de blocos do modelo | 4 no `tiny.yaml`; 16 nas seções 3.3 e 3.4; 1 a 64 na seção 5.2 | `num_layers` |
| $\ell$ | "ele cursivo" | índice do **bloco**, começando em 0 | $\ell = 0, \dots, L-1$ | `for block in blocks`, `inputs[ℓ]` |
| $k$ | "kê" | índice da **subcamada**, de 1 a $2L$. **Sentido local:** em [04-attention.md](04-attention.md), $k$ é a key | 1 a 8 com $L = 4$ | não aparece |
| $x$ | "xis" | **sentido local, depende do contexto:** no bloco, o residual stream inteiro; na LayerNorm, **um** vetor de $d$ números | `[B, T, d]` no bloco; `[d]` na LayerNorm | `x` |
| $x_i$ | "xis í" | a feature $i$ de um vetor $x$ | $i = 1, \dots, d$ | `x[..., i-1]` |
| $x'$ | "xis linha" | o stream depois da attention e antes do feed-forward | `[B, T, d]` | `x` depois da linha 37; `after_attention` no experimento |
| $x_0$ | "xis zero" | o stream inicial: a saída dos embeddings | `[32, 128, 128]`, norma média 0,315 | `emb(x)`, `h` |
| $x_k$ | "xis kê" | o stream depois de $k$ subcamadas | `[B, T, d]` | não aparece |
| $x_\ell$ | "xis ele" | o stream na **entrada do bloco** $\ell$; na contagem por subcamada, é $x_{2\ell}$ | `[B, T, d]` | `inputs[ℓ]` no experimento |
| $x_{\text{final}}$ | "xis final" | o stream depois das $2L$ subcamadas | `[B, T, d]` | saída do último bloco |
| $y$, $y_i$ | "ípsilon" | **sentido local:** a saída da LayerNorm. Em outros docs, $y$ são os alvos | `[d]` | valor de `return` na linha 29 |
| $\text{LN}$, $\text{LN}_1$, $\text{LN}_2$ | "éle ene" | LayerNorm; 1 e 2 são as duas de cada bloco | 256 parâmetros cada | `LayerNorm`, `ln_1`, `ln_2` |
| $\text{MHA}$ | "multi-head attention" | a subcamada de attention | 65.536 parâmetros | `self.attention` |
| $\text{FFN}$ | "feed-forward" | a subcamada de feed-forward | 131.712 parâmetros | `self.feed_forward` |
| $F$ | "éfe" | uma subcamada genérica. No pre-norm, inclui a LayerNorm: $F(x) = \text{MHA}(\text{LN}_1(x))$ ou $\text{FFN}(\text{LN}_2(x))$ | `[B, T, d]` → `[B, T, d]` | `self.attention(self.ln_1(x))` |
| $F_k$ | "éfe kê" | a $k$-ésima subcamada do modelo inteiro | $k = 1, \dots, 2L$ | não aparece |
| $\mu$ | "mi" | média das $d$ features de um vetor | 4,333 no exemplo | `mean` |
| $\sigma^2$ | "sigma ao quadrado" | variância das features (divide por $d$) | 2,222 no exemplo | `var` |
| $\sigma$ | "sigma" | desvio padrão, $\sqrt{\sigma^2}$. **Sentido local:** em [06-feed-forward.md](06-feed-forward.md), $\sigma$ é a ativação | 1,491 no exemplo | `var.sqrt()` |
| $\epsilon$ | "épsilon" | número minúsculo que evita divisão por zero | $10^{-5}$ | `eps` |
| $\hat x$, $\hat x_i$ | "xis chapéu" | o vetor normalizado: média 0, variância 1 | `[d]` | `x_hat` |
| $\gamma$, $\gamma_i$ | "gama" | escala aprendida, uma por feature | começa em 1; `[d]` | `weight` |
| $\beta$, $\beta_i$ | "beta" | deslocamento aprendido, um por feature | começa em 0; `[d]` | `bias` |
| $a$, $c$ | "á", "cê" | **sentido local:** dois números quaisquer na propriedade de invariância, com $a > 0$ | $a = 3$, $c = 10$ no experimento; $3$ e $7$ no teste | `3 * x + 10` |
| $\lVert x \rVert$ | "norma de xis" | tamanho de um vetor | $\sqrt{128} = 11{,}31$ depois da LayerNorm | `x.norm(dim=-1)` |
| $\mathbb{R}^d$ | "erre dê" | vetores de $d$ números reais | $d = 128$ | shape `(d,)` |
| $I$ | "í" | matriz identidade: 1 na diagonal, 0 fora. $I\,v = v$ | `[d, d]` | o `x +` do forward |
| $J_k$ | "jota kê" | Jacobiana da subcamada $k$: quanto cada saída de $F_k$ muda com cada entrada | `[d, d]` por posição | calculada pelo autograd |
| $\delta_{ij}$ | "delta í jota" | 1 se $i = j$, senão 0 | | `torch.eye(d)` |
| $\sum$, $\prod$ | "somatório", "produtório" | soma e produto de uma sequência de termos | | `sum`, laço |
| $\mathcal{L}$ | "éle cursivo" | a loss (um número) | na seção 3.3: $\sum \text{saída} \cdot r$ | `loss`, `(h * readout).sum()` |
| $\dfrac{\partial \mathcal{L}}{\partial x}$ | "d parcial da loss em relação a xis" | o gradiente: quanto a loss muda com cada número de $x$; mesmo shape de $x$ | `[B, T, d]` | `x.grad` |
| $r$ | "erre" | **readout** ("leitura"): tensor aleatório fixo usado para montar uma loss artificial (seção 3.3) | `[8, 32, 64]`, norma 127,8 | `readout` |
| $g$ | "gê" | **sentido local:** um gradiente, visto como vetor | | `inputs[i].grad.flatten()` |
| $n$ | "ene" | **sentido local:** quantidade de números num gradiente achatado | $8 \cdot 32 \cdot 64 = 16.384$ | `.flatten()` |
| $\cos$ | "cosseno" | similaridade de direção entre dois vetores, de −1 a 1 | 0,83 a 0,99 com residual | `F.cosine_similarity` |
| $\text{Var}$ | "variância" | variância de uma quantidade aleatória | | `.var()`, `.std() ** 2` |
| $s$, $s^2$ | "ésse" | **sentido local:** desvio e variância de **uma** escrita no stream. Em [02-dataset.md](02-dataset.md), $s$ é o stride | $s \approx 0{,}02$ na seção 5.2 | não aparece |
| $\mathcal{N}(0, \sigma^2)$ | "normal de média 0 e variância sigma ao quadrado" | sorteio de pesos em torno de 0 | $\mathcal{N}(0,\ 0{,}02^2)$ | `nn.init.normal_(w, std=...)` |
| $p$ | "pê" | probabilidade de dropout | 0,1 | `dropout`, `nn.Dropout(p)` |
| $m_i$ | "ême í" | **sentido local:** máscara do dropout, 0 ou 1 por número | | sorteada dentro de `nn.Dropout` |
| $\text{RMS}(x)$ | "érre ême ésse" | raiz da média dos quadrados (RMSNorm, seção 2.5) | | não implementado |
| $\propto$ | "proporcional a" | cresce ou diminui na mesma razão | | |
| $\leftarrow$ | "passa a valer" | atribuição, como `=` no código | | `x = x + ...` |
| $\approx$, $\Rightarrow$ | "aproximadamente", "implica" | | | |

**Duas contagens do stream.** $x_k$ conta **subcamadas** (seções 3.1 e 3.2) e $x_\ell$ conta
**blocos** (seção 3.3). Como cada bloco tem duas subcamadas, a entrada do bloco $\ell$ é o
stream depois de $2\ell$ subcamadas. As duas contagens coincidem em $x_0$, os embeddings.

---

## 0. Onde estamos

Todas as peças existem:

| peça | papel | doc |
|---|---|---|
| multi-head attention | **comunicação**: traz informação de outras posições | [04-attention.md](04-attention.md), [05-multi-head-attention.md](05-multi-head-attention.md) |
| feed-forward | **computação**: processa a informação de cada posição | [06-feed-forward.md](06-feed-forward.md) |

Este documento acrescenta a "cola" que permite empilhar essas peças muitas vezes: **LayerNorm**
e **conexões residuais**. O resultado é o **Transformer Block**, a unidade que o GPT repete
$L$ vezes ([08-gpt.md](08-gpt.md)).

---

## 1. A arquitetura

O bloco faz duas atualizações do stream, uma por subcamada. Cada uma lê uma versão
normalizada do stream e soma o que calculou:

$$
\begin{aligned}
x' &= x + \text{MHA}\big(\text{LN}_1(x)\big) \\
\text{saída} &= x' + \text{FFN}\big(\text{LN}_2(x')\big)
\end{aligned}
$$

**Lendo a fórmula:**

- $x$: o residual stream que entra no bloco, `[B, T, d]`.
- $\text{LN}_1(x)$: uma cópia de $x$ com cada vetor normalizado. O $x$ original não muda.
- $\text{MHA}(\dots)$: o que a attention calcula a partir dessa cópia. É a **escrita** da
  attention, também `[B, T, d]`.
- $x + \text{MHA}(\dots)$: soma número a número. Esse é o $x'$, o stream depois da primeira
  escrita.
- A segunda linha repete o padrão com a segunda LayerNorm ($\text{LN}_2$) e o feed-forward
  ($\text{FFN}$), agora lendo $x'$.
- Os índices 1 e 2 só distinguem as duas LayerNorms: cada uma tem os seus próprios $\gamma$ e
  $\beta$.

**Exemplo com os números da seção 3.5** (norma média por posição, bloco recém-inicializado):
$\lVert x \rVert = 0{,}315$; a attention escreve 0,054; $x'$ fica com 0,321; o feed-forward
escreve 0,217; a saída fica com 0,387. As normas **não** se somam diretamente
($0{,}315 + 0{,}054 \neq 0{,}321$) porque os vetores apontam em direções quase
perpendiculares: $\sqrt{0{,}315^2 + 0{,}054^2} \approx 0{,}320$ e
$\sqrt{0{,}321^2 + 0{,}217^2} \approx 0{,}387$.

```text
x ─────────────┬──────────────────────────────┐
               │                              │
               ▼                              │
           LayerNorm                          │
               ▼                              │
     Multi-Head Attention                     │
               ▼                              │
               + ◄────────────────────────────┘
               │
x' ────────────┬──────────────────────────────┐
               │                              │
               ▼                              │
           LayerNorm                          │
               ▼                              │
          Feed-Forward                        │
               ▼                              │
               + ◄────────────────────────────┘
               │
             saída
```

A melhor forma de ler o bloco é pelo **residual stream** ([architecture.md](architecture.md)):

- a linha reta à esquerda é o stream; ele atravessa o bloco sem ser modificado;
- cada subcamada **lê** o stream (através de uma LayerNorm), calcula algo e **escreve**
  somando o resultado de volta;
- a attention escreve informação vinda de outras posições; o feed-forward escreve o
  resultado de processar a posição atual.

O shape nunca muda: `[B, T, d]` entra, `[B, T, d]` sai. Por isso é possível empilhar blocos.

---

## 2. LayerNorm

### 2.1 O problema

A escala do residual stream não é fixa. Na seção 3 do experimento, os embeddings entram com
norma média 0,315 por posição, e cada bloco soma novas contribuições. A escala varia entre
posições, entre camadas e ao longo do treino. Se cada subcamada recebesse o stream "cru",
teria que funcionar bem com entradas de tamanhos muito diferentes. A LayerNorm entrega a
cada subcamada uma entrada com escala padronizada.

**Objetivo da LayerNorm (Ba et al., 2016):** transformar **cada vetor**, sozinho, num vetor com
média 0 e variância 1, e depois deixar o modelo reajustar a escala de cada feature com
parâmetros aprendidos.

### 2.2 A fórmula

Para um vetor $x \in \mathbb{R}^d$ (**uma** posição):

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2, \qquad
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_i = \gamma_i\, \hat x_i + \beta_i
$$

São quatro passos. Vamos um de cada vez, com o vetor da seção 1 do experimento:
$x = (2, 4, 4, 4, 5, 7)$, $d = 6$.

**Passo 1: a média.** Calcula o "centro" do vetor, para depois removê-lo.

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$

**Lendo a fórmula:**

- $x_i$: a feature $i$ do vetor; $i$ vai de 1 até $d$.
- $\sum_{i=1}^{d} x_i$: soma das $d$ features.
- $\frac{1}{d}$: divide pela quantidade. O resultado é um número só.

Exemplo: $\mu = (2 + 4 + 4 + 4 + 5 + 7)/6 = 26/6 = 4{,}333$.

**Passo 2: a variância.** Mede o quanto as features se espalham em torno da média, para
depois dividir por esse espalhamento.

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2
$$

**Lendo a fórmula:**

- $x_i - \mu$: a distância da feature $i$ até a média (pode ser negativa).
- $(\cdot)^2$: eleva ao quadrado, para que distâncias negativas e positivas contem igual.
- $\frac{1}{d}\sum$: média desses quadrados. O resultado é a variância $\sigma^2$; a raiz,
  $\sigma$, é o desvio padrão.

Exemplo: as distâncias são $(-2{,}333;\ -0{,}333;\ -0{,}333;\ -0{,}333;\ 0{,}667;\ 2{,}667)$. Os
quadrados somam $5{,}444 + 3 \cdot 0{,}111 + 0{,}444 + 7{,}111 = 13{,}333$. Dividindo por 6:
$\sigma^2 = 2{,}222$ e $\sigma = 1{,}491$.

Cuidado: `x.var()` do PyTorch divide por $d - 1$ e daria $13{,}333/5 = 2{,}667$. A LayerNorm
divide por $d$ (é o que a linha 26 do código faz).

**Passo 3: normalizar.** Centraliza (subtrai a média) e reescala (divide pelo desvio). O
resultado tem média 0 e variância 1, qualquer que fosse a escala de $x$.

$$
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**Lendo a fórmula:**

- $\hat x_i$ ("xis chapéu í"): a feature $i$ normalizada.
- $x_i - \mu$: move o vetor para ter média 0.
- $\sqrt{\sigma^2 + \epsilon}$: praticamente o desvio $\sigma$. O $\epsilon = 10^{-5}$ só importa
  quando $\sigma^2$ é zero ou muito pequeno: impede a divisão por zero.

Exemplo: $\hat x_1 = (2 - 4{,}333)/1{,}491 = -1{,}565$.

**Passo 4: escala e deslocamento aprendidos.** Devolve ao modelo a liberdade de escolher a
escala e o centro de cada feature.

$$
y_i = \gamma_i\, \hat x_i + \beta_i
$$

**Lendo a fórmula:**

- $\gamma_i$ ("gama í"): multiplica a feature $i$. Se $\gamma_i = 2$, a feature fica com
  desvio 2.
- $\beta_i$ ("beta í"): soma à feature $i$. Se $\beta_i = 1$, a feature fica centrada em 1.
- $y_i$: a saída final da LayerNorm na feature $i$.

Seção 1 do experimento, com $d = 6$:

```text
x                          = +2.000 +4.000 +4.000 +4.000 +5.000 +7.000
média = 4.333   variância = 2.222   desvio = 1.491
x̂ = (x − média) / √(var + eps) = -1.565 -0.224 -0.224 -0.224 +0.447 +1.789
    média de x̂ = -0.000, variância de x̂ = 1.000
γ                          = +1.000 +1.000 +2.000 +2.000 +0.500 +0.500
β                          = +0.000 +0.000 +0.000 +1.000 +1.000 +1.000
y = γ · x̂ + β              = -1.565 -0.224 -0.447 +0.553 +1.224 +1.894

LayerNorm do projeto == nn.LayerNorm: True
LayerNorm(3·x + 10) == LayerNorm(x):  True
```

Confira a primeira coordenada: $(2 - 4{,}333)/1{,}491 = -1{,}565$. Com $\gamma_1 = 1$ e
$\beta_1 = 0$, $y_1 = -1{,}565$. Na quarta: $\hat x_4 = -0{,}224$, $y_4 = 2 \cdot (-0{,}224) + 1 = 0{,}553$.

Mais duas, para praticar: na terceira, $\gamma_3 = 2$ e $\beta_3 = 0$, então
$y_3 = 2 \cdot (-0{,}224) = -0{,}447$. Na sexta, $\hat x_6 = 2{,}667/1{,}491 = 1{,}789$ e
$y_6 = 0{,}5 \cdot 1{,}789 + 1 = 1{,}894$.

### 2.3 Por que $\gamma$ e $\beta$ são aprendidos

A normalização **remove** a média e a escala de cada vetor. Às vezes essa informação é útil.
$\gamma$ (escala) e $\beta$ (deslocamento), um por feature, deixam o modelo reintroduzir a
escala e o deslocamento que forem convenientes, dimensão por dimensão. Começam em
$\gamma = 1$ e $\beta = 0$: no início, a LayerNorm só normaliza.

São $2d$ parâmetros por LayerNorm: 256 com $d = 128$.

"Aprendidos" significa: são `nn.Parameter`, recebem gradiente no `loss.backward()` e o
otimizador os atualiza a cada passo, como qualquer peso.

### 2.4 Propriedades

- **Invariância a escala e deslocamento.** $\text{LN}(a x + c) = \text{LN}(x)$ para qualquer
  $a > 0$ e $c$ escalar: média e desvio mudam juntos, e a razão fica igual.
- **A saída tem norma $\approx \sqrt{d}$** (com $\gamma = 1$, $\beta = 0$): um vetor com média
  0 e variância 1 em $d$ coordenadas tem $\|\hat x\|^2 = d$. Na seção 3 do experimento,
  `ln_1(x)` tem norma 11,24, com $\sqrt{128} = 11{,}31$.
- **Um vetor por vez.** As estatísticas são de cada posição, sobre as suas $d$ features. Não
  há mistura entre posições nem entre sequências do batch: a LayerNorm é causal e funciona
  igual com batch 1.
- **$\epsilon$ evita divisão por zero** quando todas as features são iguais ($\sigma = 0$). Nesse
  caso a saída é só $\beta$ (`test_constant_vector_does_not_produce_nan`).
- **Consequência para o gradiente.** Como a saída não depende da escala da entrada, a
  derivada é inversamente proporcional a ela: $\partial\, \text{LN}(x)/\partial x \propto 1/\sigma_x$.
  Entradas pequenas recebem gradientes grandes. Isso aparece na seção 3.3.

Cada propriedade, em detalhe:

#### Invariância a escala e deslocamento

Multiplicar o vetor por $a$ e somar $c$ não muda a saída da LayerNorm.

$$
\text{LN}(a x + c) = \text{LN}(x), \qquad a > 0
$$

**Lendo a fórmula:**

- $a x + c$: cada feature é multiplicada por $a$ e somada a $c$ (o mesmo $c$ para todas).
- A média vira $a\mu + c$. Subtraindo: $(a x_i + c) - (a\mu + c) = a(x_i - \mu)$. O $c$ some.
- O desvio vira $a\sigma$. Dividindo: $a(x_i - \mu)/(a\sigma) = (x_i - \mu)/\sigma$. O $a$ some.
- $a > 0$ é necessário: com $a$ negativo, o sinal de todas as features se inverteria.

Exemplo: $3x + 10 = (16, 22, 22, 22, 25, 31)$ tem $\mu = 23$ e $\sigma = 4{,}472 = 3 \cdot 1{,}491$.
A primeira feature normalizada é $(16 - 23)/4{,}472 = -1{,}565$, igual à de $x$. O experimento
confirma: `LayerNorm(3·x + 10) == LayerNorm(x):  True`. A igualdade é aproximada só por causa
do $\epsilon$.

#### A norma da saída é $\approx \sqrt{d}$

$$
\lVert \hat x \rVert^2 = \sum_{i=1}^{d} \hat x_i^2 = d \cdot \underbrace{\frac{1}{d}\sum_{i=1}^{d} \hat x_i^2}_{\text{variância de } \hat x \,=\, 1} = d
$$

**Lendo a fórmula:**

- $\lVert \hat x \rVert^2$: a norma ao quadrado é a soma dos quadrados das features.
- Multiplicar e dividir por $d$ não muda nada, mas deixa aparecer a média dos quadrados.
- Como $\hat x$ tem média 0, a média dos quadrados **é** a variância, que vale 1.
- Logo $\lVert \hat x \rVert = \sqrt{d}$.

Exemplo: no vetor de 6 features, $\lVert \hat x \rVert = 2{,}449 = \sqrt{6}$.

**Por que 11,24 e não 11,31?** Por causa do $\epsilon$. Os embeddings recém-inicializados são
pequenos: variância média de $\approx 0{,}00077$ por posição. Nesse caso
$\epsilon = 0{,}00001$ já é ~1,3% da variância, e a divisão por $\sqrt{\sigma^2 + \epsilon}$ fica
um pouco maior que por $\sigma$. A norma prevista, $\sqrt{d \cdot \sigma^2/(\sigma^2 + \epsilon)}$,
calculada no mesmo batch, dá exatamente 11,24.

#### Um vetor por vez

A média e a variância da posição $t$ usam só as $d$ features da posição $t$. Mudar a posição 4
não altera a saída das outras (`test_each_position_is_normalized_independently`, e a linha
`LayerNorm` da seção 2.5).

#### O papel do $\epsilon$

Com $x = (2{,}5;\ 2{,}5;\ \dots;\ 2{,}5)$: $\mu = 2{,}5$, $\sigma^2 = 0$ e $x_i - \mu = 0$. Sem
$\epsilon$, a conta seria $0/0$ (`nan`). Com $\epsilon$: $0/\sqrt{10^{-5}} = 0$, e
$y_i = \gamma_i \cdot 0 + \beta_i = \beta_i$.

#### Por que $\partial\,\text{LN}/\partial x \propto 1/\sigma$

Esta propriedade diz como a LayerNorm altera o **tamanho do gradiente** que passa por ela. Ela
sai direto da invariância.

$$
\text{LN}(a x) = \text{LN}(x) \quad\Rightarrow\quad J_{\text{LN}}(a x) \cdot a = J_{\text{LN}}(x) \quad\Rightarrow\quad J_{\text{LN}}(a x) = \frac{1}{a}\, J_{\text{LN}}(x)
$$

**Lendo a fórmula:**

- $J_{\text{LN}}(x)$: a Jacobiana da LayerNorm no ponto $x$, `[d, d]`. A entrada $(i, j)$ diz
  quanto $y_i$ muda quando $x_j$ muda um pouquinho.
- Primeira seta: deriva os dois lados em relação a $x$. No lado esquerdo, pela regra da cadeia,
  aparece a Jacobiana calculada em $a x$ vezes a derivada de $a x$, que é $a$.
- Segunda seta: divide por $a$.
- Conclusão: se a entrada fica $a$ vezes **menor** (por exemplo $a = 0{,}03$), a Jacobiana fica
  $1/a$ vezes **maior** (33 vezes). Como o desvio de $a x$ é $a\sigma$, isso é o mesmo que
  dizer $J_{\text{LN}} \propto 1/\sigma$.

A fórmula explícita (com $\gamma = 1$) confirma o $1/\sigma$ na frente:

$$
\frac{\partial \hat x_i}{\partial x_j} = \frac{1}{\sigma}\Big(\delta_{ij} - \frac{1}{d} - \frac{\hat x_i\, \hat x_j}{d}\Big)
$$

**Lendo a fórmula:**

- $\delta_{ij}$: 1 se $i = j$, senão 0. É a parte "mexer em $x_i$ muda $\hat x_i$".
- $-\frac{1}{d}$: mexer em qualquer $x_j$ muda a média, e isso afeta todas as features.
- $-\frac{\hat x_i \hat x_j}{d}$: mexer em $x_j$ também muda o desvio, e isso reescala todas.
- $\frac{1}{\sigma}$: tudo é dividido pelo desvio (na prática, $\sqrt{\sigma^2 + \epsilon}$).

Exemplo calculado com o autograd (Python, fora do experimento): com o vetor
$x = (2, 4, 4, 4, 5, 7)$ e um gradiente de saída fixo, o gradiente na entrada tem norma 2,57.
Com a entrada $0{,}03\,x$, a norma vira 85,6: 33 vezes maior, como previsto por $1/0{,}03$.

### 2.5 Por que LayerNorm e não BatchNorm

BatchNorm, comum em redes de imagem, normaliza **cada feature ao longo do batch**. Aplicada a
`[B, T, d]`, as estatísticas misturam todas as posições e todas as sequências. Seção 2 do
experimento, mudando só as posições 4 e 5 da entrada:

```text
maior |diferença| na saída de cada posição (entradas diferentes só em 4 e 5):
  LayerNorm  t0=0.00  t1=0.00  t2=0.00  t3=0.00  t4=2.97  t5=3.01
  BatchNorm  t0=1.15  t1=1.71  t2=2.18  t3=1.33  t4=2.18  t5=3.86
```

Com BatchNorm, **as posições 0 a 3 mudam por causa do futuro**: a causalidade quebra. Além
disso, a saída de uma sequência passaria a depender das outras do batch, e o comportamento
no treino (estatísticas do batch) seria diferente da inferência (médias acumuladas). A
LayerNorm (Ba et al., 2016) não tem nenhum desses problemas.

**A diferença numa fórmula.** As duas calculam média e variância; muda **sobre o quê**:

$$
\text{LayerNorm: } \mu_{b,t} = \frac{1}{d}\sum_{j=1}^{d} x_{b,t,j}
\qquad\qquad
\text{BatchNorm: } \mu_j = \frac{1}{B\,T}\sum_{b}\sum_{t} x_{b,t,j}
$$

**Lendo a fórmula:**

- $x_{b,t,j}$: a feature $j$ da posição $t$ da sequência $b$.
- LayerNorm: **uma média por vetor** (fixa $b$ e $t$, percorre as features $j$). O resultado
  tem shape `[B, T, 1]`.
- BatchNorm: **uma média por feature** (fixa $j$, percorre todas as sequências $b$ e todas as
  posições $t$). O resultado tem shape `[d]`. A média da feature $j$ inclui a posição 5, e é
  usada para normalizar a posição 0. Por isso o futuro "vaza".

No experimento, o tensor `[1, 6, 8]` vira `[6, 8]` (`t.view(-1, d_model)`): cada uma das 6
posições é tratada como uma "amostra", e cada uma das 8 features é normalizada com a média das
6 posições.

**Alternativa moderna:** RMSNorm (Zhang & Sennrich, 2019) remove o passo de centralizar e
divide só pela raiz da média dos quadrados. É mais barata e usada no LLaMA — uma extensão
possível, ainda não implementada aqui.

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}, \qquad y_i = \gamma_i\, \frac{x_i}{\sqrt{\text{RMS}(x)^2 + \epsilon}}
$$

**Lendo a fórmula:**

- $\text{RMS}(x)$ (*root mean square*): raiz da média dos quadrados. Parecido com o desvio,
  mas sem subtrair a média.
- Não há $\mu$ nem $\beta$: a saída não é centralizada.
- Exemplo (calculado em Python): $x = (2, 4, 4, 4, 5, 7)$ tem média dos quadrados
  $126/6 = 21$ e $\text{RMS} = 4{,}583$. Com $\gamma = 1$:
  $y \approx (0{,}436;\ 0{,}873;\ 0{,}873;\ 0{,}873;\ 1{,}091;\ 1{,}528)$, que **não** tem média 0.

### 2.6 O código, linha a linha

[`model/layer_norm.py`](../model/layer_norm.py) implementa as quatro etapas à mão.

**`__init__` (linhas 13 a 17): cria os parâmetros.**

```python
def __init__(self, d_model: int, eps: float = 1e-5) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(d_model))  # γ: começa sem alterar a escala
    self.bias = nn.Parameter(torch.zeros(d_model))  # β: começa sem deslocar
```

- `super().__init__()`: inicializa o `nn.Module`, que passa a registrar parâmetros e
  submódulos.
- `self.eps`: o $\epsilon$. É um número comum, não um parâmetro: não é treinado.
- `self.weight`: o $\gamma$, `[d_model]`, cheio de 1. `nn.Parameter` marca o tensor como
  treinável.
- `self.bias`: o $\beta$, `[d_model]`, cheio de 0.
- Os nomes `weight` e `bias` são os mesmos de `nn.LayerNorm`. Por isso o teste consegue copiar
  os valores de uma para a outra e comparar.

**`extra_repr` (linhas 19 a 21): como o módulo aparece em `print`.**

Devolve `"128, eps=1e-05"`. Com isso, `print(LayerNorm(128))` mostra
`LayerNorm(128, eps=1e-05)`, igual ao `nn.LayerNorm`. Não afeta nenhuma conta.

**`forward` (linhas 23 a 29): a fórmula.**

| linha | código | fórmula | shape (bloco do `tiny.yaml`) |
|---|---|---|---|
| 25 | `mean = x.mean(dim=-1, keepdim=True)` | $\mu$ | `[32, 128, 128]` → `[32, 128, 1]` |
| 26 | `var = (x - mean).pow(2).mean(dim=-1, keepdim=True)` | $\sigma^2$, dividindo por $d$ | `[32, 128, 1]` |
| 28 | `x_hat = (x - mean) / torch.sqrt(var + self.eps)` | $\hat x$ | `[32, 128, 128]` |
| 29 | `return self.weight * x_hat + self.bias` | $y = \gamma \hat x + \beta$ | `[32, 128, 128]` |

- `dim=-1`: a média é tirada ao longo da **última** dimensão, as $d$ features. Uma média por
  vetor, nunca pelo batch ou pela sequência.
- `keepdim=True`: mantém a dimensão com tamanho 1 (`[32, 128, 1]`). Assim `x - mean` funciona
  por broadcasting: a mesma média é subtraída das 128 features daquele vetor.
- `x: [..., d_model]`: os `...` significam "qualquer número de dimensões antes". A mesma
  classe normaliza um vetor `[6]` (seção 1 do experimento) ou um batch `[32, 128, 128]`.
- Linha 29: `self.weight` tem shape `[128]` e é multiplicado por broadcasting em todas as
  posições. O mesmo $\gamma$ vale para todas as posições e sequências.

---

## 3. Conexões residuais

### 3.1 O stream é a soma de todas as escritas

Cada subcamada faz $x \leftarrow x + F(x)$. Desenrolando as $2L$ subcamadas do modelo:

$$
x_{\text{final}} = x_0 + \sum_{k=1}^{2L} F_k(x_{k-1})
$$

A saída é **o embedding original mais a soma de tudo que cada subcamada escreveu**. Toda
subcamada lê a soma de tudo que veio antes. Essa ideia vem das ResNets (He et al., 2016),
criadas justamente para treinar redes profundas.

**Lendo a fórmula:**

- $x_0$: o stream inicial (os embeddings).
- $F_k(x_{k-1})$: a escrita da subcamada $k$, calculada a partir do stream **como estava
  antes dela**, $x_{k-1}$.
- $\sum_{k=1}^{2L}$: soma as $2L$ escritas. Com $L = 4$, são 8 escritas (4 da attention e 4 do
  feed-forward), alternadas.
- $x_{\text{final}}$: o stream que sai do último bloco e vai para a LayerNorm final.

**Desenrolando passo a passo.** Cada linha substitui a anterior:

$$
\begin{aligned}
x_1 &= x_0 + F_1(x_0) \\
x_2 &= x_1 + F_2(x_1) = x_0 + F_1(x_0) + F_2(x_1) \\
x_3 &= x_2 + F_3(x_2) = x_0 + F_1(x_0) + F_2(x_1) + F_3(x_2) \\
&\ \ \vdots
\end{aligned}
$$

A cada subcamada, a expressão ganha **um termo novo** e nenhum termo antigo some. Depois de
$2L$ subcamadas, sobra a fórmula acima. $x_2$ é a saída do primeiro bloco ($F_1$ = attention,
$F_2$ = feed-forward).

### 3.2 O caminho do gradiente

Para treinar, o gradiente precisa voltar da loss até cada subcamada. A pergunta é o que
acontece com ele ao atravessar uma subcamada residual.

A derivada de uma subcamada residual é

$$
\frac{\partial x_k}{\partial x_{k-1}} = I + J_k, \qquad J_k = \frac{\partial F_k}{\partial x_{k-1}}
$$

**Lendo a fórmula:**

- $\frac{\partial x_k}{\partial x_{k-1}}$: quanto o stream depois da subcamada $k$ muda quando o
  stream antes dela muda. É uma matriz `[d, d]` por posição.
- $x_k = x_{k-1} + F_k(x_{k-1})$ tem duas parcelas. Derivando a primeira, $x_{k-1}$, em
  relação a si mesma: a **identidade** $I$ (a matriz que não muda nada). Derivando a
  segunda: a Jacobiana $J_k$ da subcamada.
- A soma vira soma de derivadas: $I + J_k$.

Pela regra da cadeia, o gradiente na entrada é

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_{\text{final}}} \prod_{k} (I + J_k)
= \frac{\partial \mathcal{L}}{\partial x_{\text{final}}} \Big(I + \sum_k J_k + \dots\Big)
$$

**Lendo a fórmula:**

- $\frac{\partial \mathcal{L}}{\partial x_{\text{final}}}$: o **sinal de erro** na saída: em
  que direção a saída deveria mudar para diminuir a loss.
- $\prod_k (I + J_k)$: a regra da cadeia multiplica as derivadas de todas as subcamadas. A
  ordem é da última para a primeira: $(I + J_{2L}) \cdots (I + J_2)(I + J_1)$.
- $I + \sum_k J_k + \dots$: o produto expandido. Os "$\dots$" são os termos com duas ou mais
  Jacobianas multiplicadas.

**O caso de 2 subcamadas, expandido.** Chamando $g = \partial\mathcal{L}/\partial x_2$:

$$
\frac{\partial \mathcal{L}}{\partial x_0} = g\,(I + J_2)(I + J_1) = g\,\big(I + J_1 + J_2 + J_2 J_1\big)
= \underbrace{g}_{\text{atalho}} + \underbrace{g J_1}_{\text{só por } F_1} + \underbrace{g J_2}_{\text{só por } F_2} + \underbrace{g J_2 J_1}_{\text{por } F_2 \text{ e } F_1}
$$

Cada termo é um **caminho** que o gradiente pode seguir. Com $2L$ subcamadas, são $2^{2L}$
caminhos (256 com $L = 4$). Um deles não passa por nenhuma subcamada: o termo $g\,I = g$.

**Exemplo com números** (fingindo que as Jacobianas são números, $J_1 = 0{,}5$ e
$J_2 = -0{,}3$):

- com residual: $(1 - 0{,}3)(1 + 0{,}5) = 1 + 0{,}5 - 0{,}3 - 0{,}15 = 1{,}05$. O "1" do atalho
  garante que o sinal chega;
- sem residual: $J_2 J_1 = -0{,}15$. O sinal encolheu e **trocou de sinal**;
- sem residual e com 32 fatores de 0,5: $0{,}5^{32} \approx 2{,}3 \cdot 10^{-10}$. O gradiente
  praticamente some. Esse é o **gradiente que desaparece** (*vanishing gradient*): um produto
  de muitos fatores menores que 1 tende a zero (e de muitos fatores maiores que 1, a
  infinito).

Com matrizes, além do tamanho, a **direção** também se perde: cada $J_k$ gira o vetor. É o que
a seção 3.3 mede.

O termo $I$ é um **atalho**: o sinal de erro da saída chega a qualquer camada **sem passar por
nenhuma subcamada**. O teste `test_residual_path_passes_gradient_unchanged` isola esse termo:
com as subcamadas silenciadas, o gradiente na entrada é exatamente igual ao da saída.

**Sem residual**, cada passo é só $J_k$, e o gradiente vira o produto
$\prod_k J_k$ de 32 Jacobianas. Nada garante que esse produto preserve a direção ou o
tamanho do sinal.

(As 32 Jacobianas são as dos 16 blocos do experimento da seção 3.3: $2 \times 16 = 32$
subcamadas.)

### 3.3 Medindo na inicialização

Seção 4 do experimento: 16 blocos com $d = 64$ e uma loss $\mathcal{L} = \sum \text{saída} \cdot r$,
com $r$ aleatório, de modo que $\partial \mathcal{L}/\partial\,\text{saída} = r$. Três variantes:
**pre-norm** (o nosso bloco), **post-norm** (seção 4) e **sem residual**
($x \leftarrow F(\text{LN}(x))$).

**Por que uma loss artificial.** Para medir como o gradiente viaja, é conveniente saber
exatamente qual é o gradiente na saída. A loss acima faz isso.

$$
\mathcal{L} = \sum_{i} \text{saída}_i \cdot r_i \quad\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial\, \text{saída}_i} = r_i
$$

**Lendo a fórmula:**

- $i$ percorre todos os $8 \times 32 \times 64 = 16.384$ números da saída.
- $r$: o **readout**, um tensor aleatório fixo `[8, 32, 64]`. "Lê" a saída multiplicando e
  somando.
- Derivar $\text{saída}_i \cdot r_i$ em relação a $\text{saída}_i$ dá $r_i$. Então o gradiente
  na saída é o próprio $r$.
- Exemplo: saída $= (2, -1)$, $r = (0{,}5;\ 3)$: $\mathcal{L} = 1 - 3 = -2$ e
  $\partial\mathcal{L}/\partial\,\text{saída} = (0{,}5;\ 3)$.

No código: `(h * readout).sum().backward()`. `h.retain_grad()` guarda o gradiente na entrada
de cada bloco, $\partial\mathcal{L}/\partial x_\ell$, que normalmente seria descartado.

**Primeira tentativa: a norma do gradiente.**

```text
norma ‖∂loss/∂x_ℓ‖, com x_ℓ = entrada do bloco ℓ:
                          ℓ=0      ℓ=1      ℓ=2      ℓ=4      ℓ=8     ℓ=12     ℓ=15
  pre-norm (projeto)    155.0    153.0    150.8    147.0    140.1    133.5    128.9
  post-norm            4280.1    125.7    125.7    125.7    125.7    125.7    125.7
  sem residual           24.2    206.5    192.7    184.4    170.9    104.9     47.2
```

Como ler: a coluna $\ell = 15$ é a entrada do último bloco (o gradiente atravessou só 1 bloco);
$\ell = 0$ é a entrada do primeiro (atravessou os 16). Para comparação, a norma de $r$, o
gradiente na saída, é 127,8 (calculado em Python com a mesma semente).

O pre-norm é limpo: a norma é quase a mesma nos 16 blocos. Mas **as outras linhas enganam**.
O salto do post-norm em $\ell = 0$ não tem nada a ver com residuais: a entrada tem escala
0,03, e a primeira LayerNorm multiplica o gradiente por $\approx 1/0{,}03$ (seção 2.4). Sem
residual, as LayerNorms reescalam o sinal a cada bloco, e a norma sobe e desce sem padrão.
**A norma sozinha não mostra se o gradiente carrega informação útil.**

Por que a norma engana, em resumo:

- **A LayerNorm muda a norma sem mudar a informação.** Ela multiplica o gradiente por
  $1/\sigma$ da sua entrada. Uma norma grande pode ser só "volume alto".
- **Uma norma grande pode ser ruído puro.** Sem residual, $\ell = 1$ tem norma 206,5, a maior da
  tabela, e cosseno 0,010 (tabela abaixo): muito sinal, nenhuma relação com o erro.
- **O tamanho do passo é ajustado depois.** Learning rate e otimizador (AdamW,
  [11-training.md](11-training.md)) reescalam os passos. A **direção** é o que diz ao parâmetro
  para onde ir.

**Segunda tentativa: a direção do gradiente.** O cosseno entre $\partial\mathcal{L}/\partial x_\ell$
e $\partial\mathcal{L}/\partial\,\text{saída}$ não depende de escala. Vale 1 se o sinal de erro chega
à camada $\ell$ apontando para onde apontava na saída.

$$
\cos = \frac{g \cdot r}{\lVert g \rVert\, \lVert r \rVert}, \qquad g = \frac{\partial \mathcal{L}}{\partial x_\ell} \text{ achatado em } n = 16.384 \text{ números}
$$

**Lendo a fórmula:**

- $g$: o gradiente na entrada do bloco $\ell$, transformado num vetor único
  (`.flatten()`).
- $g \cdot r$: produto escalar; grande quando $g$ e $r$ apontam para o mesmo lado.
- $\lVert g \rVert\, \lVert r \rVert$: divide pelos tamanhos. Assim, **só a direção importa**.
- Valores: 1 = mesma direção; 0 = sem relação (perpendiculares); −1 = direção oposta.
- Exemplo: se $g = 2r$ ou $g = 0{,}001\,r$, o cosseno é 1 nos dois casos. A norma mudou 2.000
  vezes; a direção, nada.

**O que o cosseno mede aqui.** Se o gradiente viajasse só pelo atalho $I$, chegaria idêntico a
$r$ e o cosseno seria 1. As Jacobianas $J_k$ acrescentam componentes em outras direções, e o
cosseno cai. Um cosseno menor que 1 não é ruim por si só (as subcamadas transformam o sinal
de propósito). Um cosseno **≈ 0** é ruim: significa que o gradiente não tem relação
reconhecível com o erro da saída.

**Qual é o cosseno do ruído?** Dois vetores aleatórios com $n$ números têm cosseno em torno de
0, com desvio de $1/\sqrt{n}$. Com $n = 16.384$: $1/\sqrt{16.384} = 1/128 \approx 0{,}008$.
Valores entre −0,02 e +0,02 são indistinguíveis de ruído.

```text
direção: cosseno entre ∂loss/∂x_ℓ e ∂loss/∂saída (1 = o sinal de erro chega intacto):
                          ℓ=0      ℓ=1      ℓ=2      ℓ=4      ℓ=8     ℓ=12     ℓ=15
  pre-norm (projeto)   +0.832   +0.846   +0.857   +0.879   +0.921   +0.961   +0.990
  post-norm            +0.979   +0.983   +0.984   +0.984   +0.984   +0.984   +0.984
  sem residual         +0.017   +0.010   +0.001   +0.017   +0.008   +0.002   +0.008
```

Agora a diferença é inequívoca:

- **com residual** (pre ou post), o sinal de erro atravessa os 16 blocos preservando a
  direção (cosseno 0,83 a 0,99), graças ao termo $I$;
- **sem residual**, já no último bloco o cosseno é $\approx 0$: o gradiente que chega às
  camadas iniciais é praticamente **ruído** em relação ao erro da saída. Cada camada recebe
  uma instrução embaralhada sobre como mudar.

Os valores sem residual (0,001 a 0,017) estão exatamente na faixa do ruído, $\approx 0{,}008$.

**Detalhe (opcional): de onde vêm 125,7 e 0,984 no post-norm.** A última operação do bloco
post-norm é uma LayerNorm. Pela fórmula explícita da seção 2.4, a Jacobiana dela (com
$\sigma \approx 1$) remove de cada vetor duas direções: a do "tudo igual" (termo $1/d$) e a do
próprio $\hat x$ (termo $\hat x_i \hat x_j / d$). De 64 direções, sobram 62. Um vetor aleatório
perde, em média, $2/64$ da norma ao quadrado:
$\sqrt{62/64} \approx 0{,}984$, e $0{,}984 \times 127{,}8 \approx 125{,}8$. Como as escritas são
pequenas, as LayerNorms seguintes removem quase as mesmas duas direções, e os valores ficam
constantes até $\ell = 1$. Em $\ell = 0$, a entrada tem desvio 0,03, e o $1/\sigma$ multiplica a
norma por ~33: $4.280{,}1/125{,}7 \approx 34$.

### 3.4 Medindo no treino

Seção 5 do experimento: cada variante vira um pequeno modelo de linguagem (embeddings → 16
blocos → LayerNorm → `Linear(d, V)`), treinado por 150 passos no *Dom Casmurro*. É uma prévia
do GPT completo e do training loop ([08-gpt.md](08-gpt.md) a [11-training.md](11-training.md)).

```text
loss de um chute uniforme: ln 97 = 4.57

                         passo 0   passo 25   passo 50  passo 100  passo 150
  pre-norm (projeto)       4.775      2.893      2.652      2.468      2.430
  post-norm                4.768      2.822      2.599      2.419      2.406
  sem residual             4.835      3.548      3.162      3.123      3.160
```

- As três começam perto de $\ln 97$, como previsto em [02-dataset.md](02-dataset.md).
- **Sem residual, o treino estaciona em ~3,1** a partir do passo 50. Com residual, a loss
  continua caindo e chega a ~2,4 — a mesma comparação, em miniatura.
- **Pre-norm e post-norm ficam praticamente empatados** neste cenário. A seção 4 explica por
  que isso não contradiz a escolha do pre-norm.

Detalhes da montagem (do código do experimento): batch `[16, 64]`, AdamW com lr $10^{-3}$,
dropout 0, e as três variantes têm a mesma LayerNorm final antes do `Linear`. A única
diferença entre elas é o `forward` do bloco.

### 3.5 Na inicialização, o bloco muda pouco?

É comum dizer que um bloco recém-inicializado é "quase a identidade". A seção 3 do
experimento mostra que isso depende da escala do stream:

```text
  etapa                     shape            norma média por posição
  x (residual stream)       (32, 128, 128)                     0.315
  ln_1(x)                   (32, 128, 128)                    11.240
  attention(ln_1(x))        (32, 128, 128)                     0.054
  x + attention             (32, 128, 128)                     0.321
  ln_2(x + attention)       (32, 128, 128)                    11.243
  feed_forward(ln_2(...))   (32, 128, 128)                     0.217
  saída do bloco            (32, 128, 128)                     0.387

  ‖saída − x‖ / ‖x‖ = 0.707
```

- As escritas são **pequenas em valor absoluto** (0,054 e 0,217), comparadas com o que as
  subcamadas leem (norma ~11).
- Mas os embeddings também são pequenos (0,315), então **em termos relativos** o primeiro
  bloco altera bastante o stream.
- A attention escreve bem menos que o feed-forward porque, na inicialização, faz uma média
  quase uniforme das posições anteriores ([04-attention.md](04-attention.md), seção 5), e
  médias de vetores aleatórios encolhem.

**Lendo a última linha.** $\lVert \text{saída} - x \rVert / \lVert x \rVert$ é o tamanho da
mudança dividido pelo tamanho do stream. $\text{saída} - x$ é a soma das duas escritas, e
$\sqrt{0{,}054^2 + 0{,}217^2} \approx 0{,}224$; dividindo por 0,315, dá $\approx 0{,}71$. O bloco
mudou o stream em ~71% do seu tamanho: longe de "quase a identidade". "Identidade" é a função
que devolve a entrada sem mudança; o bloco só seria a identidade se as escritas fossem zero
(`test_block_is_identity_when_sublayers_write_nothing`).

---

## 4. Pre-norm × post-norm

A questão é **onde colocar a LayerNorm**: no ramo que lê o stream (antes da subcamada) ou no
próprio stream (depois da soma).

| | fórmula | usado em |
|---|---|---|
| **post-norm** | $x \leftarrow \text{LN}\big(x + F(x)\big)$ | Transformer original (Vaswani et al., 2017) |
| **pre-norm** | $x \leftarrow x + F\big(\text{LN}(x)\big)$ | GPT-2 (Radford et al., 2019) e quase todos os LLMs atuais |

**Lendo as fórmulas:**

- Aqui $F$ é **só** a subcamada (MHA ou FFN), sem LayerNorm.
- **Post-norm:** soma primeiro ($x + F(x)$), normaliza depois. O resultado normalizado
  **substitui** o stream.
- **Pre-norm:** normaliza uma **cópia** para a subcamada ler ($F(\text{LN}(x))$) e soma ao
  stream original, que nunca é normalizado.

**A diferença estrutural.** No post-norm, o próprio stream passa por uma LayerNorm depois de
cada subcamada: o atalho $I$ é "interrompido" por normalizações. No pre-norm, a LayerNorm
fica no **ramo** que lê o stream, e o caminho do embedding até a saída é uma soma pura, sem
nenhuma operação no meio.

Em termos de derivada: no pre-norm, cada passo é $I + J_k$ (seção 3.2). No post-norm, é
$J_{\text{LN}}\,(I + J_k)$: toda Jacobiana de LayerNorm multiplica o atalho, e com ela vem o
fator $1/\sigma$ da seção 2.4.

Duas consequências do pre-norm:

- o stream nunca é normalizado, então **é preciso uma LayerNorm final** antes do LM head
  ([08-gpt.md](08-gpt.md));
- a norma do stream cresce com a profundidade, e as escritas das camadas finais ficam
  relativamente menores.

**Por que o pre-norm é o padrão.** Xiong et al. (2020) mostraram que, no post-norm, os
gradientes dos parâmetros perto da saída são grandes na inicialização. Por isso ele precisa de
*learning rate warmup* (começar com lr pequeno e aumentar aos poucos) para não divergir, e
fica mais sensível ao lr e à profundidade. O pre-norm treina de forma estável sem warmup.

(**Warmup**, "aquecimento": nos primeiros passos do treino, a learning rate sobe gradualmente de
quase zero até o valor escolhido. **Divergir**: a loss explodir em vez de cair.)

**Por que o nosso experimento não mostra isso.** Com 16 blocos de $d = 64$, lr $10^{-3}$ e 150
passos, o post-norm treinou tão bem quanto o pre-norm. Os problemas descritos aparecem com
modelos mais profundos e largos, lr maiores e treinos longos. Um experimento pequeno não
reproduzir um efeito de escala não o desmente: mostra só que ele não é relevante nesta
escala. Seguimos o GPT-2 com pre-norm.

### 4.1 As variantes do experimento, em código

As três variantes das seções 3.3 e 3.4 herdam de `TransformerBlock`. Por isso têm **os mesmos
módulos e a mesma inicialização** (inclusive a escala por profundidade da seção 5). Só o
`forward` muda:

```python
class PostNormBlock(TransformerBlock):
    def forward(self, x):
        x = self.ln_1(x + self.attention(x))           # soma, depois normaliza o stream
        return self.ln_2(x + self.feed_forward(x))

class NoResidualBlock(TransformerBlock):
    def forward(self, x):
        x = self.attention(self.ln_1(x))               # sem "x +": a saída substitui x
        return self.feed_forward(self.ln_2(x))
```

| variante | linha da attention | o que a subcamada lê | o que acontece com o stream |
|---|---|---|---|
| pre-norm (`TransformerBlock`) | `x = x + self.attention(self.ln_1(x))` | cópia normalizada | recebe a escrita somada |
| post-norm (`PostNormBlock`) | `x = self.ln_1(x + self.attention(x))` | o stream cru | recebe a escrita e é normalizado |
| sem residual (`NoResidualBlock`) | `x = self.attention(self.ln_1(x))` | cópia normalizada | é **substituído** pela saída da subcamada |

Repare que o post-norm aplica a attention direto em `x`, sem LayerNorm antes. No primeiro bloco,
isso significa ler os embeddings na escala 0,03 do experimento.

---

## 5. Inicialização por profundidade

**Objetivo:** fazer com que, na inicialização, o total que os blocos somam ao stream tenha a
mesma escala, seja o modelo raso ou profundo. **Inicialização por profundidade** é escolher o
desvio inicial de alguns pesos em função de $L$.

### 5.1 O argumento

O stream final é $x_0 + \sum_{k=1}^{2L} F_k$. Se as $2L$ escritas fossem independentes, cada uma
com variância $s^2$:

$$
\text{Var}\Big(\sum_{k=1}^{2L} F_k\Big) = 2L\, s^2 \quad\Rightarrow\quad \text{desvio} = s\sqrt{2L}
$$

**Quanto mais profundo o modelo, maior a soma**, só por acumulação. A última matriz de cada
subcamada (`out_proj` na attention, `down_proj` no feed-forward) é linear na escrita: dividir
o desvio dos seus pesos por $\sqrt{2L}$ divide o desvio da escrita por $\sqrt{2L}$, e a soma
passa a ter desvio $s$, **independente de $L$**.

O GPT-2 faz exatamente isso: essas duas matrizes começam com desvio $0{,}02/\sqrt{2L}$.

**Lendo a fórmula, passo a passo:**

1. $F_k$: a escrita da subcamada $k$, vista como uma quantidade aleatória (os pesos foram
   sorteados).
2. $s^2$: a variância de **uma** escrita; $s$ é o desvio.
3. **Propriedade usada:** a variância de uma soma de quantidades **independentes** é a soma
   das variâncias. Exemplo: duas escritas com desvio 0,02 têm variância $0{,}0004$ cada; a soma
   tem variância $0{,}0008$ e desvio $\sqrt{0{,}0008} \approx 0{,}0283$.
4. Com $2L$ escritas iguais: $\text{Var} = s^2 + s^2 + \dots = 2L\, s^2$.
5. $\text{desvio} = \sqrt{2L\, s^2} = s\sqrt{2L}$. Quadruplicar $L$ dobra o desvio.

"Independentes" é uma aproximação (cada subcamada lê o stream que as anteriores escreveram).
A seção 5.2 mostra que ela funciona bem.

**Por que dividir os pesos por $\sqrt{2L}$ resolve.** Uma escrita é `out_proj` aplicada a
alguma entrada $u$: cada número da escrita é $\sum_j W_{ij} u_j$. Se todos os $W_{ij}$ ficam
$c$ vezes menores, a escrita fica $c$ vezes menor, e a variância, $c^2$ vezes menor. Com
$c = 1/\sqrt{2L}$:

$$
\text{Var}\Big(\sum_{k=1}^{2L} F_k\Big) = 2L \cdot \frac{s^2}{2L} = s^2
$$

**Lendo a fórmula:** cada escrita passa a ter variância $s^2/(2L)$; somando $2L$ delas, o $2L$
cancela, e a variância total volta a $s^2$, qualquer que seja $L$.

Exemplo com $L = 4$ ($2L = 8$ escritas) e $s = 0{,}02$:

- sem escala: $\text{Var} = 8 \cdot 0{,}0004 = 0{,}0032$, desvio $\approx 0{,}0566$;
- com escala: cada escrita tem desvio $0{,}02/\sqrt{8} \approx 0{,}0071$ e variância $0{,}00005$;
  somando 8: $0{,}0004$, desvio $0{,}02$.

`down_proj` tem bias, mas ele começa em zero, então a escrita continua proporcional aos pesos.

### 5.2 A verificação

Seção 6 do experimento: desvio de $(\text{saída} - \text{entrada})$ depois de $L$ blocos:

```text
   L |  W_O e down_proj std 0,02 |  std 0,02/√(2L)
   1 |                    0.0282 |          0.0197
   4 |                    0.0546 |          0.0198
  16 |                    0.1117 |          0.0199
  64 |                    0.2236 |          0.0200
```

Sem a escala, multiplicar $L$ por 4 dobra o desvio (0,028 → 0,055 → 0,112 → 0,224):
**crescimento exato com $\sqrt{L}$**, como a conta prevê. Com a escala, o desvio fica em 0,020
para 1 ou 64 blocos.

- $W_O$ é o nome matemático de `out_proj` ([05-multi-head-attention.md](05-multi-head-attention.md)).
- As razões entre linhas consecutivas da coluna sem escala são 1,94; 2,05 e 2,00: muito perto
  de $\sqrt{4} = 2$.
- Comparando com a fórmula $s\sqrt{2L}$, usando $s \approx 0{,}02$ (compatível com a linha
  $L = 1$: $0{,}02\sqrt{2} = 0{,}0283$): a previsão é 0,0283; 0,0566; 0,1131; 0,2263. O medido é
  0,0282; 0,0546; 0,1117; 0,2236.
- A entrada `x` desse experimento é aleatória com variância 1 por coordenada, e o dropout é 0.

### 5.3 Implementação

O bloco recebe `num_layers` (o total de blocos do modelo) só para isso:

```python
residual_std = 0.02 / math.sqrt(2 * num_layers)
nn.init.normal_(self.attention.out_proj.weight, std=residual_std)
nn.init.normal_(self.feed_forward.down_proj.weight, std=residual_std)
```

Com $L = 4$: $0{,}02/\sqrt{8} \approx 0{,}0071$.

Linha a linha ([`model/transformer_block.py`](../model/transformer_block.py), linhas 31 a 33):

- **Linha 31:** calcula o desvio. É $0{,}02/\sqrt{2L}$, com $L$ = `num_layers`.
- **Linhas 32 e 33:** `nn.init.normal_` **sorteia de novo** os pesos, in-place (o `_` no fim do
  nome indica isso no PyTorch), com média 0 e desvio `residual_std`.
- É uma **reinicialização**: `MultiHeadAttention` e `FeedForward` já tinham sorteado esses
  pesos com desvio 0,02 nos seus próprios `__init__`. O bloco sobrescreve só essas duas
  matrizes.
- Continuam com 0,02: `q_proj`, `k_proj`, `v_proj` e `up_proj`. O bias de `down_proj`
  continua 0. O teste `test_residual_projections_use_depth_scaled_init` confere as duas coisas
  (com $L = 8$: $0{,}02/\sqrt{16} = 0{,}005$).
- **Por que só essas duas:** são as matrizes que **escrevem** no stream. As outras produzem
  valores internos da subcamada.
- `num_layers` é o total do modelo, não o índice do bloco: todos os blocos usam o mesmo desvio.

---

## 6. Dropout

**Dropout** é uma regularização (técnica para reduzir overfitting): durante o treino, zera
números aleatórios, com probabilidade $p$, e aumenta os que sobraram para compensar. O modelo
não pode depender de nenhum caminho específico.

$$
\text{dropout}(x)_i = \frac{m_i\, x_i}{1 - p}, \qquad m_i = \begin{cases} 0 & \text{com probabilidade } p \\ 1 & \text{com probabilidade } 1 - p \end{cases}
$$

**Lendo a fórmula:**

- $p$: probabilidade de zerar cada número. No `tiny.yaml`, $p = 0{,}1$.
- $m_i$: a máscara, sorteada de novo a cada chamada: 0 (zera) ou 1 (mantém).
- $\frac{1}{1-p}$: os números mantidos são multiplicados por $1/0{,}9 \approx 1{,}111$. Assim a
  média do resultado continua igual à de $x$.
- Fora do treino (`model.eval()`), o dropout não faz nada: devolve $x$.

Exemplo: $x = (0{,}2;\ -0{,}5;\ 0{,}1;\ 0{,}4)$ com máscara $(1, 0, 1, 1)$ e $p = 0{,}1$ dá
$(0{,}222;\ 0;\ 0{,}111;\ 0{,}444)$.

Três pontos de dropout em cada bloco, os mesmos do GPT-2:

| onde | nome no GPT-2 | efeito |
|---|---|---|
| pesos da attention `[B, h, T, T]` | `attn_pdrop` | corta ligações aleatórias entre posições |
| saída da attention, antes da soma | `resid_pdrop` | regulariza o que a attention escreve no stream |
| saída do feed-forward, antes da soma | `resid_pdrop` | regulariza o que o feed-forward escreve no stream |

- **`attn_pdrop`** (*attention probability of dropout*): a probabilidade de dropout nos pesos da
  attention, depois da softmax. Zerar o peso $w_{ij}$ faz a posição $i$ ignorar a posição $j$
  naquele passo. As linhas deixam de somar exatamente 1 (somam 1 só em média).
- **`resid_pdrop`** (*residual probability of dropout*): a probabilidade de dropout no que vai
  ser somado ao residual stream.
- No GPT-2 são dois hiperparâmetros separados. No Mini-GPT, os três pontos usam o **mesmo**
  valor, `dropout` da config (0,1).

Onde está no código:

| ponto | arquivo | linha de código |
|---|---|---|
| pesos da attention | `model/attention.py`, `MultiHeadAttention.forward` | `weights = self.dropout(weights)` |
| saída da attention | `model/attention.py`, `MultiHeadAttention.forward` | `return self.dropout(out)` |
| saída do feed-forward | `model/feed_forward.py`, `FeedForward.forward` | `return self.dropout(self.down_proj(hidden))` |

Na attention, o mesmo módulo `self.dropout` é usado nos dois pontos (é seguro: `nn.Dropout`
não guarda estado, sorteia uma máscara nova a cada chamada).

**Ajuste feito aqui:** a `MultiHeadAttention` ([05-multi-head-attention.md](05-multi-head-attention.md))
aplicava dropout só nos pesos. Agora aplica também na saída, simétrico ao `FeedForward`. Os
testes desse documento usam dropout 0 e não mudaram.

O stream em si **nunca** sofre dropout: só as escritas. Assim o atalho $I$ continua íntegro
durante o treino.

(Isso vale dentro do bloco. Antes do primeiro bloco, os embeddings
([03-embeddings.md](03-embeddings.md)) têm o seu próprio dropout.)

---

## 7. Shapes e parâmetros

```text
x                       [B, T, d]
ln_1(x)                 [B, T, d]
attention(ln_1(x))      [B, T, d]
x + attention           [B, T, d]
ln_2(...)               [B, T, d]
feed_forward(ln_2(...)) [B, T, d]
saída                   [B, T, d]
```

No `tiny.yaml`, todas essas linhas são `[32, 128, 128]`. Dentro das subcamadas há shapes
diferentes (`[32, 4, 128, 128]` nos pesos da attention, `[32, 128, 512]` no meio do
feed-forward), mas elas sempre voltam a `[B, T, d]` antes da soma.

Seção 7 do experimento:

```text
  ln_1               256  ( 0.1%)
  attention       65,536  (33.1%)
  ln_2               256  ( 0.1%)
  feed_forward   131,712  (66.6%)
  total          197,760  (12·d² + 9·d = 197,760)
  4 blocos:      791,040
```

$$
\underbrace{4d^2}_{\text{attention}} + \underbrace{8d^2 + 5d}_{\text{feed-forward}} + \underbrace{4d}_{\text{2 LayerNorms}} = 12d^2 + 9d
$$

**Lendo a fórmula:**

- $4d^2$: quatro matrizes `[d, d]` sem bias na attention ($W_Q$, $W_K$, $W_V$, $W_O$).
  $4 \cdot 128^2 = 65.536$.
- $8d^2 + 5d$: `up_proj` `[4d, d]` com bias $4d$, e `down_proj` `[d, 4d]` com bias $d$:
  $4d^2 + 4d + 4d^2 + d$. $8 \cdot 128^2 + 5 \cdot 128 = 131.712$.
- $4d$: duas LayerNorms, cada uma com $\gamma$ e $\beta$ de $d$ números. $4 \cdot 128 = 512$.
- Total: $12 \cdot 128^2 + 9 \cdot 128 = 196.608 + 1.152 = 197.760$.

---

## 8. Da matemática ao código

Sem abstração:

```python
x = x + attention(layer_norm_1(x))
x = x + feed_forward(layer_norm_2(x))
```

A LayerNorm, escrita à mão:

```python
mean = x.mean(dim=-1, keepdim=True)                    # [..., 1]
var = (x - mean).pow(2).mean(dim=-1, keepdim=True)     # [..., 1]
x_hat = (x - mean) / torch.sqrt(var + eps)             # média 0, variância ~1
y = weight * x_hat + bias                              # γ, β
```

O módulo:

```python
from model import TransformerBlock

block = TransformerBlock(
    d_model=128, num_heads=4, d_ff=512, dropout=0.1, num_layers=4
)
out = block(h)    # h: [B, T, 128]  ->  out: [B, T, 128]
```

A LayerNorm do projeto usa os mesmos nomes de parâmetros de `nn.LayerNorm` (`weight`, `bias`),
e o teste `test_matches_pytorch_layer_norm` garante que o resultado é igual.

### 8.1 `TransformerBlock`, linha a linha

[`model/transformer_block.py`](../model/transformer_block.py):

**`__init__`, linhas 20 a 27: cria as quatro peças, na ordem em que são usadas.**

| linha | código | o que é | parâmetros ($d = 128$) |
|---|---|---|---|
| 24 | `self.ln_1 = LayerNorm(d_model)` | $\text{LN}_1$, antes da attention | 256 |
| 25 | `self.attention = MultiHeadAttention(d_model, num_heads, dropout)` | $\text{MHA}$ ([05-multi-head-attention.md](05-multi-head-attention.md)) | 65.536 |
| 26 | `self.ln_2 = LayerNorm(d_model)` | $\text{LN}_2$, antes do feed-forward | 256 |
| 27 | `self.feed_forward = FeedForward(d_model, d_ff, dropout)` | $\text{FFN}$ ([06-feed-forward.md](06-feed-forward.md)) | 131.712 |

As linhas 31 a 33 fazem a reinicialização por profundidade (seção 5.3). `num_layers` não é
guardado: só serve para calcular `residual_std`.

Como o bloco aparece em `print(block)` (saída real, `tiny.yaml`, gerada em Python):

```text
TransformerBlock(
  (ln_1): LayerNorm(128, eps=1e-05)
  (attention): MultiHeadAttention(
    (q_proj): Linear(in_features=128, out_features=128, bias=False)
    (k_proj): Linear(in_features=128, out_features=128, bias=False)
    (v_proj): Linear(in_features=128, out_features=128, bias=False)
    (out_proj): Linear(in_features=128, out_features=128, bias=False)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (ln_2): LayerNorm(128, eps=1e-05)
  (feed_forward): FeedForward(
    (up_proj): Linear(in_features=128, out_features=512, bias=True)
    (activation): GELU(approximate='none')
    (down_proj): Linear(in_features=512, out_features=128, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
)
```

A linha `LayerNorm(128, eps=1e-05)` vem do `extra_repr` da seção 2.6.

**`forward`, linhas 35 a 39: as duas linhas da fórmula.**

```python
x = x + self.attention(self.ln_1(x))  # comunicação entre posições
x = x + self.feed_forward(self.ln_2(x))  # computação em cada posição
return x  # [batch, seq_len, d_model]
```

- **Linha 37:** `self.ln_1(x)` cria uma cópia normalizada; `self.attention(...)` calcula a
  escrita; `x + ...` cria um **tensor novo** com a soma e o nome `x` passa a apontar para ele.
  Esse novo `x` é o $x'$ da seção 1. O tensor original não é alterado (não há operação
  in-place), e o autograd consegue derivar pelos dois caminhos: o atalho e a subcamada.
- **Linha 38:** mesma estrutura, com $\text{LN}_2$ e o feed-forward, lendo $x'$.
- **Linha 39:** devolve o stream atualizado, com o mesmo shape da entrada.

---

## 9. O que cada teste garante

**LayerNorm** ([`tests/test_layer_norm.py`](../tests/test_layer_norm.py)):

| teste | propriedade |
|---|---|
| `test_matches_pytorch_layer_norm` | igual a `nn.LayerNorm`, com $\gamma$ e $\beta$ quaisquer |
| `test_normalized_vectors_have_zero_mean_and_unit_variance` | $\hat x$ tem média 0 e variância 1 |
| `test_invariant_to_scaling_and_shifting_the_input` | $\text{LN}(3x + 7) = \text{LN}(x)$ |
| `test_each_position_is_normalized_independently` | não mistura posições |
| `test_constant_vector_does_not_produce_nan` | $\epsilon$ protege a divisão; a saída é $\beta$ |
| `test_initial_parameters` | $\gamma = 1$, $\beta = 0$, $2d$ parâmetros |

Os testes usam $d = 16$. A fixture `ln` sorteia $\gamma$ e $\beta$ aleatórios, para que as
comparações não passem só porque $\gamma = 1$ e $\beta = 0$.

**Transformer Block** ([`tests/test_transformer_block.py`](../tests/test_transformer_block.py)):

| teste | propriedade |
|---|---|
| `test_output_shape` | `[B, T, d]` → `[B, T, d]` |
| `test_matches_manual_composition` | igual às duas linhas da fórmula |
| `test_block_is_identity_when_sublayers_write_nothing` | sem escritas, o bloco é a identidade |
| `test_residual_path_passes_gradient_unchanged` | o atalho $I$ leva o gradiente sem alteração |
| `test_future_tokens_do_not_affect_past` | causalidade |
| `test_gradient_only_reaches_current_and_past_inputs` | Jacobiana triangular inferior |
| `test_stacked_blocks_keep_shape_and_causality` | 3 blocos empilhados continuam causais |
| `test_residual_projections_use_depth_scaled_init` | `out_proj` e `down_proj` com $0{,}02/\sqrt{2L}$; o resto com 0,02 |
| `test_parameter_count` | $12d^2 + 9d$ |
| `test_gradients_reach_all_parameters` | todos os parâmetros recebem gradiente |
| `test_dropout_only_in_train_mode` | dropout ativo só em `train()` |

Como os dois testes do atalho funcionam: a função auxiliar `silence_sublayers` zera
`out_proj.weight` e `down_proj.weight`. `out_proj` não tem bias e o bias de `down_proj` começa
em 0, então as duas escritas passam a ser exatamente zero, e as Jacobianas $J_k$ também. Sobra
só o atalho:

- no forward, $\text{saída} = x + 0 + 0 = x$ (identidade);
- no backward, com a loss `block(x).sum()`, o gradiente na saída é um tensor de 1, e
  $(I + 0)$ o entrega intacto: `x.grad` é um tensor de 1.

---

## Resumo

1. O bloco é `x = x + MHA(LN(x))` seguido de `x = x + FFN(LN(x))`: cada subcamada **lê** o
   residual stream através de uma LayerNorm e **escreve** somando o resultado.
2. A LayerNorm normaliza **cada vetor** para média 0 e variância 1 e aplica $\gamma$ e $\beta$
   aprendidos. Não mistura posições, por isso é causal; a BatchNorm não seria.
3. A conexão residual cria o atalho $I$ no gradiente. Com ele, o sinal de erro atravessa 16
   blocos com a direção preservada (cosseno 0,83 a 0,99); sem ele, chega como ruído (≈ 0,01),
   e o treino estaciona.
4. Pre-norm deixa o caminho residual livre de normalizações; é o padrão por treinar de forma
   estável em escala, embora aqui tenha empatado com o post-norm.
5. `out_proj` e `down_proj` começam com desvio $0{,}02/\sqrt{2L}$, para que a soma das $2L$
   escritas não cresça com a profundidade.
6. Para medir se o gradiente carrega informação, olhe a **direção** (cosseno), não a norma: a
   LayerNorm multiplica o gradiente por $1/\sigma$ e muda a norma sem mudar a informação.
7. Dropout atua em três pontos (pesos da attention e as duas escritas), nunca no stream, e só
   em modo `train()`.

---

## Para checar o entendimento

1. Calcule à mão a LayerNorm de $x = (1, 2, 3)$ com $\gamma = 1$, $\beta = 0$ e $\epsilon$
   desprezível. Qual a norma do resultado, comparada com $\sqrt{3}$?
2. Por que $\text{LN}(5x) = \text{LN}(x)$? O que isso implica para $\partial\,\text{LN}(x)/\partial x$
   quando $x$ é muito pequeno?
3. Dê um exemplo concreto de como uma BatchNorm deixaria a posição 2 "saber" algo sobre a
   posição 7.
4. Escreva $x_2$ (depois de dois blocos) como $x_0$ mais uma soma de termos. Quantos termos
   aparecem com $L$ blocos?
5. Com as subcamadas silenciadas, por que o gradiente na entrada é exatamente igual ao da
   saída? O que muda quando elas escrevem algo?
6. Na tabela de normas da seção 3.3, por que o post-norm tem 4.280 em $\ell = 0$ e 125,7 nas
   demais camadas?
7. Por que o pre-norm precisa de uma LayerNorm final antes do LM head, e o post-norm não?
8. Com $L = 12$ (GPT-2 small), qual é o desvio inicial de `out_proj` e `down_proj`? E se o
   modelo tivesse 48 blocos?
9. Expanda $(I + J_3)(I + J_2)(I + J_1)$. Quantos termos aparecem, e qual deles não passa por
   nenhuma subcamada?
10. No exemplo da seção 2.2, por que `x.var()` do PyTorch daria 2,667 e não 2,222? Qual das
    duas a LayerNorm usa?
11. Com $p = 0{,}1$, por que os números que sobrevivem ao dropout são multiplicados por
    $1/0{,}9$? O que aconteceria com a escala das escritas se isso não fosse feito?
12. Um gradiente com norma 206,5 e cosseno 0,010 é útil para o treino? E um com norma 0,5 e
    cosseno 0,95?
