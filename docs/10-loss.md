# Loss

> **Como o modelo sabe que errou?**

Código: [`training/loss.py`](../training/loss.py). Testes:
[`tests/test_loss.py`](../tests/test_loss.py). Experimento:
`uv run python -m experiments.e10_loss` (~15 s, por causa do mini-treino). Notação geral:
[00-notacao.md](00-notacao.md). Visão geral: [architecture.md](architecture.md).
Pré-requisito: [09-lm-head.md](09-lm-head.md).

---

## Para que serve

### O problema

Depois do LM head ([09-lm-head.md](09-lm-head.md)), o modelo produz, para cada posição, uma
distribuição de probabilidade sobre os 97 caracteres. O dataset
([02-dataset.md](02-dataset.md)) sabe qual caractere **realmente** veio em seguida. Falta
uma forma de comparar os dois e resumir a comparação num **único número**:

- pequeno quando o modelo deu probabilidade alta ao caractere certo;
- grande quando deu probabilidade baixa;
- **diferenciável**, para que o gradiente diga como ajustar cada parâmetro e reduzir o número.

Esse número é a **loss**, e a escolhida para modelos de linguagem é a **cross-entropy**.

Termos usados daqui em diante:

- **Loss** (perda, erro): o número que mede o quão ruins são as previsões. Treinar é ajustar os
  parâmetros para diminuí-lo.
- **Cross-entropy** (entropia cruzada): a média de $-\log p(\text{token correto})$.
- **Nat:** a unidade da loss quando se usa o logaritmo natural ($\ln$). **Bit:** a unidade com
  logaritmo na base 2. $1 \text{ nat} \approx 1{,}443 \text{ bit}$.
- **Perplexidade:** $e^{\text{loss}}$. Pode ser lida como "entre quantas opções igualmente
  prováveis o modelo está hesitando".
- **Verossimilhança** (*likelihood*): a probabilidade que o modelo dá aos dados reais.
- **N-grama:** um modelo de contagem que prevê o próximo caractere olhando só os $n-1$
  anteriores.

### Uma analogia

Pense num serviço de previsão do tempo avaliado todo dia:

- disse "90% de chance de chuva" e choveu: quase nenhuma penalidade;
- disse "50%" e choveu: penalidade média;
- disse "1%" e choveu: penalidade enorme, porque estava **confiante e errado**.

A cross-entropy é exatamente essa regra de pontuação, com a penalidade $-\log p$, onde $p$ é a
probabilidade dada ao que aconteceu. Um previsor que sempre diz "50%" nunca leva uma penalidade
enorme, mas também nunca leva uma pequena: só vence quem acerta **e** sabe quando está seguro.

### O que entra e o que sai

| | shape | exemplo no `tiny.yaml` |
|---|---|---|
| entrada `logits` | `[B, T, V]` | `[32, 128, 97]`: 4.096 distribuições |
| entrada `targets` | `[B, T]` | `[32, 128]`: o ID correto de cada posição (o `y` do dataset) |
| saída | escalar | um único número, por exemplo `4.6080` |

### Onde este componente entra

```text
texto: "Uma noite destas, vindo da cidade..."
  │
  ▼  tokenizer                                 str → IDs [374.785]
  ▼  dataset e batches                         IDs → x, y: [32, 128]
  ▼  embeddings                                [32, 128] → [32, 128, 128]
  ▼  blocos Transformer × 4                    [32, 128, 128] → [32, 128, 128]
  ▼  LM head                                   [32, 128, 128] → logits [32, 128, 97]
  ▼  loss                                      logits e y → um número            ← este componente
  ▼  backward + otimizador                     o número → ajuste dos 820.096 parâmetros
```

### O que daria errado sem esta parte

- **Não haveria treino.** O gradiente é a derivada **da loss** em relação a cada parâmetro. Sem
  loss, não há o que derivar.
- **Com uma medida ruim, o treino não andaria.** A **acurácia** ("acertou o caractere mais
  provável?") não serve como loss: pequenas mudanças nos pesos quase nunca mudam qual caractere
  é o mais provável, então a derivada é zero em quase todo lugar (seção 1.3).
- **Com a conta ingênua, logits grandes gerariam `nan`** e derrubariam o treino (seção 4).

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $V$ | "vê" | tamanho do vocabulário | 97 | `vocab_size` |
| $z$, $z_i$ | "zê", "zê i" | logits de uma posição; $z_i$ é o logit do token $i$ | `[97]` | `logits[b, t]` |
| $p$, $p_i$ | "pê", "pê i" | probabilidades: $p_i = \text{softmax}(z)_i$ | `[97]`, soma 1 | `torch.softmax(z, -1)` |
| $y$ | "ípsilon" | ID do token correto (o alvo) de uma posição | um inteiro de 0 a 96 | `targets[b, t]` |
| $p_y$ | "pê ípsilon" | probabilidade dada ao token correto | entre 0 e 1 | `p[y]` |
| $\ell$ | "ele" cursivo | loss de **uma** posição: $\ell = -\log p_y$. Aqui **não** é índice de camada | um número $\ge 0$ | `-target_log_probs` |
| $\mathcal{L}$ | "ele caligráfico" | loss do batch: a média dos $\ell$ | um número | `cross_entropy(...)` |
| $N$ | "ene" | número de posições com alvo no batch: $B \cdot T$ | 4.096 | `targets.numel()` |
| $n$ | "ene" minúsculo | índice de uma posição, de 1 a $N$ | | |
| $B$, $T$ | "bê", "tê" | batch e contexto | 32, 128 | |
| $\log$, $\ln$ | "log", "logaritmo natural" | logaritmo na base $e$ | $\ln 97 = 4{,}575$ | `torch.log` |
| $\log_2$ | "log na base 2" | logaritmo em bits | $\log_2 2 = 1$ | `math.log2` |
| $\text{LSE}(z)$ | "log-sum-exp de zê" | $\ln \sum_j e^{z_j}$ | $\text{LSE}(2; 1; 0{,}1; -1) = 2{,}449$ | `torch.logsumexp(z, -1)` |
| $m$ | "eme" | o maior logit, usado no truque do log-sum-exp | | |
| $\mathbb{1}[i = y]$ | "indicadora de i igual a ípsilon" | 1 se $i$ é o token correto, 0 se não | | `F.one_hot(y, V)[i]` |
| $\text{one\_hot}(y)$ | "one-hot de ípsilon" | vetor de tamanho $V$ com 1 na posição $y$ e 0 no resto | `[97]` | `F.one_hot(y, V)` |
| $\dfrac{\partial \mathcal{L}}{\partial z_i}$ | "d parcial de ele em relação a zê i" | quanto a loss muda se o logit $i$ subir um pouco | | `logits.grad` |
| $\text{PPL}$ | "perplexidade" | $e^{\mathcal{L}}$ | 97 para o chute uniforme | `math.exp(loss)` |
| $H$ | "agá" | entropia: a menor loss possível para uma fonte de dados | desconhecida para o português | |
| $c(\cdot)$ | "contagem de" | quantas vezes uma sequência de caracteres aparece no treino | $c(\texttt{qu})$ | `torch.bincount` |
| $\alpha$ | "alfa" | constante de suavização dos n-gramas | 1; 0,1; 0,01 | `alpha` |
| $\theta$ | "teta" | todos os parâmetros do modelo | 820.096 números | `model.parameters()` |

---

## 1. A loss de uma previsão: $-\log p_y$

### 1.1 A fórmula

Para uma posição, o modelo deu a distribuição $p$ e o token correto era $y$. A loss dessa
posição é:

$$
\ell = -\log p_y
$$

**Lendo a fórmula:**

- $p_y$: a probabilidade que o modelo deu ao token **que realmente veio**. As probabilidades
  dos outros tokens não aparecem diretamente, mas pesam: como tudo soma 1, dar muita
  probabilidade aos errados deixa pouca para o certo.
- $\log$: logaritmo natural. Como $0 < p_y \le 1$, o log é zero ou negativo.
- o sinal de menos transforma o valor num número **positivo**: a penalidade.

### 1.2 Como a penalidade cresce

Seção 1 do experimento:

```text
  p(correto) | −ln p (nats) | −log₂ p (bits)
           1 |        0.000 |          0.000
         0.9 |        0.105 |          0.152
         0.5 |        0.693 |          1.000
        0.25 |        1.386 |          2.000
         0.1 |        2.303 |          3.322
     0.01031 |        4.575 |          6.600
       0.001 |        6.908 |          9.966
       1e-06 |       13.816 |         19.932
```

- **Certeza no acerto ($p = 1$):** loss 0, o mínimo possível.
- **Cada vez que a probabilidade cai pela metade, a loss sobe o mesmo tanto:** $\ln 2 = 0{,}693$
  nat, ou exatamente 1 bit (0,5 → 0,25 soma 1 bit).
- **$p = 1/97$:** a loss é $\ln 97 = 4{,}575$, a do chute uniforme ([02-dataset.md](02-dataset.md)).
- **Confiança no erro ($p \to 0$):** a loss cresce sem limite. Com $p = 10^{-6}$ já é 13,8.

### 1.3 Por que não usar a acurácia

A acurácia pergunta só "o token mais provável era o certo?". Imagine $p_y$ subindo de 0,20 para
0,21 enquanto outro token continua com 0,30: a acurácia não muda (continua errando), mas o
modelo melhorou. A cross-entropy registra essa melhora ($-\ln 0{,}20 = 1{,}609$ para
$-\ln 0{,}21 = 1{,}561$). Como ela muda suavemente com cada logit, tem derivada útil em toda
parte, e é isso que o treino precisa.

---

## 2. Do token ao batch: a média

### 2.1 A fórmula

Um batch tem $N = B \cdot T = 4.096$ posições, cada uma com o seu alvo. A loss do batch é a
**média** das perdas individuais:

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} -\log p^{(n)}_{y_n}
$$

**Lendo a fórmula:**

- $n$: percorre as $N$ posições do batch (todas as sequências e todas as posições de cada uma).
- $y_n$: o token correto da posição $n$.
- $p^{(n)}_{y_n}$: a probabilidade que o modelo deu a $y_n$ na posição $n$. O $^{(n)}$ em cima é
  um rótulo ("da posição $n$"), não um expoente.
- $\sum_{n=1}^{N}$: soma as perdas das $N$ posições.
- $\frac{1}{N}$: divide pelo número de posições, transformando a soma em média.

Exemplo da seção 2 do experimento, com três posições cujos alvos são `'a'`, `'c'` e `'d'`:

```text
num batch, a loss é a média das posições:
  alvos 'a', 'c', 'd' -> (0.449 2.349 3.449) / 3 = 2.083
```

### 2.2 Por que média, e não soma

Com a soma, a loss e o gradiente cresceriam junto com o tamanho do batch: dobrar $B$ dobraria o
passo do otimizador. Com a média, a escala não depende de $B$ nem de $T$, e a mesma learning
rate funciona para batches de tamanhos diferentes.

### 2.3 Minimizar a loss é maximizar a verossimilhança

Em [02-dataset.md](02-dataset.md) vimos que a probabilidade de um texto inteiro é o produto
das probabilidades de cada token dado o anterior. O log transforma esse produto em soma:

$$
-\log \prod_{n} p^{(n)}_{y_n} = \sum_{n} -\log p^{(n)}_{y_n} = N \cdot \mathcal{L}
$$

**Lendo a fórmula:**

- $\prod_n p^{(n)}_{y_n}$: a probabilidade que o modelo dá ao **texto real** do batch (a
  verossimilhança).
- $-\log$ desse produto: pela propriedade $\ln(ab) = \ln a + \ln b$, vira a soma das perdas.
- $N \cdot \mathcal{L}$: a soma é $N$ vezes a média.

Logo, **diminuir $\mathcal{L}$ é o mesmo que aumentar a probabilidade que o modelo dá ao texto
de verdade**. Esse princípio se chama *máxima verossimilhança*.

O log também é necessário na prática: o produto de 4.096 probabilidades em torno de 0,01 seria
$0{,}01^{4096} = 10^{-8192}$, muito abaixo do menor número que um float representa. A soma de
logs fica perto de $4.096 \times 4{,}6 \approx 19$ mil, sem problema algum.

---

## 3. Calculando: logits → log-probabilidades → loss

### 3.1 O caminho em três passos

A loss parte dos **logits**, não das probabilidades. O cálculo tem três passos:

$$
\log p_i = z_i - \text{LSE}(z), \qquad \text{LSE}(z) = \ln \sum_{j=1}^{V} e^{z_j}, \qquad \ell = -\log p_y
$$

**Lendo a fórmula:**

- $z_i$: o logit do token $i$.
- $e^{z_j}$: a exponencial de cada logit (sempre positiva).
- $\sum_j e^{z_j}$: a soma das exponenciais, o denominador da softmax.
- $\text{LSE}(z)$: o log dessa soma. É o mesmo número para todos os $i$ de uma posição.
- $z_i - \text{LSE}(z)$: pela definição da softmax,
  $\log p_i = \log \frac{e^{z_i}}{\sum_j e^{z_j}} = z_i - \ln \sum_j e^{z_j}$.
- $\ell = -\log p_y$: pega o log-probabilidade do token correto e troca o sinal.

### 3.2 Um exemplo completo

Seção 2 do experimento: um vocabulário de 4 tokens (`a`, `b`, `c`, `d`) e o alvo `'c'`.

```text
tokens:              a      b      c      d
logits z:            +2.000 +1.000 +0.100 -1.000
exp(z):              +7.389 +2.718 +1.105 +0.368   soma = 11.580
p = exp(z) / soma:   +0.638 +0.235 +0.095 +0.032   soma = 1.000
logsumexp(z) = ln 11.580 = 2.449
log p = z − 2.449:  -0.449 -1.449 -2.349 -3.449
alvo = 'c': loss = −log p[2] = 2.349
F.cross_entropy: 2.349; cross_entropy do projeto: 2.349
```

Passo a passo:

1. **Exponenciar:** $e^{2} = 7{,}389$, $e^{1} = 2{,}718$, $e^{0,1} = 1{,}105$, $e^{-1} = 0{,}368$.
2. **Somar:** $7{,}389 + 2{,}718 + 1{,}105 + 0{,}368 = 11{,}580$.
3. **LSE:** $\ln 11{,}580 = 2{,}449$.
4. **Log-probabilidades:** subtrair 2,449 de cada logit. Para `'c'`: $0{,}1 - 2{,}449 = -2{,}349$.
5. **Loss:** $\ell = -(-2{,}349) = 2{,}349$. Conferindo: $p_c = 0{,}095$ e $-\ln 0{,}095 \approx 2{,}35$.

O mesmo cálculo para cada possível alvo:

```text
a loss se o alvo fosse cada um dos tokens:
  alvo 'a': p = 0.638  loss = 0.449
  alvo 'b': p = 0.235  loss = 1.449
  alvo 'c': p = 0.095  loss = 2.349
  alvo 'd': p = 0.032  loss = 3.449
```

Como os logits diferem de 1 em 1 entre `a`, `b` e `d`, as perdas também diferem de 1 em 1: a
loss é $\text{LSE}(z) - z_y$, então subir o logit do alvo em 1 reduz a loss em exatamente 1.

---

## 4. Estabilidade numérica: o truque do log-sum-exp

### 4.1 O problema

Um `float32` guarda números até $3{,}4 \times 10^{38}$. Mas $e^{1000}$ tem mais de 400 dígitos.
Seção 3 do experimento, com logits `1000, 0, -1000`:

```text
logits: 1000 0 -1000
  exp(z) / soma(exp(z)):         nan 0.000 0.000
  log(softmax(z)):               0.0 -inf -inf
  z − logsumexp(z) (o projeto):  0.0 -1000.0 -2000.0
  maior float32 finito: 3.403e+38; exp(1000) não cabe
```

- **A conta ingênua** calcula $e^{1000} = \infty$ e depois $\infty / \infty$ = `nan` ("não é um
  número"). Um único `nan` na loss contamina todos os gradientes e destrói o modelo.
- **`log(softmax)`** evita o `nan`, mas as probabilidades minúsculas viram exatamente 0, e
  $\log 0 = -\infty$. Se um desses fosse o alvo, a loss seria infinita.
- **`z − logsumexp(z)`** dá os valores corretos: $-1000$ e $-2000$.

### 4.2 O truque

O `torch.logsumexp` usa uma identidade que evita números enormes. Seja $m = \max_j z_j$:

$$
\text{LSE}(z) = m + \ln \sum_{j} e^{z_j - m}
$$

**Lendo a fórmula:**

- $m$: o maior logit.
- $z_j - m$: cada logit menos o maior. Todos ficam $\le 0$, então $e^{z_j - m} \le 1$: nada
  explode.
- $m + \dots$: devolve o que foi subtraído. Vale porque
  $\ln \sum_j e^{z_j} = \ln\big(e^m \sum_j e^{z_j - m}\big) = m + \ln \sum_j e^{z_j - m}$.

Exemplo: com $z = (1000, 0, -1000)$, $m = 1000$, e
$\text{LSE} = 1000 + \ln(e^{0} + e^{-1000} + e^{-2000}) = 1000 + \ln(1 + \text{quase } 0) = 1000$.
O log-probabilidade do terceiro token é $-1000 - 1000 = -2000$, que é a loss que o teste
`test_large_logits_are_numerically_stable` confere.

---

## 5. O gradiente: $p - \text{one\_hot}(y)$

### 5.1 A derivação

O treino precisa de $\partial \mathcal{L} / \partial z_i$: quanto a loss muda quando cada logit
sobe um pouco. Para uma posição, a loss é $\ell = \text{LSE}(z) - z_y$. Derivando cada parte:

**Parte 1, o LSE.** Pela regra da cadeia, com $\ln u$ e $u = \sum_j e^{z_j}$:

$$
\frac{\partial\, \text{LSE}(z)}{\partial z_i} = \frac{1}{\sum_j e^{z_j}} \cdot e^{z_i} = p_i
$$

**Lendo a fórmula:**

- $\frac{1}{\sum_j e^{z_j}}$: a derivada de $\ln u$ é $1/u$.
- $e^{z_i}$: a derivada de $u$ em relação a $z_i$. Só o termo $j = i$ da soma depende de $z_i$.
- o produto é exatamente a softmax $p_i$.

**Parte 2, o $-z_y$.** A derivada de $-z_y$ em relação a $z_i$ é $-1$ se $i = y$ e $0$ se não:

$$
\frac{\partial (-z_y)}{\partial z_i} = -\mathbb{1}[i = y]
$$

**Juntando** e dividindo por $N$, por causa da média:

$$
\boxed{\;\frac{\partial \mathcal{L}}{\partial z_i} = \frac{p_i - \mathbb{1}[i = y]}{N}\;}
\qquad\Longleftrightarrow\qquad
\frac{\partial \mathcal{L}}{\partial z} = \frac{p - \text{one\_hot}(y)}{N}
$$

**Lendo a fórmula:**

- $p_i$: a probabilidade atual do token $i$.
- $\mathbb{1}[i = y]$: vale 1 só para o token correto.
- $p - \text{one\_hot}(y)$: a diferença entre "o que o modelo previu" e "a resposta ideal"
  (probabilidade 1 no token certo e 0 no resto).
- $\frac{1}{N}$: cada posição contribui com uma fração do gradiente total.

### 5.2 O que o gradiente diz

O otimizador anda **contra** o gradiente (descida do gradiente, [00-notacao.md](00-notacao.md),
seção 7.5). Então:

| token | gradiente (sem o $1/N$) | sinal | efeito do passo |
|---|---|---|---|
| o correto ($i = y$) | $p_y - 1$ | negativo | o logit **sobe** |
| cada errado ($i \ne y$) | $p_i$ | positivo | o logit **desce** |

Três propriedades importantes:

- **O empurrão é proporcional ao erro.** Um token errado com $p_i = 0{,}4$ é empurrado para baixo
  com força 0,4; um com $p_i = 0{,}001$ quase não é tocado.
- **O gradiente desliga sozinho.** Se o modelo já dá $p_y \approx 1$, então $p_y - 1 \approx 0$
  e os $p_i$ dos errados também são $\approx 0$: nada a corrigir.
- **A soma é zero em cada posição:** $\sum_i (p_i - \mathbb{1}[i = y]) = 1 - 1 = 0$. O quanto o
  alvo sobe é exatamente o quanto os outros descem, somados. É por isso que somar uma
  constante a todos os logits não muda nada ([09-lm-head.md](09-lm-head.md), seção 2.2).

### 5.3 A verificação no modelo real

Seção 4 do experimento: o GPT recém-inicializado num batch real, com `logits.retain_grad()` para
guardar o gradiente dos logits.

```text
batch (32, 128): loss = 4.5656; N = B·T = 4,096 posições
logits.grad == (softmax − one_hot) / N: True

posição 10 da sequência 0: contexto 'alguma cous', alvo 'a'
  alvo:  p = 0.0089  gradiente = p − 1 = -0.9911  (sobe)
    's': p = 0.0358  gradiente = p     = +0.0358  (desce)
    ' ': p = 0.0154  gradiente = p     = +0.0154  (desce)
    "'": p = 0.0152  gradiente = p     = +0.0152  (desce)
  soma do gradiente nesta posição: +2.1e-07
```

- **A fórmula bate com o autograd** do PyTorch em todas as 4.096 × 97 entradas.
- **O contexto é "alguma cous"** e o alvo `'a'` (de "cousa", grafia de 1899). O modelo ainda não
  sabe nada: dá 0,89% ao `'a'`, e o gradiente pede para subir esse logit com força ~0,99.
- **O `'s'` é o token errado mais provável**, com 3,6%. É o viés do weight tying de "repetir o
  caractere atual" ([09-lm-head.md](09-lm-head.md), seção 4.5), e é justamente o que leva o
  maior empurrão para baixo.
- **A soma é $2 \times 10^{-7}$:** zero, a menos de arredondamento.

Esse gradiente denso, que toca os 97 logits de cada posição, é o que dá sinal a **todas** as
linhas do embedding quando há weight tying (mesmo documento, seção 4.4).

---

## 6. Réguas: o que significa uma loss de 2,4?

### 6.1 O problema de interpretar

A loss é um número em nats, e sozinho ele diz pouco. 2,4 é bom? Para responder, comparamos com
modelos simples, que servem de **régua**. Todas as medidas abaixo usam a **validação** (os 10%
finais do livro, que o modelo nunca usa para aprender).

### 6.2 Modelos de contagem (n-gramas)

Um **n-grama** prevê o próximo caractere contando, no texto de treino, o que costuma vir depois
dos $n - 1$ caracteres anteriores:

| modelo | olha para trás | pergunta que responde |
|---|---|---|
| unigrama ($n = 1$) | nada | "qual a frequência de cada caractere?" |
| bigrama ($n = 2$) | 1 caractere | "o que costuma vir depois de `'q'`?" |
| trigrama ($n = 3$) | 2 caracteres | "o que costuma vir depois de `'qu'`?" |

A probabilidade de um bigrama, por exemplo, é:

$$
p(\text{próximo} = b \mid \text{anterior} = a) = \frac{c(ab) + \alpha}{c(a\,\cdot) + \alpha V}
$$

**Lendo a fórmula:**

- $c(ab)$: quantas vezes o par "$a$ seguido de $b$" aparece no treino.
- $c(a\,\cdot)$: quantas vezes $a$ aparece seguido de qualquer caractere (a soma de $c(ab)$
  sobre todos os $b$).
- sem o $\alpha$, seria só a frequência relativa: "das vezes que veio $a$, em quantas veio $b$
  depois".
- $\alpha$ (**suavização add-α**): um número pequeno somado a toda contagem. Sem ele, um par que
  nunca apareceu no treino teria probabilidade 0, e bastaria um par novo na validação para a
  loss ser infinita (seção 1.2).
- $\alpha V$ no denominador: compensa os $V$ valores de $\alpha$ somados no numerador, para as
  probabilidades continuarem somando 1.

Exemplo com números inventados: se `'q'` apareceu 1.000 vezes, sempre seguido de `'u'`, com
$\alpha = 0{,}01$ e $V = 97$:
$p(\texttt{u} \mid \texttt{q}) = \frac{1000 + 0{,}01}{1000 + 0{,}97} = 0{,}999$ e cada um dos
outros 96 caracteres fica com $\frac{0{,}01}{1000{,}97} \approx 0{,}00001$: pequeno, mas não zero.

O código conta todas as combinações de uma vez: cada sequência vira um número único, como
dígitos na base $V$ (para o trigrama, $c_1 \cdot V^2 + c_2 \cdot V + \text{próximo}$), e
`torch.bincount` conta quantas vezes cada número aparece.

### 6.3 O efeito de $\alpha$

```text
modelos de contagem (n-gramas) com suavização add-α, loss por α:
                                    α = 1  α = 0,1  α = 0,01
  unigrama (nenhum contexto)        3.058    3.058    3.058
  bigrama (1 caractere antes)       2.338    2.328    2.329
  trigrama (2 caracteres antes)     2.023    1.931    1.923
```

- **Unigrama:** $\alpha$ não faz diferença, porque cada caractere tem milhares de contagens.
- **Trigrama:** há $97^2 = 9.409$ contextos possíveis, muitos vistos poucas vezes. Com
  $\alpha = 1$, cada contexto ganha 97 contagens falsas, que abafam as verdadeiras dos
  contextos raros. Um $\alpha$ menor funciona melhor.

Escolher o $\alpha$ olhando a validação deixa essas réguas um pouco otimistas. Para uma régua,
isso é aceitável.

### 6.4 A tabela de referência

Seção 5 do experimento:

```text
melhor α de cada n-grama, comparado com outras referências:
  régua                                loss (nats)  perplexidade  bits/caractere
  chute uniforme (ln 97)                     4.575         97.00           6.600
  GPT recém-inicializado                     4.570         96.50           6.592
  unigrama                                   3.058         21.29           4.412
  bigrama                                    2.328         10.26           3.359
  GPT após 150 passos (prévia)               2.402         11.04           3.465
  trigrama                                   1.923          6.84           2.774
```

As duas colunas novas são outras formas de ler a mesma loss:

$$
\text{PPL} = e^{\mathcal{L}}, \qquad \text{bits por caractere} = \frac{\mathcal{L}}{\ln 2}
$$

**Lendo a fórmula:**

- $\text{PPL}$ (**perplexidade**): se o modelo escolhesse entre $k$ opções igualmente prováveis,
  a loss seria $\ln k$ e a perplexidade seria $k$. Por isso ela se lê como "número efetivo de
  opções". O chute uniforme hesita entre 97; o bigrama, entre ~10; o trigrama, entre ~7.
- $\ln 2 = 0{,}693$: dividir por ele converte nats em bits. **Bits por caractere** é a medida
  usual em modelos de caracteres: quantos bits, em média, seriam necessários para codificar cada
  caractere usando as probabilidades do modelo.
- Exemplo: bigrama, $\mathcal{L} = 2{,}328$. $e^{2{,}328} = 10{,}26$ e $2{,}328 / 0{,}693 = 3{,}359$.

### 6.5 Onde o GPT está

- **Recém-inicializado (4,570):** praticamente o chute uniforme, como previsto em
  [09-lm-head.md](09-lm-head.md). Fica um pouco abaixo por causa do viés de repetição do
  weight tying.
- **Depois de 150 passos (2,402):** já é muito melhor que o unigrama, mas ainda **pior que o
  bigrama**. Em 150 passos, o modelo aprendeu aproximadamente "que letra costuma vir depois de
  qual", e ainda não usa bem o contexto mais longo que tem à disposição (128 caracteres).
- **A meta mínima do training loop:** ficar **abaixo do trigrama (1,923)**. Isso mostraria que o
  modelo usa mais do que 2 caracteres de contexto, algo que nenhum modelo de contagem simples
  consegue sem explodir o número de contextos.

Uma ressalva de medição: os n-gramas avaliam todos os caracteres da validação, e o GPT avalia os
que cabem nas janelas completas (36.864 dos 37.478, por causa do `drop_last`). A diferença não
muda as conclusões.

### 6.6 Existe um piso?

Nenhum modelo chega a loss 0 no texto real, porque o próximo caractere nunca é totalmente
previsível. A menor loss possível é a **entropia** $H$ da fonte (o português do *Dom Casmurro*),
um valor desconhecido. A cross-entropy de qualquer modelo fica sempre acima dela:

$$
\mathcal{L}_{\text{modelo}} = H + (\text{distância entre a distribuição real e a do modelo}) \ge H
$$

**Lendo a fórmula:**

- $H$: a imprevisibilidade intrínseca do texto, que nenhum modelo remove.
- a "distância" (em estatística, a divergência KL) é zero só se o modelo reproduzir exatamente as
  probabilidades reais.

Para ter uma ordem de grandeza: Shannon (1951) estimou que o inglês escrito tem entre ~0,6 e
~1,3 bit por letra, usando pessoas como previsores. Modelos grandes de linguagem chegam perto
da parte baixa dessa faixa. O Mini-GPT, com 820 mil parâmetros e 337 mil caracteres de treino,
vai ficar bem acima disso.

---

## 7. Código

### 7.1 A função `cross_entropy`

> **Atualização.** O cálculo passou para `token_losses(logits, targets)`, que devolve a loss
> de **cada** posição (shape de `targets`), e `cross_entropy` agora é `token_losses(...).mean()`. As
> contas abaixo são as mesmas; a separação permite analisar onde o modelo erra
> ([14-evaluation.md](14-evaluation.md)).

```python
def cross_entropy(logits, targets):
    if logits.shape[:-1] != targets.shape:
        raise ValueError(...)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)   # [..., V]
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [...]
    return -target_log_probs.mean()
```

Linha por linha, com `logits: [32, 128, 97]` e `targets: [32, 128]`:

1. **Validação dos shapes.** `logits.shape[:-1]` é o shape sem a última dimensão, `(32, 128)`,
   e precisa ser igual ao de `targets`. Sem essa checagem, um erro de shape poderia passar
   calado pelo broadcasting e produzir uma loss sem sentido
   (`test_shape_mismatch_raises`).
2. **Log-probabilidades.** `torch.logsumexp(logits, dim=-1, keepdim=True)` calcula o LSE de cada
   posição: shape `[32, 128, 1]`. O `keepdim=True` mantém a dimensão de tamanho 1 para o
   broadcasting subtrair o mesmo valor dos 97 logits. Resultado: `[32, 128, 97]`.
3. **Escolher o log-probabilidade do alvo.** `gather` pega, em cada posição, o elemento cujo
   índice está em `targets`:
   - `targets.unsqueeze(-1)`: `[32, 128]` → `[32, 128, 1]` (o `gather` exige o mesmo número de
     dimensões);
   - `log_probs.gather(-1, ...)`: para cada `[b, t]`, pega `log_probs[b, t, targets[b, t]]`,
     com shape `[32, 128, 1]`;
   - `.squeeze(-1)`: remove a dimensão de tamanho 1, voltando a `[32, 128]`.
4. **Média e sinal.** `-target_log_probs.mean()` calcula $\mathcal{L}$: um escalar.

A função aceita qualquer número de dimensões antes de $V$: `[N, V]`, `[B, T, V]` etc.

### 7.2 Por que não usar direto `F.cross_entropy`

`F.cross_entropy` calcula a mesma coisa, e os testes garantem que os valores são iguais. Seção 6
do experimento:

```text
  cross_entropy do projeto   0.686 ms por chamada, entrada [32, 128, 97]
  F.cross_entropy            0.543 ms por chamada, entrada [32, 128, 97]
```

A versão do PyTorch é um pouco mais rápida (funde as operações num único kernel), mas a
diferença é de ~0,14 ms por passo, desprezível perto do forward e backward do modelo. O projeto
usa a própria função pelo mesmo motivo da LayerNorm
([07-transformer-block.md](07-transformer-block.md)): as três linhas mostram exatamente
a conta. `F.cross_entropy` tem recursos que o projeto ainda não usa, como `ignore_index` para
pular posições de padding e `label_smoothing`.

### 7.3 Integração com o `train.py`

```text
modelo:     GPT com 820,096 parâmetros (com weight tying)
forward:    ids (32, 128) -> logits (32, 128, 97)
loss:       4.6080 no primeiro batch (chute uniforme: 4.5747)
Training loop ainda não implementado — nada para treinar.
```

A loss inicial fica perto de $\ln 97$, como previsto. A peça que falta é usar esse número para
atualizar os pesos: o training loop ([11-training.md](11-training.md)).

---

## 8. O que cada teste garante

[`tests/test_loss.py`](../tests/test_loss.py):

| teste | propriedade |
|---|---|
| `test_matches_pytorch_for_flat_inputs` | igual a `F.cross_entropy` com `[N, V]` |
| `test_matches_pytorch_for_sequences` | igual a `F.cross_entropy` com `[B, T, V]` |
| `test_small_example_by_hand` | logits (2, 1, 0), alvo 0: $-\ln 0{,}665$ |
| `test_uniform_logits_give_log_vocab_size` | logits iguais dão $\ln V$ |
| `test_confident_correct_prediction_gives_almost_zero_loss` | certeza no acerto: loss ≈ 0 |
| `test_confident_wrong_prediction_is_heavily_penalized` | certeza no erro, com margem 50: loss ≈ 50 |
| `test_adding_a_constant_to_the_logits_does_not_change_the_loss` | só diferenças entre logits importam |
| `test_large_logits_are_numerically_stable` | logits ±1000 dão loss finita e correta (2.000) |
| `test_shape_mismatch_raises` | shapes incompatíveis geram `ValueError` |
| `test_gradient_is_softmax_minus_one_hot` | $\partial\mathcal{L}/\partial z = (p - \text{one\_hot}(y))/N$ |
| `test_gradient_of_each_position_sums_to_zero` | o gradiente de cada posição soma 0 |
| `test_loss_reaches_every_parameter_of_the_model` | a loss do GPT gera gradiente em todos os parâmetros |

---

## Resumo

1. A loss de uma previsão é $-\log p_y$: zero para certeza no acerto, crescendo sem limite para
   confiança no erro. A cross-entropy do batch é a média dessas perdas.
2. Minimizar a cross-entropy é maximizar a probabilidade que o modelo dá ao texto real (máxima
   verossimilhança). O log transforma o produto de probabilidades numa soma estável.
3. A conta vai dos logits às log-probabilidades por $z_i - \text{LSE}(z)$. O `logsumexp` evita os
   `nan` e $-\infty$ que a conta ingênua gera com logits grandes.
4. O gradiente nos logits é $(p - \text{one\_hot}(y))/N$: o alvo sobe com força $1 - p_y$, cada
   token errado desce com força $p_i$, e o gradiente some quando a previsão fica certa.
5. As réguas dão sentido aos números: uniforme 4,575; unigrama 3,058; bigrama 2,328; trigrama
   1,923. O GPT com 150 passos (2,402) ainda não passa do bigrama; a meta do training loop
   ([11-training.md](11-training.md)) é ficar abaixo do trigrama.

---

## Para checar o entendimento

1. Um modelo dá $p_y = 0{,}2$ ao token certo. Qual a loss em nats e em bits? E se ele dobrar essa
   probabilidade?
2. Por que a acurácia não serve como loss para o treino, mesmo sendo o que queremos no fim?
3. Com logits $z = (3, 1, 0)$ e alvo no segundo token, calcule $\text{LSE}(z)$, os
   log-probabilidades e a loss.
4. Mostre que $\sum_i \partial \ell / \partial z_i = 0$. O que isso diz sobre somar uma constante a
   todos os logits?
5. Se o modelo já dá $p_y = 0{,}99$, qual é o gradiente no logit do alvo? E num token errado com
   $p_i = 0{,}005$? O que isso significa para o treino?
6. Por que um trigrama sem suavização teria loss infinita na validação?
7. O que significa uma perplexidade de 10,26? Por que o chute uniforme tem perplexidade 97?
8. O GPT com 150 passos tem loss 2,402, pior que o bigrama (2,328), apesar de ver 128 caracteres
   de contexto. O que isso sugere sobre o que ele aprendeu até ali?
9. Por que a média, e não a soma, das perdas das posições?
10. Por que um único `nan` na loss é tão grave para o treino?
