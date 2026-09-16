# GPT Model

> **O que acontece quando empilhamos os blocos?**

Código: [`model/gpt.py`](../model/gpt.py). Testes: [`tests/test_gpt.py`](../tests/test_gpt.py).
Experimento: `uv run python -m experiments.e08_gpt` (~15 s, por causa do mini-treino). Visão
geral: [architecture.md](architecture.md). Notação geral: [00-notacao.md](00-notacao.md).

> **Atualização depois do LM head ([09-lm-head.md](09-lm-head.md)).** O `GPT` ganhou o LM head:
> `model(ids)` passou a devolver logits `[B, T, V]`, e os estados finais descritos neste
> documento estão em `model.hidden_states(ids)`. Os números abaixo já foram gerados com essa
> versão.

---

## Para que serve

### O problema

Até aqui, cada peça funcionava sozinha: o tokenizer, o dataset, os embeddings e o
Transformer block. Faltava **juntar tudo num único modelo**: um objeto que recebe IDs de
caracteres e devolve, para cada posição, um vetor com tudo o que foi calculado sobre ela.

Este documento faz essa montagem. Ele responde três perguntas:

1. **Como ligar as peças?** Embeddings, depois $L$ blocos em sequência, depois uma LayerNorm
   final.
2. **O que ganhamos repetindo o bloco?** Não é alcance (um bloco já vê todo o contexto). É
   **composição**: um bloco usa o que o anterior calculou.
3. **O modelo inteiro continua correto?** Continua causal, não mistura sequências do batch e
   tem o número de parâmetros que a fórmula prevê.

### Uma analogia

Pense numa **linha de revisão de um texto** com 4 revisores em fila. Todos trabalham sobre a
mesma folha (o **residual stream**). Cada revisor lê a folha, faz anotações e passa adiante,
sem apagar o que os anteriores escreveram. O segundo revisor pode usar as anotações do
primeiro ("aqui antes vinha a letra `a`") para fazer algo que sozinho não conseguiria. No fim,
um editor (a **LayerNorm final**) passa a limpo a folha, num tamanho padrão, para quem vai ler
depois (o **LM head**).

### O que entra e o que sai

Com o `tiny.yaml` e o vocabulário do *Dom Casmurro*:

| | shape | valores |
|---|---|---|
| **entra** `ids` | `[B, T]` = `[32, 128]` | inteiros de 0 a 96 (IDs de caracteres; $V = 97$) |
| **sai de** `model.hidden_states(ids)` | `[B, T, d]` = `[32, 128, 128]` | um **estado final** de 128 números por posição, com média 0 e norma ~11,3 na inicialização |
| **sai de** `model(ids)` (desde o LM head, [09-lm-head.md](09-lm-head.md)) | `[B, T, V]` = `[32, 128, 97]` | logits: um score por caractere do vocabulário |

**Estado oculto** (*hidden state*) é o vetor de $d$ números que representa uma posição
**dentro** do modelo. "Oculto" porque não é nem a entrada (IDs) nem a saída (logits).
**Estado final** é o estado oculto depois do último bloco e da LayerNorm final.

### Onde este componente fica no pipeline

```text
texto ("Dom Casmurro")
   │
   ▼
tokenizer                      caracteres → IDs                  (01-tokenizer.md)
   │
   ▼
batches                        [B, T] = [32, 128]                (02-dataset.md)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ embeddings                   [32, 128] → [32, 128, 128]      │ ← este documento
│    │                                                         │   (montagem)
│    ▼                                                         │
│ 4× Transformer block         [32, 128, 128] → [32, 128, 128] │
│    │                                                         │
│    ▼                                                         │
│ LayerNorm final              [32, 128, 128]                  │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
LM head                        [32, 128, 128] → [32, 128, 97]    (09-lm-head.md)
   │
   ▼
logits → loss                  compara com os alvos y            (10-loss.md)
```

### O que daria errado

- **Com um único bloco:** o modelo ainda vê todo o contexto, mas não consegue **compor**
  operações. Mecanismos como as induction heads (seção 4), que precisam de uma camada lendo o
  que a anterior escreveu, ficam impossíveis.
- **Sem a LayerNorm final:** o LM head receberia o residual stream cru. A escala desse stream
  cresce **11×** em 50 passos de treino e varia 2,5× entre posições (seção 6). Como os logits
  são proporcionais à entrada, a confiança das previsões mudaria o tempo todo por causa da
  escala, e não do conteúdo.

---

## Notação deste documento

Todas as letras seguem o [guia de notação](00-notacao.md). **Atenção:** neste documento $h$ é
**estado oculto** (um tensor), e **não** o número de heads. Quando o número de heads aparecer,
ele é escrito por extenso ("4 heads") ou como `num_heads`.

### Tamanhos

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $B$ | "bê" | tamanho do batch | 32 | `batch_size` |
| $T$ | "tê" | tamanho do contexto (posições por sequência) | 128 | `context_length`, `seq_len` |
| $V$ | "vê" | tamanho do vocabulário | 97 (vem do tokenizer) | `vocab_size` |
| $d$ | "dê" | largura de cada vetor do modelo | 128 | `d_model` |
| $d_{ff}$ | "dê éfe éfe" | dimensão interna do feed-forward, $4d$ | 512 | `config.d_ff` |
| $L$ | "ele" | número de blocos empilhados | 4 | `num_layers` |
| $N_{\text{params}}$ | "ene params" | número total de parâmetros do modelo | 820.096 | `model.num_parameters()` |

### Dados e estados

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $x$ | "xis" | a sequência de IDs de entrada | `[B, T]` | `ids` |
| $x_t$ | "xis tê" | o ID na posição $t$ | um inteiro de 0 a 96 | `ids[:, t]` |
| $x_{\le t}$ | "xis até tê" | os IDs das posições $0, \dots, t$ | | `ids[:, :t+1]` |
| $t$ | "tê" minúsculo | índice de posição, de $0$ a $T-1$ | 0 a 127 | `t` |
| $\ell$ | "ele cursivo" | índice de camada (bloco): $\ell = 1, \dots, L$ | 1 a 4 (no código, 0 a 3) | `i` em `enumerate(model.blocks)` |
| $h$ | "agá" | **estado oculto** / residual stream | `[B, T, d]` | `h` |
| $h_0$ | "agá zero" | estado depois dos embeddings, antes de qualquer bloco | `[32, 128, 128]`, norma média 0,315 | `model.embeddings(ids)` |
| $h_\ell$ | "agá ele" | estado depois do bloco $\ell$ | `[32, 128, 128]` | `h` depois de `h = block(h)` |
| $h_L$ | "agá ele maiúsculo" | estado depois do último bloco | norma média 0,556 | `stream_before_final_norm(...)` no experimento |
| $\Delta$ | "delta" | o que um bloco somou ao stream: $h_\ell - h_{\ell-1}$ | `[B, T, d]` | `new_h - h` |
| $\|\cdot\|$ | "norma de" | comprimento de um vetor | $\|h_0\| \approx 0{,}315$ | `h.norm(dim=-1)` |
| $\sqrt{d}$ | "raiz de dê" | norma de um vetor com média 0 e variância 1 em $d$ coordenadas | $\sqrt{128} = 11{,}314$ | `d_model**0.5` |

### Parâmetros e módulos

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $E$ | "é" | matriz de token embedding: linha $i$ = vetor do token $i$ | `[V, d]` = `[97, 128]` | `embeddings.token_embedding.weight` |
| $E_x$ | "é índice xis" | as linhas de $E$ escolhidas pelos IDs (um lookup por posição) | `[B, T, d]` | `token_embedding(ids)` |
| $P$ | "pê" | matriz de positional embedding: linha $t$ = vetor da posição $t$ | `[T, d]` = `[128, 128]` | `embeddings.position_embedding.weight` |
| $P_{0..T-1}$ | "pê de zero a tê menos um" | as linhas $0, 1, \dots, T-1$ de $P$ | `[T, d]` | `position_embedding(torch.arange(T))` |
| $\text{Block}_\ell$ | "bloco ele" | o Transformer block número $\ell$ ([07-transformer-block.md](07-transformer-block.md)) | 197.760 parâmetros cada | `model.blocks[ℓ-1]` |
| $\text{LN}_{\text{final}}$ | "éle-éne final" | a LayerNorm depois do último bloco | $2d = 256$ parâmetros | `model.ln_final` |
| $W$ | "dáblio" | matriz do LM head ([09-lm-head.md](09-lm-head.md)); com weight tying, $W = E$ | `[V, d]` | `lm_head.proj.weight` |

### Símbolos da LayerNorm (seções 3, 6 e 7)

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $\mu$ | "mi" | média das $d$ coordenadas de um vetor | um número por posição | `x.mean(dim=-1, keepdim=True)` |
| $\sigma^2$ | "sigma ao quadrado" | variância das $d$ coordenadas de um vetor | ~$2{,}4 \times 10^{-3}$ na entrada de `ln_final`, na inicialização | `var` |
| $\epsilon$ | "épsilon" | número minúsculo somado a $\sigma^2$ para não dividir por zero | $10^{-5}$ | `eps` |
| $\hat x$, $\hat x_i$ | "xis chapéu" | o vetor **normalizado** (média 0, variância ~1); $\hat x_i$ é a coordenada $i$ | `[..., d]` | `x_hat` |
| $\gamma$, $\gamma_i$ | "gama" | escala aprendida, uma por coordenada | começa em 1 | `ln_final.weight` |
| $\beta$, $\beta_i$ | "beta" | deslocamento aprendido, um por coordenada | começa em 0 | `ln_final.bias` |
| $y$, $y_i$ | "ípsilon" | saída da LayerNorm: $y_i = \gamma_i \hat x_i + \beta_i$ | `[..., d]` | retorno de `LayerNorm.forward` |
| $i$ | "i" | índice de coordenada (feature), de 1 a $d$ | 1 a 128 | `[..., i]` |
| $\sum_i$ | "soma sobre i" | somar sobre todas as coordenadas | | `.sum(dim=-1)` |

### Termos do código

| termo | o que significa | onde |
|---|---|---|
| `nn.ModuleList` | lista que registra os módulos dentro dela como submódulos | `GPT.__init__`, seção 10.2 |
| forward hook | função chamada automaticamente logo depois do forward de um módulo | seção 5.2 e 10.8 |
| dispositivo `meta` | "device" do PyTorch em que tensores têm shape e dtype, mas nenhum dado | seção 9 e 10.7 |
| weight tying | usar a mesma matriz para o token embedding e o LM head | seção 10.6 e [09-lm-head.md](09-lm-head.md) |

---

## 0. Onde estamos

Todas as peças estão prontas e testadas isoladamente:

| peça | entrada → saída | doc |
|---|---|---|
| tokenizer + dataset | texto → `[B, T]` | [01](01-tokenizer.md), [02](02-dataset.md) |
| embeddings | `[B, T]` → `[B, T, d]` | [03](03-embeddings.md) |
| Transformer block (attention, feed-forward, LayerNorm, residual) | `[B, T, d]` → `[B, T, d]` | [04](04-attention.md)–[07](07-transformer-block.md) |

Este documento monta o **corpo** do GPT: embeddings, $L$ blocos empilhados e uma LayerNorm final.
O resultado ainda não são probabilidades: é um vetor de $d$ dimensões por posição, o
**estado final** com tudo o que o modelo calculou sobre aquele ponto do texto. Transformar
esse estado em logits sobre o vocabulário é o trabalho do LM head ([09-lm-head.md](09-lm-head.md)).

**Empilhar** blocos significa ligá-los em série: a saída de um é a entrada do próximo. Isso só
funciona porque cada bloco devolve o mesmo shape que recebe.

---

## 1. A arquitetura

As três linhas abaixo descrevem o modelo inteiro. A primeira transforma IDs em vetores, a
segunda aplica os blocos um depois do outro, e a terceira padroniza o resultado.

$$
\begin{aligned}
h_0 &= E_{x} + P_{0..T-1} && \text{embeddings: } [B, T] \to [B, T, d] \\
h_\ell &= \text{Block}_\ell(h_{\ell-1}), \quad \ell = 1, \dots, L && \text{cada bloco: } [B, T, d] \to [B, T, d] \\
\text{saída} &= \text{LN}_{\text{final}}(h_L) && [B, T, d]
\end{aligned}
$$

**Lendo a fórmula:**

- **Linha 1.** $E_x$ troca cada ID pela linha correspondente da matriz $E$ (`[B, T]` →
  `[B, T, d]`). $P_{0..T-1}$ são os vetores das posições $0$ a $T-1$ (`[T, d]`). O $+$ soma o
  vetor da posição $t$ ao vetor do caractere na posição $t$; o mesmo $P$ é somado a todas as
  $B$ sequências (broadcasting). O resultado é $h_0$.
- **Linha 2.** O bloco $\ell$ recebe o estado anterior $h_{\ell-1}$ e devolve $h_\ell$. A
  vírgula "$\ell = 1, \dots, L$" diz que a linha vale para cada bloco: $h_1 = \text{Block}_1(h_0)$,
  $h_2 = \text{Block}_2(h_1)$, e assim por diante até $h_L$. Cada $\text{Block}_\ell$ tem os
  **seus próprios** pesos.
- **Linha 3.** $\text{LN}_{\text{final}}$ normaliza cada vetor de $h_L$ separadamente.
- **Os textos à direita** (`[B, T] → [B, T, d]`) dizem o shape de entrada e de saída.

**Exemplo numérico (ilustrativo, com $d = 2$).** Se o caractere da posição 0 tem
$E_{x_0} = (0{,}01;\ -0{,}02)$ e a posição 0 tem $P_0 = (0{,}03;\ 0{,}00)$, então
$h_0$ na posição 0 vale $(0{,}01 + 0{,}03;\ -0{,}02 + 0{,}00) = (0{,}04;\ -0{,}02)$. Com o
`tiny.yaml`, a conta é a mesma, só que com 128 coordenadas e para as $32 \times 128$ posições
do batch.

```text
ids [B, T]
   │
   ▼
Embeddings (token + posição)      [B, T, d]
   │
   ▼
Block 1                           [B, T, d]
   │
   ▼
Block 2                           [B, T, d]
   │
   ▼
Block 3                           [B, T, d]
   │
   ▼
Block 4                           [B, T, d]
   │
   ▼
LayerNorm final                   [B, T, d]
   │
   ▼
(LM head, 09-lm-head.md)          [B, T, V]
```

Em código, o forward inteiro são quatro linhas (hoje elas formam o método `hidden_states`;
veja a seção 10):

```python
h = self.embeddings(ids)            # [B, T, d]
for block in self.blocks:
    h = block(h)                    # [B, T, d]
return self.ln_final(h)             # [B, T, d]
```

A correspondência com a fórmula é direta: a primeira linha é $h_0$, o `for` é a linha 2
repetida $L$ vezes (a variável `h` é reaproveitada: vale $h_1$, depois $h_2$, …), e o `return`
é a linha 3.

---

## 2. Anatomia

### 2.1 A árvore de módulos

`print(model)` na seção 1 do experimento (o PyTorch agrupa blocos idênticos em `(0-3): 4 x`):

```text
GPT(
  (embeddings): Embeddings(
    (token_embedding): Embedding(97, 128)
    (position_embedding): Embedding(128, 128)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (blocks): ModuleList(
    (0-3): 4 x TransformerBlock(
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
  )
  (ln_final): LayerNorm(128, eps=1e-05)
  (lm_head): LMHead(
    (proj): Linear(in_features=128, out_features=97, bias=False)
  )
)
```

Cada linha corresponde a um componente já visto. O único componente novo aqui é
`ln_final`; o `lm_head` é o assunto de [09-lm-head.md](09-lm-head.md).

Como ler a árvore:

| linha | o que é | doc |
|---|---|---|
| `Embedding(97, 128)` | matriz $E$, `[V, d]` | [03](03-embeddings.md) |
| `Embedding(128, 128)` | matriz $P$, `[T, d]` | [03](03-embeddings.md) |
| `(0-3): 4 x TransformerBlock` | os blocos 0, 1, 2 e 3, com a mesma estrutura e pesos **diferentes** | [07](07-transformer-block.md) |
| `q_proj`, `k_proj`, `v_proj`, `out_proj` | projeções da attention, sem bias | [04](04-attention.md), [05](05-multi-head-attention.md) |
| `up_proj` → `GELU` → `down_proj` | feed-forward $d \to 4d \to d$ | [06](06-feed-forward.md) |
| `ln_1`, `ln_2` | LayerNorms de dentro do bloco (pre-norm) | [07](07-transformer-block.md) |
| `ln_final` | **a novidade deste documento** | aqui |
| `lm_head` | projeção `[d]` → `[V]` | [09](09-lm-head.md) |

### 2.2 Onde estão os parâmetros

```text
  token_embedding         12,416    1.5%   V·d = 97·128
  position_embedding      16,384    2.0%   T·d = 128·128
  blocks[0]              197,760   24.1%   12·d² + 9·d
  blocks[1]              197,760   24.1%   12·d² + 9·d
  blocks[2]              197,760   24.1%   12·d² + 9·d
  blocks[3]              197,760   24.1%   12·d² + 9·d
  ln_final                   256    0.0%   2·d
  total                  820,096  (V·d + T·d + L·(12·d² + 9·d) + 2·d = 820,096)
```

A fórmula abaixo conta **quantos números aprendíveis** o modelo tem. Serve para conferir a
implementação (o teste `test_parameter_count` compara com ela) e para prever o tamanho de
outras configurações sem criar o modelo.

$$
N_{\text{params}} = \underbrace{Vd + Td}_{\text{embeddings}} + \underbrace{L\,(12d^2 + 9d)}_{\text{blocos}} + \underbrace{2d}_{\text{LN final}}
$$

**Lendo a fórmula:**

- $N_{\text{params}}$: total de parâmetros.
- $\underbrace{\ \cdots\ }_{\text{texto}}$: a chave embaixo só **rotula** o grupo; não é uma
  operação.
- $Vd$: a matriz $E$ tem $V$ linhas (uma por caractere) e $d$ colunas.
- $Td$: a matriz $P$ tem $T$ linhas (uma por posição) e $d$ colunas.
- $12d^2 + 9d$: parâmetros de **um** bloco (derivado em [07-transformer-block.md](07-transformer-block.md)):
  - attention: 4 matrizes $d \times d$ sem bias = $4d^2$;
  - feed-forward: $d \cdot 4d + 4d$ (subida, com bias) $+\ 4d \cdot d + d$ (descida, com bias)
    $= 8d^2 + 5d$;
  - 2 LayerNorms: cada uma tem $\gamma$ e $\beta$ com $d$ números = $4d$;
  - soma: $4d^2 + 8d^2 + 5d + 4d = 12d^2 + 9d$.
- $L\,(\cdots)$: os $L$ blocos têm a mesma estrutura, então o número por bloco é multiplicado
  por $L$.
- $2d$: a LayerNorm final tem $\gamma$ ($d$ números) e $\beta$ ($d$ números).
- **O LM head não aparece** porque, com weight tying (padrão), ele usa a própria matriz $E$ e
  não acrescenta parâmetros (seção 10.6).

**Passo a passo com o `tiny.yaml`** ($V = 97$, $T = 128$, $d = 128$, $L = 4$):

| passo | conta | resultado |
|---|---|---|
| 1. token embedding | $Vd = 97 \times 128$ | 12.416 |
| 2. positional embedding | $Td = 128 \times 128$ | 16.384 |
| 3. embeddings (1 + 2) | $12.416 + 16.384$ | 28.800 |
| 4. $d^2$ | $128 \times 128$ | 16.384 |
| 5. $12d^2$ | $12 \times 16.384$ | 196.608 |
| 6. $9d$ | $9 \times 128$ | 1.152 |
| 7. um bloco (5 + 6) | $196.608 + 1.152$ | 197.760 |
| 8. $L$ blocos | $4 \times 197.760$ | 791.040 |
| 9. LN final | $2 \times 128$ | 256 |
| 10. **total** (3 + 8 + 9) | $28.800 + 791.040 + 256$ | **820.096** |

**96,5%** dos parâmetros estão nos blocos. Os embeddings são só 3,5%, porque o vocabulário de
caracteres é minúsculo (seção 9 compara com o GPT-2). Conta: $791.040 / 820.096 = 0{,}9646$ e
$28.800 / 820.096 = 0{,}0351$.

### 2.3 Por que `vocab_size` é um argumento separado

```python
model = GPT(config.model, vocab_size=tokenizer.vocab_size)
```

O `ModelConfig` descreve a **arquitetura** (quantas camadas, qual largura). O tamanho do
vocabulário é uma propriedade do **corpus e do tokenizer**: com outro texto, seriam outros
caracteres. Se `vocab_size` estivesse fixo no YAML, poderia divergir do tokenizer, e um ID
válido para o tokenizer quebraria o `nn.Embedding` (`test_id_outside_vocab_raises`).

Exemplo: se o YAML dissesse `vocab_size: 90` e o tokenizer produzisse o ID 95, a matriz $E$
teria só 90 linhas (0 a 89), e o lookup da linha 95 geraria `IndexError`.

---

## 3. O forward, camada por camada

Seção 2 do experimento, modelo recém-inicializado, batch real `[32, 128]`:

```text
  etapa              shape               norma média   ‖Δ‖ / ‖entrada‖
  ids                (32, 128)                     —                 —
  embeddings         (32, 128, 128)            0.315                 —
  bloco 0            (32, 128, 128)            0.384             0.700
  bloco 1            (32, 128, 128)            0.444             0.583
  bloco 2            (32, 128, 128)            0.500             0.505
  bloco 3            (32, 128, 128)            0.556             0.481
  ln_final           (32, 128, 128)           11.290                 —

  igual a model.hidden_states(ids): True
  √d_model = 11.31
```

O experimento coloca o modelo em `.eval()` (dropout desligado) e executa cada peça à mão, na
ordem de `hidden_states`, medindo duas coisas.

A **norma média** resume o tamanho dos $B \cdot T = 4.096$ vetores de uma etapa num único
número:

$$
\text{norma média}(h) = \frac{1}{B\,T} \sum_{b=1}^{B} \sum_{t=0}^{T-1} \|h_{b,t}\|
$$

**Lendo a fórmula:** $h_{b,t}$ é o vetor de $d$ números da sequência $b$, posição $t$.
$\|h_{b,t}\|$ é o seu comprimento. As duas somas percorrem todas as posições de todas as
sequências, e $\frac{1}{BT}$ transforma a soma em média. No código: `h.norm(dim=-1).mean()`.

A **mudança relativa** mede quanto um bloco alterou o stream, em proporção ao tamanho do stream
que ele recebeu:

$$
\frac{\|\Delta\|}{\|\text{entrada}\|} = \frac{\text{norma média}(h_\ell - h_{\ell-1})}{\text{norma média}(h_{\ell-1})}
$$

**Lendo a fórmula:** $\Delta = h_\ell - h_{\ell-1}$ é exatamente o que o bloco $\ell$ somou ao
stream (as escritas da attention e do feed-forward). O numerador é o tamanho dessa escrita; o
denominador, o tamanho do stream antes do bloco. No código:
`(new_h - h).norm(dim=-1).mean() / h.norm(dim=-1).mean()`.

**Exemplo numérico.** No bloco 0: $0{,}700 \times 0{,}315 \approx 0{,}22$. Ou seja, o bloco 0
escreveu vetores de norma média ~0,22 num stream de norma 0,315. Fazendo a mesma conta para os
outros blocos: $0{,}583 \times 0{,}384 \approx 0{,}22$; $0{,}505 \times 0{,}444 \approx 0{,}22$;
$0{,}481 \times 0{,}500 \approx 0{,}24$. As escritas têm tamanho parecido.

Como ler:

- **O shape nunca muda** depois dos embeddings. É isso que permite empilhar qualquer
  quantidade de blocos.
- **O stream cresce ~0,06 por bloco.** Cada bloco soma suas escritas; com a escala por
  profundidade ([07-transformer-block.md](07-transformer-block.md)), cada escrita é pequena e
  parecida com a anterior.
- **Por que cresce só ~0,06 se a escrita tem ~0,22?** Porque a escrita aponta numa direção quase
  **perpendicular** ao stream (em 128 dimensões, vetores aleatórios são quase perpendiculares).
  Aí as normas se combinam como os lados de um triângulo retângulo (Pitágoras), e não somando
  direto: $\sqrt{0{,}315^2 + 0{,}22^2} \approx 0{,}385$, bem perto dos 0,384 medidos (somar
  direto daria 0,535). Isso é uma aproximação feita com médias, mas confere nos quatro blocos
  (0,444; 0,497; 0,555 contra 0,444; 0,500; 0,556).
- **A mudança relativa cai** (0,70 → 0,48): as escritas têm tamanho parecido, mas o stream
  onde elas são somadas está maior.
- **`ln_final` leva a norma para ~$\sqrt{d}$.** O valor é 11,290 e não 11,314 por causa do
  $\epsilon$ da LayerNorm. A norma do stream é 0,556, então a variância por coordenada é
  $0{,}556^2/128 \approx 2{,}4 \times 10^{-3}$. A saída tem variância
  $\sigma^2/(\sigma^2 + \epsilon) = 0{,}9959$ e norma $\sqrt{0{,}9959} \cdot 11{,}314 = 11{,}290$.
  Com um stream tão pequeno, $\epsilon = 10^{-5}$ já aparece. O teste
  `test_final_layer_norm_standardizes_each_position` usa tolerância de 0,05 por isso.
- **`igual a model.hidden_states(ids): True`** confirma que a execução manual, peça por peça,
  dá o mesmo resultado que o método do modelo.

### 3.1 A conta do $\epsilon$, passo a passo

Por que a norma depois da LayerNorm não é exatamente $\sqrt{128} = 11{,}314$? Com $\gamma = 1$
e $\beta = 0$, a saída é $\hat x$, e a sua norma ao quadrado é:

$$
\|\hat x\|^2 = \sum_{i=1}^{d} \hat x_i^2 = \sum_{i=1}^{d} \frac{(x_i - \mu)^2}{\sigma^2 + \epsilon} = \frac{d\,\sigma^2}{\sigma^2 + \epsilon}
\quad\Rightarrow\quad
\|\hat x\| = \sqrt{d} \cdot \sqrt{\frac{\sigma^2}{\sigma^2 + \epsilon}}
$$

**Lendo a fórmula:**

- $\hat x_i = (x_i - \mu)/\sqrt{\sigma^2 + \epsilon}$ é a coordenada normalizada; ao elevar ao
  quadrado, a raiz some do denominador.
- $\sum_i (x_i - \mu)^2 = d\,\sigma^2$, pela própria definição de variância
  ($\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$).
- Se $\epsilon$ fosse 0, a fração $\sigma^2/(\sigma^2 + \epsilon)$ valeria 1 e a norma seria
  exatamente $\sqrt{d}$. O $\epsilon$ torna a fração um pouco menor que 1.
- A fração só fica visivelmente menor que 1 quando $\sigma^2$ não é muito maior que $\epsilon$.

**Com os números do experimento:**

| passo | conta | resultado |
|---|---|---|
| 1. norma do stream antes de `ln_final` | medido | $0{,}556$ |
| 2. soma dos quadrados das 128 coordenadas | $0{,}556^2$ | $0{,}309$ |
| 3. variância por coordenada (a média é ~0) | $0{,}309 / 128$ | $0{,}002415$ |
| 4. somar $\epsilon$ | $0{,}002415 + 0{,}00001$ | $0{,}002425$ |
| 5. fração | $0{,}002415 / 0{,}002425$ | $0{,}9959$ |
| 6. raiz da fração | $\sqrt{0{,}9959}$ | $0{,}9979$ |
| 7. $\sqrt{d}$ | $\sqrt{128}$ | $11{,}314$ |
| 8. **norma da saída** | $0{,}9979 \times 11{,}314$ | **11,290** |

O passo 3 é uma aproximação: usa a norma **média** e supõe média das coordenadas ~0. Mesmo
assim, reproduz o valor medido. Depois do treino, com o stream de norma ~6,5, a fração vira
$0{,}99997$ e a norma volta a $11{,}314$: o efeito do $\epsilon$ desaparece (conta feita em
Python para este documento).

---

## 4. O que a profundidade acrescenta

### 4.1 Não é "enxergar mais longe"

O **receptive field** (campo receptivo) de uma posição é o conjunto de entradas que podem
influenciar a sua saída.

Numa rede convolucional, cada camada amplia o campo de visão: são precisas várias camadas para
um pixel "ver" pixels distantes. **Num Transformer, uma única camada já enxerga todo o
contexto**: a attention da posição $t$ lê diretamente as posições $0, \dots, t$. Empilhar
blocos não aumenta o alcance.

Exemplo: numa convolução 1D com janela de 3, cada camada olha 1 vizinho de cada lado. Uma camada
vê 3 entradas; duas camadas, 5; três, 7. Para ver 128 posições seriam precisas dezenas de
camadas. No Mini-GPT, a posição 127 já lê as posições 0 a 127 no **primeiro** bloco.

### 4.2 É compor operações

O que a profundidade dá é **composição**: o bloco 2 lê o stream **depois** de o bloco 1 ter
escrito nele. Uma head da camada 2 pode, portanto, procurar por informações que não estavam
nos embeddings e que só existem porque uma head da camada 1 as colocou lá.

**Composição**, em matemática, é aplicar uma função sobre o resultado de outra: $g(f(x))$. Aqui,
$\text{Block}_2(\text{Block}_1(h_0))$: o segundo bloco trabalha sobre o que o primeiro produziu.

O exemplo clássico são as **induction heads** (Olsson et al., 2022), um mecanismo que modelos
treinados aprendem para continuar padrões que já apareceram no contexto. Imagine o texto
`"... o gato dorme ... o ga"`: qual o próximo caractere?

1. **Camada 1: previous-token head** (a head construída à mão em [04-attention.md](04-attention.md)). Cada posição copia
   para o seu vetor "o caractere antes de mim era $c$". Depois dela, a posição do primeiro
   `'t'` de "gato" carrega a informação "antes de mim veio `'a'`".
2. **Camada 2: induction head.** A posição atual (`'a'`, no final) faz a query "procuro
   posições cujo caractere anterior era `'a'`". A key do primeiro `'t'` responde a essa
   pergunta, graças ao que a camada 1 escreveu. A head copia o value dessa posição: `'t'`.
   Resultado: prever `'t'`.

Definições:

- **Previous-token head:** uma head de attention em que cada posição $t$ olha para a posição
  $t - 1$ e copia o caractere de lá. Aqui, $c$ é esse caractere anterior.
- **Induction head:** uma head que procura, no passado, o lugar onde o caractere atual já
  apareceu e copia **o que veio logo depois** dele. Em resumo: "se já vi `a` seguido de `t`,
  depois de `a` provavelmente vem `t`".

**Exemplo com posições.** Tomando o texto concreto `"o gato dorme o ga"` (17 caracteres,
posições 0 a 16):

| $t$ | caractere | caractere anterior (escrito pela camada 1) |
|---:|:---:|:---:|
| 2 | `g` | `' '` |
| 3 | `a` | `g` |
| **4** | **`t`** | **`a`** ← a única posição cujo anterior é `a` |
| 5 | `o` | `t` |
| … | … | … |
| 15 | `g` | `' '` |
| **16** | **`a`** (posição atual) | `g` |

Na camada 2, a posição 16 (caractere `a`) procura "anterior = `a`". Só a posição 4 combina.
A head lê o caractere da posição 4, que é `t`, e escreve no stream da posição 16: "o próximo
deve ser `t`".

**Com uma camada só, isso é impossível.** Uma key da camada 1 só pode anunciar o que está nos
embeddings: o **próprio** caractere e a **própria** posição. Ela não tem como anunciar "o
caractere antes de mim", porque essa informação ainda não foi movida para lá.

Esse é o sentido de "circuitos": **attention de uma camada, lida pela attention da seguinte**.
Com 4 blocos, o Mini-GPT tem 16 heads e 4 feed-forwards para montar esse tipo de composição
durante o treino.

Um **circuito** é um conjunto de componentes (heads, neurônios) de camadas diferentes que,
juntos, implementam um comportamento reconhecível, como a previous-token head + induction head
acima. O vocabulário vem da linha de pesquisa de interpretabilidade que descreveu as induction
heads em modelos pequenos, só de attention (Elhage et al., 2021, *A Mathematical Framework for
Transformer Circuits*).

**Importante:** o Mini-GPT não recebe essas heads prontas. A arquitetura só **permite** que
elas surjam; se surgem ou não depende do treino.

---

## 5. O modelo inteiro é causal

**Causal** significa: a saída na posição $t$ depende só das entradas $x_0, \dots, x_t$, nunca das
posições futuras. Em [04-attention.md](04-attention.md) isso foi garantido para uma attention.
Agora é preciso mostrar que continua valendo depois de embeddings, 4 blocos e LayerNorm final.

### 5.1 A prova

**O que é uma prova por indução.** É uma forma de provar que algo vale para **todos** os
números $0, 1, 2, \dots$ em dois passos:

1. **Base:** mostrar que vale para o primeiro caso ($\ell = 0$).
2. **Passo:** mostrar que, **se** vale para um caso qualquer $\ell - 1$ (a *hipótese de
   indução*), **então** vale para o seguinte, $\ell$.

É como uma fila de dominós: a base derruba o primeiro, e o passo garante que cada dominó
derruba o próximo. Logo, todos caem. Aqui, "vale para $\ell$" quer dizer "$h_\ell$ é causal".

Por indução nas camadas:

- **$h_0$** é causal: o embedding da posição $t$ depende só de $x_t$ e de $t$.
- **Se $h_{\ell-1}$ é causal, $h_\ell$ também é.** A attention mascarada da posição $t$ lê
  $h_{\ell-1}$ nas posições $\le t$, que só dependem de $x_{\le t}$. LayerNorm e feed-forward
  atuam por posição. A soma residual combina valores da mesma posição.
- **`ln_final`** atua por posição.

Logo, a saída na posição $t$ depende só de $x_0, \dots, x_t$.

**Onde a hipótese de indução é usada.** No passo, a attention da posição $t$ lê os vetores
$h_{\ell-1}$ das posições $j \le t$. A máscara garante $j \le t$, mas isso não basta: é preciso
saber que **o conteúdo** desses vetores não traz informação do futuro. É exatamente a hipótese
"$h_{\ell-1}$ é causal" que diz que o vetor na posição $j$ só depende de $x_{\le j}$, e como
$j \le t$, de $x_{\le t}$.

**Passo detalhado, dentro de um bloco** ($x' = x + \text{MHA}(\text{LN}_1(x))$, depois
$x + \text{FFN}(\text{LN}_2(x'))$, [07-transformer-block.md](07-transformer-block.md)):

| operação | lê quais posições? | continua causal? |
|---|---|---|
| $\text{LN}_1$ | só a posição $t$ | sim |
| attention mascarada | posições $j \le t$, que por hipótese dependem de $x_{\le j}$ | sim |
| soma residual | posição $t$ com posição $t$ | sim |
| $\text{LN}_2$ e feed-forward | só a posição $t$ | sim |
| soma residual | posição $t$ com posição $t$ | sim |

### 5.2 A verificação

Seção 3 do experimento, com os 4 blocos:

```text
A = 'o gato dorme', B = 'o gato corre'; 4 blocos
maior |saída_A − saída_B| em cada posição:
  t=0  'o' 'o'  0.00e+00
  ...
  t=6  ' ' ' '  0.00e+00
  t=7  'd' 'c'  2.74e+00   <- entrada diferente
  t=8  'o' 'o'  3.01e-01
  t=9  'r' 'r'  2.63e-01
  t=10 'm' 'r'  3.25e+00   <- entrada diferente
  t=11 'e' 'e'  2.63e-01

o gradiente da saída na posição 5 chega aos embeddings das posições: [0, 1, 2, 3, 4, 5]
```

Posições 0 a 6: diferença **exatamente zero** mesmo depois de 4 blocos. A última linha usa um
*forward hook* para capturar a saída dos embeddings e mostra que o gradiente da posição 5 chega
às posições 0–5, e a nenhuma posterior.

Como ler:

- **Primeira parte.** Duas frases iguais até a posição 6 e diferentes a partir da 7. A coluna
  mostra o maior $|\text{saída}_A - \text{saída}_B|$ entre as 128 coordenadas de cada posição.
  Posições 0 a 6 dão `0.00e+00`: o futuro diferente não vazou para trás. As posições 8, 9 e 11
  têm o mesmo caractere nas duas frases, mas diferem (~0,3) porque **leem** as posições 7 e 10.
- **Segunda parte.** Teste pelo gradiente: pede-se "quanto a saída da posição 5 muda se cada
  embedding mudar". Só as posições 0 a 5 recebem gradiente diferente de zero.

**Como o forward hook funciona nessa medição** (`experiments/e08_gpt.py`, linhas 110–122):

1. `model.embeddings.register_forward_hook(keep_embeddings)` pede ao PyTorch: "sempre que
   `embeddings` terminar o forward, chame `keep_embeddings(module, inputs, output)`".
2. Dentro da função, `output.retain_grad()` pede para guardar o gradiente desse tensor
   intermediário (por padrão, o PyTorch só guarda gradientes de parâmetros), e
   `captured["embeddings"] = output` guarda uma referência a ele.
3. `handle.remove()` desliga o hook depois do forward, para não afetar outros usos do modelo.
4. `(out[0, 5] * torch.randn(d)).sum().backward()` cria uma loss que depende só da posição 5
   (com pesos aleatórios, pelo motivo da seção 7).
5. `captured["embeddings"].grad[0].norm(dim=-1)` mede o tamanho do gradiente em cada posição;
   as posições com valor $> 0$ são as listadas.

A vantagem do hook: dá para observar um valor **de dentro** do modelo sem alterar o código de
`GPT`.

### 5.3 E entre sequências do batch

O modelo também não mistura **sequências diferentes** do batch: nenhuma operação (attention,
LayerNorm, feed-forward) calcula estatísticas ao longo da dimensão $B$. Trocar a sequência 1
não altera a saída da sequência 0 (`test_sequences_in_a_batch_are_independent`). Uma
BatchNorm quebraria essa propriedade ([07-transformer-block.md](07-transformer-block.md), seção 2.5).

Exemplo: a LayerNorm calcula $\mu$ e $\sigma^2$ sobre as 128 coordenadas **de um vetor**
(`dim=-1`). Uma BatchNorm calcularia sobre as 32 sequências do batch, e a sequência 0 passaria a
depender das outras 31.

---

## 6. Por que a LayerNorm final

### 6.1 O argumento

No pre-norm ([07-transformer-block.md](07-transformer-block.md)), cada subcamada **lê** o stream
através de uma LayerNorm, mas o stream em si **nunca** é normalizado. O LM head
([09-lm-head.md](09-lm-head.md)) vai ler esse stream. Sem uma normalização antes, a escala da
entrada do LM head seria a escala do stream, e ela muda.

A LayerNorm final é a mesma operação do Transformer block, aplicada uma vez a $h_L$. Ela padroniza
cada vetor e depois aplica uma escala e um deslocamento aprendidos:

$$
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_i = \gamma_i\, \hat x_i + \beta_i
$$

**Lendo a fórmula:**

- $x$ é um vetor de $h_L$ (uma posição de uma sequência), com coordenadas $x_1, \dots, x_d$.
- $\mu = \frac{1}{d}\sum_i x_i$ é a média das suas coordenadas; $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$
  é a variância.
- $x_i - \mu$ centraliza (média passa a ser 0). Dividir por $\sqrt{\sigma^2 + \epsilon}$ ajusta a
  escala (variância passa a ser ~1). O resultado é $\hat x$.
- $\epsilon = 10^{-5}$ evita divisão por zero se todas as coordenadas forem iguais.
- $\gamma_i$ multiplica e $\beta_i$ soma, coordenada a coordenada. Começam em 1 e 0, então no
  início $y = \hat x$.
- No código (`model/layer_norm.py`, linhas 25–29): `mean`, `var`, `x_hat` e
  `self.weight * x_hat + self.bias`.

**Exemplo numérico** (o mesmo de [07-transformer-block.md](07-transformer-block.md), com $d = 6$): $x = (2, 4, 4, 4, 5, 7)$ tem
$\mu = 4{,}333$ e $\sigma^2 = 2{,}222$. Então
$\hat x = (-1{,}565;\ -0{,}224;\ -0{,}224;\ -0{,}224;\ 0{,}447;\ 1{,}789)$. Primeira coordenada:
$(2 - 4{,}333)/\sqrt{2{,}222} = -2{,}333/1{,}491 = -1{,}565$.

**O ponto central:** se $x$ fosse $10 \times$ maior, $\mu$ e $\sqrt{\sigma^2}$ também seriam
$10 \times$ maiores, e $\hat x$ seria o mesmo. A LayerNorm **remove a escala** da entrada.

### 6.2 A medição

Seção 5 do experimento. Norma do stream por posição antes de `ln_final` (média, mínima e
máxima no batch) e norma média depois dela:

```text
na inicialização, variando a profundidade:
      L |    média      mín      máx |  depois
      1 |    0.545    0.436    0.777 |  11.289
      4 |    0.556    0.457    0.741 |  11.290
     16 |    0.562    0.453    0.766 |  11.290

durante um mini-treino (tiny.yaml + Linear(d_model, vocab) provisório, AdamW lr 1e-3):
  passo |    média      mín      máx |  depois |   loss
      0 |    0.556    0.457    0.741 |  11.290 |  4.714
     50 |    6.097    3.478    7.908 |  11.764 |  2.463
    100 |    6.323    3.540    8.041 |  11.971 |  2.376
    150 |    6.512    3.592    8.823 |  12.114 |  2.311
```

Como o experimento foi montado:

- **Primeira tabela:** cria modelos com $L = 1$, 4 e 16 blocos e mede $\|h_L\|$ por posição
  (média, mínimo e máximo sobre as 4.096 posições do batch) e a norma média depois de
  `ln_final`.
- **Segunda tabela:** treina o modelo do `tiny.yaml` por 150 passos com um `nn.Linear`
  provisório no lugar do LM head e mede as mesmas normas nos passos 0, 50, 100 e 150. A coluna
  `loss` é a cross-entropy daquele passo ([10-loss.md](10-loss.md)).

Leitura:

- **Na inicialização a profundidade quase não muda a escala** (0,545 com 1 bloco, 0,562 com
  16), graças à escala $1/\sqrt{2L}$ de [07-transformer-block.md](07-transformer-block.md).
- **O treino muda muito a escala.** Em 50 passos a norma média do stream cresce **11×**
  (0,56 → 6,10), à medida que as subcamadas aprendem a escrever vetores maiores. Entre
  posições do mesmo batch, a norma varia 2,5× (3,6 a 8,8).
- **Depois de `ln_final`, a escala fica estável** (11,3 a 12,1). A pequena variação vem de
  $\gamma$, que é aprendido: o modelo ajusta a escala por conta própria, mas de forma
  controlada.

Contas: $6{,}097 / 0{,}556 \approx 10{,}97$ (o "11×"); $8{,}823 / 3{,}592 \approx 2{,}46$ (o "2,5×",
no passo 150).

Sem a LayerNorm final, o LM head veria entradas cuja escala muda 11× em poucos passos e varia
entre posições. Como os logits são proporcionais à entrada, isso equivaleria a mudar a
"temperatura" da softmax o tempo todo. Por isso o GPT-2 introduziu essa LayerNorm quando
adotou o pre-norm.

**O que "temperatura" quer dizer aqui.** Os logits são $z_i = h \cdot W_i$ ([09-lm-head.md](09-lm-head.md)). Se $h$
dobra, todos os $z_i$ dobram, e a softmax fica mais "confiante" (concentra probabilidade no
maior logit) sem que o conteúdo de $h$ tenha mudado. Exemplo do guia de notação:
$\text{softmax}(2, 1, 0) = (0{,}665;\ 0{,}245;\ 0{,}090)$. Com $h$ dobrado os logits seriam
$(4, 2, 0)$, e a maior probabilidade subiria para cerca de 0,867.

---

## 7. Uma armadilha: `out.sum()` depois da LayerNorm

Ao escrever testes de gradiente, é tentador usar `model(ids).sum().backward()` (antes do LM head,
`model(ids)` devolvia os estados finais; hoje o equivalente é
`model.hidden_states(ids).sum().backward()`). Com uma LayerNorm no final, isso não funciona.
Seção 4 do experimento, que usa `model.hidden_states(x)`:

```text
  out.sum()                        max |grad| token_embedding 1.98e-04 | ln_final.bias 4.10e+03
  (out · pesos aleatórios).sum()   max |grad| token_embedding 2.81e+03 | ln_final.bias 1.91e+02
```

As duas linhas usam o mesmo modelo e o mesmo batch; só muda a "loss" de teste:

- `out.sum()`: soma todos os $32 \times 128 \times 128$ números da saída;
- `(out · pesos aleatórios).sum()`: multiplica cada número por um peso aleatório antes de somar.

Com `out.sum()`, o gradiente que chega aos embeddings é ~$10^{-4}$, arredondamento numérico,
contra ~$10^{3}$ com uma loss de pesos aleatórios. **Por quê:** o vetor normalizado tem média
zero por construção, $\sum_i \hat x_i = 0$. Com $\gamma = 1$,

$$
\sum_i y_i = \sum_i (\hat x_i + \beta_i) = \sum_i \beta_i
$$

**Lendo a fórmula:**

- $\sum_i y_i$: a soma das $d$ coordenadas da saída de **uma** posição.
- $y_i = \gamma_i \hat x_i + \beta_i$ com $\gamma_i = 1$ vira $\hat x_i + \beta_i$.
- A soma se separa em $\sum_i \hat x_i + \sum_i \beta_i$.
- $\sum_i \hat x_i = 0$ **sempre**: $\hat x$ é $x - \mu$ dividido por um número, e
  $\sum_i (x_i - \mu) = \sum_i x_i - d\,\mu = d\,\mu - d\,\mu = 0$.
- Sobra só $\sum_i \beta_i$, que não contém $x$.

**Exemplo numérico.** Com $x = (2, 4, 4, 4, 5, 7)$, $\hat x = (-1{,}565;\ -0{,}224;\ -0{,}224;\ -0{,}224;\ 0{,}447;\ 1{,}789)$,
e a soma é $0$. Trocando o último valor, $x = (2, 4, 4, 4, 5, 8)$, obtém-se
$\hat x = (-1{,}387;\ -0{,}277;\ -0{,}277;\ -0{,}277;\ 0{,}277;\ 1{,}941)$: cada coordenada mudou,
mas a soma continua $0$. Com $\beta_i = 0{,}1$ em todas as 6 coordenadas, $\sum_i y_i = 0{,}6$
nos dois casos (conferido em Python).

A soma da saída **não depende da entrada** da LayerNorm, só de $\beta$. O gradiente para tudo
que vem antes é zero. E o gradiente de $\beta$ é o número de posições somadas:
$B \cdot T = 32 \cdot 128 = 4{.}096 \approx 4{,}10 \times 10^3$.

**Por que "não depende" implica "gradiente zero".** O gradiente mede quanto a loss muda quando a
entrada muda um pouco (guia de notação, seção 7). Se a loss é constante em relação a $x$, a
mudança é zero, e pela regra da cadeia esse zero se propaga para todos os parâmetros antes da
LayerNorm: blocos e embeddings. O $1{,}98 \times 10^{-4}$ medido não é um sinal real; é o erro de
arredondamento das contas em ponto flutuante.

**Por que o gradiente de $\beta$ é 4.096.** Cada $\beta_i$ aparece somado uma vez em **cada**
posição. A loss é a soma sobre $B \cdot T$ posições, então $\partial(\text{loss}) / \partial\beta_i = 1 + 1 + \dots + 1 = B \cdot T = 4.096$.

**Por que a loss com pesos aleatórios funciona.** Com pesos diferentes $r_i$, a loss vira
$\sum_i r_i \hat x_i$, que **muda** quando a direção de $\hat x$ muda. Por isso os testes
(`test_gradients_reach_all_parameters`) e a verificação da seção 5.2 usam
`(out * torch.randn_like(out)).sum()`.

A lição vale além dos testes: **uma loss precisa depender da direção dos vetores, não só da
sua soma.** A cross-entropy ([10-loss.md](10-loss.md)) depende, porque compara logits entre tokens.

---

## 8. Integração com o `train.py`

Saída do `uv run python train.py` **antes do LM head existir**, quando o modelo ainda terminava
nos estados finais:

```text
PyTorch 2.14.0 | device: cpu
corpus:     dom_casmurro.txt (374,785 caracteres)
vocab_size: 97
treino:     337,307 tokens | 2,635 janelas | 82 batches/época
validação:  37,478 tokens | 292 janelas | 9 batches
batch:      x (32, 128), y (32, 128)
modelo:     GPT com 820,096 parâmetros
forward:    ids (32, 128) -> estados finais (32, 128, 128)
LM head ainda não implementado (logits mais adiante) — nada para treinar.
```

O pipeline texto → tokenizer → batches → modelo funciona de ponta a ponta. Falta a peça que
compara a saída com `y`.

Linha a linha:

| linha | o que mostra |
|---|---|
| `corpus`, `vocab_size` | o texto tem 374.785 caracteres, 97 deles distintos ($V = 97$) |
| `treino`, `validação` | divisão 90%/10% dos tokens; janelas de $T = 128$ |
| `batch` | `x` e `y` com shape `[B, T]` = `[32, 128]`; `y` é `x` deslocado uma posição |
| `modelo` | $N_{\text{params}} = 820.096$, o total da seção 2.2 |
| `forward` | `[32, 128]` → `[32, 128, 128]`: um estado final de $d = 128$ números por posição |

**Hoje (depois do LM head, [09-lm-head.md](09-lm-head.md))**, a mesma execução termina assim
(rodado para este documento):

```text
modelo:     GPT com 820,096 parâmetros (com weight tying)
forward:    ids (32, 128) -> logits (32, 128, 97)
Loss e training loop ainda não implementados — nada para treinar.
```

O número de parâmetros não mudou porque o LM head reusa a matriz $E$ (seção 10.6). O shape
final passou de $d = 128$ para $V = 97$: um logit por caractere do vocabulário.

---

## 9. Escala: do Mini-GPT ao GPT-2 small

Seção 6 do experimento, contando parâmetros com a **mesma** classe `GPT`:

```text
                       V     T    d   L |  embeddings      blocos        total
  tiny.yaml           97   128  128   4 |      28,800     791,040      820,096
  d = 256, L = 8      97   256  256   8 |      90,368   6,309,888    6,400,768
  GPT-2 small      50257  1024  768  12 |  39,383,808  85,017,600  124,402,944
```

**Conferindo a linha do GPT-2 small com a fórmula** ($V = 50.257$, $T = 1.024$, $d = 768$,
$L = 12$):

| parte | conta | resultado |
|---|---|---|
| token embedding | $50.257 \times 768$ | 38.597.376 |
| positional embedding | $1.024 \times 768$ | 786.432 |
| embeddings | soma das duas | 39.383.808 |
| um bloco | $12 \times 768^2 + 9 \times 768 = 7.077.888 + 6.912$ | 7.084.800 |
| blocos | $12 \times 7.084.800$ | 85.017.600 |
| LN final | $2 \times 768$ | 1.536 |
| **total** | $39.383.808 + 85.017.600 + 1.536$ | **124.402.944** |

**Conferindo com o GPT-2 real.** O GPT-2 small oficial tem 124.439.808 parâmetros. A diferença
é de exatamente $36{.}864 = 12 \times 4 \cdot 768$: os biases das projeções da attention, que
optamos por não usar ([04-attention.md](04-attention.md)). O LM head do GPT-2 reaproveita a
matriz do token embedding (*weight tying*), por isso não soma parâmetros; é a decisão de
[09-lm-head.md](09-lm-head.md). **Ou seja, a arquitetura do Mini-GPT é a do GPT-2, em miniatura.**

Lendo a conta $12 \times 4 \cdot 768$: 12 blocos; em cada um, 4 vetores de bias (query, key,
value e saída) com 768 números cada.

**Onde ficam os parâmetros muda com a escala:**

| | embeddings | blocos |
|---|---:|---:|
| Mini-GPT | 3,5% | 96,5% |
| GPT-2 small | 31,7% | 68,3% |

Com um vocabulário BPE de 50 mil tokens, só o token embedding tem 38,6 milhões de parâmetros.

Por quê: os embeddings crescem com $V$ e $T$ (linear em $d$), enquanto os blocos crescem com
$d^2$. No Mini-GPT, $V = 97$ é pequeno; no GPT-2, $V = 50.257$ é enorme. Só o token embedding
é $38.597.376 / 124.402.944 \approx 31\%$ do GPT-2 small, contra $12.416 / 820.096 \approx 1{,}5\%$
do Mini-GPT.

**O dispositivo `meta`.** Criar o GPT-2 small na memória custaria ~500 MB. Dentro de
`with torch.device("meta"):` o PyTorch cria tensores só com shape e dtype, sem alocar dados:
dá para contar parâmetros de modelos de qualquer tamanho instantaneamente.

De onde vêm os ~500 MB: cada parâmetro em `float32` ocupa 4 bytes, e
$124{,}4 \text{ milhões} \times 4 \approx 498$ MB. Detalhes do `meta` na seção 10.7.

---

## 10. Código, parte por parte

O código atual de [`model/gpt.py`](../model/gpt.py), já com as adições do LM head ([09-lm-head.md](09-lm-head.md)):

```python
class GPT(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(
            vocab_size, config.context_length, config.d_model, config.dropout
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.d_model, config.num_heads, config.d_ff, config.dropout, config.num_layers
            )
            for _ in range(config.num_layers)
        )
        # No pre-norm o residual stream nunca é normalizado; esta LayerNorm padroniza a saída.
        self.ln_final = LayerNorm(config.d_model)
        self.lm_head = LMHead(config.d_model, vocab_size)
        if config.tie_weights:
            # Mesma matriz [vocab_size, d_model]: a linha i representa o token i na entrada
            # (lookup) e pontua o token i na saída (produto escalar).
            self.lm_head.proj.weight = self.embeddings.token_embedding.weight

    def hidden_states(self, ids: torch.Tensor) -> torch.Tensor:
        """IDs [batch, seq_len] -> estados finais [batch, seq_len, d_model]."""
        h = self.embeddings(ids)  # [batch, seq_len, d_model]
        for block in self.blocks:
            h = block(h)  # [batch, seq_len, d_model]
        return self.ln_final(h)  # [batch, seq_len, d_model]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [batch, seq_len]
        return self.lm_head(self.hidden_states(ids))  # [batch, seq_len, vocab_size]

    def num_parameters(self) -> int:
        # parameters() não repete tensores compartilhados: com tying, a matriz conta uma vez.
        return sum(p.numel() for p in self.parameters())
```

A versão de antes do LM head era esta; o seu `forward` é o `hidden_states` de hoje:

```python
class GPT(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        self.embeddings = Embeddings(vocab_size, config.context_length, config.d_model, config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout, config.num_layers)
            for _ in range(config.num_layers)
        )
        self.ln_final = LayerNorm(config.d_model)

    def forward(self, ids):              # [B, T]
        h = self.embeddings(ids)         # [B, T, d]
        for block in self.blocks:
            h = block(h)                 # [B, T, d]
        return self.ln_final(h)          # [B, T, d]
```

### 10.1 `GPT.__init__`: criar as peças

**Objetivo:** construir e registrar todos os submódulos a partir da config. Nenhum cálculo
acontece aqui; só criação de pesos.

Passo a passo (linhas 19–37 de `model/gpt.py`):

1. `super().__init__()` inicializa o `nn.Module`. Sem isso, o PyTorch não consegue registrar
   submódulos.
2. `self.config = config` guarda a config para consulta posterior.
3. `self.embeddings = Embeddings(...)` cria $E$ (`[V, d]`) e $P$ (`[T, d]`), ambos com
   $\mathcal{N}(0,\ 0{,}02^2)$, e o dropout de entrada ([03-embeddings.md](03-embeddings.md)).
4. `self.blocks = nn.ModuleList(...)` cria `num_layers` blocos **independentes** (seção 10.2).
5. `self.ln_final = LayerNorm(config.d_model)` cria $\gamma = 1$ e $\beta = 0$ com $d$ números
   cada (seção 10.5).
6. `self.lm_head = LMHead(...)` cria a projeção `[d] → [V]` ([09-lm-head.md](09-lm-head.md)).
7. `if config.tie_weights:` faz o LM head usar a matriz $E$ (seção 10.6).

Detalhes:

- **Cada bloco recebe `num_layers`** para a inicialização por profundidade
  ([07-transformer-block.md](07-transformer-block.md)): as matrizes que escrevem no stream
  (`out_proj` e `down_proj`) usam desvio $0{,}02/\sqrt{2L}$. Com $L = 4$: $0{,}02/\sqrt{8} \approx 0{,}00707$.
- **`config.d_ff`** não está no YAML: é uma propriedade de `ModelConfig` que vale
  $4 \times d = 512$.
- **`vocab_size` vem de fora** da config (seção 2.3).

### 10.2 `nn.ModuleList`: guardar os blocos

**Objetivo:** guardar os $L$ blocos numa lista que o PyTorch **enxerga**.

- **`nn.ModuleList`, e não uma lista Python.** Só assim o PyTorch registra os parâmetros dos
  blocos: eles aparecem em `model.parameters()`, vão para a GPU com `model.to(device)` e entram
  no `state_dict` dos checkpoints.
- **`nn.ModuleList` em vez de `nn.Sequential`.** O forward fica explícito e é fácil
  inspecionar ou instrumentar um bloco de cada vez, como fazem os experimentos.
- **Cada bloco recebe `num_layers`** para a inicialização por profundidade ([07-transformer-block.md](07-transformer-block.md)).

O que aconteceria com uma lista comum (`self.blocks = [TransformerBlock(...) for ...]`): o
forward ainda rodaria, mas `model.parameters()` não incluiria os 791.040 parâmetros dos blocos.
O otimizador não os treinaria, `model.to("cuda")` os deixaria na CPU, e o checkpoint não os
salvaria. `num_parameters()` devolveria só 28.800 + 256 = 29.056.

O gerador `TransformerBlock(...) for _ in range(config.num_layers)` chama o construtor 4 vezes.
Cada chamada sorteia pesos novos: são 4 blocos com a mesma estrutura, mas pesos diferentes
(não é o mesmo bloco repetido).

### 10.3 `hidden_states`: do ID ao estado final

**Objetivo:** executar a arquitetura da seção 1 e devolver os estados finais `[B, T, d]`.

Passo a passo (linhas 39–44):

1. `h = self.embeddings(ids)`: `[B, T]` → `[B, T, d]`. Aqui também é verificado que
   $T \le$ `context_length` (senão, `ValueError`). É $h_0$.
2. `for block in self.blocks: h = block(h)`: aplica os blocos na ordem 0, 1, 2, 3. A cada volta,
   `h` passa de $h_{\ell-1}$ para $h_\ell$. O shape continua `[B, T, d]`.
3. `return self.ln_final(h)`: normaliza cada um dos $B \cdot T$ vetores de $h_L$.

Uso:

```python
from model import GPT

model = GPT(config.model, vocab_size=tokenizer.vocab_size)
model.num_parameters()      # 820_096
hidden = model(x)           # x: [B, T] int64  ->  hidden: [B, T, 128]
```

Esse era o uso antes do LM head. **Hoje**, `model(x)` devolve logits; os estados finais vêm de
`hidden_states`:

```python
hidden = model.hidden_states(x)   # x: [B, T] int64  ->  hidden: [B, T, 128]
logits = model(x)                 # x: [B, T] int64  ->  logits: [B, T, 97]
```

### 10.4 `forward`: estados finais → logits

**Objetivo:** ser o que roda quando se chama `model(ids)`. Desde o LM head
([09-lm-head.md](09-lm-head.md)), é uma linha (linhas 46–48):
`self.lm_head(self.hidden_states(ids))`, `[B, T]` → `[B, T, d]` → `[B, T, V]`.

Por que separar em dois métodos: o treino precisa dos logits (`model(ids)`), mas os testes e
experimentos deste documento precisam dos estados finais (`hidden_states`). O teste
`test_matches_manual_composition` verifica os dois: `hidden_states` = embeddings → blocos →
`ln_final`, e `model(ids)` = `lm_head(hidden_states)`.

Chamar `model(ids)`, e não `model.forward(ids)`, é o jeito certo no PyTorch: `__call__` executa
também os hooks registrados (seção 10.8).

### 10.5 `ln_final`: a LayerNorm de saída

**Objetivo:** entregar ao LM head vetores de escala estável (seção 6).

- É a classe `LayerNorm` de [07-transformer-block.md](07-transformer-block.md) (`model/layer_norm.py`), igual a `nn.LayerNorm(d_model)`.
- `weight` é $\gamma$ (começa em 1) e `bias` é $\beta$ (começa em 0): $2d = 256$ parâmetros.
- `eps=1e-05` é o $\epsilon$ da seção 3.1.
- Atua em `dim=-1`: cada posição de cada sequência é normalizada sozinha, o que preserva a
  causalidade e a independência entre sequências.

### 10.6 `tie_weights` (resumo; detalhes em [09-lm-head.md](09-lm-head.md))

**Objetivo:** usar a mesma matriz `[V, d]` como token embedding (entrada) e como pesos do LM
head (saída).

- A linha 37, `self.lm_head.proj.weight = self.embeddings.token_embedding.weight`, **não copia**
  valores: os dois módulos passam a apontar para o mesmo `Parameter`.
- `model.parameters()` não repete tensores compartilhados. Por isso `num_parameters()` conta a
  matriz uma vez, e o total continua 820.096.
- Sem tying (`tie_weights: false`), o LM head teria a sua própria matriz: $820.096 + 97 \times 128 = 832.512$
  parâmetros ([09-lm-head.md](09-lm-head.md), seção 4.3).
- Nos gradientes, as contribuições dos dois usos são somadas no mesmo tensor.

### 10.7 `num_parameters` e o dispositivo `meta`

**Objetivo de `num_parameters`** (linhas 50–52): somar `p.numel()` (quantidade de números de um
tensor) para todo parâmetro. Exemplo: $E$ tem shape `(97, 128)`, então `numel()` = 12.416.

**Dispositivo `meta`.** Um *device* diz onde os dados de um tensor moram (`cpu`, `cuda`, …). O
`meta` é um device "de mentira": o tensor guarda shape, dtype e device, mas **nenhum número**.

- `with torch.device("meta"):` faz todo tensor criado dentro do bloco nascer em `meta`.
- `GPT(cfg, vocab_size)` roda normalmente: shapes são calculados, os módulos são registrados, e
  `num_parameters()` funciona, porque `numel()` só precisa do shape.
- A inicialização (`nn.init.normal_`) não gera números de verdade; não há memória para eles.
- Não dá para rodar um forward útil nem ler valores: não existem.

No experimento (`experiments/e08_gpt.py`, linhas 219–220), é assim que o GPT-2 small é "criado"
sem gastar ~500 MB.

### 10.8 Forward hooks (usados no experimento)

**Objetivo:** observar ou alterar valores intermediários **sem mudar o código do modelo**.

- `handle = modulo.register_forward_hook(fn)` registra `fn`, que o PyTorch chama como
  `fn(module, inputs, output)` logo depois do forward daquele módulo.
- Se `fn` devolver `None`, a saída não muda (é o caso do experimento). Se devolver um tensor, ele
  substitui a saída.
- `handle.remove()` desregistra o hook.
- Hooks só rodam quando o módulo é chamado como `modulo(x)` (via `__call__`), e não com
  `modulo.forward(x)`.

O uso concreto, passo a passo, está na seção 5.2.

---

## 11. O que cada teste garante

| teste | propriedade |
|---|---|
| `test_hidden_states_shape` | `[B, T]` → `[B, T, d]` para $T \in \{1, 7, 16\}$ |
| `test_sequence_longer_than_context_raises` | $T >$ `context_length` gera `ValueError` |
| `test_id_outside_vocab_raises` | ID $\ge V$ gera `IndexError` |
| `test_has_one_block_per_layer` | $L$ blocos |
| `test_matches_manual_composition` | forward = embeddings → blocos → `ln_final` (e, desde o LM head, `model(ids)` = `lm_head(hidden_states)`) |
| `test_final_layer_norm_standardizes_each_position` | cada posição sai com média 0 e variância ~1 |
| `test_parameter_count` | $Vd + Td + L(12d^2 + 9d) + 2d$ |
| `test_future_tokens_do_not_affect_past` | o modelo inteiro é causal |
| `test_sequences_in_a_batch_are_independent` | sequências do batch não se misturam |
| `test_gradients_reach_all_parameters` | todos os parâmetros recebem gradiente |
| `test_sum_of_hidden_states_gives_almost_no_gradient_before_final_layer_norm` | a armadilha da seção 7 |
| `test_all_blocks_use_depth_scaled_init` | todos os blocos usam $0{,}02/\sqrt{2L}$ |
| `test_dropout_only_in_train_mode` | dropout ativo só em `train()` |

Os testes usam um modelo pequeno para rodar rápido: $V = 20$, $T = 16$, $d = 32$, 4 heads,
$L = 3$, dropout 0 (`tests/test_gpt.py`, linhas 10–11). Detalhes que valem notar:

- `test_future_tokens_do_not_affect_past` altera os IDs das posições 8 a 11 e exige saídas
  **iguais** nas posições 0 a 7 e **diferentes** a partir da 8.
- `test_final_layer_norm_standardizes_each_position` usa tolerância 0,05 na variância por causa
  do $\epsilon$ (seção 3.1).
- `test_all_blocks_use_depth_scaled_init` usa $d = 256$ e $L = 8$ para ter amostras suficientes
  e compara o desvio medido com $0{,}02/\sqrt{16} = 0{,}005$, com 5% de tolerância.

---

## Resumo

1. O GPT é `embeddings → L blocos → LayerNorm final`. O shape `[B, T, d]` se mantém do primeiro
   bloco ao último, e a saída é um estado final de $d$ dimensões por posição. Desde o LM head
   ([09-lm-head.md](09-lm-head.md)), esse caminho é `model.hidden_states(ids)`, e `model(ids)`
   acrescenta o LM head e devolve logits `[B, T, V]`.
2. Com o `tiny.yaml` e o vocabulário do *Dom Casmurro*, são 820.096 parâmetros, 96,5% nos
   blocos. A mesma classe, com a configuração do GPT-2 small, tem 124,4 milhões: a diferença
   para o original são só os biases da attention.
3. A profundidade não aumenta o alcance (uma camada já vê todo o contexto); ela permite
   **compor**: uma camada lê o que a anterior escreveu, como nas induction heads.
4. O modelo inteiro é causal, por indução nas camadas, e sequências do batch não se misturam.
5. A LayerNorm final é necessária no pre-norm: durante o treino a escala do stream cresce 11×
   em 50 passos, e ela entrega ao LM head uma entrada de escala estável.
6. Uma loss de teste como `out.sum()` depois de uma LayerNorm não gera gradiente, porque
   $\sum_i \hat x_i = 0$. Losses precisam depender da direção dos vetores.
7. `nn.ModuleList` registra os blocos; o dispositivo `meta` conta parâmetros sem alocar memória;
   forward hooks observam valores internos sem mudar o modelo.

---

## Para checar o entendimento

1. Calcule o número de parâmetros com $d = 256$, $L = 8$, $T = 256$ e $V = 97$ pela fórmula, e
   confira com a tabela da seção 9.
2. Por que uma única camada de attention já "enxerga" o contexto inteiro? Então para que
   servem as outras camadas?
3. Descreva, com as suas palavras, por que uma induction head precisa de pelo menos duas
   camadas.
4. Complete a prova de causalidade da seção 5.1: onde exatamente a hipótese de indução é
   usada?
5. Por que a norma depois de `ln_final` é 11,290 e não $\sqrt{128} = 11{,}314$? O que aconteceria
   com essa diferença depois do treino, quando o stream tem norma ~6,7?
6. Por que `out.sum().backward()` quase não gera gradiente nos embeddings? Por que o gradiente
   de `ln_final.bias` é ~4.096?
7. O que aconteceria se os blocos fossem guardados numa lista Python comum em vez de
   `nn.ModuleList`?
8. No GPT-2 small, que fração dos parâmetros está no token embedding? Por que essa fração é tão
   menor no Mini-GPT?
9. Na tabela de induction head da seção 4.2, troque o texto por `"o pato e o pa"`. Que posição a
   induction head encontra, e que caractere ela prevê? E com `"o bala e a ba"`, por que a
   resposta deixa de ser única?
10. Se `model(ids).sum().backward()` fosse feito hoje, com logits no final, o gradiente nos
    embeddings ainda seria ~0? (Dica: a soma dos logits de uma posição é $h \cdot \sum_i W_i$.)
11. Por que `hidden_states` e `forward` são métodos separados? Qual deles os experimentos deste
    documento usam, e por quê?
