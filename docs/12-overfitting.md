# Overfitting proposital

> **O sistema realmente consegue aprender?**

Código: [`experiments/e12_overfit.py`](../experiments/e12_overfit.py). Teste:
[`tests/test_overfit.py`](../tests/test_overfit.py). Experimento:
`uv run python -m experiments.e12_overfit` (~30 s). Notação geral: [00-notacao.md](00-notacao.md).
Visão geral: [architecture.md](architecture.md). Pré-requisito: [11-training.md](11-training.md).

---

## Para que serve

### O problema

No treino ([11-training.md](11-training.md)) a loss caiu de 4,6 para 1,45. Isso é bom sinal, mas
**não prova que o pipeline está certo**. Muitos bugs deixam a loss cair mesmo assim, e alguns
fazem ela cair **mais rápido**. Um deles aparece na seção 7.

Antes de qualquer treino grande, vale uma prova de sanidade: dar ao modelo um conjunto de
dados **minúsculo** e verificar se ele consegue **decorá-lo**. A lógica é simples:

- o modelo tem ~810 mil parâmetros;
- a frase tem 20 previsões;
- com essa folga, um modelo funcional **precisa** chegar a loss ~0 e acertar tudo.

Se não chegar, há um bug na arquitetura, na loss, no backward ou no otimizador, e é muito mais
barato descobrir isso em 5 segundos do que depois de horas de treino.

Termos usados daqui em diante:

- **Overfitting** (sobreajuste): o modelo se ajusta tanto aos dados de treino que passa a
  **decorá-los**, e piora em dados novos. Normalmente é um problema; aqui, é o objetivo.
- **Generalização:** acertar em dados que o modelo nunca viu. É o contrário de decorar.
- **Prova de sanidade** (*sanity check*): um teste rápido que qualquer sistema correto passa. Não
  prova que tudo está certo, mas reprovar prova que algo está errado.
- **Teacher forcing:** na hora de treinar e medir, cada posição recebe o texto **verdadeiro** como
  contexto, e não o que o modelo previu antes.
- **Acurácia:** a fração de posições em que o token mais provável é o correto.
- **Geração gulosa** (*greedy*): gerar texto escolhendo sempre o token mais provável e
  acrescentando-o ao contexto. É uma prévia de [13-generation.md](13-generation.md).
- **Atalho** (*shortcut*): uma forma "barata" de reduzir a loss que não é a que queríamos que o
  modelo aprendesse.
- **Vazamento** (*leakage*): quando a entrada contém, sem querer, a resposta.

### Uma analogia

Antes de usar uma calculadora nova para fazer o imposto de renda, você digita 2 + 2. Se não der 4,
não adianta seguir. E se der 4, ainda não significa que ela faz raiz quadrada certo.

Decorar uma frase é o "2 + 2" do treino. Passar prova que forward, loss, backward e otimizador se
conversam. As seções 6 e 7 mostram o que ele **não** prova.

### O que entra e o que sai

| | o quê | valor |
|---|---|---|
| entra | a frase `"o gato está na casa"`, com `<BOS>` e `<EOS>` | 21 tokens: `x` e `y` com shape `[1, 20]` |
| entra | o modelo do `tiny.yaml`, sem dropout | 809.472 parâmetros (vocabulário de 14 tokens) |
| entra | 300 passos de AdamW, lr 1e-3, sem weight decay | |
| sai | loss na frase | de 1,79 para **0,0012** |
| sai | acurácia | **100%** |
| sai | geração a partir de `<BOS>` | `'o gato está na casa'` |

### Onde isto se encaixa

Este documento não cria um componente novo. Ele **exercita o pipeline inteiro** num caso em que a
resposta certa é conhecida:

```text
frase: "o gato está na casa"
  │
  ▼  tokenizer                                 str → [<BOS>, o, ' ', g, ..., a, <EOS>]
  ▼  pares x, y                                x: [1, 20], y: [1, 20]
  ▼  GPT (embeddings, blocos, LM head)         x → logits [1, 20, 14]
  ▼  loss                                      logits e y → um número
  ▼  train_step, 300 vezes                     ajuste dos pesos
  │
  ▼  a verificação                             loss ≈ 0? acurácia 100%? gera a frase?  ← este documento
```

### O que daria errado sem esta parte

- **Bugs silenciosos passariam para o treino grande.** A loss cairia um pouco, e ninguém saberia se
  1,45 é o melhor possível ou o resultado de um bug.
- **Um bug que ajuda a loss** (como a máscara faltando, seção 7) poderia parecer uma melhoria.
- **Horas de CPU seriam gastas** para descobrir um problema que um teste de 5 segundos revela.

---

## Notação deste documento

| símbolo | como se lê | o que significa | valor neste experimento | no código |
|---|---|---|---|---|
| $x$, $x_t$ | "xis", "xis tê" | a sequência de entrada e o token na posição $t$ | `[1, 20]`; $x_0$ = `<BOS>` | `x`, `x[0, t]` |
| $y$, $y_t$ | "ípsilon", "ípsilon tê" | os alvos: $y_t = x_{t+1}$ | `[1, 20]`; $y_{19}$ = `<EOS>` | `y`, `y[0, t]` |
| $t$ | "tê" | posição na sequência | de 0 a 19 | `t` |
| $n$ | "ene" | número de previsões | 20 | `y.numel()` |
| $V$ | "vê" | tamanho do vocabulário | 14 (frase); 97 (parágrafo) | `tok.vocab_size` |
| $\mathcal{L}$ | "ele caligráfico" | loss (cross-entropy média) | de 1,79 a 0,0012 | `cross_entropy` |
| $p(y_t)$ | "pê de ípsilon tê" | probabilidade dada ao alvo correto | ~0,999 no fim | `probs[t, target]` |
| $\arg\max$ | "argumento do máximo" | o índice do maior valor: o token mais provável | | `.argmax(dim=-1)` |
| $a$, $b$ | "a", "bê" | um caractere e o caractere que vem logo depois dele | $a$ = `'q'`, $b$ = `'u'` | |
| $c(ab)$ | "contagem de a-bê" | quantas vezes o par "$a$ seguido de $b$" aparece na frase | $c(\texttt{' 'g}) = 1$ | `successors[a][b]` |
| $c(a\,\cdot)$ | "contagem de a seguido de qualquer um" | quantas vezes $a$ aparece com um sucessor | $c(\texttt{' '}\,\cdot) = 4$ | `sum(successors[a].values())` |
| `<BOS>`, `<EOS>` | "bós", "iós" | início e fim de sequência (ver [01-tokenizer.md](01-tokenizer.md)) | IDs 2 e 3 | `bos_id`, `eos_id` |
| $T$ | "tê" maiúsculo | tamanho da janela (seção 7) | 32 | `window` |

---

## 1. A ideia: decorar de propósito

### 1.1 Por que decorar prova alguma coisa

Um modelo funcional com ~810 mil parâmetros e **20** previsões para acertar tem capacidade de sobra
para decorá-las. Se isso não acontece, o problema não é falta de capacidade nem de dados: é um bug.

A decoração exercita tudo ao mesmo tempo:

| peça | o que precisa funcionar para a loss chegar a 0 |
|---|---|
| tokenizer e pares `x`/`y` | `y` precisa ser `x` deslocado em uma posição |
| embeddings e blocos | a informação precisa chegar das entradas aos logits |
| LM head e loss | os logits do alvo precisam poder crescer até dominar a softmax |
| backward | o gradiente precisa chegar a todos os parâmetros |
| otimizador | os parâmetros precisam de fato mudar a cada passo |

### 1.2 O que muda em relação ao treino normal

- **Um único exemplo, repetido 300 vezes.** Não há batches nem épocas: o mesmo `x` e `y` em todo passo.
- **Sem dropout e sem weight decay.** Os dois são **regularizadores**: técnicas que dificultam
  decorar, de propósito. Aqui queremos o contrário.
- **Tokens especiais.** A frase vira `<BOS>` + frase + `<EOS>`. Assim o modelo aprende também onde a
  frase começa e termina, e a geração pode partir de `<BOS>` e parar sozinha no `<EOS>`.

---

## 2. O dataset: uma frase

Seção 1 do experimento:

```text
frase: 'o gato está na casa' (19 caracteres)
vocabulário: 14 tokens (10 caracteres + 4 especiais)
x: (1, 20)   y: (1, 20)   -> 20 previsões para decorar

   t  entrada x[t]  alvo y[t]
   0  <BOS>         'o'      
   1  'o'           ' '      
   2  ' '           'g'      
   3  'g'           'a'      
   4  'a'           't'      
   5  't'           'o'      
   6  'o'           ' '      
   7  ' '           'e'      
   8  'e'           's'      
   9  's'           't'      
  10  't'           'á'      
  11  'á'           ' '      
  12  ' '           'n'      
  13  'n'           'a'      
  14  'a'           ' '      
  15  ' '           'c'      
  16  'c'           'a'      
  17  'a'           's'      
  18  's'           'a'      
  19  'a'           <EOS>    
```

- **O vocabulário** foi construído com esta frase e com a frase de comparação `"a gata está na casa"`,
  que usa os mesmos 10 caracteres: 10 + 4 especiais = 14 tokens.
- **21 tokens viram 20 pares:** `x` é tudo menos o último, e `y` é tudo menos o primeiro (a mesma
  regra de [02-dataset.md](02-dataset.md), numa única janela).
- **A primeira previsão** é "o que vem depois de `<BOS>`", ou seja, como a frase começa.
- **A última** é "o que vem depois do último `'a'`": o `<EOS>`.

---

## 3. A régua: olhar só o caractere anterior não basta

### 3.1 Caracteres ambíguos

Em [10-loss.md](10-loss.md), o bigrama previa o próximo caractere olhando só o anterior. Nesta
frase, isso não basta, porque alguns caracteres aparecem seguidos de coisas diferentes. Seção 2
do experimento:

```text
caracteres seguidos por mais de um caractere diferente:
  depois de ' '   vem: 'g' (1×), 'e' (1×), 'n' (1×), 'c' (1×)
  depois de 'a'   vem: 't' (1×), ' ' (1×), 's' (1×), <EOS> (1×)
  depois de 't'   vem: 'o' (1×), 'á' (1×)
  depois de 's'   vem: 't' (1×), 'a' (1×)

menor loss possível olhando só o caractere anterior: 0.693 nats
para chegar a ~0, o modelo precisa de mais informação: a posição ou o contexto
```

### 3.2 A menor loss possível para um bigrama

O melhor que um modelo de "caractere anterior" consegue é dar a cada sucessor a sua frequência
real na frase:

$$
\mathcal{L}_{\text{bigrama}} = -\frac{1}{n} \sum_{t=0}^{n-1} \log \frac{c(x_t\, y_t)}{c(x_t\, \cdot)}
$$

**Lendo a fórmula:**

- $x_t\, y_t$: o par (caractere atual, caractere seguinte) na posição $t$.
- $c(x_t\, y_t)$: quantas vezes esse par aparece na frase.
- $c(x_t\, \cdot)$: quantas vezes o caractere $x_t$ aparece seguido de qualquer coisa.
- a fração é a probabilidade que o melhor bigrama dá ao alvo certo.
- $-\frac{1}{n}\sum \log$: a cross-entropy média sobre as $n = 20$ posições (ver [10-loss.md](10-loss.md)).

A conta, agrupando as posições pelo caractere anterior:

| caractere anterior | posições | sucessores diferentes | probabilidade do certo | loss por posição | total |
|---|---:|---:|---:|---:|---:|
| `' '` | 4 | 4 | 1/4 | $\ln 4 = 1{,}386$ | 5,545 |
| `'a'` | 4 | 4 | 1/4 | $\ln 4 = 1{,}386$ | 5,545 |
| `'t'` | 2 | 2 | 1/2 | $\ln 2 = 0{,}693$ | 1,386 |
| `'s'` | 2 | 2 | 1/2 | $\ln 2 = 0{,}693$ | 1,386 |
| `<BOS>`, `'o'` (2×), `'g'`, `'e'`, `'á'`, `'n'`, `'c'` | 8 | 1 | 1 | 0 | 0 |
| **soma** | **20** | | | | **13,863** |

$\mathcal{L}_{\text{bigrama}} = 13{,}863 / 20 = 0{,}693$ nats (coincide com $\ln 2$). **Para ficar
abaixo disso, o modelo precisa usar outra informação além do caractere anterior**: a posição na
sequência, ou os caracteres mais antigos do contexto.

---

## 4. Treinando até decorar

### 4.1 A curva

Seção 3 do experimento. A cada 25 passos, medimos a loss e a acurácia **na frase** (sem dropout) e,
para comparação, a loss numa **outra** frase que o modelo nunca viu, `"a gata está na casa"`:

```text
809,472 parâmetros, AdamW lr 1e-3, sem dropout e sem weight decay

  passo | loss (frase) | acurácia |  grad | loss em 'a gata está na casa'
      1 |       1.7945 |      65% |  8.91 |                        1.895
     25 |       0.0888 |     100% |  0.38 |                        0.649
     50 |       0.0153 |     100% |  0.06 |                        0.705
     75 |       0.0080 |     100% |  0.03 |                        0.761
    100 |       0.0055 |     100% |  0.02 |                        0.791
    125 |       0.0041 |     100% |  0.02 |                        0.815
    150 |       0.0032 |     100% |  0.01 |                        0.837
    175 |       0.0026 |     100% |  0.01 |                        0.855
    200 |       0.0021 |     100% |  0.01 |                        0.872
    225 |       0.0018 |     100% |  0.01 |                        0.887
    250 |       0.0015 |     100% |  0.01 |                        0.901
    275 |       0.0013 |     100% |  0.01 |                        0.914
    300 |       0.0012 |     100% |  0.01 |                        0.926
```

### 4.2 Lendo a curva

- **Passo 1:** a medição acontece **depois** do primeiro passo. Um único passo já levou a acurácia a
  65% e a loss a 1,79 (o chute uniforme seria $\ln 14 = 2{,}64$).
- **Passo 25:** acurácia 100% e loss 0,089. O modelo já passou da régua do bigrama (0,693). **A prova
  de sanidade passou.**
- **Passos 25 a 300:** a acurácia fica em 100%, e a loss continua caindo até 0,0012. O modelo não
  aprende nada novo: só fica **mais confiante**, empurrando $p(\text{alvo})$ de ~0,92 para ~0,999.
- **A norma do gradiente** cai de 8,91 para 0,01. Quando as previsões estão certas e confiantes,
  $p - \text{one\_hot}(y) \approx 0$ e o gradiente some sozinho (ver [10-loss.md](10-loss.md), seção 5.2).

### 4.3 A assinatura do overfitting

A última coluna mostra, em miniatura, o que o overfitting faz:

| trecho | loss na frase de treino | loss na outra frase | o que está acontecendo |
|---|---|---|---|
| passos 1 → 25 | 1,79 → 0,089 | 1,90 → **0,649** | o modelo aprende o que as duas frases têm em comum (`" está na casa"`) |
| passos 25 → 300 | 0,089 → 0,0012 | 0,649 → **0,926** | o modelo se especializa na frase de treino e piora na outra |

A loss de treino cai **sempre**; a da outra frase cai e depois **sobe**. É exatamente o sinal de
overfitting descrito em [11-training.md](11-training.md) (seção 9.3), agora medido de forma bem
visível. Num treino de verdade, o melhor ponto seria o passo ~25, onde a loss na frase nova é a
menor.

---

## 5. O modelo decorou

### 5.1 Posição por posição

Seção 4 do experimento, depois dos 300 passos:

```text
   t  contexto               alvo    previsto   p(alvo)
   0  ''                     'o'     'o'         0.9989
   1  'o'                    ' '     ' '         0.9990
   2  'o '                   'g'     'g'         0.9985
   3  'o g'                  'a'     'a'         0.9991
   4  'o ga'                 't'     't'         0.9988
   5  'o gat'                'o'     'o'         0.9989
   6  'o gato'               ' '     ' '         0.9990
   7  'o gato '              'e'     'e'         0.9985
   8  'o gato e'             's'     's'         0.9987
   9  'o gato es'            't'     't'         0.9988
  10  'o gato est'           'á'     'á'         0.9988
  11  'o gato está'          ' '     ' '         0.9990
  12  'o gato está '         'n'     'n'         0.9986
  13  'o gato está n'        'a'     'a'         0.9991
  14  'o gato está na'       ' '     ' '         0.9990
  15  'o gato está na '      'c'     'c'         0.9986
  16  'o gato está na c'     'a'     'a'         0.9991
  17  'o gato está na ca'    's'     's'         0.9987
  18  'o gato está na cas'   'a'     'a'         0.9991
  19  'o gato está na casa'  <EOS>   <EOS>       0.9988
```

(O contexto da posição 0 aparece vazio porque é só o `<BOS>`, que o `decode` omite.)

- **Todas as 20 previsões estão certas,** com probabilidade ~0,999.
- **As posições ambíguas para o bigrama também:** depois do primeiro `'a'` (posição 4) vem `'t'`;
  depois do segundo (14), `' '`; do terceiro (17), `'s'`; do último (19), `<EOS>`.
- A loss 0,0012 é a média de $-\ln 0{,}999 \approx 0{,}001$ nas 20 posições.

### 5.2 Gerando a frase

Com teacher forcing, cada posição recebe o texto verdadeiro. Um teste mais forte é deixar o modelo
**gerar sozinho**, usando as próprias previsões como contexto:

```python
ids = [bos_id]
repita:
    próximo = argmax(model(ids)[última posição])
    se próximo == eos_id: pare
    ids.append(próximo)
```

```text
geração a partir de <BOS>, sempre o mais provável: 'o gato está na casa'
igual à frase: True
```

O modelo recita a frase inteira, caractere por caractere, e para sozinho no ponto certo.

---

## 6. O atalho: decorar posições em vez do texto

### 6.1 A pergunta

A régua da seção 3 diz que o modelo precisa de "a posição **ou** o contexto". Qual dos dois ele
usou? Com uma única frase, **a posição sozinha basta**: a posição 13 sempre tem alvo `'a'`, a 14
sempre tem `' '`, e assim por diante. O modelo não precisa ler os caracteres anteriores. Basta
decorar "posição → caractere".

### 6.2 O teste

Seção 5 do experimento: damos ao modelo decorado a **outra** frase como entrada e vemos o que ele
prevê.

```text
entrada (outra frase): 'a gata está na casa'
previsões do modelo:   'o gato está na casa'
previsões iguais aos alvos da frase decorada: 100%
previsões iguais aos alvos certos desta frase: 90%

posições onde as duas frases pedem caracteres diferentes (ou onde o modelo errou):
   t  entrada  alvo certo  alvo decorado  previsto
   0  <BOS>    'a'         'o'            'o'     
   5  't'      'a'         'o'            'o'     
```

- A entrada diz "a gata", mas o modelo continua prevendo "o gato". As previsões batem **100%** com a
  frase decorada.
- Nas duas posições em que as frases diferem, o modelo escolhe o alvo **decorado**. Na posição 5, a
  entrada é `'t'` nas duas frases, e o contexto anterior é `"a gat"` (e não `"o gat"`), mas ele prevê
  `'o'` de qualquer jeito.
- Os 90% de acerto na frase certa são coincidência: as duas frases compartilham o final.

### 6.3 A lição

O modelo reduziu a loss pelo caminho **mais fácil**, e o mais fácil aqui era a posição. Isso tem
duas consequências:

- **A prova de sanidade é válida,** mas limitada. Ela mostrou que loss, backward e otimizador
  funcionam, e que a informação chega das entradas aos logits. **Não** mostrou que a attention usa o
  contexto.
- **Um bug na attention poderia passar despercebido.** Na primeira versão deste experimento,
  treinamos a frase **sem** a causal mask (o bug da seção 7), e o modelo decorou e gerou a frase
  perfeitamente do mesmo jeito: a posição não precisa do futuro. Um teste de decoração que o bug
  também passa não detecta o bug.

Esse comportamento tem nome: **shortcut learning**. Modelos exploram qualquer regularidade que
reduza a loss, inclusive as que não queríamos. Um bom teste precisa fechar os atalhos.

---

## 7. Um bug proposital: sem a causal mask

### 7.1 Fechando o atalho

Para testar se o modelo usa o **contexto**, precisamos de dados em que a posição não diz nada. O
experimento usa os 400 primeiros caracteres do *Dom Casmurro* (a partir de "Uma noite") e janelas
de $T = 32$ com **stride 1**. A mesma posição da janela recebe, em janelas diferentes, caracteres
diferentes: só o contexto identifica o alvo.

Treinamos duas vezes, com a mesma seed e 400 passos:

- **com máscara:** o modelo normal;
- **sem máscara:** trocando a função `causal_mask` por uma que deixa toda posição ver todas as outras,
  inclusive as futuras.

Depois, damos ao modelo os 32 primeiros caracteres do parágrafo e deixamos que ele gere os próximos 80
de forma gulosa.

### 7.2 O resultado

Seção 6 do experimento:

```text
parágrafo de 400 caracteres; janelas de 32 com stride 1
(a mesma posição da janela recebe caracteres diferentes: só o contexto identifica o alvo)

  com máscara: loss de treino passo 1: 4.5776 | passo 100: 0.6804 | passo 200: 0.2106 | passo 400: 0.1316
  continuação dos 32 primeiros caracteres: 'Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz aqui do bairro, que'
  dos 80 caracteres gerados, iguais ao original: 80

  sem máscara: loss de treino passo 1: 4.5638 | passo 100: 0.9264 | passo 200: 0.0593 | passo 400: 0.0034
  continuação dos 32 primeiros caracteres: 'Uma noite destas, vindo da cidade parararararararararararaorarara, que cecu      verse cs nçade mimitapapapapapa'
  dos 80 caracteres gerados, iguais ao original: 10
```

| | loss de treino no passo 400 | geração (80 caracteres) |
|---|---:|---|
| com máscara | 0,1316 | **80 de 80 corretos** |
| sem máscara | **0,0034** (39× menor) | 10 de 80: `"parararara…"` |

### 7.3 Por que a loss menor gerou lixo

Sem máscara, a posição $t$ enxerga $x_{t+1}$, que **é** o alvo $y_t$ (ver [02-dataset.md](02-dataset.md)). O caminho mais fácil
para reduzir a loss é **copiar o próximo caractere da entrada**. É um vazamento: a resposta está na
pergunta. Por isso a loss de treino chega a 0,0034.

Na geração, a última posição **não tem próximo caractere**: ele ainda não existe. O circuito que o
modelo aprendeu não tem o que copiar, e a saída degenera em repetições (`"rararara"`). O modelo com
máscara nunca teve esse atalho: foi obrigado a aprender a prever a partir do passado, que é
exatamente a situação da geração.

**Por que, com máscara, a loss não chega a ~0:** cada janela começa num ponto arbitrário do
parágrafo. Nas primeiras posições de uma janela, o contexto tem 1, 2, 3 caracteres, e isso nem sempre
basta para saber onde se está no texto. Por exemplo, depois de `"da "` pode vir `"cidade"`,
`"Central"` ou `"lua"`. Essas previsões são genuinamente ambíguas, e a loss média fica acima de
zero. Nas
posições com contexto longo, o modelo acerta tudo, e é isso que a geração usa.

### 7.4 As lições

1. **Loss de treino menor não significa modelo melhor.** Um vazamento produz a menor loss de todas.
2. **Uma prova de sanidade precisa verificar o comportamento real** (a geração), e não só a loss com
   teacher forcing.
3. **É uma versão em miniatura** de "remover a causal mask e observar o impacto no treinamento",
   e a confirmação prática do argumento de [04-attention.md](04-attention.md), seção 3.

### 7.5 Como o bug foi injetado

A `MultiHeadAttention` chama a função `causal_mask` do módulo `model.attention` a cada forward. O
experimento **substitui temporariamente** essa função (uma técnica chamada *monkeypatching*):

```python
original = attention_module.causal_mask
attention_module.causal_mask = see_everything     # máscara toda True
try:
    ...treina e gera...
finally:
    attention_module.causal_mask = original       # sempre restaura, mesmo com erro
```

O `try`/`finally` garante que a função original volta mesmo que algo dê errado no meio. Nenhum
arquivo do modelo foi alterado.

---

## 8. Se o modelo não decorasse: um guia de diagnóstico

A prova de sanidade serve para quando **algo dá errado**. O sintoma costuma apontar a causa:

| sintoma | causas prováveis | como verificar |
|---|---|---|
| a loss fica parada perto de $\ln V$ | o otimizador não atualiza: lr = 0, falta `optimizer.step()`, parâmetros fora do otimizador, forward dentro de `torch.no_grad()` | os pesos mudam depois de um passo? (`test_train_step_changes_parameters`) |
| a loss cai e **estaciona** perto da régua do bigrama (0,69) | o modelo só usa o caractere atual: posições não somadas, attention quebrada | o modelo prevê diferente para o mesmo caractere em posições diferentes? |
| aparece `nan` ou `inf` | lr alto demais, `log(softmax)` em vez de `logsumexp`, linha da máscara toda proibida | testes de estabilidade de [04-attention.md](04-attention.md) e [10-loss.md](10-loss.md) |
| a loss oscila ou sobe | lr alto demais, gradientes acumulando (falta `zero_grad`), sem clipping | norma do gradiente no log; `test_gradients_do_not_accumulate_between_steps` |
| a loss vai a ~0 **rápido demais** e a geração falha | vazamento: máscara faltando, `y` igual a `x` (alvos não deslocados) | gerar texto (seção 7); testes de causalidade de [04-attention.md](04-attention.md) e [08-gpt.md](08-gpt.md) |
| acurácia 100% com teacher forcing, mas a geração difere | diferença entre treino e inferência: dropout ligado na geração, prompt sem `<BOS>`, tokenização diferente | comparar a geração com as previsões posição por posição (seção 5) |

---

## 9. Código

### 9.1 Funções do experimento

| função | objetivo | o que faz |
|---|---|---|
| `pairs(tok, text)` | montar o exemplo | `<BOS>` + texto + `<EOS>`, recortado em `x = ids[:, :-1]` e `y = ids[:, 1:]` |
| `measure(model, x, y)` | medir sem ruído | loss e acurácia em `eval()` (sem dropout) e sem gradientes; restaura o modo depois |
| `greedy_from_bos(model, tok)` | testar a geração da frase | parte de `<BOS>`, acrescenta o `argmax` a cada passo e para no `<EOS>` |
| `greedy_continue(model, tok, prompt, ...)` | testar a geração do parágrafo | continua um prompt, olhando só os últimos `window` tokens (o contexto máximo das janelas de treino) |
| `see_everything(seq_len)` | injetar o bug | devolve uma máscara toda `True`: nenhuma posição fica proibida |
| `bigram_floor(tok)` | calcular a régua | conta os pares da frase e calcula a loss do melhor bigrama |

O treino usa diretamente `configure_optimizer` e `train_step` de [11-training.md](11-training.md),
sem a função `train`: com um único exemplo, não há épocas nem validação para organizar.

### 9.2 O teste

[`tests/test_overfit.py`](../tests/test_overfit.py) transforma a prova de sanidade num teste que roda
com os demais:

```python
def test_gpt_memorizes_a_short_sentence():
    ...                                                    # modelo pequeno: d=32, 2 blocos
    for _ in range(200):
        result = train_step(model, x, y, optimizer, gradient_clip=1.0)
    ...
    assert result.loss < 0.05
    assert torch.equal(predictions, y)                     # acurácia de 100%
```

- **Um modelo menor** (`d_model = 32`, 2 blocos, lr 3e-3) deixa o teste rápido (~1 s).
- **Se uma mudança futura quebrar o treino** (um sinal trocado, um `zero_grad` removido, os alvos
  desalinhados), este teste falha.
- **Ele herda a limitação da seção 6:** não detecta a falta da causal mask. Essa garantia fica com os
  testes de causalidade de [04](04-attention.md), [05](05-multi-head-attention.md),
  [07](07-transformer-block.md) e [08](08-gpt.md), que verificam diretamente que o futuro não
  afeta o passado.

| teste | propriedade |
|---|---|
| `test_gpt_memorizes_a_short_sentence` | em 200 passos, loss < 0,05 e 100% das previsões certas na frase |

---

## Resumo

1. A prova de sanidade: um modelo com ~810 mil parâmetros precisa decorar uma frase de 20
   previsões. Em 25 passos, a acurácia chegou a 100% e a loss (0,089) passou da régua do bigrama
   (0,693); em 300, a loss chegou a 0,0012 e a geração reproduziu a frase.
2. A loss numa frase parecida caiu até o passo 25 (0,649) e depois subiu (0,926): a assinatura do
   overfitting, com treino caindo e dados novos piorando.
3. Com uma única frase, o modelo tomou um **atalho**: decorou "posição → caractere". Recebendo outra
   frase, continuou recitando a decorada. A prova confirma loss, backward e otimizador, mas não o uso
   do contexto.
4. Fechando o atalho (parágrafo real, stride 1), a falta da causal mask aparece: a loss de treino
   ficou 39× menor (vazamento do próximo caractere), e a geração virou `"rararara"`. Com máscara, 80 de
   80 caracteres corretos.
5. Loss de treino baixa não prova que o modelo está certo: é preciso verificar o comportamento real,
   como a geração.

---

## Para checar o entendimento

1. Por que um modelo funcional **precisa** conseguir decorar 20 previsões? O que significaria não
   conseguir?
2. Refaça a conta da régua do bigrama (0,693) sem olhar a tabela. Por que as posições depois de
   `'o'` contribuem com zero, se `'o'` aparece duas vezes?
3. Por que desligamos dropout e weight decay neste experimento?
4. Entre os passos 25 e 300, a acurácia não mudou, mas a loss caiu de 0,089 para 0,0012. O que o
   modelo "aprendeu" nesse intervalo?
5. Em que passo você pararia o treino se o objetivo fosse ir bem na frase `"a gata está na casa"`?
   Por quê?
6. Explique por que o modelo continua prevendo `'o'` na posição 5 mesmo quando a entrada é
   `"a gat"`.
7. Por que o treino da frase **sem** máscara também decorou e gerou a frase certa?
8. Por que janelas com stride 1 fecham o atalho da posição?
9. Sem máscara, a loss de treino foi 39× menor, mas a geração falhou. Explique usando o fato de que
   $y_t = x_{t+1}$.
10. O teste `test_gpt_memorizes_a_short_sentence` detectaria alvos não deslocados (`y = x`)? E a falta
    da causal mask? Justifique.
