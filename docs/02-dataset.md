# Dataset

> **Como um texto vira exemplos de treino?**

Código: [`data/dataset.py`](../data/dataset.py), [`data/loader.py`](../data/loader.py) e
[`scripts/prepare_corpus.py`](../scripts/prepare_corpus.py). Testes:
[`tests/test_dataset.py`](../tests/test_dataset.py). Experimento:
`uv run python -m experiments.e02_dataset`. Notação geral: [00-notacao.md](00-notacao.md).
Visão geral: [architecture.md](architecture.md).

---

## Para que serve

### O problema

O tokenizer ([01-tokenizer.md](01-tokenizer.md)) entrega uma única sequência longa: os 374.785 IDs do
livro. O modelo não aprende "com o livro" de uma vez. Ele aprende com muitos exemplos
pequenos do tipo "dado este trecho, qual é o próximo caractere?". Este documento cobre:

1. prepara o texto do livro, removendo o que não é o romance;
2. separa o trecho final para validação;
3. recorta a sequência em **janelas** de tamanho fixo, cada uma com uma entrada `x` e as
   respostas `y`;
4. agrupa as janelas em **batches** embaralhados.

Termos usados daqui em diante:

- **Exemplo de treino:** um par (entrada, resposta esperada). A resposta esperada se chama
  **alvo** (*target*).
- **Contexto:** os tokens que o modelo pode ver antes de fazer uma previsão.
- **Janela:** um trecho contíguo (sem buracos) recortado da sequência de IDs.
- **Batch:** um grupo de janelas processadas juntas, empilhadas num único tensor. Processar em
  grupo é muito mais rápido que processar uma janela por vez.
- **Época:** uma passada completa por todas as janelas do treino.
- **Treino e validação:** o modelo aprende com o treino. A validação é um texto separado, que
  ele nunca usa para aprender, só para medir se aprendeu algo que vale para texto novo.
- **Hiperparâmetro:** um valor escolhido antes do treino, na config (`context_length`,
  `stride`, `batch_size`, `val_fraction`).
- **Loss:** o número que mede o erro das previsões. O treino tenta diminuí-lo
  ([10-loss.md](10-loss.md)).

### Uma analogia

Imagine estudar o livro com um cartão vazado que mostra 128 letras. Você lê o que aparece e
tenta adivinhar a letra seguinte. Depois desliza o cartão e repete. O quanto você desliza é o
**stride**. As últimas páginas do livro ficam guardadas para a prova (a validação): se você
tivesse estudado por elas, a prova não mediria nada.

### O que entra e o que sai

| etapa | entra | sai |
|---|---|---|
| `scripts/prepare_corpus.py` | o arquivo do Project Gutenberg | `corpus/dom_casmurro.txt`, com 374.785 caracteres |
| tokenizer | o texto | `ids`: tensor `[374.785]`, tipo `int64` |
| `split_train_val` | `ids: [374.785]` | treino `[337.307]` e validação `[37.478]` |
| `NextTokenDataset[i]` | o número $i$ de uma janela | `x: [128]` e `y: [128]` |
| `DataLoader` | as 2.635 janelas do treino | 82 batches por época, cada um com `x: [32, 128]` e `y: [32, 128]` |

### Onde este componente entra

```text
texto: "Uma noite destas, vindo da cidade..."
  │
  ▼  tokenizer                                 str → IDs [374.785]
  ▼  dataset e batches                         IDs → x, y: [32, 128]            ← este componente
  ▼  embeddings                                [32, 128] → [32, 128, 128]
  ▼  blocos Transformer × 4                    [32, 128, 128] → [32, 128, 128]
  ▼  LM head                                   [32, 128, 128] → logits [32, 128, 97]
  ▼  loss (ainda não implementada)             logits e y → um número
```

O `x` segue pelo modelo. O `y` fica guardado e só é usado no fim, para comparar com a previsão
e calcular a loss.

### O que daria errado sem esta parte

- **Sem pares (x, y),** não haveria resposta certa para comparar com a previsão, e não
  existiria loss.
- **Com trechos de tamanhos diferentes,** não daria para empilhá-los num tensor.
- **Com a validação sorteada no meio do treino,** a validation loss mediria memorização, não
  aprendizado (seção 7).
- **Sem embaralhar,** batches seguidos viriam do mesmo trecho do livro (seção 8).
- **Sem seed,** duas execuções com a mesma config veriam batches diferentes, e os experimentos
  não seriam comparáveis.

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $P(\cdot)$ | "pê de" | probabilidade. Atenção: em [03-embeddings.md](03-embeddings.md), $P$ é outra coisa (a tabela de posições) | | |
| $P(A \mid B)$ | "pê de A dado B" | probabilidade de A, sabendo que B aconteceu | $P(\texttt{i} \mid \texttt{o})$: chance de vir `'i'` depois de `'o'` | |
| $\prod_{t=1}^{N}$ | "produtório de tê igual a um até ene" | multiplicar os fatores para $t = 1, 2, \dots, N$ | | `math.prod` |
| $\sum_{t}$ | "somatório em tê" | somar os termos para todos os $t$ | | `sum`, `.sum()` |
| $\log$, $\ln$ | "log", "logaritmo natural" | logaritmo na base $e \approx 2{,}718$; nos documentos, $\log$ é sempre natural | $\ln 97 \approx 4{,}57$ | `math.log`, `torch.log` |
| nats | "nats" | unidade da loss quando ela usa o logaritmo natural | 4,57 nats | |
| $x_t$ | "xis tê" | o token na posição $t$ da sequência | | `ids[t]` |
| $x_1, \dots, x_N$ | "xis um até xis ene" | a sequência inteira. Na seção 1 a contagem começa em 1, costume da probabilidade; no código, começa em 0 | | `ids` |
| $x_{<t}$ | "xis antes de tê" | todos os tokens antes da posição $t$ | | `ids[:t]` |
| $N$ | "ene" | tamanho da sequência de IDs que está sendo recortada | 374.785 no corpus; 337.307 no treino | `len(ids)` |
| $T$ | "tê" | `context_length`: quantos tokens tem cada entrada `x` | 128 | `context_length` |
| $s$ | "esse" | **stride**: distância entre o início de uma janela e o início da seguinte. Neste documento, $s$ é sempre o stride (em [09-lm-head.md](09-lm-head.md) a letra tem outro sentido) | 128 | `stride` |
| $i$ | "i" | número (índice) de uma janela | de 0 a 2.634 no treino | `index` |
| $\text{start}$ | "start" | posição em `ids` onde a janela $i$ começa | janela 1: 128 | `start` |
| $N_w$ | "ene dáblio" | número de janelas (*w* de *window*) | 2.635 no treino | `len(dataset)` |
| $y_t$ | "ípsilon tê" | alvo da posição $t$: o token que vem depois de `x[0..t]` | | `y[t]` |
| $B$ | "bê" | `batch_size`: janelas por batch | 32 | `batch_size` |
| $f$ | "éfe" | fração da sequência reservada para validação | 0,1 | `val_fraction` |
| $n_{\text{val}}$, $n_{\text{treino}}$ | "ene val", "ene treino" | quantos tokens vão para a validação e para o treino | 37.478 e 337.307 | `len(val_ids)`, `len(train_ids)` |
| $V$ | "vê" | tamanho do vocabulário | 97 | `tokenizer.vocab_size` |
| $d$ | "dê" | tamanho dos vetores do modelo (só na seção 10) | 128 | `d_model` |
| $\lfloor \cdot \rfloor$ | "piso de" | arredonda para baixo (fica com a parte inteira) | $\lfloor 82{,}34 \rfloor = 82$ | `//`, `int()` |
| $\bmod$ | "módulo" | resto da divisão inteira | $7 \bmod 3 = 1$ | `%` |
| $\cdot$, $\times$ | "vezes" | multiplicação | $32 \cdot 128 = 4.096$ | `*` |
| $\approx$ | "aproximadamente" | igual depois de arredondar | | |

---

## 1. O que o modelo precisa aprender

Um **modelo de linguagem** atribui uma probabilidade a uma sequência de tokens
$x_1, \dots, x_N$. Pela regra da cadeia da probabilidade, essa distribuição conjunta se
decompõe **exatamente** num produto de condicionais.

A fórmula abaixo quebra "qual a probabilidade deste texto inteiro?" numa série de perguntas
menores, uma por posição: "qual a probabilidade deste token, dado o que veio antes?".

$$
P(x_1, \dots, x_N) = \prod_{t=1}^{N} P(x_t \mid x_1, \dots, x_{t-1})
$$

**Lendo a fórmula:**

- $P(x_1, \dots, x_N)$: a probabilidade **conjunta**, isto é, a chance de o texto ser
  exatamente essa sequência de $N$ tokens, nessa ordem.
- $\prod_{t=1}^{N}$: multiplica os fatores da direita para $t = 1, 2, \dots, N$.
- $P(x_t \mid x_1, \dots, x_{t-1})$: a probabilidade **condicional** de o token da posição
  $t$ ser $x_t$, sabendo quais foram os anteriores. Para $t = 1$ não há anteriores, e o fator
  é só $P(x_1)$.
- "Exatamente": não é aproximação. A igualdade vale para qualquer distribuição de
  probabilidade.

**Exemplo** (probabilidades inventadas, só para ilustrar a conta): para o texto `"oi"`, se
$P(\texttt{o}) = 0{,}08$ e $P(\texttt{i} \mid \texttt{o}) = 0{,}05$, então
$P(\texttt{oi}) = 0{,}08 \times 0{,}05 = 0{,}004$.

Um GPT modela cada fator: dado o que veio antes, qual a distribuição do próximo token?
Treinar é maximizar a log-verossimilhança do corpus, $\sum_t \log P(x_t \mid x_{<t})$,
o que equivale a minimizar a cross-entropy do próximo token ([10-loss.md](10-loss.md)).

Escrita em destaque, a quantidade que o treino tenta deixar o maior possível é:

$$
\sum_{t=1}^{N} \log P(x_t \mid x_{<t})
$$

**Lendo a fórmula:**

- $\log P(\dots)$: o logaritmo de cada fator. Como uma probabilidade fica entre 0 e 1, o
  logaritmo é negativo ou zero, e só vale 0 quando a probabilidade é 1.
- $\sum_{t=1}^{N}$: soma as parcelas. O logaritmo transforma o produto da fórmula anterior
  numa soma, porque $\ln(ab) = \ln a + \ln b$. Somar é mais seguro no computador: multiplicar
  centenas de milhares de números menores que 1 daria um resultado pequeno demais para ser
  representado, que viraria 0.
- $x_{<t}$: forma curta de escrever $x_1, \dots, x_{t-1}$.
- **Log-verossimilhança** (*log-likelihood*) é o nome dessa soma. Maximizá-la significa deixar
  o texto real o mais provável possível segundo o modelo.
- **Cross-entropy** é a média dessas mesmas parcelas com o sinal trocado. Por isso maximizar
  uma é o mesmo que minimizar a outra.

**Exemplo** (continuação): $\ln 0{,}08 + \ln 0{,}05 = -2{,}526 + (-2{,}996) = -5{,}521$, que
é igual a $\ln 0{,}004$. A cross-entropy média desses 2 tokens seria $5{,}521 / 2 \approx 2{,}76$
nats.

Portanto, **um exemplo de treino é um par (contexto, próximo token)**. O dataset existe
para produzir esses pares em massa, empacotados em tensores.

Há uma aproximação inevitável: o modelo só enxerga os últimos $T$ = `context_length`
tokens. Na prática, ele estima

$$
P(x_t \mid x_{t-T}, \dots, x_{t-1})
$$

e tudo o que ficou antes da janela é esquecido. Com caracteres e $T = 128$, isso dá
algo como uma ou duas frases.

**Lendo a fórmula:**

- $x_t$: o token a prever.
- $x_{t-T}, \dots, x_{t-1}$: os $T$ tokens imediatamente anteriores. O mais antigo que o
  modelo ainda vê é $x_{t-T}$.
- Tudo antes de $x_{t-T}$ não entra na conta.
- $T$ é o **máximo**. Dentro de uma janela, as primeiras posições têm contexto menor
  (seção 5).

**Exemplo:** com $T = 128$, para prever o token de número 1.000 o modelo vê, no máximo, os
tokens 872 a 999. São $999 - 872 + 1 = 128$ tokens.

---

## 2. O corpus

O **corpus** é o texto de onde saem o vocabulário e todos os exemplos de treino.

### 2.1 Fonte

*Dom Casmurro*, Machado de Assis, edição H. Garnier de 1899, via
[Project Gutenberg #55752](https://www.gutenberg.org/ebooks/55752). A obra está em
domínio público.

A edição mantém a **ortografia original**: *chapéo*, *elle*, *succedeu*, *cançado*,
*theatro*, *photographia*. O modelo vai aprender essa grafia. Um prompt em ortografia
atual ainda é tokenizável (os caracteres existem), mas terá combinações de letras que o
modelo viu pouco ou nunca.

### 2.2 Limpeza: `clean()` em `scripts/prepare_corpus.py`

**Objetivo:** transformar o arquivo baixado num texto que contenha só o romance, com
formatação que faça sentido aprender.

O arquivo do Gutenberg não é só o romance. O `scripts/prepare_corpus.py` aplica:

| passo | motivo |
|---|---|
| `\r\n` → `\n` | um caractere a menos, repetido em toda linha |
| remove licença, folha de rosto e índice final | não é texto do romance; o índice é uma lista de 148 títulos em formato de tabela |
| remove `_` | é marcação de itálico do Gutenberg (`_Dom Casmurro._`), não texto |
| junta as linhas de cada parágrafo | o Gutenberg quebra a prosa a cada ~70 colunas; sem isso, o modelo aprenderia a inserir `\n` por contagem de caracteres, um artefato de formatação |
| preserva as quebras dos versos | nos 13 versos indentados, a quebra de linha faz parte do texto |
| normaliza para NFC | mesmo motivo do tokenizer: acentos compostos e decompostos viram o mesmo caractere |

Os 13 versos são 13 linhas indentadas (com espaços no início), distribuídas em 10 blocos
(conferido no arquivo original do Gutenberg).

Passo a passo no código:

1. **Linha 19**, `raw.replace("\r\n", "\n")`. `\r\n` é a quebra de linha no estilo Windows,
   feita de dois caracteres (*carriage return* e *line feed*). `\n` é a quebra de um caractere
   só.
2. **Linha 20**, `text[text.index(FIRST_CHAPTER) : text.index(INDEX)]`. Guarda só o que está
   entre o início do capítulo I (`"\nI\n\nDo titulo."`) e o título do índice (`"\nINDICE\n"`).
   Licença e folha de rosto (antes) e índice e licença final (depois) saem.
3. **Linha 21**, `text.replace("_", "")`. Remove a marcação de itálico.
4. **Linha 24**, `re.split(r"\n\s*\n", text.strip())`. Divide o texto em blocos a cada linha
   em branco. A expressão regular `\n\s*\n` significa "quebra de linha, zero ou mais espaços,
   quebra de linha", então aceita também linhas "em branco" que contêm espaços.
5. **Linhas 25 a 28.** Para cada bloco: se a primeira linha começa com dois espaços, é verso,
   e as linhas são unidas com `\n`. Senão, é prosa, e as linhas são unidas com um espaço. Em
   ambos os casos, `strip()` remove os espaços das pontas de cada linha.
6. **Linha 30.** Junta os parágrafos com `\n\n`, normaliza para NFC e acrescenta um `\n` no
   fim.

**Exemplo** (quebra em posição ilustrativa): o bloco de prosa
`"Uma noite destas, vindo da cidade\npara o Engenho Novo"` vira
`"Uma noite destas, vindo da cidade para o Engenho Novo"`.

A função `main()` baixa o arquivo, chama `clean` e grava o resultado em
`corpus/dom_casmurro.txt`. Para rodar: `uv run python scripts/prepare_corpus.py`.

Parágrafos ficam separados por uma linha em branco (`\n\n`). Capítulos começam com o
numeral romano e o título em parágrafos próprios (`I\n\nDo titulo.\n\n`).

### 2.3 Estatísticas

| medida | valor |
|---|---|
| caracteres | 374.785 |
| parágrafos | 1.719 |
| caracteres distintos | 93 |
| `vocab_size` (+ 4 especiais) | 97 |
| caractere mais frequente | espaço, 63.398 (16,9%) |
| letra mais frequente | `a`, 38.045 (10,2%) |

Saída da seção "Corpus" do experimento:

```text
=== Corpus ===
caracteres: 374,785
parágrafos: 1,719
vocab_size: 97
mais frequentes: ' ' 63,398  'a' 38,045  'e' 36,138  'o' 29,021  's' 22,114  'i' 17,638  'r' 17,447  'm' 15,060
raros (< 10x):   '+' 1  '=' 1  'Z' 1  'À' 1  'Ó' 1  '9' 2  'à' 2  'W' 3  '6' 4  'k' 4  'Á' 4  '$' 5  'è' 7  '2' 9
```

As porcentagens da tabela são frequências relativas: a fração do texto ocupada por um
caractere.

$$
\text{frequência relativa} = \frac{\text{ocorrências do caractere}}{\text{total de caracteres}}
$$

**Lendo a fórmula:**

- numerador: quantas vezes o caractere aparece no corpus;
- denominador: o tamanho do corpus, 374.785 caracteres;
- o resultado fica entre 0 e 1; multiplicado por 100, vira porcentagem.

**Exemplo:** para o espaço, $63.398 / 374.785 \approx 0{,}169 = 16{,}9\%$. Para `'a'`,
$38.045 / 374.785 \approx 0{,}102 = 10{,}2\%$.

A distribuição tem cauda longa. `+`, `=`, `Z`, `À` e `Ó` aparecem **uma única vez**; `9`,
`à` e `W`, duas ou três. Na prática, o embedding desses caracteres quase não recebe
gradiente, e o modelo praticamente nunca vai gerá-los.

- **Cauda longa:** poucos caracteres muito frequentes e muitos caracteres raros.
- **Gradiente:** o sinal que diz como ajustar cada parâmetro para diminuir a loss. Um
  caractere que quase não aparece quase não gera sinal para o seu próprio vetor
  ([03-embeddings.md](03-embeddings.md)).

**Uma referência útil para a loss ([10-loss.md](10-loss.md)):** um modelo sem nenhum conhecimento, que dá
probabilidade uniforme $1/97$ a cada token, tem cross-entropy
$\ln 97 \approx 4{,}57$ nats. A loss inicial de um modelo recém-inicializado deve ficar
perto desse valor. Muito acima disso indica um problema na inicialização.

A conta, para uma posição qualquer: a loss é o logaritmo da probabilidade dada ao token
certo, com o sinal trocado.

$$
-\ln \frac{1}{97} = \ln 97 \approx 4{,}57 \text{ nats}
$$

**Lendo a fórmula:**

- $\frac{1}{97}$: a probabilidade que o modelo "sem conhecimento" dá a cada um dos 97 tokens,
  inclusive ao certo.
- $\ln$: logaritmo natural. $\ln(1/97) \approx -4{,}57$ é negativo, porque $1/97 < 1$.
- o sinal de menos deixa a loss positiva. Quanto menor a loss, melhor.
- $-\ln(1/x) = \ln x$: por isso o resultado é $\ln 97$.
- **nats:** a unidade da loss quando se usa o logaritmo natural.
- **Inicialização:** os valores aleatórios dos parâmetros antes do treino
  ([03-embeddings.md](03-embeddings.md)).

**Exemplo:** $1/97 \approx 0{,}0103$, e $-\ln 0{,}0103 \approx 4{,}575$. Um modelo perfeito,
que desse probabilidade 1 ao token certo, teria loss $-\ln 1 = 0$.

---

## 3. De texto a tensor

```text
texto (str, 374.785 caracteres)
  │  tokenizer.encode
  ▼
list[int] (374.785 IDs)
  │  torch.tensor(..., dtype=torch.long)
  ▼
ids: [N]  int64  (~3 MB)
```

O texto é tokenizado **uma vez**, inteiro, antes de qualquer divisão. O tipo é `int64`
porque `nn.Embedding` e `F.cross_entropy` exigem índices inteiros longos.

- **`int64`** (ou `torch.long`): inteiro de 64 bits, que ocupa 8 bytes.
- **`nn.Embedding`:** a tabela de vetores de [03-embeddings.md](03-embeddings.md), que usa cada
  ID como número de linha.
- **`F.cross_entropy`:** a função de loss de [10-loss.md](10-loss.md), que usa cada alvo como
  índice.

O tamanho de ~3 MB sai de uma multiplicação:

$$
\text{memória} = N \times 8 \text{ bytes}
$$

**Lendo a fórmula:** $N$ é o número de IDs, e cada ID `int64` ocupa 8 bytes.

**Exemplo:** $374.785 \times 8 = 2.998.280$ bytes $\approx 3$ MB.

O vocabulário é construído com o corpus inteiro, treino e validação juntos. Com
caracteres, essa é a prática usual. A alternativa seria construí-lo só com o treino, e
caracteres que aparecem apenas na validação virariam `<UNK>`.

---

## 4. Janelas deslizantes: `NextTokenDataset`

**Objetivo:** transformar a sequência `ids` numa coleção de pares (x, y) que pode ser
acessada por número, sem copiar dados.

Um `Dataset` do PyTorch é qualquer objeto com dois métodos: `__len__` (quantos exemplos
existem) e `__getitem__` (o exemplo de número $i$). O `DataLoader` (seção 8) só precisa disso.

### 4.1 `__init__`: validar e guardar

Passo a passo (linhas 20 a 33 de `dataset.py`):

1. `torch.as_tensor(ids, dtype=torch.long)`: aceita uma lista ou um tensor e garante o tipo
   `int64` (teste `test_accepts_list_and_returns_long`). Se `ids` já é um tensor `int64`, não
   há cópia.
2. Rejeita `ids` que não seja 1-D (`ValueError`): a entrada precisa ser uma fila única de
   tokens, não uma matriz.
3. Rejeita `context_length < 1` ou `stride < 1`.
4. Rejeita sequências com menos de $T + 1$ tokens: nelas não cabe nenhuma janela.
5. Guarda `ids`, `context_length` e `stride`. **Nenhuma janela é criada aqui.** Cada janela é
   recortada só quando alguém pede por ela.

### 4.2 `__getitem__`: recortar uma janela

`NextTokenDataset` recorta a sequência `ids` em janelas. Para a janela $i$, com
$T$ = `context_length` e passo $s$ = `stride`, o início é $\text{start} = i \cdot s$ e:

```text
x = ids[start     : start + T]
y = ids[start + 1 : start + T + 1]
```

A posição inicial de cada janela é um múltiplo do stride:

$$
\text{start} = i \cdot s
$$

**Lendo a fórmula:**

- $i$: o número da janela (0, 1, 2, ...).
- $s$: o stride, quantos tokens se anda de uma janela para a seguinte.
- $\text{start}$: onde a janela $i$ começa dentro de `ids`.

**Exemplo:** com $s = 128$, a janela 3 começa em $3 \cdot 128 = 384$. Então
`x = ids[384:512]` e `y = ids[385:513]`.

Elemento por elemento, `x` e `y` são a mesma fatia de `ids`, com `y` deslocado uma posição:

$$
x_t = \text{ids}_{\,\text{start} + t}, \qquad
y_t = \text{ids}_{\,\text{start} + t + 1}, \qquad
t = 0, 1, \dots, T - 1
$$

**Lendo a fórmula:**

- $x_t$: o token da posição $t$ da entrada.
- $y_t$: o token seguinte na sequência original, que é o alvo da posição $t$.
- $\text{ids}_{\,k}$: o elemento $k$ da sequência completa.
- Consequência: $y_t = x_{t+1}$ para $t < T - 1$ (teste `test_target_is_input_shifted_by_one`).
  O último alvo, $y_{T-1} = \text{ids}_{\,\text{start}+T}$, não aparece em `x`.

**Exemplo:** com `ids = [0 1 2 3 4 5 6 7 8 9]`, $T = 3$ e $s = 3$, a janela 1 começa em
$\text{start} = 3$: $x = (3, 4, 5)$ e $y = (4, 5, 6)$.

Exemplo com `ids = [0 1 2 3 4 5 6 7 8 9]` e $T = 3$:

```text
stride = 1                          stride = 3
janela 0:  x=[0 1 2]  y=[1 2 3]     janela 0:  x=[0 1 2]  y=[1 2 3]
janela 1:  x=[1 2 3]  y=[2 3 4]     janela 1:  x=[3 4 5]  y=[4 5 6]
   ...                              janela 2:  x=[6 7 8]  y=[7 8 9]
janela 6:  x=[6 7 8]  y=[7 8 9]
```

Passo a passo no código (linhas 39 a 45):

1. Rejeita índices fora de $0, \dots, N_w - 1$ com `IndexError`. Sem isso, um índice negativo
   seria aceito pelo Python e devolveria uma janela errada em silêncio.
2. Calcula `start = index * self.stride`.
3. Recorta `window = self.ids[start : start + T + 1]`, com shape `[T + 1]`.
4. Devolve `window[:-1]` (x, todos menos o último) e `window[1:]` (y, todos menos o primeiro),
   ambos com shape `[T]`.

`window[:-1]` e `window[1:]` são *views* do mesmo tensor: nada é copiado em
`__getitem__`. A cópia só acontece quando o DataLoader empilha as janelas num batch.

Uma **view** é um tensor que aponta para os mesmos números na memória, só com outro recorte.
Mudar a view mudaria o original.

### 4.3 `__len__`: quantas janelas cabem

Cada janela consome $T + 1$ tokens, porque o último só aparece em `y`. O número de
janelas é:

$$
N_w = \left\lfloor \frac{N - T - 1}{s} \right\rfloor + 1
$$

**Lendo a fórmula:**

- $N$: quantos tokens tem a sequência.
- $T + 1$: quantos tokens cada janela ocupa.
- $N - T - 1$: o **último início válido**. Uma janela que começa ali termina exatamente no fim
  da sequência.
- $\frac{\dots}{s}$: quantos passos de tamanho $s$ cabem entre o início 0 e esse último início.
- $\lfloor \cdot \rfloor$: só contam passos inteiros.
- $+1$: conta a janela 0, que começa no início.

É exatamente a linha 37: `(len(self.ids) - self.context_length - 1) // self.stride + 1`.

**Exemplos:**

| $N$ | $T$ | $s$ | conta | $N_w$ |
|---:|---:|---:|---|---:|
| 10 | 3 | 1 | $\lfloor 6 / 1 \rfloor + 1$ | 7 |
| 10 | 3 | 3 | $\lfloor 6 / 3 \rfloor + 1$ | 3 |
| 11 | 3 | 3 | $\lfloor 7 / 3 \rfloor + 1 = 2 + 1$ | 3 |
| 337.307 | 128 | 128 | $\lfloor 337.178 / 128 \rfloor + 1 = 2.634 + 1$ | 2.635 |

As duas primeiras linhas batem com o desenho acima (janelas 0 a 6, e janelas 0 a 2).

Os tokens depois do fim da última janela são descartados (menos de $s$ tokens). No
treino atual, são 26 de 337.307.

A quantidade descartada é o resto da mesma divisão:

$$
\text{tokens descartados} = (N - T - 1) \bmod s
$$

**Lendo a fórmula:**

- $(N - T - 1)$: o último início válido, como antes.
- $\bmod s$: o resto da divisão por $s$. É a "sobra" que não completa mais um passo, por isso
  é sempre menor que $s$.

**Exemplos:** com $N = 11$, $T = 3$ e $s = 3$: $7 \bmod 3 = 1$. As janelas começam em 0, 3 e
6; a última usa os tokens 6 a 9, e o token 10 fica de fora. No treino real:
$337.178 \bmod 128 = 26$.

---

## 5. Uma janela são T exemplos, não um

`y[t]` é o token que vem depois de `x[0..t]`. Então uma janela de 128 tokens contém 128
problemas de previsão, com contextos de tamanho 1 a 128. O início da primeira janela do
corpus:

```text
t=0   'I'              -> '\n'
t=1   'I\n'            -> '\n'
t=2   'I\n\n'          -> 'D'
t=3   'I\n\nD'         -> 'o'
t=4   'I\n\nDo'        -> ' '
 ...
t=11  'I\n\nDo titulo' -> '.'
```

O tamanho do contexto cresce com a posição dentro da janela:

$$
\text{tamanho do contexto na posição } t = t + 1
$$

**Lendo a fórmula:** na posição $t$, o modelo pode usar `x[0]` até `x[t]`, que são $t + 1$
tokens. **Exemplo:** em $t = 0$ o contexto é só `'I'` (1 token); em $t = 127$, a janela
inteira (128 tokens).

O GPT resolve os $T$ problemas **em paralelo, num único forward pass**: a saída na
posição $t$ é a previsão de `y[t]`. Por isso o treinamento de Transformers é eficiente, e
também por isso a **causal mask** ([04-attention.md](04-attention.md)) é indispensável:

- `y[t]` é igual a `x[t+1]`, ou seja, a resposta da posição $t$ está na própria entrada,
  uma posição à frente.
- Se a posição $t$ pudesse olhar para $x_{t+1}$, bastaria copiar. A loss cairia para
  perto de zero sem o modelo aprender nada de linguagem.
- Na geração não existe futuro para copiar, e um modelo assim seria inútil na prática.

Dois termos desta seção:

- **Forward pass:** uma passada da entrada pelo modelo, do começo ao fim, produzindo as
  previsões.
- **Causal mask:** a regra que impede cada posição de olhar para as posições seguintes.

Como os contextos variam de 1 a $T$ tokens, o modelo aprende a prever tanto com
contexto curto quanto longo. Na geração ele precisa exatamente disso, porque começa com
um prompt de poucos caracteres.

---

## 6. O stride

O **stride** é o passo entre os inícios de janelas consecutivas. Ele decide quantas janelas
existem e o quanto elas se sobrepõem.

Números do experimento (treino, $T = 128$, `batch_size = 32`):

| stride | janelas | batches/época | alvos por época |
|-------:|--------:|--------------:|----------------:|
| 1      | 337.179 | 10.536        | 43.158.912      |
| 16     | 21.074  | 658           | 2.697.472       |
| 64     | 5.269   | 164           | 674.432         |
| 128    | 2.635   | 82            | 337.280         |

As duas últimas colunas saem de $N_w$ (seção 4.3):

$$
\text{batches por época} = \left\lfloor \frac{N_w}{B} \right\rfloor
\qquad\qquad
\text{alvos por época} = N_w \cdot T
$$

**Lendo a fórmula:**

- $N_w$: número de janelas.
- $B$: janelas por batch (`batch_size`).
- $\lfloor \cdot \rfloor$: arredonda para baixo, porque o batch incompleto do fim é descartado
  (`drop_last`, seção 8).
- $T$: cada janela tem $T$ alvos (seção 5).

**Exemplos:**

- stride 128: $\lfloor 2.635 / 32 \rfloor = \lfloor 82{,}34 \rfloor = 82$ batches e
  $2.635 \cdot 128 = 337.280$ alvos;
- stride 16: $\lfloor 21.074 / 32 \rfloor = \lfloor 658{,}56 \rfloor = 658$ batches e
  $21.074 \cdot 128 = 2.697.472$ alvos.

A coluna "alvos por época" conta todas as janelas. Com `drop_last`, as 11 janelas do batch
incompleto ficam de fora, e o loop de treino vê de fato $82 \cdot 32 \cdot 128 = 335.872$
alvos por época (conta feita à parte, com os mesmos números).

- **`stride = 1`** gera todos os pares (contexto, alvo) possíveis. Cada token aparece como
  alvo em até $T$ janelas, uma vez em cada posição. Janelas vizinhas são quase idênticas
  (diferem por um token), então uma "época" contém muita redundância.
- **`stride = T`** faz cada token ser alvo exatamente uma vez por época. É a definição
  mais limpa de época. A desvantagem é que as fronteiras das janelas são fixas: um token
  que cai na posição 0 de uma janela sempre é previsto com contexto de 1 caractere.

O `tiny.yaml` começa com `stride: 128`. Isso dá 82 batches por época, e as 10 épocas
configuradas somam 820 passos de otimização, provavelmente pouco. A escolha deve ser
revista no training loop ([11-training.md](11-training.md)), medindo tokens/segundo. Uma alternativa comum
(nanoGPT) é sortear inícios aleatórios a cada batch, o que elimina as fronteiras fixas.

Um **passo de otimização** é uma atualização dos parâmetros, feita uma vez por batch. Por isso:

$$
\text{passos} = \text{épocas} \times \text{batches por época} = 10 \times 82 = 820
$$

**Lendo a fórmula:** cada época percorre todos os batches, e cada batch gera um passo.

---

## 7. Treino × validação: `split_train_val`

**Objetivo:** reservar um texto que o modelo nunca vê no treino, para medir se ele
**generaliza**, isto é, se acerta em texto novo e não só no que já leu.

`split_train_val` corta a sequência de IDs **por posição**: os primeiros 90% vão para o
treino (337.307 tokens) e os últimos 10% para a validação (37.478 tokens).

Passo a passo (linhas 11 a 15 de `loader.py`):

1. Rejeita `val_fraction` fora do intervalo entre 0 e 1 (exclusive) com `ValueError`. Os
   testes usam 0, 1 e −0,1.
2. Calcula `split = len(ids) - int(len(ids) * val_fraction)`.
3. Devolve `ids[:split]` (treino) e `ids[split:]` (validação). Os dois pedaços, colados, dão
   exatamente a sequência original (teste `test_split_is_contiguous`).

Em fórmula:

$$
n_{\text{val}} = \lfloor N \cdot f \rfloor,
\qquad
n_{\text{treino}} = N - n_{\text{val}}
$$

**Lendo a fórmula:**

- $N$: tokens do corpus inteiro.
- $f$: a fração de validação (`val_fraction`).
- $\lfloor N \cdot f \rfloor$: a parte inteira do produto. `int()` faz isso para números
  positivos.
- $n_{\text{treino}}$: o que sobra fica com o treino.

**Exemplos:**

- corpus: $374.785 \cdot 0{,}1 = 37.478{,}5$, então $n_{\text{val}} = 37.478$ e
  $n_{\text{treino}} = 374.785 - 37.478 = 337.307$;
- teste: com $N = 100$ e $f = 0{,}1$, são 10 tokens de validação e 90 de treino.

```text
ids: [━━━━━━━━━━━━━━━━━━━━━━━━━━━ treino ━━━━━━━━━━━━━━━━━━━━━━━━━━━|━━ validação ━━]
                                         "...Eu não sabia q" | "ue lhe respondesse..."
```

A saída do experimento mostra o ponto de corte, no meio de uma palavra:

```text
treino:    337,307 tokens
validação: 37,478 tokens
fim do treino:       ...'dios aconselhados aos melancolicos. Eu não sabia q'
início da validação:    'ue lhe respondesse; recusei as diversões. Como ins'...
```

**Por que não sortear janelas para cada lado?** Com `stride < T`, janelas se sobrepõem.
Uma janela de validação compartilharia a maior parte dos tokens com janelas de treino, e
a validation loss mediria memorização, não generalização. Mesmo sem sobreposição,
janelas vizinhas continuam a mesma frase e a mesma cena. O corte contíguo garante que a
validação é texto que o modelo nunca viu.

- **Validation loss:** a loss calculada no texto de validação.
- **Memorização:** acertar porque já viu aquele trecho, não porque aprendeu padrões.

O preço é uma **mudança de distribuição**: a validação são os capítulos finais do livro,
com eventos e nomes que aparecem pouco no início. É esperado que a validation loss fique
um pouco acima da training loss mesmo sem overfitting.

- **Mudança de distribuição:** o texto de validação é, em média, diferente do texto de treino.
- **Overfitting:** quando o modelo se ajusta tanto ao treino que piora em texto novo. O sinal
  típico é a training loss continuar caindo enquanto a validation loss sobe.

O loader de validação usa sempre `stride = context_length` e `shuffle = False`,
independentemente da config: cada token de validação é avaliado uma única vez, sempre na
mesma ordem. Isso está em [`train.py`](../train.py), onde o `val_loader` é criado.

---

## 8. Batching: `create_dataloader`

**Objetivo:** entregar as janelas em batches de shape fixo, numa ordem embaralhada mas
reproduzível.

Passo a passo (linhas 28 a 38 de `loader.py`):

1. Cria o `NextTokenDataset` com `context_length` e `stride`.
2. Se o dataset tem menos janelas que `batch_size`, lança `ValueError` (explicado abaixo).
3. Devolve um `DataLoader` do PyTorch com `batch_size`, `shuffle`, `drop_last=True` e um
   `torch.Generator` inicializado com a `seed`.

```text
Sampler  ──►  índices de janelas  ──►  dataset[i] → (x_i, y_i)  ──►  collate (stack)
(permutação                             x_i, y_i: [T]                 x: [B, T]
 aleatória com seed)                                                  y: [B, T]
```

- **Sampler:** a peça do `DataLoader` que decide a ordem dos índices $i$. Com `shuffle=True`,
  ele sorteia uma permutação (uma ordem embaralhada) de $0, \dots, N_w - 1$.
- **Collate:** a peça que junta os exemplos num batch. O padrão empilha (`torch.stack`) $B$
  tensores `[T]` num tensor `[B, T]`.

- **`shuffle`** embaralha a **ordem das janelas**, nunca os caracteres dentro delas. Sem
  isso, cada batch viria de um mesmo trecho do livro, e os gradientes de batches
  consecutivos seriam correlacionados.
- **`seed`** fixa o `torch.Generator` do sampler: a mesma config produz a mesma sequência
  de batches, o que torna experimentos comparáveis.
- **`drop_last=True`** descarta o último batch incompleto, para que todo batch tenha
  shape `[B, T]` (no treino, 11 janelas ficam de fora). Se o dataset tiver menos janelas
  que `batch_size`, isso daria **zero batches sem erro**, e o loop de treino simplesmente
  não rodaria. Por isso `create_dataloader` lança `ValueError` nesse caso.

Detalhes que valem conferir:

- **Seed** é o número que inicializa um gerador de números aleatórios. Mesma seed, mesma
  sequência de sorteios.
- O gerador é criado **uma vez**, dentro de `create_dataloader`. Cada nova passada pelo
  loader (cada época) sorteia uma permutação nova a partir dele. Resultado: as épocas têm
  ordens diferentes, mas a sequência de ordens se repete sempre que a config é a mesma
  (conferido rodando o loader duas vezes com seed 0).
- **Exemplo de `drop_last`** (teste `test_batches_have_consistent_shape`): com 1.000 tokens,
  $T = 16$ e $s = 16$, são $\lfloor 983 / 16 \rfloor + 1 = 62$ janelas. Com $B = 8$, saem
  $\lfloor 62 / 8 \rfloor = 7$ batches completos, e 6 janelas ficam de fora.

Cada batch tem $B \cdot T = 32 \cdot 128 = 4.096$ alvos.

$$
\text{alvos por batch} = B \cdot T
$$

**Lendo a fórmula:** $B$ janelas, cada uma com $T$ alvos. **Exemplo:**
$32 \cdot 128 = 4.096$, o `tokens por batch` da saída abaixo.

A saída da seção "Batches" do experimento. As três primeiras janelas do primeiro batch vêm de
pontos diferentes do livro, efeito do `shuffle`:

```text
x: (32, 128) torch.int64
y: (32, 128) torch.int64
batches por época: 82 | tokens por batch: 4,096
  x[0] = 'alguma cousa menos austera, que pede umas linhas de repouso e preparaç'...
  x[1] = 'e ides ler:\n\nOh! flòr do ceu! oh! flòr candida e pura!\n\nComo e porque '...
  x[2] = 'sa. Minha mãe disse-lhe que fosse falar-me ao quarto.\n\n--Dá licença? p'...
```

---

## 9. Interface

```python
import torch
from data.loader import create_dataloader, split_train_val
from tokenizer import CharTokenizer

tokenizer = CharTokenizer.from_text(text)
ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)   # [N]
train_ids, val_ids = split_train_val(ids, val_fraction=0.1)

train_loader = create_dataloader(
    train_ids, context_length=128, stride=128, batch_size=32, shuffle=True, seed=42
)
for x, y in train_loader:   # x, y: [32, 128] int64
    ...
```

---

## 10. Decisões e limitações

| decisão | consequência |
|---|---|
| sem `<BOS>`/`<EOS>` | o livro é um único documento contínuo, então não há fronteiras entre documentos a marcar |
| janelas atravessam parágrafos e capítulos | intencional: o modelo aprende `\n\n` e o formato dos títulos de capítulo |
| vocabulário do corpus inteiro | a validação não gera `<UNK>`; ver seção 3 |
| fronteiras de janela fixas entre épocas | ver seção 6 |
| `vocab_size` saiu do `tiny.yaml` | vem de `tokenizer.vocab_size` (97); um número fixo na config poderia divergir do corpus |

**Uma estimativa de escala.** Com $d = 128$ e 4 camadas, cada bloco Transformer tem
cerca de $12 d^2 \approx 197$ mil parâmetros: $4d^2$ na attention e $8d^2$ no
feed-forward. O modelo inteiro fica perto de **0,8 milhão de parâmetros**; a contagem
exata sai em [08-gpt.md](08-gpt.md). O treino tem ~337 mil tokens, **menos de um token por parâmetro**.
Heurísticas de escala para modelos grandes (Chinchilla) sugerem ~20 tokens por
parâmetro. Aqui, várias épocas sobre o mesmo livro vão levar a overfitting em algum
momento, e acompanhar a validation loss ([14-evaluation.md](14-evaluation.md)) é o que vai mostrar quando.

A estimativa por bloco, escrita por partes:

$$
\text{parâmetros por bloco} \approx 12\, d^2 = \underbrace{4\, d^2}_{\text{attention}} + \underbrace{8\, d^2}_{\text{feed-forward}}
$$

**Lendo a fórmula:**

- $d$: tamanho dos vetores do modelo (`d_model` = 128).
- $d^2 = d \cdot d$: o número de parâmetros de uma matriz $d \times d$.
- $4\, d^2$: a attention tem 4 matrizes $d \times d$ ([04-attention.md](04-attention.md),
  [05-multi-head-attention.md](05-multi-head-attention.md)).
- $8\, d^2$: o feed-forward tem 2 matrizes $d \times 4d$, cada uma com $4\, d^2$
  ([06-feed-forward.md](06-feed-forward.md)).
- $\approx$: ignora os biases e as LayerNorms, que são poucos parâmetros.
- $\underbrace{\cdots}_{\text{nome}}$: a chave embaixo só dá nome à parte da conta.

**Exemplo:** $12 \cdot 128^2 = 12 \cdot 16.384 = 196.608 \approx 197$ mil. A contagem exata do
modelo inteiro, com weight tying, é 820.096 ([architecture.md](architecture.md)).

A comparação com a heurística do Chinchilla:

$$
\frac{\text{tokens de treino}}{\text{parâmetros}} = \frac{337.307}{820.096} \approx 0{,}41
$$

**Lendo a fórmula:** quantos tokens de treino existem para cada parâmetro do modelo. O
Chinchilla sugere cerca de 20; para isso, este modelo precisaria de
$20 \times 820.096 \approx 16{,}4$ milhões de tokens.

---

## Resumo

- Um exemplo de treino é um par (contexto, próximo token). A regra da cadeia mostra que prever
  o próximo token, posição por posição, é o mesmo que modelar o texto inteiro.
- `clean()` deixa só o romance, junta as linhas da prosa, preserva os versos e normaliza para
  NFC: 374.785 caracteres, 97 tokens.
- O texto é tokenizado uma vez, num tensor `int64` de ~3 MB.
- `NextTokenDataset` recorta janelas sob demanda: `x = ids[start:start+T]` e `y` deslocado uma
  posição. São $N_w = \lfloor (N - T - 1)/s \rfloor + 1$ janelas (2.635 no treino).
- Uma janela contém $T$ problemas de previsão, resolvidos em paralelo; por isso a causal mask é
  indispensável.
- O stride troca número de janelas por redundância. Com `stride = 128`: 82 batches por época.
- A validação é o trecho final (10%), cortado por posição para não vazar tokens do treino.
- O `DataLoader` embaralha janelas com seed, descarta o batch incompleto e entrega
  `x, y: [32, 128]`.

## Para checar o entendimento

1. Por que uma janela precisa de $T + 1$ tokens e não $T$?
2. Com `stride = 1`, quantas vezes um token do meio do corpus aparece como alvo numa época?
   Em quais posições da janela?
3. Se as janelas (com `stride = 16`) fossem sorteadas aleatoriamente entre treino e
   validação, o que aconteceria com a validation loss? Por quê?
4. Por que um modelo sem causal mask chegaria a uma loss perto de zero neste dataset, e
   por que ele não conseguiria gerar texto?
5. Por que a loss inicial esperada é $\ln 97$? O que significaria uma loss inicial de 10?
6. Por que o `NextTokenDataset` não guarda todas as janelas prontas numa lista?
7. Com `stride = 64`, quantos tokens do treino ficam de fora da última janela?
