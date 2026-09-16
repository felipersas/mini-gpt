# Tokenizer

> **Como texto vira números?**

Código: [`tokenizer/vocab.py`](../tokenizer/vocab.py) e
[`tokenizer/tokenizer.py`](../tokenizer/tokenizer.py). Testes:
[`tests/test_tokenizer.py`](../tests/test_tokenizer.py). Experimento:
`uv run python -m experiments.e01_tokenizer`. Notação geral: [00-notacao.md](00-notacao.md).
Visão geral: [architecture.md](architecture.md).

---

## Para que serve

### O problema

Uma rede neural só opera sobre números. O tokenizer é a ponte: ele corta o texto em
pedaços (**tokens**) e dá a cada pedaço um número inteiro (**ID**). O resto do GPT
nunca vê texto, só esses IDs.

```text
"a casa"  ──encode──►  [16, 5, 18, 16, 32, 16]  ──decode──►  "a casa"
```

Termos usados daqui em diante:

- **Token:** a menor unidade de texto que o modelo enxerga. No Mini-GPT, um token é um
  caractere: uma letra, um espaço, uma vírgula, uma quebra de linha.
- **ID:** o número inteiro que representa um token.
- **Vocabulário:** a lista de todos os tokens que o tokenizer conhece, cada um com o seu ID.
- **Corpus:** o texto usado para construir o vocabulário e treinar o modelo. Aqui, o livro
  *Dom Casmurro* ([02-dataset.md](02-dataset.md)).
- **encode** ("codificar"): texto → lista de IDs.
- **decode** ("decodificar"): lista de IDs → texto.

Os IDs do exemplo acima (`'a'` = 16, `' '` = 5, `'c'` = 18, `'s'` = 32) vêm do vocabulário
pequeno do experimento, montado com um único parágrafo. No vocabulário do corpus completo, os
números são outros (`'a'` = 53, `'c'` = 55; ver [03-embeddings.md](03-embeddings.md)). O ID
depende do vocabulário.

### Uma analogia

Pense na lista de chamada de uma turma. Cada aluno recebe um número, que é só a posição dele
na lista em ordem alfabética. O número 16 não diz nada sobre o aluno: diz apenas em que linha
ele está. Se alguém reordenar a lista, o número 16 passa a ser outra pessoa.

O vocabulário é essa lista, só que de caracteres. E, como na chamada, todo mundo precisa usar
**a mesma lista**: o modelo aprende o que significa "16" e depende de 16 ser sempre `'a'`.

### O que entra e o que sai

| operação | entra | sai |
|---|---|---|
| construir (`from_text`) | o corpus inteiro: `str` com 374.785 caracteres | vocabulário com $V = 97$ tokens (93 caracteres + 4 especiais) |
| `encode` | `str`, ex.: `"a casa"` (6 caracteres) | `list[int]` com 6 IDs; para o corpus inteiro, 374.785 IDs |
| `decode` | `list[int]` ou tensor de IDs | `str` |

Dois termos do PyTorch que aparecem em todos os documentos:

- **Tensor:** a estrutura de dados do PyTorch, um bloco de números com um ou mais eixos.
- **Shape:** o tamanho de cada eixo. Os 374.785 IDs do corpus viram, no dataset
  ([02-dataset.md](02-dataset.md)), um tensor de shape `[374.785]` (um eixo só).

O tokenizer trabalha com **uma sequência por vez** e não sabe nada de batches. Quem recorta os
IDs em pedaços de shape `[32, 128]` é o dataset (`02-dataset.md`).

### Onde este documento entra

```text
texto: "Uma noite destas, vindo da cidade..."
  │
  ▼  tokenizer                                 str → IDs [374.785]              ← este documento
  ▼  dataset e batches (02-dataset.md)         IDs → x, y: [32, 128]
  ▼  embeddings (03-embeddings.md)             [32, 128] → [32, 128, 128]
  ▼  blocos Transformer × 4 (04 a 08)          [32, 128, 128] → [32, 128, 128]
  ▼  LM head (09-lm-head.md)                   [32, 128, 128] → logits [32, 128, 97]
  ▼  loss (10-loss.md, ainda não implementada) logits e y → um número
```

### O que daria errado sem esta parte

- **A rede não teria como receber texto.** A primeira camada (`03-embeddings.md`) só aceita
  índices inteiros.
- **Sem uma tabela fixa, os IDs mudariam de uma execução para outra.** Um modelo treinado com
  `'a'` = 16 leria errado um texto codificado com `'a'` = 17.
- **Sem um token para "desconhecido",** um único emoji num prompt derrubaria o programa.

---

## Notação deste documento

Este documento usa poucas fórmulas. Os símbolos gerais estão em [00-notacao.md](00-notacao.md).

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $V$ | "vê" | tamanho do vocabulário: quantos tokens diferentes existem, contando os 4 especiais | 97 no corpus; 41 no trecho do experimento | `tok.vocab_size`, `len(vocab)` |
| $C$ | "cê" | conjunto dos caracteres distintos do texto, depois da normalização NFC (seção 8) | 93 caracteres no corpus | `set(_normalize(text))` |
| $\lvert C \rvert$ | "tamanho de cê" | quantos elementos o conjunto $C$ tem (cardinalidade) | 93 | `len(set(...))` |
| $n$ | "ene" | número de caracteres de um texto | `"a casa"`: $n = 6$ | `len(text)` |
| $t$ | "tê" | posição de um caractere no texto, contando de 0 | de 0 a $n - 1$ | índice em `text[t]` |
| $c_t$ | "cê tê" | o caractere na posição $t$ | em `"a casa"`, $c_2$ = `'c'` | `text[t]` |
| $x_t$ | "xis tê" | o ID do caractere $c_t$ | em `"a casa"`, $x_2 = 18$ (vocabulário do trecho) | `ids[t]` |
| $\text{id}(c)$ | "id de cê" | ID do token $c$; se $c$ não está no vocabulário, o ID de `<UNK>` (1) | $\text{id}(\texttt{a}) = 16$ no trecho | `vocab.get_id(c)` |
| $\text{token}(x)$ | "token de xis" | a operação inversa: o token que tem o ID $x$ | $\text{token}(16)$ = `'a'` | `vocab.get_token(x)` |
| $\text{encode}$, $\text{decode}$ | "encoude", "dicoude" | texto → IDs e IDs → texto | | `tok.encode`, `tok.decode` |
| $T$ | "tê" maiúsculo | tamanho do contexto: quantos tokens o modelo lê de uma vez (só na seção 6) | 128 | `context_length` |
| $d$ | "dê" | tamanho de cada vetor de embedding (só na seção 10 e nas perguntas) | 128 | `d_model` |
| $\in$ | "pertence a" | o elemento da esquerda faz parte do conjunto da direita | $x_t \in \{0, \dots, 96\}$ | `in` |
| $\{0, 1, \dots, V - 1\}$ | "conjunto de zero a vê menos um" | todos os inteiros de 0 até $V - 1$, sem pular nenhum | $\{0, \dots, 96\}$ | `range(vocab_size)` |
| $=$, $\neq$ | "igual a", "diferente de" | comparação | | `==`, `!=` |
| $\approx$ | "aproximadamente" | igual depois de arredondar | $485 / 86 \approx 5{,}64$ | |
| $\times$ | "vezes" ou "por" | multiplicação; em tamanhos de matriz, "linhas por colunas" | $97 \times 128$ | `*` |
| `U+00E1` | "U mais zero zero E um" | um **code point** Unicode, escrito em hexadecimal (base 16) | `'á'` = `U+00E1` | `hex(ord("á"))` |

Atenção a $t$ e $T$: minúsculo é **uma posição**; maiúsculo é **o tamanho do contexto**.

---

## 1. O vocabulário é só uma tabela

**Objetivo:** guardar a correspondência token ↔ ID de forma que as duas direções sejam
rápidas e sempre concordem.

O `Vocab` guarda duas estruturas que precisam concordar sempre:

| estrutura      | tipo             | exemplo                   |
|----------------|------------------|---------------------------|
| `id_to_token`  | `list[str]`      | `id_to_token[16] == "a"`  |
| `token_to_id`  | `dict[str, int]` | `token_to_id["a"] == 16`  |

- `id_to_token` responde "qual token tem o ID 16?" (ler uma posição da lista).
- `token_to_id` responde "qual o ID de `'a'`?" (procurar uma chave no dicionário). Procurar
  numa lista seria lento; o dicionário encontra a resposta direto.

O ID de um token é **a posição dele na lista**. Não há nada de especial no número 16:
ele só diz em que linha da tabela o "a" está. Em [03-embeddings.md](03-embeddings.md), esse
número vai escolher uma linha da matriz de embeddings. É por isso que os IDs precisam ir de `0`
a `vocab_size - 1`, sem buracos.

Em notação matemática, a regra "sem buracos" fica assim:

$$
x \in \{0, 1, \dots, V - 1\}
$$

**Lendo a fórmula:**

- $x$: um ID qualquer produzido pelo tokenizer.
- $\in$: "pertence a".
- $\{0, 1, \dots, V - 1\}$: o conjunto dos inteiros de 0 até $V - 1$. As reticências
  ($\dots$) significam "e todos os inteiros no meio".
- $V$: o tamanho do vocabulário.

**Exemplo:** com $V = 97$, os IDs vão de 0 a 96. A matriz de embeddings (`03-embeddings.md`) tem
97 linhas, numeradas de 0 a 96. Um ID 97 não teria linha, e o PyTorch lançaria `IndexError`
(teste `test_id_outside_vocab_raises`, no mesmo documento).

### 1.1 `Vocab.__init__`: montar as duas estruturas

**Objetivo:** receber uma lista de tokens e criar as duas tabelas, com os especiais sempre
nos primeiros IDs.

```python
self.id_to_token: list[str] = list(dict.fromkeys(SPECIAL_TOKENS + tokens))
self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.id_to_token)}
```

Passo a passo (linhas 20 a 23 de `vocab.py`):

1. `SPECIAL_TOKENS + tokens` coloca os 4 tokens especiais (seção 3) na frente da lista
   recebida.
2. `dict.fromkeys(...)` remove duplicatas. Ele mantém a **primeira** ocorrência de cada
   token e preserva a ordem. Se um especial aparecer de novo em `tokens`, ele não sai do lugar.
3. `list(...)` transforma o resultado em `id_to_token`. O índice na lista é o ID.
4. `enumerate` percorre essa lista, gerando pares (ID, token), e monta o dicionário inverso
   `token_to_id`.

**Exemplo** (teste `test_vocab_removes_duplicates`): `Vocab(["a", "a", "b", "<PAD>"])` gera
`id_to_token = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "a", "b"]`. São $V = 6$ tokens, e `<PAD>`
continua no ID 0.

Os IDs dos especiais ficam guardados como atributos da classe (`pad_id`, `unk_id`, `bos_id`,
`eos_id`), calculados com `SPECIAL_TOKENS.index(...)`. O resto do código usa esses nomes, e
não os números soltos.

### 1.2 `__len__`, `get_id` e `get_token`

**Objetivo:** consultar a tabela nas duas direções, com um comportamento definido para
entradas inválidas.

| método | o que faz | exemplo (vocabulário do trecho, $V = 41$) |
|---|---|---|
| `len(vocab)` | devolve $V$, o número de tokens | `41` |
| `get_id(token)` | procura o token no dicionário; se não achar, devolve `unk_id` (1) | `get_id("a") == 16`; `get_id("á") == 1` |
| `get_token(id)` | devolve `id_to_token[id]`; lança `IndexError` se o ID não está entre 0 e $V - 1$ | `get_token(16) == "a"`; `get_token(41)` → erro |

A regra do `get_id`, escrita como fórmula:

$$
\text{id}(c) =
\begin{cases}
\text{posição de } c \text{ na lista} & \text{se } c \text{ está no vocabulário} \\
1 \ (\text{o ID de UNK}) & \text{caso contrário}
\end{cases}
$$

**Lendo a fórmula:**

- $c$: um token (no Mini-GPT, um caractere).
- $\text{id}(c)$: o número que o tokenizer usa no lugar de $c$.
- A chave `{` significa "depende do caso". Cada linha é um caso, com a condição à direita.
- Primeira linha: se o caractere existe no vocabulário, o ID é a posição dele na lista.
- Segunda linha: se não existe, o ID é 1, o de `<UNK>` ("desconhecido").

**Exemplo:** no vocabulário do trecho, $\text{id}(\texttt{a}) = 16$. O trecho não tem nenhum
`'á'`, então $\text{id}(\texttt{á}) = 1$.

**Por que `get_token` checa o intervalo.** Em Python, `lista[-1]` devolve o **último**
elemento. Sem a checagem, `get_token(-1)` devolveria `'ê'` (ID 40) em silêncio, e um bug que
gerou um ID negativo passaria despercebido. Os testes cobrem `-1` e um ID igual a $V$.

---

## 2. Construindo o vocabulário: `CharTokenizer.from_text`

**Objetivo:** gerar o vocabulário a partir de um corpus, sempre igual para o mesmo corpus.

```python
chars = sorted(set(_normalize(text)))
return cls(Vocab(chars))
```

Passo a passo (linhas 16 a 20 de `tokenizer.py`):

1. `_normalize(text)` converte o texto para a forma NFC (seção 8), para que um mesmo
   caractere visual tenha uma única representação.
2. `set(...)` cria um **conjunto**: cada caractere distinto aparece uma vez só. Os 374.785
   caracteres do corpus viram 93 caracteres distintos.
3. `sorted(...)` ordena pelo **code point**, o número que o padrão Unicode atribui a cada
   caractere. Por isso `'\n'` (code point 10) e `' '` (32) vêm primeiro, as maiúsculas
   (`'A'` = 65) vêm antes das minúsculas (`'a'` = 97), e as letras acentuadas (`'ã'` = 227)
   ficam no fim.
4. `Vocab(chars)` acrescenta os 4 especiais na frente e monta as duas tabelas.

O `CharTokenizer.from_text` monta a tabela com os caracteres distintos do texto, em ordem
(`sorted`). A ordem fixa garante que o mesmo corpus gere sempre os mesmos IDs. Sem isso,
um modelo treinado hoje leria errado os IDs gerados amanhã.

O motivo concreto: a ordem em que um `set` de strings é percorrido em Python pode mudar de uma
execução para outra. `sorted` elimina essa variação. O teste
`test_vocab_does_not_depend_on_char_order` confirma que `"abc"` e `"cba"` geram o mesmo
vocabulário.

O vocabulário do trecho usado no experimento (primeiro parágrafo do capítulo I, em ortografia
atualizada) mostra essa ordem:

```text
=== Vocabulário ===
vocab_size = 41
  0 '<PAD>'     1 '<UNK>'     2 '<BOS>'     3 '<EOS>'     4 '\n'     5 ' '     6 ','     7 '-'
  8 '.'     9 ';'    10 'A'    11 'C'    12 'E'    13 'N'    14 'S'    15 'U'
 16 'a'    17 'b'    18 'c'    19 'd'    20 'e'    21 'f'    22 'g'    23 'h'
 24 'i'    25 'l'    26 'm'    27 'n'    28 'o'    29 'p'    30 'q'    31 'r'
 32 's'    33 't'    34 'u'    35 'v'    36 'z'    37 'ã'    38 'ç'    39 'é'
 40 'ê'
```

Quantos tokens o vocabulário terá? Os especiais mais os caracteres distintos:

$$
V = 4 + \lvert C \rvert
$$

**Lendo a fórmula:**

- $V$: tamanho do vocabulário.
- $4$: os tokens especiais `<PAD>`, `<UNK>`, `<BOS>` e `<EOS>`.
- $C$: o conjunto dos caracteres distintos do texto, depois da normalização NFC.
- $\lvert C \rvert$: quantos elementos há em $C$.

**Exemplos:**

- corpus completo: $\lvert C \rvert = 93$, então $V = 4 + 93 = 97$;
- trecho do experimento: $V = 41$, então $\lvert C \rvert = 41 - 4 = 37$ caracteres distintos.

A propriedade `vocab_size` devolve exatamente esse número (`len(self.vocab)`). O construtor
`CharTokenizer(vocab)` só guarda o vocabulário recebido. Isso permite criar um tokenizer a
partir de um vocabulário já pronto, por exemplo um vocabulário salvo junto com o modelo
([15-checkpoints.md](15-checkpoints.md)).

---

## 3. Tokens especiais

**Objetivo:** reservar IDs para sinais que não são texto (início, fim, preenchimento,
desconhecido).

Os quatro primeiros IDs são sempre os mesmos, em qualquer vocabulário:

| ID | token   | para que serve                                                          |
|----|---------|-------------------------------------------------------------------------|
| 0  | `<PAD>` | completa sequências curtas até um tamanho comum, para caberem juntas no mesmo batch |
| 1  | `<UNK>` | substitui qualquer token que não está no vocabulário                    |
| 2  | `<BOS>` | marca o início de um texto                                              |
| 3  | `<EOS>` | marca o fim; na geração ([13-generation.md](13-generation.md)), é o sinal de que o modelo pode parar |

**Batch** é um grupo de sequências processadas juntas ([02-dataset.md](02-dataset.md)). Para
empilhar sequências num tensor, todas precisam ter o mesmo tamanho; o `<PAD>` ("padding",
preenchimento) serve para completar as mais curtas. No Mini-GPT atual, o dataset não usa
nenhum dos especiais:
todas as janelas têm exatamente o mesmo tamanho, e o livro é um único documento.

Eles não são texto: só entram na sequência por ID, via `encode(..., add_special_tokens=True)`.
Escrever `"<EOS>"` no texto gera 5 IDs de caracteres, não o ID 3. Se o texto do usuário
pudesse produzir tokens especiais, qualquer pessoa conseguiria "encerrar" a sequência
por dentro.

Isso acontece naturalmente: o `encode` olha **um caractere por vez** (`'<'`, `'E'`, `'O'`,
`'S'`, `'>'`), e nenhum caractere sozinho é igual à string `"<EOS>"`. O teste
`test_special_token_text_is_not_a_special_token` verifica isso.

No `decode`, `<PAD>`, `<BOS>` e `<EOS>` são omitidos por padrão. O `<UNK>` **nunca** é
omitido: se algo se perdeu no encode, isso precisa aparecer no texto.

---

## 4. `encode`: texto → IDs

**Objetivo:** transformar uma string numa lista de IDs, um por caractere.

```python
ids = [self.vocab.get_id(char) for char in _normalize(text)]
if add_special_tokens:
    ids = [self.vocab.bos_id, *ids, self.vocab.eos_id]
return ids
```

Passo a passo (linhas 31 a 34):

1. Normaliza o texto para NFC (seção 8).
2. Percorre o texto caractere por caractere e troca cada um pelo seu ID (`get_id`).
   Caracteres desconhecidos viram 1 (`<UNK>`).
3. Se `add_special_tokens=True`, coloca `<BOS>` (2) no início e `<EOS>` (3) no fim.

A mesma coisa em fórmula:

$$
\text{encode}(c_0\, c_1 \cdots c_{n-1}) = (x_0, x_1, \dots, x_{n-1}), \qquad x_t = \text{id}(c_t)
$$

**Lendo a fórmula:**

- $c_0\, c_1 \cdots c_{n-1}$: o texto, escrito como os seus caracteres lado a lado.
- $n$: quantos caracteres o texto tem.
- $(x_0, x_1, \dots, x_{n-1})$: a lista de IDs, com o mesmo tamanho $n$.
- $x_t = \text{id}(c_t)$: o ID da posição $t$ depende **só** do caractere da posição $t$. O
  encode não olha os vizinhos.
- $t$: vai de 0 a $n - 1$.

**Exemplo** (vocabulário do trecho):

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $c_t$ | `'a'` | `' '` | `'c'` | `'a'` | `'s'` | `'a'` |
| $x_t$ | 16 | 5 | 18 | 16 | 32 | 16 |

O `'a'` aparece três vezes e recebe sempre o ID 16.

Com `add_special_tokens=True`, o resultado é $(2, x_0, \dots, x_{n-1}, 3)$: são $n + 2$ IDs.
No experimento, `'a casa do rapaz'` tem 15 caracteres e vira 17 IDs:

```text
texto:   'a casa do rapaz'
IDs:     [2, 16, 5, 18, 16, 32, 16, 5, 19, 28, 5, 31, 16, 29, 16, 36, 3]
decode:  'a casa do rapaz'
round-trip ok: True
sem pular especiais: '<BOS>a casa do rapaz<EOS>'
```

---

## 5. `decode`: IDs → texto

**Objetivo:** fazer o caminho inverso, para ler o que o modelo gerou.

```python
skipped = {vocab.pad_id, vocab.bos_id, vocab.eos_id} if skip_special_tokens else set()
tokens = (vocab.get_token(int(i)) for i in ids if int(i) not in skipped)
return "".join(tokens)
```

Passo a passo (linhas 38 a 42):

1. Monta o conjunto de IDs a pular: $\{0, 2, 3\}$ por padrão, ou nenhum se
   `skip_special_tokens=False`. O 1 (`<UNK>`) nunca está nesse conjunto.
2. Para cada ID, converte com `int(i)`. Isso permite receber um tensor do PyTorch: os
   elementos de um tensor são tensores de 0 dimensões, e um tensor nunca seria encontrado
   dentro de um `set` de inteiros.
3. Troca cada ID restante pelo seu token (`get_token`).
4. Junta tudo numa string (`"".join`).

Em fórmula (com `skip_special_tokens=True`, omitindo os IDs 0, 2 e 3):

$$
\text{decode}(x_0, x_1, \dots, x_{n-1}) = \text{token}(x_0)\ \text{token}(x_1) \cdots \text{token}(x_{n-1})
$$

**Lendo a fórmula:**

- $(x_0, \dots, x_{n-1})$: a lista de IDs recebida.
- $\text{token}(x_t)$: o token de cada ID (`get_token`).
- Escrever os tokens lado a lado significa **concatenar**: colar as strings em sequência.

**Exemplo:** `decode([2, 16, 5, 18, 16, 32, 16, 3])` pula o 2 e o 3 e junta
`'a'`, `' '`, `'c'`, `'a'`, `'s'`, `'a'`: o resultado é `"a casa"`.

### O critério de ida e volta

O decode precisa desfazer o encode exatamente:

$$
\text{decode}(\text{encode}(\text{texto})) = \text{texto}
$$

**Lendo a fórmula:**

- $\text{encode}(\text{texto})$: primeiro, o texto vira IDs.
- $\text{decode}(\dots)$: depois, os IDs voltam a ser texto.
- $= \text{texto}$: o resultado precisa ser **idêntico** ao original. Essa propriedade se chama
  *round-trip* ("ida e volta").

**Exemplo:** `'a casa do rapaz'` passa no teste (`round-trip ok: True` na saída acima). A
seção 7 mostra um texto que não passa. O critério vale para textos cujos caracteres estão
todos no vocabulário e que já estão em NFC (seção 8).

---

## 6. Por que caracteres, e não palavras

Resultado de `uv run python -m experiments.e01_tokenizer` sobre o primeiro parágrafo de
*Dom Casmurro*:

|                                          | caracteres | palavras |
|------------------------------------------|-----------:|---------:|
| tokens no trecho                         | 485        | 86       |
| tamanho do vocabulário                   | 41         | 70       |
| `<UNK>` em `"o gato está na casa"`       | 1 de 19    | 4 de 5   |

A saída do experimento que gerou a tabela:

```text
'o gato está na casa' com o vocabulário do trecho:
  por caracteres: 1 <UNK> em 19 tokens
  por palavras:   4 <UNK> em 5 tokens ['gato', 'está', 'na', 'casa']

=== Caracteres × palavras (no mesmo trecho) ===
caracteres:  485 tokens | vocabulário 41
palavras:     86 tokens | vocabulário 70
palavras que aparecem uma única vez: 61/70
'versos' e 'versos.' contam como palavras diferentes: True
context_length=128 em caracteres ≈ 23 palavras
```

- **Por palavras**, 61 das 70 palavras aparecem **uma única vez**. O modelo veria cada
  uma delas num único contexto, pouco para aprender algo. E `split()` trata `"versos"` e
  `"versos."` como palavras diferentes.
- **Por caracteres**, o vocabulário é pequeno e quase tudo é coberto. O preço são sequências
  ~5,6× mais longas: `context_length = 128` enxerga só ~23 palavras.

Os números 5,6 e 23 saem de duas divisões:

$$
\text{caracteres por palavra} = \frac{485}{86} \approx 5{,}64
\qquad\qquad
\text{palavras no contexto} \approx \frac{T}{5{,}64} = \frac{128}{5{,}64} \approx 23
$$

**Lendo a fórmula:**

- $485$: tokens do trecho quando cada caractere é um token (inclui espaços e pontuação).
- $86$: tokens do mesmo trecho quando cada palavra é um token.
- $485 / 86$: quantos caracteres "custa", em média, cada palavra.
- $T$: o **context_length**, quantos tokens o modelo lê de uma vez (128 no `tiny.yaml`).
- $T / 5{,}64$: quantas palavras cabem nesses 128 caracteres.

**Exemplo:** com 61 de 70 palavras aparecendo uma única vez, $61 / 70 \approx 87\%$ do
vocabulário por palavras teria um único exemplo de uso no trecho.

O modelo precisa aprender sozinho a formar sílabas e palavras a partir de letras. Isso é
mais difícil para ele, mas é exatamente o que torna o treinamento interessante de observar.
O BPE é um meio-termo possível para uma evolução futura do projeto: pedaços frequentes de
palavras viram um token só.

**BPE** (*Byte Pair Encoding*) é um algoritmo que começa com caracteres e junta, repetidamente,
os pares mais frequentes num token novo. Assim, `"que"` pode virar um token só, enquanto
palavras raras continuam divididas em pedaços.

---

## 7. `<UNK>` também acontece com caracteres

O trecho do livro não tem nenhum `á`. Então:

```text
"o gato está na casa 🐱"  ──►  "o gato est<UNK> na casa <UNK>"
```

O emoji 🐱 é um único code point (`U+1F431`), por isso vira **um** `<UNK>`.

O vocabulário só conhece o que viu no corpus. Consequência prática para o dataset
([02-dataset.md](02-dataset.md)): o vocabulário deve ser construído a partir do corpus de
treino inteiro, e caracteres raros que ficarem de fora viram `<UNK>`. O critério de ida e volta,
`decode(encode(text)) == text`, vale para **textos suportados**: os que só usam caracteres do
vocabulário.

---

## 8. Um detalhe de Unicode: NFC (`_normalize`)

**Unicode** é o padrão que dá um número (o **code point**) a cada caractere de todas as
escritas. O problema: alguns caracteres podem ser escritos de mais de um jeito.

`"á"` pode ser escrito de duas formas que parecem idênticas na tela:

- 1 code point: `U+00E1` (forma **NFC**);
- 2 code points: `"a"` + acento combinante `U+0301` (forma **NFD**, comum em textos
  vindos do macOS).

Sem normalização, a forma NFD viraria `"a"` + `<UNK>`. Por isso `encode` e `from_text`
normalizam tudo para NFC antes de tokenizar.

**Objetivo de `_normalize`:** garantir que o vocabulário e o encode vejam o mesmo caractere
da mesma forma. A função tem uma linha, `unicodedata.normalize("NFC", text)`, e é chamada nos
dois lugares. Assim, um texto NFD e um texto NFC com a mesma aparência geram os mesmos IDs
(teste `test_equivalent_unicode_forms_encode_equally`). O script que prepara o corpus
(`02-dataset.md`) também normaliza para NFC.

**Exemplo:** em NFC, `"á"` tem `len` igual a 1; em NFD, `len` igual a 2 (`U+0061` + `U+0301`).
Depois do `_normalize`, os dois viram o mesmo caractere e o mesmo ID.

---

## 9. Interface

```python
from tokenizer import CharTokenizer

tok = CharTokenizer.from_text(corpus)           # constrói o vocabulário
tok.vocab_size                                  # 4 especiais + caracteres distintos

ids = tok.encode("a casa")                      # list[int], um ID por caractere
ids = tok.encode("a casa", add_special_tokens=True)  # [<BOS>, ..., <EOS>]

tok.decode(ids)                                 # "a casa"
tok.decode(ids, skip_special_tokens=False)      # "<BOS>a casa<EOS>"
```

`decode` também aceita um tensor do PyTorch, o que vai ser útil na geração.

---

## 10. O que muda nos próximos componentes

- **`vocab_size` vem do tokenizer.** Ele saiu do `configs/tiny.yaml` em `02-dataset.md`; com o
  corpus completo (*Dom Casmurro*) são 97 tokens. A matriz de embeddings
  ([03-embeddings.md](03-embeddings.md)) e o LM head (`09-lm-head.md`) usam `tok.vocab_size`.
- **O vocabulário precisa ser salvo junto com o modelo** ([15-checkpoints.md](15-checkpoints.md)).
  Um modelo sem o vocabulário que o treinou é só uma matriz de números sem significado.

O tamanho do vocabulário define o tamanho da primeira matriz do modelo. A **matriz de
embeddings** (`03-embeddings.md`) tem uma linha para cada token e $d$ colunas:

$$
\text{números na matriz de embeddings} = V \times d
$$

**Lendo a fórmula:**

- $V$: número de linhas, uma por token do vocabulário.
- $d$: número de colunas, o tamanho do vetor de cada token (`d_model`).
- $\times$: linhas vezes colunas dá o total de números (parâmetros) da matriz.

**Exemplo:** $97 \times 128 = 12.416$ parâmetros no Mini-GPT. Um vocabulário maior aumenta essa
matriz na mesma proporção.

---

## Resumo

- O tokenizer converte texto em IDs e IDs em texto. O resto do modelo só vê IDs.
- O vocabulário é uma lista: o ID de um token é a posição dele. IDs vão de 0 a $V - 1$, sem
  buracos.
- `from_text` normaliza (NFC), pega os caracteres distintos e os ordena: $V = 4 + \lvert C \rvert = 97$.
- Os 4 especiais têm IDs fixos (0 a 3) e não podem ser produzidos por texto. O `decode` omite
  `<PAD>`, `<BOS>` e `<EOS>`, mas nunca `<UNK>`.
- `encode` gera um ID por caractere; caractere desconhecido vira `<UNK>` (1).
- Tokenizar por caracteres dá vocabulário pequeno e quase sem `<UNK>`, ao preço de sequências
  ~5,6× mais longas que por palavras.
- `vocab_size` vem do tokenizer, e o vocabulário precisa ser salvo junto com o modelo.

## Para checar o entendimento

1. Se você trocar a ordem dos caracteres no vocabulário depois de treinar, o que acontece
   com as respostas do modelo?
2. Por que `<PAD>` tem ID 0, e não um número qualquer?
3. Um corpus de 10 MB tem, digamos, 150 caracteres distintos e 200 mil palavras distintas.
   Qual o tamanho da matriz de embeddings (`03-embeddings.md`, `d_model = 128`) em cada caso?
4. Por que esconder o `<UNK>` no `decode` seria uma má ideia?
5. Um texto em NFD passa no critério `decode(encode(texto)) == texto`? Por quê?
6. Por que `from_text` precisa de `sorted`, se `set` já remove as repetições?
