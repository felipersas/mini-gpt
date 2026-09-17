# Tokenizer BPE

> **Como dar ao modelo um vocabulário de pedaços de palavra, em vez de só caracteres?**

Código: [`tokenizer/bpe.py`](../tokenizer/bpe.py) (`BPETokenizer`); `trim_trailing_partial_word`
em [`generation/generate.py`](../generation/generate.py). Testes:
[`tests/test_bpe_tokenizer.py`](../tests/test_bpe_tokenizer.py) e a seção "trim_trailing_partial_word"
em [`tests/test_generation.py`](../tests/test_generation.py). Experimento:
`uv run python -m experiments.e18_bpe_tokenizer`. Notação geral: [00-notacao.md](00-notacao.md).
Tokenizer por caracteres (o ponto de partida deste documento): [01-tokenizer.md](01-tokenizer.md).

---

## Para que serve

### O problema

O [`CharTokenizer`](01-tokenizer.md) trata cada caractere como um token. É simples e funcionou
bem para construir o resto do modelo, mas carrega um custo: `context_length` é medido em
caracteres, então uma janela de 128 tokens só enxerga 128 caracteres de texto — pouco mais que
uma frase. E a geração para exatamente no `max_new_tokens`-ésimo **caractere**, o que quase
sempre cai no meio de uma palavra: "medicina" vira "medi", "coração" vira "cora".

Esse segundo problema apareceu na prática: gerar texto e ver a última palavra cortada ao meio.
A causa raiz é a granularidade do tokenizer, não a lógica de geração — `generate`
([13-generation.md](13-generation.md)) já fazia exatamente o que deveria, parar depois de N
tokens; só que, com um token por caractere, N tokens quase nunca coincide com o fim de uma
palavra.

Este documento troca a unidade do tokenizer: em vez de "um token por caractere", **"um token por
pedaço de palavra, aprendido a partir de quão comum ele é no corpus"** — Byte-Pair Encoding (BPE),
o mesmo princípio usado por GPT-2, GPT-3 e a maioria dos modelos de linguagem atuais (na prática
eles usam BPE em *bytes*; aqui, para continuar no mesmo alfabeto do `CharTokenizer`, os "bytes"
são caracteres Unicode).

Termos usados daqui em diante:

- **Merge**: a regra "sempre que os símbolos X e Y aparecerem lado a lado, funda-os num símbolo
  só, X+Y". O treino do BPE aprende uma lista ordenada de merges.
- **Alfabeto**: o vocabulário inicial, antes de qualquer merge — um símbolo por caractere distinto
  do corpus. Os merges só combinam símbolos que já existem; nunca inventam um caractere novo.
- **Subpalavra** (*subword*): um token que é maior que um caractere e (em geral) menor que uma
  palavra inteira — o resultado de aplicar um ou mais merges.
- **Compressão**: quantos caracteres, em média, cabem num único token (`caracteres / tokens`).
  Mede o quanto o vocabulário aprendido "economiza" tokens para o mesmo texto.

### Uma analogia

`CharTokenizer` é como escrever um texto letra por letra num teclado que só tem 1.130 teclas, uma
por caractere. `BPETokenizer` é como ganhar teclas extras para as combinações mais usadas: uma
tecla só para "que", outra para "ção", outra para "Capitú" — as 2.962 combinações mais frequentes
do corpus, uma tecla por vez. Digitar "coração" ainda é possível letra por letra se for preciso
(o alfabeto original continua lá), mas na prática sai numa tecla só. Quanto mais um pedaço de
texto se repete no material de treino, mais cedo ele ganha sua própria tecla — é por isso que
"de", "que" e "os" são os três primeiros merges aprendidos (seção 1 do experimento).

### O que entra e o que sai

| | o quê | exemplo |
|---|---|---|
| entra | o corpus (texto bruto) | `corpus/machado_e_wikipedia.txt`, 16.729.864 caracteres |
| entra | `vocab_size` alvo (`data.vocab_size` na config) | 4.096 |
| sai | um alfabeto + uma lista ordenada de merges | 1.130 caracteres + 2.962 merges |
| sai | `encode`/`decode`, como qualquer tokenizer do projeto | texto ↔ IDs |
| sai | uma taxa de compressão | 2,10 caracteres/token neste corpus |

### Onde isso entra no pipeline

```text
corpus/machado_e_wikipedia.txt
  │
  │  BPETokenizer.train(text, vocab_size)     ← este documento; ~15 s no corpus inteiro
  ▼
tokenizer.encode(text) -> ids                  ← mesma interface do CharTokenizer
  │
  ▼
data/, model/, training/                       ← nada muda: só veem IDs
  │
  ▼
generation/generate.py: generate() + trim_trailing_partial_word()  ← geração sem palavra cortada
```

`train.py` e `inference.py` trocaram `CharTokenizer.from_text(text)` por
`BPETokenizer.train(text, vocab_size=...)`; `training/checkpoint.py` passou a salvar `merges`
junto com `vocab`, para que um checkpoint continue se bastando sozinho
([15-checkpoints.md](15-checkpoints.md)).

### O que daria errado sem esta parte

- **A geração continuaria cortando palavras.** É o problema que motivou este documento.
- **`context_length` continuaria "caro".** Cobrir mais texto exigiria aumentar `context_length`
  diretamente, o que custa caro: a attention manual ([04-attention.md](04-attention.md)) escala
  com $T^2$.
- **Nada no resto do pipeline quebraria** — e é isso que este documento prova: `data/`, `model/`
  e `training/` só enxergam listas de IDs inteiros; trocar o que cada ID representa não exige
  mudar nenhum deles.

---

## Notação deste documento

| termo | como se lê | o que significa | no `tiny.yaml` | no código |
|---|---|---|---|---|
| `vocab_size` (alvo) | | especiais + alfabeto + merges desejados | 4.096 | `data.vocab_size` |
| alfabeto | | quantos caracteres distintos existem no corpus | 1.130 | `base_chars` |
| merge | | um par de símbolos fundido num só, na ordem aprendida | 2.962 | `tokenizer.merges` |
| rank | "posição do merge" | quão cedo um merge foi aprendido; ranks menores vencem no encode | | `self._merge_rank` |
| compressão | | caracteres por token, em média | 2,10 | `len(texto) / len(ids)` |
| pré-tokenização | | separar o texto em letras / dígitos / pontuação / espaço antes de aprender merges | | `_WORD_PATTERN` |

---

## 1. Treinando o BPE: aprender os merges mais frequentes

O algoritmo (Sennrich et al., 2015) começa com um alfabeto — um símbolo por caractere distinto —
e repete, `vocab_size − especiais − alfabeto` vezes:

1. conte quantas vezes cada **par de símbolos adjacentes** aparece no corpus inteiro;
2. funda o par mais frequente num símbolo novo (empate: o par lexicograficamente menor vence, para
   o treino ser determinístico);
3. substitua toda ocorrência desse par pelo símbolo novo, em todo o corpus.

```python
for _ in range(num_merges):
    best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
    merges.append(best_pair)
    merged_token = best_pair[0] + best_pair[1]
    _apply_merge_everywhere(best_pair, merged_token, ...)
```

Seção 1 do experimento, no corpus inteiro (agora Machado de Assis + Wikipedia,
[02-dataset.md](02-dataset.md)):

```text
  16,729,864 caracteres | alvo vocab_size 4096 | treino em 14.6s
  alfabeto: 1130 caracteres | merges aprendidos: 2962
  vocab_size real: 4096

  primeiros 10 merges (os pares mais frequentes do corpus inteiro):
    ('d', 'e') -> 'de'
    ('r', 'a') -> 'ra'
    ('d', 'o') -> 'do'
    ('e', 's') -> 'es'
    ('n', 't') -> 'nt'
    ('d', 'a') -> 'da'
    ('c', 'o') -> 'co'
    ('e', 'r') -> 'er'
    ('a', 's') -> 'as'
    ('o', 's') -> 'os'

  10 merges por volta do meio (menos óbvios, mais específicos):
    ('Car', 'los') -> 'Carlos'
    ('ér', 'ica') -> 'érica'
    ('a', 'gora') -> 'agora'
    ('Pa', 'ul') -> 'Paul'
    ('T', 'amb') -> 'Tamb'
```

- **Os primeiros merges são pares de letras**, não palavras: "de", "ra", "do" são fragmentos
  baratos e frequentíssimos (aparecem dentro de muitas palavras diferentes).
- **Os merges do meio já são pedaços de palavras e nomes próprios específicos** ("Carlos", "Paul",
  "agora") — nomes próprios aparecem bastante nos artigos biográficos da Wikipedia, então entram
  cedo na lista.
- **O alfabeto saltou de 108 para 1.130 caracteres** com a Wikipedia: nomes estrangeiros, símbolos
  e pontuação que a literatura do século XIX não usa. Isso consome uma fatia bem maior do
  `vocab_size` de 4.096 antes mesmo do primeiro merge — por isso sobram 2.962 merges agora, contra
  3.984 antes, e a compressão cai um pouco (seção 4).
- **14,6 segundos para quase 3 mil merges**, porque cada merge só atualiza as palavras que o
  contêm (seção "Por que é rápido" abaixo) em vez de recontar o corpus inteiro a cada passo — o
  bastante para treinar de novo a cada `train.py`, sem precisar salvar um artefato à parte.

### Por que é rápido: atualização incremental, não recontagem

Recontar os pares do zero a cada um dos ~4 mil merges custaria uma passada pelo corpus inteiro
por merge — minutos, não segundos, numa primeira versão testada durante o desenvolvimento deste
documento. A versão usada guarda, para cada par, **quais palavras únicas o contêm**
(`pair_words`), e só revisita essas palavras quando o par é fundido:

```python
def _apply_merge_everywhere(pair, merged, symbols_by_word, word_counts, pair_counts, pair_words):
    for word in list(pair_words[pair]):
        ...                           # remove as contagens antigas dessa palavra
        symbols_by_word[word] = _merge_symbols(symbols, pair, merged)
        ...                           # soma as contagens novas dessa palavra
```

Como o corpus tem 159.792 palavras **únicas** (repetições não contam de novo) e a maioria dos
merges afeta só uma fração pequena delas, cada passo custa muito menos que revisitar tudo — é a
diferença entre 14,6 s e alguns minutos.

---

## 2. Palavras comuns viram um token; raras, viram pedaços

Seção 2 do experimento:

```text
  'o'                  -> ['o']  (um token só: sim)
  'que'                -> ['que']  (um token só: sim)
  'não'                -> ['não']  (um token só: sim)
  'Capitú'             -> ['Capitú']  (um token só: sim)
  'coração'            -> ['co', 'ração']  (um token só: não)
  'escrivaninha'       -> ['escri', 'van', 'inha']  (um token só: não)
  'retroactivamente'   -> ['re', 'tro', 'ac', 'tivamente']  (um token só: não)
```

Palavras muito frequentes no corpus inteiro (artigos, "Capitú") ganharam merges suficientes para
virar um token inteiro. "Coração" era um token só quando o corpus era só os 4 romances — um tema
central do livro —, mas na mistura com a Wikipedia (16x maior, majoritariamente enciclopédica) sua
frequência relativa caiu, e ela para em 2 pedaços. Palavras raras ("escrivaninha") continuam saindo
como uma sequência de pedaços maiores que letras soltas ("escri", "van", "inha"), não caractere
por caractere.

`_encode_word` aplica os merges na **ordem em que foram aprendidos** (por rank, não pela posição
no texto): a cada passo, funde o par de menor rank disponível, até sobrar um símbolo só ou não
haver mais nenhum par com merge conhecido.

---

## 3. Pontuação nunca gruda na palavra

A pré-tokenização (antes de aprender qualquer merge) separa o texto em blocos que nunca se
misturam: uma sequência de letras, **ou** uma de dígitos, **ou** uma de espaços em branco, **ou**
uma da mesma pontuação — nunca dois tipos no mesmo bloco:

```python
_WORD_PATTERN = re.compile(r"\s+|[^\W\d]+|\d+|[^\w\s]+")
```

Seção 3 do experimento:

```text
  'gato'   -> ['ga', 'to']
  'gato.'  -> ['ga', 'to', '.']
  'gato,'  -> ['ga', 'to', ',']
  'gato!'  -> ['ga', 'to', '!']
  'gato?'  -> ['ga', 'to', '?']
```

"Gato" agora sai em 2 pedaços em vez de 1 (a Wikipedia diluiu sua frequência, como "coração" na
seção 2) — mas o ponto desta seção não muda: os mesmos dois pedaços, `'ga'` e `'to'`, aparecem
**idênticos** nas 5 variações, e a pontuação sempre sai como token à parte. Sem essa separação, o
par mais frequente aprendido cedo poderia facilmente ser "o"+"." (fim de frase é comum), e a
partir daí o BPE aprenderia "gato" e "gato." como **palavras completamente diferentes**, cada uma
exigindo seus próprios merges — a mesma raiz do problema que [15-checkpoints.md](15-checkpoints.md)
descreveu para o vocabulário do tokenizer por caracteres (um corpus novo desloca todos os IDs), só
que multiplicada por toda combinação de palavra + pontuação do corpus. Separar a pontuação faz os
merges de "gato" valerem para qualquer frase em que a palavra aparece, com ou sem pontuação atrás.

---

## 4. Compressão: mais texto no mesmo `context_length`

Seção 4 do experimento:

```text
  corpus inteiro: 16,729,864 caracteres
  CharTokenizer:  16,729,864 tokens (1,00 caractere/token, por definição)
  BPETokenizer:   7,980,360 tokens (2.10 caracteres/token)
  context_length=128: CharTokenizer cobre 128 caracteres | BPETokenizer cobre ~268 caracteres
  context_length=256: CharTokenizer cobre 256 caracteres | BPETokenizer cobre ~537 caracteres
```

**O mesmo `context_length: 128` do `tiny.yaml` passa a cobrir ~268 caracteres em vez de 128** —
mais que o dobro, ainda que menos que os ~293 de antes da Wikipedia (o alfabeto maior deixa menos
`vocab_size` para merges, seção 1) — sem mudar nenhum hiperparâmetro do modelo (`d_model`,
`num_layers`, etc.) e sem aumentar o custo quadrático da attention, porque o número de *posições*
que o modelo processa continua sendo 128. A compressão é "de graça": o mesmo tensor `[B, T, d]`,
cobrindo mais texto.

---

## 5. Caracteres nunca vistos no treino viram `<UNK>`

Seção 5 do experimento:

```text
  'gato 🐱' -> [1305, 1166, 6, 1]
  decode: 'gato <UNK>'
  o emoji não existe no alfabeto do corpus (só português + pontuação usual)
```

Igual ao [`CharTokenizer`](01-tokenizer.md): um caractere fora do alfabeto vira `<UNK>` — o
`Vocab.get_id` (compartilhado pelos dois tokenizers) já faz isso automaticamente. A diferença é
só de escala: o alfabeto do `CharTokenizer` é qualquer caractere que apareça no corpus (1.130
aqui), e o mesmo vale para o alfabeto **base** do BPE — os merges nunca criam um caractere novo,
só combinam os que já existem.

---

## 6. Round-trip exato no corpus inteiro

Seção 6 do experimento:

```text
  decode(encode(corpus)) == corpus: True
  (16,729,864 caracteres -> 7,980,360 tokens -> 16,729,864 caracteres)
```

A pré-tokenização (seção 3) é uma partição exaustiva do texto — todo caractere pertence a
exatamente um bloco — e cada merge só reescreve como um bloco é representado internamente, nunca
descarta informação. `decode` é só concatenar as strings dos tokens, na ordem; por isso o
round-trip é exato, não aproximado. `test_round_trip` (parametrizado com o corpus inteiro,
strings vazias, `"\n\n"`, texto com pontuação) confere isso.

---

## 7. O corte de segurança que ainda sobra: `trim_trailing_partial_word`

BPE reduz drasticamente o corte no meio de palavra (a maioria das palavras comuns já é um token
só), mas não o elimina: uma palavra rara como "escrivaninha" ainda é 4 tokens, e nada garante que
a geração pare bem no último deles. Seção 7 do experimento:

```text
  'a escrivaninha' -> ['a', ' ', 'escri', 'van', 'inha']
  parando em 3 tokens: 'a escri'              -> trim: 'a '
  parando em 4 tokens: 'a escrivan'           -> trim: 'a '
  parando em 5 tokens: 'a escrivaninha'       -> trim: 'a '
```

```python
def trim_trailing_partial_word(text: str) -> str:
    if not text or not text[-1].isalnum():
        return text
    for i in range(len(text) - 1, -1, -1):
        if not text[i].isalnum():
            return text[: i + 1]
    return text  # o texto inteiro é uma única "palavra", sem fronteira nenhuma
```

**Reparem que os três casos (parar em 3, 4 ou 6 tokens) dão o mesmo resultado: `'a '`.** Isso não
é uma limitação escondida — é a única escolha honesta. Olhando só para o texto decodificado, não
tem como distinguir "a palavra terminou de verdade" de "a palavra foi cortada no meio": `"dorme"`
(completa) e `"dor"` (cortada) têm exatamente a mesma cara, uma sequência de letras no fim da
string. Por segurança, `trim_trailing_partial_word` corta sempre que o texto termina em
letra/dígito — na pior hipótese, perde uma última palavra que já estava completa; na melhor
(e mais comum, com BPE), evita mostrar um pedaço quebrado. A única exceção é um texto que é, do
início ao fim, uma única sequência de letras sem nenhuma fronteira — aí cortar tudo devolveria uma
string vazia, pior que manter o que veio.

`inference.py` aplica essa função no texto final, depois do `decode`:

```python
print(trim_trailing_partial_word(tokenizer.decode(ids)))
```

---

## 8. Usando

```bash
uv run python -m experiments.e18_bpe_tokenizer   # as seções deste documento
uv run python train.py                            # treina o BPE a partir do config.data.vocab_size
uv run python inference.py "Capitú"                # geração já sai sem palavra cortada
```

| opção | onde | padrão | efeito |
|---|---|---|---|
| `vocab_size` | `configs/*.yaml`, `data:` | 4.096 | alvo de especiais + alfabeto + merges |

Não há uma flag de linha de comando para `vocab_size`: como `corpus_path`, é uma propriedade dos
**dados**, não algo que faça sentido trocar passo a passo como `--epochs` ou `--device`.

---

## 9. O que cada teste garante

[`tests/test_bpe_tokenizer.py`](../tests/test_bpe_tokenizer.py):

| teste | propriedade |
|---|---|
| `test_vocab_size_is_specials_plus_alphabet_plus_merges` | `vocab_size` é sempre especiais + alfabeto + merges, e nunca passa do alvo |
| `test_train_is_deterministic` | treinar duas vezes no mesmo texto dá o mesmo vocabulário e os mesmos merges, na mesma ordem |
| `test_more_merges_never_shrinks_a_previous_merge_list` | os primeiros $N$ merges não mudam se `vocab_size` aumentar — cada merge só depende do que veio antes |
| `test_merges_favor_frequent_words` | uma palavra repetida no texto de treino vira um token só |
| `test_vocab_size_too_small_for_alphabet_raises` | pedir menos vocabulário que especiais + alfabeto levanta `ValueError` |
| `test_small_corpus_stops_before_the_target_vocab_size` | um corpus pequeno para de aprender merges antes do alvo, sem travar |
| `test_common_word_encodes_as_a_single_token` | uma palavra frequente vira um único ID |
| `test_punctuation_does_not_merge_into_the_word` | `"gato"` e `"gato."` compartilham o primeiro token; o `"."` sai separado |
| `test_round_trip` | `decode(encode(texto)) == texto`, incluindo vazio, espaços e pontuação |
| `test_unknown_char_becomes_unk` | um caractere fora do alfabeto do treino vira `<UNK>` |
| `test_special_tokens_wrap_sequence` | `add_special_tokens` envolve a sequência em `<BOS>`/`<EOS>` |
| `test_reconstructed_tokenizer_encodes_identically` | reconstruir o tokenizer só a partir de `vocab` + `merges` (como um checkpoint faz) dá o mesmo `encode`/`decode` |

[`tests/test_generation.py`](../tests/test_generation.py):

| teste | propriedade |
|---|---|
| `test_trim_trailing_partial_word` | corta a última palavra quando o texto termina em letra/dígito; não mexe se já termina numa fronteira; mantém o texto se não houver fronteira nenhuma |

---

## Resumo

1. `BPETokenizer` aprende um vocabulário de subpalavras: começa com um alfabeto de caracteres e
   funde repetidamente o par adjacente mais frequente do corpus, até `vocab_size`. O algoritmo é
   de Sennrich et al. (2015) — o mesmo princípio usado por GPT-2 e GPT-3.
2. A pré-tokenização separa letras, dígitos, espaço e pontuação em blocos que nunca se misturam,
   para que "gato" e "gato." aprendam o mesmo primeiro token em vez de vocabulários diferentes.
3. Medido no corpus completo (4 romances + Wikipedia): 1.130 caracteres de alfabeto, 2.962 merges,
   treino em 14,6 s, compressão de 2,10 caracteres/token — `context_length: 128` passa a cobrir
   ~268 caracteres, mais que o dobro de antes, sem mudar o custo da attention.
4. `decode(encode(texto)) == texto` sempre, porque a pré-tokenização é uma partição exaustiva do
   texto e nenhum merge descarta informação — só muda como um trecho é representado por dentro.
5. BPE reduz, mas não elimina, o corte de palavra na geração: palavras raras continuam sendo
   vários tokens. `trim_trailing_partial_word` corta a última palavra sempre que o texto termina
   em letra/dígito, porque não há como saber, só pelo texto, se ela terminou de verdade.
6. Nada em `data/`, `model/` ou `training/` mudou: os três só enxergam listas de IDs inteiros, e
   trocar o que cada ID representa não exige tocar em nenhum deles.

---

## Para checar o entendimento

1. Por que os primeiros merges aprendidos ("de", "ra", "os") são pares de letras soltas, e não
   palavras inteiras, mesmo palavras comuns já sendo muito frequentes desde o início?
2. `escrivaninha` virou `['es', 'cri', 'van', 'inha']`, quatro tokens. O que precisaria ser
   diferente no corpus de treino para essa palavra virar um único token?
3. Por que a pré-tokenização impede que `"gato"` e `"gato."` aprendam vocabulários totalmente
   separados? O que aconteceria com o número de merges "úteis" se a pontuação não fosse separada?
4. `context_length: 128` passou a cobrir ~268 caracteres em vez de 128, sem mudar `d_model` nem
   `num_layers`. De onde vem esse ganho — o modelo ficou "mais inteligente" ou é outra coisa?
5. Por que `decode(encode(texto)) == texto` vale sempre, mesmo para um texto que o BPE nunca viu
   no treino (fora os caracteres desconhecidos, que viram `<UNK>`)?
6. `trim_trailing_partial_word("a escrivaninha")` corta pra `"a "`, mesmo a palavra estando
   completa naquele texto. Por que a função não tem como saber que ela estava completa?
7. Por que `test_more_merges_never_shrinks_a_previous_merge_list` é verdade — por que os primeiros
   $N$ merges aprendidos com `vocab_size` pequeno são exatamente os primeiros $N$ merges de um
   treino com `vocab_size` maior?
8. `training/checkpoint.py` agora salva `merges` além de `vocab`. O que quebraria, especificamente,
   se um checkpoint salvasse só `vocab` e tentasse reconstruir o `BPETokenizer` sem os merges?
9. Por que treinar o BPE de novo a cada `uv run python train.py` (em vez de treinar uma vez e
   salvar um arquivo à parte) é uma escolha razoável aqui, e o que teria que mudar no corpus para
   essa escolha deixar de fazer sentido?
10. Rodando `CharTokenizer.from_text` no mesmo corpus, o `vocab_size` dá 1.134. O alfabeto do BPE
    (seção 1) tem 1.130. Por que a diferença é exatamente 4?
