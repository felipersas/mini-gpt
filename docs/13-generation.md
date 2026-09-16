# Text Generation

> **Como uma sequência é produzida token por token?**

Código: [`generation/generate.py`](../generation/generate.py),
[`training/checkpoint.py`](../training/checkpoint.py) e [`inference.py`](../inference.py). Testes:
[`tests/test_generation.py`](../tests/test_generation.py) e
[`tests/test_checkpoint.py`](../tests/test_checkpoint.py). Experimento:
`uv run python -m experiments.e13_generation` (precisa do modelo salvo por `uv run python train.py`).
Notação geral: [00-notacao.md](00-notacao.md). Visão geral: [architecture.md](architecture.md).

---

## Para que serve

### O problema

O modelo treinado ([11-training.md](11-training.md)) faz uma única coisa: dado um texto, devolve **uma distribuição de
probabilidade** para o próximo caractere. Ele não escreve textos. Para escrever, é preciso:

1. **escolher** um caractere a partir dessa distribuição;
2. **acrescentá-lo** ao texto;
3. **perguntar de novo** ao modelo, agora com o texto um caractere maior;
4. repetir.

Esse processo se chama **geração autoregressiva**. Os passos 2 a 4 são mecânicos. O passo 1, a
**estratégia de escolha**, decide se o texto sai repetitivo, criativo ou sem sentido, e é o assunto
principal deste documento.

Este documento também antecipa um mínimo de [15-checkpoints.md](15-checkpoints.md): **salvar e
carregar** o modelo treinado. Sem isso, cada geração exigiria treinar de novo por ~5 minutos.

Termos usados daqui em diante:

- **Prompt:** o texto inicial que o usuário fornece, e que o modelo continua.
- **Autoregressivo:** cada token gerado vira entrada para gerar o seguinte.
- **Decodificação:** a estratégia de transformar distribuições em tokens escolhidos.
- **Greedy** (guloso): escolher sempre o token mais provável.
- **Amostragem** (*sampling*): sortear o token de acordo com as probabilidades.
- **Temperature:** um número que deixa a distribuição mais concentrada ou mais espalhada antes do
  sorteio.
- **Top-k e top-p:** filtros que descartam os tokens improváveis antes do sorteio.
- **Seed:** o número que inicializa o gerador de números aleatórios. A mesma seed produz os mesmos
  sorteios.
- **Checkpoint:** um arquivo com tudo o que é preciso para recriar o modelo treinado.
- **Entropia:** uma medida de quão espalhada é uma distribuição.

### Uma analogia

Pense no teclado do celular, que sugere a próxima palavra:

- apertar **sempre a primeira sugestão** é o greedy. Em pouco tempo, a frase entra em loop ("eu vou
  te ligar para te ligar para te ligar…");
- escolher **ao acaso entre todas as palavras do dicionário**, com qualquer peso, dá lixo;
- o equilíbrio é **sortear entre as sugestões razoáveis**, dando mais chance às melhores. É o que
  temperature, top-k e top-p fazem.

### O que entra e o que sai

| | o quê | exemplo |
|---|---|---|
| entra | o checkpoint | `checkpoints/latest.pt`, 3,1 MB: pesos, vocabulário e config |
| entra | o prompt | `"Capitú"` → IDs `[31, 53, 68, 61, 72, 96]` |
| entra | a estratégia | temperature, top-k, top-p, seed |
| a cada passo | logits da **última** posição | `[97]` |
| sai | o texto gerado | prompt + até `max_new_tokens` caracteres novos |

### Onde isso entra no pipeline

O treino usa o modelo em todas as posições ao mesmo tempo. A geração usa só a **última** posição, e
repete:

```text
prompt: "Capitú"
  │
  ▼  tokenizer (01-tokenizer.md)         "Capitú" → [31, 53, 68, 61, 72, 96]
  ▼  GPT (03 a 09), pesos do checkpoint
  │                                      [1, 6] → logits [1, 6, 97]
  ▼  logits da última posição            [97]
  ▼  estratégia de escolha (este doc)    temperature → top-k → top-p → sorteio
  ▼  novo token                          ' '
  │
  └──► acrescenta ao prompt e repete:    "Capitú " → GPT → ... → "Capitú e a prima Justina..."
```

### O que daria errado sem esta parte

- **O modelo seria inútil na prática:** ele saberia prever, mas não escrever.
- **Com greedy puro,** o texto entraria em loop (seção 4).
- **Com sorteio sem controle,** caracteres improváveis se acumulariam e o texto viraria lixo (seção 5).
- **Sem checkpoint,** cada geração custaria um treino inteiro.

---

## Notação deste documento

| símbolo | como se lê | o que significa | no Mini-GPT (valor/shape) | no código |
|---|---|---|---|---|
| $x_t$ | "xis tê" | o token na posição $t$ do texto (prompt ou gerado) | um ID de 0 a 96 | `ids[t]` |
| $x_{\le t}$ | "xis até tê" | todos os tokens até a posição $t$, inclusive | | `ids[: t + 1]` |
| $V$ | "vê" | tamanho do vocabulário | 97 | `vocab_size` |
| $z$, $z_i$ | "zê", "zê i" | logits da última posição; $z_i$ é o do token $i$ | `[97]` | `logits` |
| $p$, $p_i$ | "pê", "pê i" | probabilidades: $\text{softmax}(z)$ | `[97]`, soma 1 | `probs` |
| $\tau$ | "tau" | temperature. **Não** usamos $T$ aqui, porque $T$ é o `context_length` | 0,8 no `inference.py` | `temperature` |
| $k$ | "cá" | quantos tokens o top-k mantém | 1, 5, 20 | `top_k` |
| $p^\ast$ | "pê estrela" | limiar do top-p: a massa de probabilidade que o conjunto mantido precisa atingir. Diferente de $p_i$ | 0,95 no `inference.py` | `top_p` |
| $\mathcal{S}$ | "esse caligráfico" | conjunto de tokens que sobram depois de um filtro | | tokens com logit finito |
| $\arg\max_i$ | "argumento do máximo" | o índice do maior valor | | `.argmax()` |
| $\sim$ | "sorteado de" | $x \sim p$: $x$ é sorteado com as probabilidades $p$ | | `torch.multinomial` |
| $H$ | "agá" | entropia: $-\sum_i p_i \ln p_i$ | 0,76 depois de `"Capitú"` | `-(p * p.log()).sum()` |
| $e^H$ | "e elevado a agá" | número efetivo de opções | 2,14 depois de `"Capitú"` | `math.exp(entropy)` |
| $T$ | "tê" | contexto máximo que o modelo aceita | 128 | `context_length` |

---

## 1. Salvando o modelo treinado

### 1.1 O que precisa ser salvo

Para recriar o modelo num outro processo, três coisas são indispensáveis:

| peça | por que é indispensável | no arquivo |
|---|---|---|
| **config da arquitetura** | sem ela, não se sabe quantas camadas, heads e dimensões criar | `"model_config"`: `{"context_length": 128, "d_model": 128, ...}` |
| **vocabulário** | os pesos só fazem sentido com a mesma correspondência ID ↔ caractere ([01-tokenizer.md](01-tokenizer.md)). Com outra ordem, o ID 53 deixaria de ser `'a'` | `"vocab"`: a lista `id_to_token` |
| **pesos** | é o que o treino aprendeu | `"model_state"`: o `state_dict` |

### 1.2 O `state_dict`

`model.state_dict()` é um dicionário que associa o **nome** de cada tensor de parâmetros ao próprio
tensor: `"blocks.0.attention.q_proj.weight"` → tensor `[128, 128]`, e assim por diante. Os nomes vêm
da estrutura de módulos ([08-gpt.md](08-gpt.md), seção 2.1). Carregar é o inverso: criar um modelo com a mesma
estrutura e copiar cada tensor para o parâmetro de mesmo nome (`load_state_dict`).

### 1.3 O código

```python
def save_checkpoint(path, model, tokenizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": asdict(model.config),      # dataclass → dicionário simples
        "vocab": tokenizer.vocab.id_to_token,      # a ordem dos tokens define os IDs
        "model_state": model.state_dict(),         # todos os pesos, por nome
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = CharTokenizer(Vocab(checkpoint["vocab"]))
    model = GPT(ModelConfig(**checkpoint["model_config"]), vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer
```

Parte por parte:

- **`asdict(model.config)`** transforma o `ModelConfig` num dicionário de números e booleanos, e
  **`ModelConfig(**...)`** faz o caminho de volta.
- **`Vocab(checkpoint["vocab"])`** recebe a lista salva, que já começa com os 4 especiais. O `Vocab`
  remove duplicatas mantendo a ordem ([01-tokenizer.md](01-tokenizer.md)), então os IDs ficam idênticos aos do treino
  (`test_round_trip_preserves_config_vocab_and_outputs`).
- **O modelo é criado com pesos aleatórios** e, em seguida, `load_state_dict` sobrescreve todos eles.
- **O weight tying continua valendo.** O construtor do `GPT` amarra o LM head ao token embedding
  antes do carregamento, e `load_state_dict` copia valores para dentro dos parâmetros existentes, sem
  criar novos (`test_loaded_model_keeps_weight_tying`).
- **`map_location="cpu"`** carrega na CPU mesmo que o arquivo tenha sido salvo em GPU.
- **`weights_only=True`** é uma medida de segurança: arquivos `.pt` podem conter código Python
  arbitrário, e esta opção só aceita tensores e tipos simples.
- **`model.eval()`** desliga o dropout: o modelo sai pronto para gerar.

**Tamanho do arquivo:** 820.096 parâmetros × 4 bytes (float32) = 3.280.384 bytes ≈ 3,1 MiB. É o
tamanho de `checkpoints/latest.pt`.

O `train.py` salva o checkpoint ao terminar. [15-checkpoints.md](15-checkpoints.md) vai acrescentar
o que falta para **retomar** um treino interrompido: o estado do otimizador, o passo atual e
checkpoints periódicos.

---

## 2. O loop autoregressivo

### 2.1 A ideia em uma fórmula

$$
x_{t+1} \sim \text{escolha}\big(p(\,\cdot \mid x_{\le t})\big)
$$

**Lendo a fórmula:**

- $x_{\le t}$: todo o texto até agora (prompt mais o que já foi gerado).
- $p(\,\cdot \mid x_{\le t})$: a distribuição que o modelo dá para o próximo token, dado esse texto.
  O ponto ($\cdot$) indica "para cada token possível".
- $\text{escolha}$: a estratégia (greedy, temperature, top-k, top-p).
- $\sim$: o próximo token $x_{t+1}$ é **sorteado** (ou escolhido) a partir dela.
- depois, $x_{t+1}$ entra no texto, e a fórmula é aplicada de novo para $x_{t+2}$.

### 2.2 O código

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, *, temperature=1.0, top_k=None,
             top_p=None, eos_id=None, generator=None):
    if not prompt_ids:
        raise ValueError("o prompt precisa ter pelo menos um token")
    was_training = model.training
    model.eval()
    ids = list(prompt_ids)
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-context_length:]])      # [1, T]
        logits = model(context)[0, -1]                        # [vocab_size]
        next_id = sample_next_token(logits, temperature=temperature, top_k=top_k,
                                    top_p=top_p, generator=generator)
        ids.append(next_id)
        if next_id == eos_id:
            break

    model.train(was_training)
    return ids
```

Parte por parte:

| linha | objetivo | o que faz |
|---|---|---|
| `@torch.no_grad()` | economizar | na geração não há backward; não é preciso gravar o grafo ([11-training.md](11-training.md), seção 9.2) |
| `model.eval()` | tirar o ruído | desliga o dropout; restaurado no fim com `model.train(was_training)` |
| `ids[-context_length:]` | respeitar o limite do modelo | o positional embedding só tem 128 linhas ([03-embeddings.md](03-embeddings.md)). Com mais texto, ficam os 128 tokens mais recentes |
| `torch.tensor([...])` | criar o batch | shape `[1, T]`: um batch com uma sequência |
| `model(context)[0, -1]` | pegar a previsão certa | logits `[1, T, 97]`; `[0, -1]` escolhe a sequência 0 e a **última** posição, a única que prevê um token ainda inexistente |
| `sample_next_token(...)` | escolher | aplica a estratégia (seções 4 a 7) |
| `ids.append(next_id)` | crescer o texto | o token escolhido vira contexto do próximo passo |
| `if next_id == eos_id` | parar no fim | se o modelo gerar `<EOS>`, a geração termina antes do limite |

**Por que só a última posição:** as posições anteriores preveem tokens que **já existem** no
texto. Durante o treino elas eram úteis (cada uma tinha um alvo); na geração, só a última prevê algo
novo.

**Por que o modelo é causal ajuda aqui:** com a causal mask, as saídas das posições antigas não
mudam quando o texto cresce. Recalcular tudo a cada passo é, portanto, desperdício. A seção 10 volta
a esse ponto.

---

## 3. Um passo de geração

Seção 1 do experimento, com o modelo treinado:

```text
prompt 'Capitú' -> IDs [31, 53, 68, 61, 72, 96] -> logits (97,) na última posição
os mais prováveis: ' ' 0.783  ',' 0.136  '.' 0.059  ';' 0.008  '!' 0.005  '?' 0.002  's' 0.002  ':' 0.001
opções efetivas (e^entropia): 2.14 de 97
```

- **O modelo aprendeu que "Capitú" é uma palavra completa:** 98% da probabilidade vai para espaço
  ou pontuação (`' '`, `','`, `'.'`, `';'`).
- **"Opções efetivas"** resume o quão indeciso o modelo está.

A **entropia** mede o espalhamento de uma distribuição, e $e^H$ a traduz num número de opções:

$$
H = -\sum_{i=1}^{V} p_i \ln p_i, \qquad \text{opções efetivas} = e^{H}
$$

**Lendo a fórmula:**

- $p_i \ln p_i$: cada token contribui com sua probabilidade vezes o log dela. Como $\ln p_i < 0$, o
  sinal de menos deixa $H$ positiva.
- tokens com $p_i = 0$ não contribuem (o código usa `clamp_min(1e-12)` para evitar $\ln 0$).
- **se há $k$ opções igualmente prováveis** ($p_i = 1/k$), $H = -k \cdot \frac{1}{k} \ln \frac{1}{k} = \ln k$
  e $e^H = k$. Por isso $e^H$ se lê como "o modelo está hesitando entre quantas opções".
- aqui, $e^H = 2{,}14$: praticamente "espaço ou vírgula".

É a mesma ideia da perplexidade ([10-loss.md](10-loss.md)), aplicada a uma única posição.

---

## 4. Greedy: sempre o mais provável

### 4.1 A regra

$$
x_{t+1} = \arg\max_i\, p_i
$$

**Lendo a fórmula:** o próximo token é o de maior probabilidade. Não há sorteio: o mesmo prompt
sempre gera o mesmo texto. No código, `temperature=0` ativa esse modo: `int(logits.argmax())`.
(Como a softmax não muda a ordem dos valores, o maior logit é também o de maior probabilidade.)

### 4.2 O resultado

Seção 2 do experimento:

```text
'Capitú e a prima Justina e a minha mãe de um primeiro de um dia de um primeiro de um primeiro de um primeiro de um casa de um caracitulo de um anno do carro de Capitú. E o que ella me deu a minha mãe de min'
palavras distintas 50% | maior trecho idêntico ao livro: 18 caracteres
```

- **O começo é plausível:** "Capitú e a prima Justina e a minha mãe". Justina e a mãe são personagens
  do livro.
- **Depois entra em loop:** "de um primeiro de um primeiro de um primeiro". Como a escolha é
  determinística, sempre que o contexto recente se repete, a próxima escolha também se repete. O texto
  cai num ciclo de onde só sai por acaso (uma variação num caractere antigo que ainda está no contexto).
- **Só 50% das palavras são distintas.**

A frase mais provável caractere a caractere **não** é um texto bom: é um texto genérico e repetitivo.
Esse fenômeno é conhecido como **degeneração** do texto gerado.

---

## 5. Temperature

### 5.1 A fórmula

Antes da softmax, os logits são divididos por $\tau$:

$$
p_i(\tau) = \text{softmax}\!\left(\frac{z}{\tau}\right)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
$$

**Lendo a fórmula:**

- $z/\tau$: todos os logits divididos pelo mesmo número.
- **$\tau < 1$** multiplica as diferenças entre os logits: o mais provável fica ainda mais provável.
  A distribuição **concentra**.
- **$\tau = 1$**: a distribuição original do modelo.
- **$\tau > 1$** encolhe as diferenças: as probabilidades ficam mais parecidas. A distribuição
  **espalha**.
- **$\tau \to 0$**: tudo vai para o maior logit, e vira greedy. **$\tau \to \infty$**: tudo fica
  uniforme, $1/V$.

Exemplo com $z = (2, 1, 0)$:

| $\tau$ | $z / \tau$ | probabilidades | efeito |
|---:|---|---|---|
| 0,5 | (4, 2, 0) | (0,867; 0,117; 0,016) | concentra |
| 1,0 | (2, 1, 0) | (0,665; 0,245; 0,090) | original |
| 2,0 | (1; 0,5; 0) | (0,506; 0,307; 0,186) | espalha |

A ordem dos tokens nunca muda: a temperature só altera **quão decisiva** é a diferença entre eles.

### 5.2 Na distribuição real

Seção 3 do experimento, depois de `"olhos de "`:

```text
distribuição do próximo caractere depois de 'olhos de ':
  T = 0.2: opções efetivas  3.35 | 'u' 0.492  'm' 0.299  'c' 0.170  's' 0.012  'p' 0.011
  T = 0.7: opções efetivas 16.41 | 'u' 0.178  'm' 0.155  'c' 0.131  's' 0.062  'p' 0.060
  T = 1.0: opções efetivas 24.31 | 'u' 0.121  'm' 0.110  'c' 0.098  's' 0.058  'p' 0.057
  T = 1.5: opções efetivas 35.90 | 'u' 0.079  'm' 0.074  'c' 0.068  's' 0.048  'p' 0.047
```

(No experimento, `T` na saída é a temperature $\tau$.)

- Depois de "olhos de", o modelo está genuinamente indeciso: com $\tau = 1$, hesita entre ~24
  caracteres (o início de qualquer palavra).
- Com $\tau = 0{,}2$, sobram ~3 opções efetivas (`'u'`, `'m'`, `'c'`: "um", "minha", "cara"…); com $\tau = 1{,}5$,
  ~36.
- Curiosidade: no livro, a expressão famosa é "olhos de **r**essaca". O modelo não dá destaque ao `'r'`:
  com 128 caracteres de contexto e 820 mil parâmetros, ele aprendeu estatísticas de palavras, não
  expressões específicas.

### 5.3 Texto mais longo, comparando as temperaturas

Duzentos caracteres a partir de `"Capitú"`, com a **mesma seed**:

```text
  T = 0.2: 'Capitú se a mais que não fizesse achava a minha mãe de minha mãe a casa de casa, e a alma de Capitú estava a pressa de Capitú, e o que eu não achava nem casa de minha mãe. E a minha mãe deu a cabeça de minh'
  palavras distintas 54% | maior trecho idêntico ao livro: 19 caracteres

  T = 0.7: 'Capitú se faça que ver as destas relações. Alguma dos seus seus articas fui casada, como vivida de assim e durações de com a porta do cousa, quando se é que é o riso de a Justinaria e entre o conselho dia a'
  palavras distintas 77% | maior trecho idêntico ao livro: 13 caracteres

  T = 1.0: 'Capitú se faça que verdadei nem mal ficou.\n\n--Fui dos seus seus?\n\n--Sim, um casama, falando de lado esperar um capitulo sentipo. Que eu ultimo apantente; já sinha serra a Juste-fadidade e uma conservo de Ca'
  palavras distintas 94% | maior trecho idêntico ao livro: 13 caracteres

  T = 1.5: 'Capitú se façou-lhe chegar desa7); LôrNos.\n\n--Fui dolhogo não fala como fui uma Ezequeate. Pridade, eu quer um capitulo sentipo. QuAtendulo da apantença; já sinha serrara Juste-fady?\n\nEmeiu:\n\n--Ess.\n\nPerdia'
  palavras distintas 100% | maior trecho idêntico ao livro: 13 caracteres
```

| $\tau$ | como o texto fica | palavras distintas |
|---:|---|---:|
| 0,2 | quase greedy: palavras reais, mas repetitivas ("minha mãe de minha mãe", "casa de casa") | 54% |
| 0,7 | variado, quase só palavras reais, frases sem sentido global | 77% |
| 1,0 | a "voz" do livro aparece: diálogos com `--`, parágrafos (`\n\n`); também palavras inventadas ("verdadei", "apantente") | 94% |
| 1,5 | caracteres raros começam a aparecer ("desa7); LôrNos."); lixo misturado a trechos plausíveis | 100% |

**O dilema:** temperaturas baixas dão texto seguro e repetitivo; altas dão texto diverso e errático.
Não existe temperature "correta": é uma escolha entre fidelidade e variedade.

**Por que os textos compartilham trechos** ("--Fui do…", "um capitulo sentipo", "já sinha serra…", "Juste-fa…"
aparecem em $\tau = 1{,}0$ e em $\tau = 1{,}5$):
com a mesma seed, os números aleatórios sorteados são os mesmos. Quando duas distribuições são
parecidas, o mesmo número aleatório escolhe o mesmo caractere, e os textos seguem juntos até
divergirem.

**"Palavras distintas" não mede qualidade:** o texto de $\tau = 1{,}5$ tem 100% e é o pior. A métrica
só detecta repetição.

---

## 6. Top-k

### 6.1 A regra

Antes do sorteio, mantém só os $k$ tokens de maior logit e zera a probabilidade dos outros:

$$
\mathcal{S}_k = \{\text{os } k \text{ tokens de maiores } z_i\}, \qquad
p_i^{\text{top-}k} =
\begin{cases}
\dfrac{e^{z_i}}{\sum_{j \in \mathcal{S}_k} e^{z_j}} & \text{se } i \in \mathcal{S}_k \\[2ex]
0 & \text{se não}
\end{cases}
$$

**Lendo a fórmula:**

- $\mathcal{S}_k$: o conjunto dos $k$ candidatos mais prováveis.
- $\sum_{j \in \mathcal{S}_k}$: a soma só sobre esses candidatos. As probabilidades são
  **renormalizadas**: voltam a somar 1 só entre os que ficaram.
- os tokens fora do conjunto têm probabilidade exatamente 0: nunca são sorteados
  (`test_sampling_never_picks_a_filtered_token`).

No código, "zerar a probabilidade" é trocar o logit por $-\infty$, porque $e^{-\infty} = 0$ (a mesma
ideia da causal mask, [04-attention.md](04-attention.md)):

```python
kth_largest = torch.topk(logits, k).values[..., -1, None]   # o k-ésimo maior logit
return logits.masked_fill(logits < kth_largest, float("-inf"))
```

### 6.2 O resultado

Seção 4 do experimento (temperature 1,0):

```text
  k = 1: depois de 'olhos de ', mantém 1 tokens com 12.1% da probabilidade
  'Capitú e a prima Justina e a minha mãe de um primeiro de um dia de um primeiro de um primeiro de um primeiro de um casa de um caracitulo de um anno do carro de Capitú. E o que ella me deu a minha mãe de min'
  palavras distintas 50% | maior trecho idêntico ao livro: 18 caracteres

  k = 5: depois de 'olhos de ', mantém 5 tokens com 44.3% da probabilidade
  'Capitú se faça de sacristribue. Agora, naturalmente acabando a mão do ceu a casa, e a amborta. A contava não é para sem ser de amoria dous. Aqui, até mais que não sei praza com ella essentraria, as dous cal'
  palavras distintas 90% | maior trecho idêntico ao livro: 15 caracteres

  k = 20: depois de 'olhos de ', mantém 20 tokens com 88.2% da probabilidade
  'Capitú se faça que verdadei nem mal ficou.\n\n--Fui dos seus seus?\n\n--Sim, um casama, falando de lado esperar um capitulo sentipo. Que eu ultimo apantente; já sinha serra a visinhar não emeiras; mas deu discl'
  palavras distintas 97% | maior trecho idêntico ao livro: 13 caracteres
```

- **$k = 1$ é exatamente o greedy:** o texto é idêntico ao da seção 4 (`test_top_k_of_one_is_greedy`).
- **$k = 5$ e $k = 20$** eliminam a cauda de caracteres raros que estragava o texto de $\tau = 1{,}5$.

**O problema do $k$ fixo:** a mesma quantidade de candidatos serve mal a contextos diferentes.

- Depois de `"olhos de "`, onde o modelo está indeciso, $k = 5$ mantém só 44% da probabilidade: corta
  opções perfeitamente razoáveis.
- Depois de `"qu"`, onde praticamente só `'e'`, `'a'` e `'i'` fazem sentido, $k = 20$ manteria 17
  caracteres absurdos na disputa.

---

## 7. Top-p (nucleus sampling)

### 7.1 A regra

Em vez de um **número fixo** de candidatos, o top-p mantém uma **massa fixa de probabilidade**
(Holtzman et al., 2020). Ordenando os tokens do mais para o menos provável, $p_{(1)} \ge p_{(2)} \ge \dots$:

$$
\mathcal{S}_{p^\ast} = \text{os primeiros } m \text{ tokens, com } m \text{ o menor número tal que } \sum_{j=1}^{m} p_{(j)} \ge p^\ast
$$

**Lendo a fórmula:**

- $p_{(j)}$: a $j$-ésima maior probabilidade (o parêntese no índice indica "em ordem").
- $\sum_{j=1}^{m} p_{(j)}$: a probabilidade acumulada dos $m$ mais prováveis.
- $m$: o **menor** número de tokens que chega a $p^\ast$.
- depois de filtrar, as probabilidades são renormalizadas, como no top-k.

Exemplo (do teste `test_top_p_keeps_the_smallest_set_that_reaches_p`): probabilidades 0,5; 0,3;
0,15 e 0,05, com $p^\ast = 0{,}7$.

| em ordem | probabilidade | acumulada até ele | acumulada **antes** dele | fica? |
|---|---:|---:|---:|---|
| 1º | 0,50 | 0,50 | 0,00 | sim |
| 2º | 0,30 | 0,80 | 0,50 | sim: antes dele, ainda não tinha chegado a 0,7 |
| 3º | 0,15 | 0,95 | 0,80 | não: antes dele, já passava de 0,7 |
| 4º | 0,05 | 1,00 | 0,95 | não |

Resultado: $m = 2$. A regra de implementação é justamente a última coluna: **um token sai se a
probabilidade acumulada antes dele já chegou a $p^\ast$**. O mais provável sempre fica, porque antes
dele a soma é 0.

```python
sorted_logits, order = torch.sort(logits, descending=True)
probs = torch.softmax(sorted_logits, dim=-1)
remove_sorted = (torch.cumsum(probs, dim=-1) - probs) >= p          # acumulada antes de cada um
remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)   # volta à ordem original
return logits.masked_fill(remove, float("-inf"))
```

- `torch.sort` devolve os logits em ordem **e** `order`, a posição original de cada um.
- `torch.cumsum(probs) - probs` é a coluna "acumulada antes dele".
- `scatter(-1, order, ...)` desfaz a ordenação: coloca cada decisão de "remover" de volta na posição
  original do token.
- com $p^\ast = 1$, a função devolve os logits intactos. Sem esse atalho, erros de arredondamento
  poderiam remover um token com probabilidade minúscula.

### 7.2 O conjunto se adapta ao contexto

Seção 5 do experimento: quantos tokens ficam em três contextos:

```text
                    'qu'   'olhos de '   'Capitú e '   (tokens mantidos)
  p = 0.5              1             7             6
  p = 0.9              3            22            15
  p = 0.95             3            27            18

  temperature 1,0 e p = 0,9: 'Capitú se faça que verdadei nem mal ficou.\n\n--Fui dos seus seus articios com estudes, até lembradorava-me, disse-me a casa, como era andal-os, outras e acabou o riscordia a casa, dissimulação com um dia car'
  palavras distintas 89% | maior trecho idêntico ao livro: 14 caracteres
```

- **Depois de `"qu"`** (confiante): mesmo com $p^\ast = 0{,}95$, ficam só 3 candidatos. O top-p não
  deixa entrar lixo.
- **Depois de `"olhos de "`** (indeciso): com $p^\ast = 0{,}95$, ficam 27. O top-p não corta opções
  razoáveis.

É exatamente o que o $k$ fixo não conseguia fazer (seção 6.2).

---

## 8. Combinando as técnicas

### 8.1 A ordem

`sample_next_token` aplica as etapas nesta ordem:

```python
if temperature == 0:
    return int(logits.argmax())                    # greedy
logits = logits / temperature                      # 1. temperature
if top_k is not None:
    logits = top_k_filter(logits, top_k)           # 2. top-k
if top_p is not None:
    logits = top_p_filter(logits, top_p)           # 3. top-p
probs = torch.softmax(logits, dim=-1)
return int(torch.multinomial(probs, num_samples=1, generator=generator))   # 4. sorteio
```

- **A temperature vem primeiro** porque o top-p depende das probabilidades: com $\tau$ baixo, a
  distribuição concentra e o top-p mantém menos tokens. (O top-k não é afetado, porque a temperature
  não muda a ordem.)
- **`torch.multinomial`** sorteia um índice com as probabilidades dadas. O `generator` com seed torna o
  sorteio reproduzível (`test_same_seed_gives_the_same_text`).

### 8.2 O padrão do `inference.py`: $\tau = 0{,}8$ e $p^\ast = 0{,}95$

A escolha é subjetiva, baseada nas seções anteriores: $\tau$ entre 0,7 (palavras reais, pouca repetição)
e 1,0 (mais estilo, mais erros), com top-p 0,95 para cortar a cauda de caracteres absurdos sem
restringir contextos indecisos. Seção 6 do experimento:

```text
  'o gato se a falar da cara, e palavras commigo.\n\n--Não ser ser as recordações, semoravam para o lador, para outra passou se treta era anguem da algum exemplo dizer que não acceitou não entre o conservo de Ca'
  palavras distintas 82% | maior trecho idêntico ao livro: 15 caracteres

  'Capitú tempo e os seus descever-lhe elle.\n\n--Por eu respondeu-se uma si minha verdade.\n\n--A tinha mãe dar-se alguns escobar com outros, entendendo que não o perso dia rapida. O rato de José Dias de minha mã'
  palavras distintas 95% | maior trecho idêntico ao livro: 15 caracteres

  'Escobar o carto do que eu pensenhor tem e foi a casa. Tudo me deixou a minha mãe crer de casa; aqui assim feliz da primeira vez da cocerdade. As sensações, e instanternou-se não posso para mal que vinta mal,'
  palavras distintas 90% | maior trecho idêntico ao livro: 15 caracteres
```

### 8.3 Avaliando com honestidade

O que o modelo **aprendeu**:

- **ortografia de 1899:** "commigo", "acceitou", "ella", "elle";
- **nomes dos personagens:** José Dias, Escobar, Capitú, a mãe;
- **formato do livro:** diálogos com `--`, parágrafos separados por linha em branco, pontuação;
- **a maioria das palavras é real,** e pedaços de frase soam como português ("Tudo me deixou a minha
  mãe", "aqui assim feliz da primeira vez").

O que o modelo **não** aprendeu:

- **sentido:** as frases não se conectam, e não há narrativa;
- **gramática consistente:** "semoravam", "pensenhor", "instanternou-se" são palavras inventadas,
  combinando pedaços plausíveis;
- **contexto longo:** o modelo enxerga 128 caracteres (~23 palavras) e não tem como manter um assunto.

É o resultado esperado para 820 mil parâmetros treinados por 5 minutos com 337 mil caracteres. O
critério que importa aqui não é qualidade textual, e sim que "o sistema realmente aprendeu a prever
tokens". Ele aprendeu.

---

## 9. O modelo está copiando o livro?

Uma preocupação legítima: um modelo que decora trechos longos só pareceria escrever bem. A métrica
"maior trecho idêntico ao livro" procura, em cada texto gerado, o maior pedaço que aparece
**literalmente** no *Dom Casmurro*.

```python
def longest_copied(text, corpus):
    best = 0
    for start in range(len(text)):
        length = best + 1
        while start + length <= len(text) and text[start : start + length] in corpus:
            best = length
            length += 1
    return best
```

(A busca começa em `best + 1` porque um trecho menor que o recorde atual não interessa.)

| estratégia | maior trecho copiado |
|---|---:|
| greedy | 18 caracteres |
| $\tau = 0{,}2$ | 19 |
| $\tau = 0{,}7$ a $1{,}5$, top-k, top-p | 13 a 15 |

Treze a dezenove caracteres correspondem a duas ou três palavras comuns ("e a minha mãe de"). **O
modelo recombina, não recita.** Estratégias mais determinísticas copiam um pouco mais, porque seguem os
caminhos mais frequentes do treino.

---

## 10. O custo de gerar

Seção 7 do experimento:

```text
   64 caracteres: 0.05 s (1215 caracteres/s)
  256 caracteres: 0.33 s (777 caracteres/s)
  o contexto cresce até 128 tokens e depois é cortado
```

- **Cada caractere novo é um forward completo** sobre todo o contexto.
- **A velocidade cai conforme o contexto cresce:** o 200º caractere processa 128 posições, e o 10º, só
  16. A attention custa $O(T^2)$ ([04-attention.md](04-attention.md), seção 6).
- **Quase todo esse trabalho é repetido.** Por causa da causal mask, as keys e values das posições
  antigas são idênticas às do passo anterior (seção 2.2). Guardar esses tensores e calcular só a
  posição nova é o **KV cache**, uma das otimizações possíveis discutidas em
  [17-performance.md](17-performance.md).

---

## 11. Usando o `inference.py`

```bash
uv run python train.py                                   # treina e salva checkpoints/latest.pt (~5 min)
uv run python inference.py "o gato" --seed 1 --max-new-tokens 120
```

```text
o gato de Capitú, e a ser um perpo, e a diga e podia a minha mãe eram mesmo accordadecendo. Então que a pressa de escolhava a
```

| opção | padrão | efeito |
|---|---|---|
| `prompt` | `"o gato"` | texto inicial |
| `--max-new-tokens` | 300 | quantos caracteres gerar |
| `--temperature` | 0,8 | 0 = greedy |
| `--top-k` | desligado | só os $k$ mais prováveis |
| `--top-p` | 0,95 | massa de probabilidade mantida |
| `--seed` | aleatória | fixar para repetir a mesma geração |
| `--checkpoint` | `checkpoints/latest.pt` | outro modelo salvo |

O critério de aceitação ("dado o prompt 'o gato', o modelo deverá produzir tokens adicionais") está
cumprido.

---

## 12. O que cada teste garante

[`tests/test_generation.py`](../tests/test_generation.py):

| teste | propriedade |
|---|---|
| `test_top_k_keeps_the_k_largest` | top-k mantém exatamente os $k$ maiores |
| `test_top_k_at_least_vocab_size_changes_nothing` | $k \ge V$ não filtra nada |
| `test_top_p_keeps_the_smallest_set_that_reaches_p` | o exemplo 0,5/0,3/0,15/0,05 com $p^\ast = 0{,}7$ mantém 2 |
| `test_top_p_always_keeps_the_most_likely_token` | mesmo com $p^\ast$ minúsculo, sobra o mais provável |
| `test_top_p_of_one_keeps_everything` | $p^\ast = 1$ não filtra nada |
| `test_temperature_zero_is_greedy` | $\tau = 0$ escolhe o argmax |
| `test_low_temperature_sharpens_and_high_temperature_flattens` | $\tau$ baixo concentra; $\tau$ alto espalha |
| `test_sampling_follows_the_probabilities` | em 5.000 sorteios, as frequências batem com as probabilidades |
| `test_top_k_of_one_is_greedy` | $k = 1$ equivale ao greedy |
| `test_sampling_never_picks_a_filtered_token` | tokens filtrados nunca são sorteados |
| `test_generate_appends_max_new_tokens` | prompt + exatamente `max_new_tokens` tokens |
| `test_greedy_generation_matches_a_manual_loop` | `generate` = o loop escrito à mão |
| `test_prompt_longer_than_context_is_cropped` | prompts maiores que o contexto funcionam |
| `test_generation_stops_at_eos` | a geração para ao gerar `eos_id` |
| `test_same_seed_gives_the_same_text` | mesma seed, mesmo texto |
| `test_generate_restores_training_mode` | `generate` devolve o modo original do modelo |
| `test_empty_prompt_raises` | prompt vazio gera `ValueError` |

[`tests/test_checkpoint.py`](../tests/test_checkpoint.py):

| teste | propriedade |
|---|---|
| `test_round_trip_preserves_config_vocab_and_outputs` | salvar e carregar mantém config, vocabulário e as saídas do modelo |
| `test_loaded_model_keeps_weight_tying` | o LM head continua compartilhando a matriz do embedding |
| `test_loaded_model_is_ready_for_inference` | o modelo carregado vem em modo `eval` |

---

## Resumo

1. Gerar é um loop: prever a distribuição do próximo token com a última posição, escolher um token,
   acrescentá-lo e repetir, mantendo no máximo 128 tokens de contexto.
2. Greedy é determinístico e degenera em loops ("de um primeiro de um primeiro…"); a frase mais provável
   passo a passo não é um bom texto.
3. A temperature divide os logits: $\tau < 1$ concentra (seguro e repetitivo), $\tau > 1$ espalha (variado
   e errático). Na seção 5.3, 0,2 repete, 0,7 e 1,0 soam como o livro, 1,5 vira lixo.
4. Top-k corta a cauda com um número fixo de candidatos; top-p usa uma massa fixa de probabilidade e se
   adapta à confiança do modelo (3 candidatos depois de "qu", 27 depois de "olhos de").
5. O modelo aprendeu ortografia, personagens e o formato do livro, sem sentido global, e recombina em vez
   de copiar (trechos idênticos de no máximo 13–19 caracteres). Um checkpoint simples permite gerar sem
   treinar de novo.

---

## Para checar o entendimento

1. Por que a geração usa só os logits da **última** posição, se o modelo calcula logits para todas?
2. Por que o greedy entra em loop? Por que um loop, uma vez iniciado, tende a continuar?
3. Com $z = (3, 1, 0)$, calcule as probabilidades com $\tau = 0{,}5$, $\tau = 1$ e $\tau = 2$.
4. O que acontece com a distribuição quando $\tau \to 0$? E quando $\tau \to \infty$?
5. Por que $k = 1$ dá exatamente o mesmo texto que o greedy?
6. Com probabilidades 0,4; 0,3; 0,2; 0,1 e $p^\ast = 0{,}6$, quantos tokens o top-p mantém? E com
   $p^\ast = 0{,}9$?
7. Dê um exemplo de contexto em que top-k com $k = 10$ é pior que top-p com $p^\ast = 0{,}9$, e explique.
8. Por que a temperature é aplicada antes do top-p? O resultado mudaria se fosse depois?
9. Os textos de $\tau = 1{,}0$ e $\tau = 1{,}5$ compartilham trechos. Por quê?
10. O que é preciso salvar num checkpoint para gerar texto em outro computador? O que aconteceria se o
    vocabulário fosse salvo em outra ordem?
11. Por que a geração fica mais lenta depois de alguns caracteres? Que informação o KV cache reaproveita?
