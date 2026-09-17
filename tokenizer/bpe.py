"""Tokenizer BPE (byte-pair encoding): aprende subpalavras a partir do corpus."""

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from tokenizer.vocab import SPECIAL_TOKENS, Vocab

# Uma "palavra" é uma sequência de letras (e _), uma de dígitos, um espaço em branco, ou uma
# sequência da mesma pontuação — nunca uma mistura das quatro. Os merges nunca atravessam essas
# fronteiras, então "gato" e "gato." aprendem o mesmo token para "gato", em vez de dois tokens
# diferentes que só diferem pela pontuação colada.
_WORD_PATTERN = re.compile(r"\s+|[^\W\d]+|\d+|[^\w\s]+")


class BPETokenizer:
    """Converte texto em IDs (encode) e IDs em texto (decode), com subpalavras aprendidas.

    Cada token é uma sequência de caracteres aprendida por frequência (Sennrich et al., 2015):
    começa com um alfabeto de caracteres e vai fundindo o par adjacente mais frequente num token
    novo, repetidamente, até chegar em `vocab_size`.
    """

    def __init__(self, vocab: Vocab, merges: list[tuple[str, str]]) -> None:
        self.vocab = vocab
        self.merges = merges
        # Ordem de aprendizado -> prioridade: um merge aprendido mais cedo (mais frequente na
        # época do treino) sempre vence um aprendido depois, não importa onde apareça no texto.
        self._merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """Aprende os merges a partir de `text`: alfabeto de caracteres + pares mais frequentes.

        `vocab_size` é o alvo (especiais + alfabeto + merges); pode sair um pouco menor se o
        corpus for pequeno demais para tantos merges.
        """
        text = _normalize(text)
        word_counts = Counter(_WORD_PATTERN.findall(text))
        base_chars = sorted({char for word in word_counts for char in word})

        num_merges = vocab_size - len(SPECIAL_TOKENS) - len(base_chars)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) é menor que especiais + alfabeto "
                f"({len(SPECIAL_TOKENS) + len(base_chars)}); aumente vocab_size"
            )

        symbols_by_word = {word: list(word) for word in word_counts}
        pair_counts, pair_words = _count_pairs(symbols_by_word, word_counts)

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            if not pair_counts:
                break  # o corpus se esgotou antes do alvo (corpus pequeno demais, ou repetitivo)
            # Empate: o par lexicograficamente menor vence, para o treino ser determinístico.
            best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            _apply_merge_everywhere(
                best_pair, merged_token, symbols_by_word, word_counts, pair_counts, pair_words
            )

        tokens = base_chars + [left + right for left, right in merges]
        return cls(Vocab(tokens), merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Texto -> IDs. Um caractere nunca visto no treino vira <UNK>.

        Com `add_special_tokens`, a sequência fica entre <BOS> e <EOS>.
        """
        ids = []
        for word in _WORD_PATTERN.findall(_normalize(text)):
            ids.extend(self.vocab.get_id(symbol) for symbol in self._encode_word(word))
        if add_special_tokens:
            ids = [self.vocab.bos_id, *ids, self.vocab.eos_id]
        return ids

    def _encode_word(self, word: str) -> list[str]:
        """Aplica os merges aprendidos, sempre o de maior prioridade primeiro (menor rank)."""
        symbols = list(word)
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:], strict=False))
            ranked = [
                (self._merge_rank[p], i) for i, p in enumerate(pairs) if p in self._merge_rank
            ]
            if not ranked:
                break
            _, i = min(ranked)
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2 :]
        return symbols

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """IDs -> texto. Omite <PAD>, <BOS> e <EOS> por padrão; <UNK> é sempre mantido."""
        vocab = self.vocab
        skipped = {vocab.pad_id, vocab.bos_id, vocab.eos_id} if skip_special_tokens else set()
        # int() normaliza tensores 0-d, que nunca seriam encontrados no set.
        tokens = (vocab.get_token(int(i)) for i in ids if int(i) not in skipped)
        return "".join(tokens)


def _count_pairs(
    symbols_by_word: dict[str, list[str]], word_counts: dict[str, int]
) -> tuple[Counter, dict[tuple[str, str], set[str]]]:
    """Conta cada par adjacente, ponderado pela frequência da palavra, e quem o contém."""
    pair_counts: Counter = Counter()
    pair_words: dict[tuple[str, str], set[str]] = defaultdict(set)
    for word, symbols in symbols_by_word.items():
        freq = word_counts[word]
        for pair in zip(symbols, symbols[1:], strict=False):
            pair_counts[pair] += freq
            pair_words[pair].add(word)
    return pair_counts, pair_words


def _apply_merge_everywhere(
    pair: tuple[str, str],
    merged: str,
    symbols_by_word: dict[str, list[str]],
    word_counts: dict[str, int],
    pair_counts: Counter,
    pair_words: dict[tuple[str, str], set[str]],
) -> None:
    """Aplica `pair` -> `merged` só nas palavras que o contêm, atualizando as contagens no lugar.

    Recontar do zero a cada merge custaria uma passada pelo corpus inteiro por merge; como só as
    palavras com `pair` mudam, atualizar só essas é o que torna milhares de merges rápidos.
    """
    for word in list(pair_words[pair]):
        symbols = symbols_by_word[word]
        freq = word_counts[word]
        for old_pair in zip(symbols, symbols[1:], strict=False):
            pair_counts[old_pair] -= freq
            if pair_counts[old_pair] <= 0:
                del pair_counts[old_pair]
            pair_words[old_pair].discard(word)
        new_symbols = _merge_symbols(symbols, pair, merged)
        symbols_by_word[word] = new_symbols
        for new_pair in zip(new_symbols, new_symbols[1:], strict=False):
            pair_counts[new_pair] += freq
            pair_words[new_pair].add(word)
    pair_words.pop(pair, None)


def _merge_symbols(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    """Substitui toda ocorrência não sobreposta de `pair`, da esquerda para a direita."""
    result = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result


def _normalize(text: str) -> str:
    # NFC: "á" composto (U+00E1) e decomposto ("a" + U+0301) viram o mesmo token.
    return unicodedata.normalize("NFC", text)
