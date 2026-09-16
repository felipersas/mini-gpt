"""Tokenizer por caracteres: cada caractere Unicode é um token."""

import unicodedata
from collections.abc import Iterable

from tokenizer.vocab import Vocab


class CharTokenizer:
    """Converte texto em IDs (encode) e IDs em texto (decode), um ID por caractere."""

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Cria o vocabulário com os caracteres distintos de `text`."""
        # Ordenar deixa os IDs determinísticos para o mesmo texto.
        chars = sorted(set(_normalize(text)))
        return cls(Vocab(chars))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Texto -> IDs. Caracteres desconhecidos viram <UNK>.

        Com `add_special_tokens`, a sequência fica entre <BOS> e <EOS>.
        """
        ids = [self.vocab.get_id(char) for char in _normalize(text)]
        if add_special_tokens:
            ids = [self.vocab.bos_id, *ids, self.vocab.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """IDs -> texto. Omite <PAD>, <BOS> e <EOS> por padrão; <UNK> é sempre mantido."""
        vocab = self.vocab
        skipped = {vocab.pad_id, vocab.bos_id, vocab.eos_id} if skip_special_tokens else set()
        # int() normaliza tensores 0-d, que nunca seriam encontrados no set.
        tokens = (vocab.get_token(int(i)) for i in ids if int(i) not in skipped)
        return "".join(tokens)


def _normalize(text: str) -> str:
    # NFC: "á" composto (U+00E1) e decomposto ("a" + U+0301) viram o mesmo token.
    return unicodedata.normalize("NFC", text)
