import unicodedata

import pytest
import torch

from tokenizer import BPETokenizer, Vocab
from tokenizer.vocab import BOS, EOS, SPECIAL_TOKENS, UNK

TEXT = (
    "O gato está na casa. O gato dorme. O cachorro late, o gato foge.\n\n"
    "O gato volta à casa! Onde está o gato? O gato dorme de novo."
)


@pytest.fixture
def tok() -> BPETokenizer:
    return BPETokenizer.train(TEXT, vocab_size=80)


# --- treino ---


def test_vocab_size_is_specials_plus_alphabet_plus_merges(tok):
    alphabet = len(set(TEXT))
    assert tok.vocab_size == len(SPECIAL_TOKENS) + alphabet + len(tok.merges)
    assert tok.vocab_size <= 80  # o alvo é um teto: o corpus pode não ter pares pra chegar nele


def test_train_is_deterministic():
    a = BPETokenizer.train(TEXT, vocab_size=80)
    b = BPETokenizer.train(TEXT, vocab_size=80)
    assert a.vocab.id_to_token == b.vocab.id_to_token
    assert a.merges == b.merges


def test_more_merges_never_shrinks_a_previous_merge_list():
    # Cada merge só depende dos pares mais frequentes até ali; os primeiros N não mudam.
    short = BPETokenizer.train(TEXT, vocab_size=60)
    long = BPETokenizer.train(TEXT, vocab_size=90)
    assert long.merges[: len(short.merges)] == short.merges


def test_merges_favor_frequent_words():
    # "gato" aparece 4 vezes no texto de treino: deve virar um token só.
    tok = BPETokenizer.train(TEXT, vocab_size=200)
    assert "gato" in tok.vocab.id_to_token


def test_vocab_size_too_small_for_alphabet_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train(TEXT, vocab_size=len(SPECIAL_TOKENS))


def test_small_corpus_stops_before_the_target_vocab_size():
    # "ab" só tem 1 par possível: o treino para de aprender bem antes do alvo, sem travar.
    tok = BPETokenizer.train("ab", vocab_size=1000)
    assert tok.vocab_size < 1000


# --- encode / decode ---


def test_common_word_encodes_as_a_single_token(tok):
    ids = tok.encode("gato")
    assert len(ids) == 1
    assert tok.vocab.get_token(ids[0]) == "gato"


def test_punctuation_does_not_merge_into_the_word(tok):
    # "gato" e "gato." usam o mesmo primeiro token; o "." é sempre separado.
    with_dot = tok.encode("gato.")
    without_dot = tok.encode("gato")
    assert with_dot[: len(without_dot)] == without_dot
    assert tok.vocab.get_token(with_dot[-1]) == "."


@pytest.mark.parametrize("text", [TEXT, "", " ", "gato", "à casa", "\n\n", "O gato! O gato?"])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_becomes_unk(tok):
    ids = tok.encode("gato 🐱")
    assert ids[-1] == tok.vocab.unk_id
    assert tok.decode(ids) == "gato " + UNK


def test_special_tokens_wrap_sequence(tok):
    ids = tok.encode("gato", add_special_tokens=True)
    assert ids[0] == tok.vocab.bos_id
    assert ids[-1] == tok.vocab.eos_id
    assert tok.decode(ids) == "gato"
    assert tok.decode(ids, skip_special_tokens=False) == f"{BOS}gato{EOS}"


def test_pad_is_skipped_on_decode(tok):
    ids = tok.encode("gato") + [tok.vocab.pad_id] * 3
    assert tok.decode(ids) == "gato"


def test_special_token_text_is_not_a_special_token():
    tok = BPETokenizer.train("<EOS>" * 5, vocab_size=20)
    ids = tok.encode("<EOS>")
    assert tok.decode(ids) == "<EOS>"
    assert tok.vocab.eos_id not in ids


def test_equivalent_unicode_forms_encode_equally(tok):
    nfc = unicodedata.normalize("NFC", "está")
    nfd = unicodedata.normalize("NFD", "está")
    assert nfc != nfd
    assert tok.encode(nfd) == tok.encode(nfc)
    assert tok.vocab.unk_id not in tok.encode(nfd)


def test_decode_accepts_torch_tensor(tok):
    ids = torch.tensor(tok.encode("gato", add_special_tokens=True))
    assert tok.decode(ids) == "gato"


def test_reconstructed_tokenizer_encodes_identically(tok):
    # É assim que training/checkpoint.py reconstrói o tokenizer a partir do checkpoint.
    clone = BPETokenizer(Vocab(tok.vocab.id_to_token), list(tok.merges))
    text = "o gato dorme na casa"
    assert clone.encode(text) == tok.encode(text)
    assert clone.decode(clone.encode(text)) == tok.decode(tok.encode(text))
