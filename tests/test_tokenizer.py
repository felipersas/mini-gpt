import unicodedata

import pytest
import torch

from tokenizer import CharTokenizer, Vocab
from tokenizer.vocab import BOS, EOS, PAD, SPECIAL_TOKENS, UNK

TEXT = "o gato está na casa.\nEle dorme, às vezes!"


@pytest.fixture
def tok() -> CharTokenizer:
    return CharTokenizer.from_text(TEXT)


# --- Vocabulário ---


def test_special_tokens_have_fixed_ids():
    vocab = Vocab(["a", "b"])
    assert vocab.id_to_token[:4] == [PAD, UNK, BOS, EOS]
    assert (vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id) == (0, 1, 2, 3)


def test_vocab_maps_are_inverse():
    vocab = Vocab(["x", "y", "z"])
    assert len(vocab.token_to_id) == len(vocab) == 7
    for token, token_id in vocab.token_to_id.items():
        assert vocab.get_token(token_id) == token


def test_vocab_removes_duplicates():
    vocab = Vocab(["a", "a", "b", PAD])
    assert len(vocab) == len(SPECIAL_TOKENS) + 2
    assert vocab.pad_id == 0


def test_unknown_token_maps_to_unk():
    assert Vocab(["a"]).get_id("z") == Vocab.unk_id


@pytest.mark.parametrize("bad_id", [-1, 7])
def test_get_token_rejects_out_of_range_ids(bad_id):
    with pytest.raises(IndexError):
        Vocab(["a", "b", "c"]).get_token(bad_id)


# --- Tokenizer ---


def test_vocab_has_one_entry_per_distinct_char(tok):
    assert tok.vocab_size == len(SPECIAL_TOKENS) + len(set(TEXT))


def test_vocab_does_not_depend_on_char_order():
    abc = CharTokenizer.from_text("abc").vocab.id_to_token
    cba = CharTokenizer.from_text("cba").vocab.id_to_token
    assert abc == cba


def test_encode_produces_one_id_per_char(tok):
    ids = tok.encode("gato")
    assert ids == [tok.vocab.get_id(char) for char in "gato"]


@pytest.mark.parametrize("text", [TEXT, "", " ", "casa às vezes", "\n\n"])
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
    # Escrever "<EOS>" no texto não pode gerar o ID de <EOS>: são só 5 caracteres.
    tok = CharTokenizer.from_text("<EOS>")
    ids = tok.encode("<EOS>")
    assert len(ids) == 5
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
