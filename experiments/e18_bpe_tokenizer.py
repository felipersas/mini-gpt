"""Experimento: tokenizer BPE — merges aprendidos, compressão, e por que evita palavras cortadas.

Rode com: uv run python -m experiments.e18_bpe_tokenizer
"""

import time
from pathlib import Path

from generation.generate import trim_trailing_partial_word
from tokenizer import BPETokenizer, CharTokenizer
from tokenizer.bpe import _WORD_PATTERN

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "machado_de_assis.txt"
VOCAB_SIZE = 4096


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def train_tokenizer(text: str) -> BPETokenizer:
    section("1. Treinando o BPE no corpus real")
    start = time.perf_counter()
    tok = BPETokenizer.train(text, vocab_size=VOCAB_SIZE)
    elapsed = time.perf_counter() - start
    alphabet = len({c for c in text})
    print(f"  {len(text):,} caracteres | alvo vocab_size {VOCAB_SIZE} | treino em {elapsed:.1f}s")
    print(f"  alfabeto: {alphabet} caracteres | merges aprendidos: {len(tok.merges)}")
    print(f"  vocab_size real: {tok.vocab_size}")
    print("\n  primeiros 10 merges (os pares mais frequentes do corpus inteiro):")
    for pair in tok.merges[:10]:
        print(f"    {pair} -> {pair[0] + pair[1]!r}")
    print("\n  10 merges por volta do meio (menos óbvios, mais específicos):")
    mid = len(tok.merges) // 2
    for pair in tok.merges[mid : mid + 10]:
        print(f"    {pair} -> {pair[0] + pair[1]!r}")
    return tok


def common_words_become_one_token(tok: BPETokenizer) -> None:
    section("2. Palavras comuns viram um token só; raras, viram pedaços")
    words = ["o", "que", "não", "Capitú", "coração", "escrivaninha", "retroactivamente"]
    for word in words:
        ids = tok.encode(word)
        pieces = [tok.vocab.get_token(i) for i in ids]
        one_token = "sim" if len(pieces) == 1 else "não"
        print(f"  {word!r:<20} -> {pieces}  (um token só: {one_token})")


def punctuation_never_glues_to_the_word(tok: BPETokenizer) -> None:
    section("3. Pontuação nunca gruda na palavra")
    for text in ["gato", "gato.", "gato,", "gato!", "gato?"]:
        pieces = [tok.vocab.get_token(i) for i in tok.encode(text)]
        print(f"  {text!r:<8} -> {pieces}")
    print("\n  o primeiro token é sempre 'gato': aprender esse merge uma vez serve pra toda")
    print("  pontuação depois dele, em vez de um merge por combinação palavra+pontuação.")


def compression(tok: BPETokenizer, char_tok: CharTokenizer, text: str) -> None:
    section("4. Compressão: quantos caracteres cabem no mesmo context_length")
    bpe_ids = tok.encode(text)
    char_ids = char_tok.encode(text)
    ratio = len(text) / len(bpe_ids)
    print(f"  corpus inteiro: {len(text):,} caracteres")
    print(f"  CharTokenizer:  {len(char_ids):,} tokens (1,00 caractere/token, por definição)")
    print(f"  BPETokenizer:   {len(bpe_ids):,} tokens ({ratio:.2f} caracteres/token)")
    for context_length in (128, 256):
        print(
            f"  context_length={context_length}: CharTokenizer cobre {context_length} caracteres"
            f" | BPETokenizer cobre ~{context_length * ratio:.0f} caracteres"
        )


def unknown_characters(tok: BPETokenizer) -> None:
    section("5. Um caractere nunca visto no treino vira <UNK>")
    text = "gato 🐱"
    ids = tok.encode(text)
    print(f"  {text!r} -> {ids}")
    print(f"  decode: {tok.decode(ids)!r}")
    print("  o emoji não existe no alfabeto do corpus (só português + pontuação usual)")


def round_trip_the_whole_corpus(tok: BPETokenizer, text: str) -> None:
    section("6. Round-trip exato no corpus inteiro")
    ids = tok.encode(text)
    back = tok.decode(ids, skip_special_tokens=False)
    print(f"  decode(encode(corpus)) == corpus: {back == text}")
    print(f"  ({len(text):,} caracteres -> {len(ids):,} tokens -> {len(back):,} caracteres)")


def why_trimming_matters(tok: BPETokenizer) -> None:
    section("7. Por que ainda existe um corte de segurança na geração")
    # Simula uma geração que parou bem no meio de uma palavra rara (não vira 1 token só).
    ids = tok.encode("a escrivaninha")
    pieces = [tok.vocab.get_token(i) for i in ids]
    print(f"  'a escrivaninha' -> {pieces}")
    cut_points = [3, 4, len(ids)]
    for cut in cut_points:
        partial = tok.decode(ids[:cut])
        trimmed = trim_trailing_partial_word(partial)
        print(f"  parando em {cut} tokens: {partial!r:<22} -> trim: {trimmed!r}")
    print("\n  mesmo com BPE, uma palavra rara ainda pode ser vários tokens: se a geração parar")
    print("  entre eles, o corte de segurança evita mostrar o pedaço incompleto.")


def main() -> None:
    if not CORPUS.is_file():
        raise SystemExit(
            f"Corpus não encontrado: {CORPUS}\nRode: uv run python scripts/prepare_corpus.py"
        )
    text = CORPUS.read_text(encoding="utf-8")

    tok = train_tokenizer(text)
    common_words_become_one_token(tok)
    punctuation_never_glues_to_the_word(tok)
    char_tok = CharTokenizer.from_text(text)
    compression(tok, char_tok, text)
    unknown_characters(tok)
    round_trip_the_whole_corpus(tok, text)
    why_trimming_matters(tok)

    section("Padrão de pré-tokenização usado (tokenizer/bpe.py)")
    print(f"  {_WORD_PATTERN.pattern}")


if __name__ == "__main__":
    main()
