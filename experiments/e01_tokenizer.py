"""Experimento: tokenizar um texto real e comparar caracteres × palavras.

Rode com:  uv run python -m experiments.e01_tokenizer
"""

from collections import Counter

from tokenizer import CharTokenizer

# Machado de Assis, "Dom Casmurro" (1899), capítulo I, em ortografia atualizada.
CORPUS = """\
Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central \
um rapaz aqui do bairro, que eu conheço de vista e de chapéu. Cumprimentou-me, \
sentou-se ao pé de mim, falou da lua e dos ministros, e acabou recitando-me versos. \
A viagem era curta, e os versos pode ser que não fossem inteiramente maus. Sucedeu, \
porém, que, como eu estava cansado, fechei os olhos três ou quatro vezes; tanto \
bastou para que ele interrompesse a leitura e metesse os versos no bolso.
"""


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    tok = CharTokenizer.from_text(CORPUS)

    section("Vocabulário")
    print(f"vocab_size = {tok.vocab_size}")
    for token_id, token in enumerate(tok.vocab.id_to_token):
        print(f"{token_id:>3} {token!r}", end="   " if (token_id + 1) % 8 else "\n")
    print()

    section("encode / decode")
    text = "a casa do rapaz"
    ids = tok.encode(text, add_special_tokens=True)
    print(f"texto:   {text!r}")
    print(f"IDs:     {ids}")
    print(f"decode:  {tok.decode(ids)!r}")
    print(f"round-trip ok: {tok.decode(ids) == text}")
    print("sem pular especiais:", repr(tok.decode(ids, skip_special_tokens=False)))

    section("Tokens desconhecidos")
    # O trecho do livro não tem nenhum 'á', então "está" já perde um caractere.
    unknown = "o gato está na casa 🐱"
    decoded = tok.decode(tok.encode(unknown))
    print(f"{unknown!r} -> {decoded!r}  (round-trip ok: {decoded == unknown})")

    words = CORPUS.split()
    word_counts = Counter(words)
    phrase = "o gato está na casa"
    char_unks = tok.encode(phrase).count(tok.vocab.unk_id)
    word_unks = [word for word in phrase.split() if word not in word_counts]
    print(f"\n{phrase!r} com o vocabulário do trecho:")
    print(f"  por caracteres: {char_unks} <UNK> em {len(phrase)} tokens")
    print(f"  por palavras:   {len(word_unks)} <UNK> em {len(phrase.split())} tokens {word_unks}")

    section("Caracteres × palavras (no mesmo trecho)")
    singletons = sum(1 for count in word_counts.values() if count == 1)
    chars_per_word = len(tok.encode(CORPUS)) / len(words)
    print(f"caracteres: {len(tok.encode(CORPUS)):>4} tokens | vocabulário {tok.vocab_size}")
    print(f"palavras:   {len(words):>4} tokens | vocabulário {len(word_counts)}")
    print(f"palavras que aparecem uma única vez: {singletons}/{len(word_counts)}")
    print(f"'versos' e 'versos.' contam como palavras diferentes: {'versos.' in word_counts}")
    print(f"context_length=128 em caracteres ≈ {128 / chars_per_word:.0f} palavras")


if __name__ == "__main__":
    main()
