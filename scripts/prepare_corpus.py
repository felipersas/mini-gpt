"""Baixa "Dom Casmurro" (Project Gutenberg #55752) e grava o texto limpo em corpus/.

Uso: uv run python scripts/prepare_corpus.py
"""

import re
import unicodedata
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"
OUTPUT = Path(__file__).resolve().parent.parent / "corpus" / "dom_casmurro.txt"

FIRST_CHAPTER = "\nI\n\nDo titulo."  # antes: licença do Gutenberg e folha de rosto
INDEX = "\nINDICE\n"  # depois: índice de capítulos e licença


def clean(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    text = text[text.index(FIRST_CHAPTER) : text.index(INDEX)]
    text = text.replace("_", "")  # marcação de itálico do Gutenberg

    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        # Prosa tem quebra fixa a cada ~70 colunas; versos vêm indentados e mantêm as quebras.
        separator = "\n" if lines[0].startswith("  ") else " "
        paragraphs.append(separator.join(line.strip() for line in lines))

    return unicodedata.normalize("NFC", "\n\n".join(paragraphs)) + "\n"


def main() -> None:
    with urllib.request.urlopen(URL) as response:
        raw = response.read().decode("utf-8")
    text = clean(raw)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"{OUTPUT} ({len(text):,} caracteres)")


if __name__ == "__main__":
    main()
