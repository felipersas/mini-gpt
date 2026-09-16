"""Baixa romances de Machado de Assis do Project Gutenberg e grava o texto limpo em corpus/.

Uso: uv run python scripts/prepare_corpus.py
"""

import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
GUTENBERG_END = re.compile(r"\n\*\*\* END OF THE PROJECT GUTENBERG EBOOK")


@dataclass
class Book:
    slug: str
    gutenberg_id: int
    start: str  # primeira ocorrência marca o início do texto (depois de rosto/prefácio)
    end: str | None = None  # se None, corta no marcador padrão do Gutenberg

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gutenberg_id}/pg{self.gutenberg_id}.txt"


BOOKS = [
    Book("dom_casmurro", 55752, start="\nI\n\nDo titulo.", end="\nINDICE\n"),
    Book("memorias_postumas", 54829, start="\nCAPITULO I\n", end="\nÍNDICE\n"),
    Book("quincas_borba", 55682, start="\nCAPITULO PRIMEIRO\n"),
    Book("esau_e_jacob", 56737, start="\nCAPITULO PRIMEIRO\n", end="\nÍNDICE\n"),
]


def clean(raw: str, book: Book) -> str:
    text = raw.replace("\r\n", "\n")
    start = text.index(book.start)
    end = text.index(book.end) if book.end else GUTENBERG_END.search(text, start).start()
    text = text[start:end]
    text = text.replace("_", "")  # marcação de itálico do Gutenberg

    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        # Prosa tem quebra fixa a cada ~70 colunas; versos vêm indentados e mantêm as quebras.
        separator = "\n" if lines[0].startswith("  ") else " "
        paragraphs.append(separator.join(line.strip() for line in lines))

    return unicodedata.normalize("NFC", "\n\n".join(paragraphs)) + "\n"


def main() -> None:
    CORPUS_DIR.mkdir(exist_ok=True)
    texts = []
    for book in BOOKS:
        with urllib.request.urlopen(book.url) as response:
            raw = response.read().decode("utf-8")
        text = clean(raw, book)
        path = CORPUS_DIR / f"{book.slug}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"{path} ({len(text):,} caracteres)")
        texts.append(text)

    # Um romance por parágrafo de separação: cada um já termina com \n.
    merged = "\n".join(texts)
    merged_path = CORPUS_DIR / "machado_de_assis.txt"
    merged_path.write_text(merged, encoding="utf-8")
    print(f"{merged_path} ({len(merged):,} caracteres, {len(BOOKS)} romances)")


if __name__ == "__main__":
    main()
