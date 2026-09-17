"""Baixa um recorte da Wikipedia em português e junta ao corpus literário.

Fonte: dataset `wikimedia/wikipedia` no Hugging Face, um recorte já processado da Wikipedia
(sem marcação wiki na maior parte dos artigos) distribuído em arquivos Parquet — um único
download grande, sem o limite de taxa por IP da API de busca da própria Wikipedia.

Este script baixa **um** dos 6 fragmentos (shards) do recorte em português (o menor, ~193 MB),
embaralha os artigos com uma seed fixa e acumula texto até `TARGET_CHARS`. Descarta artigos
curtos demais e os poucos (~1-2%) que ainda têm sobras de marcação wiki (templates, tabelas,
tags de referência) que a extração do Hugging Face não limpou.

Uso: uv run --group data python scripts/prepare_wikipedia_corpus.py
"""

import random
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
SHARD_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.pt/train-00002-of-00006.parquet"
)

TARGET_CHARS = 15_000_000  # ~10x o corpus literário atual (1,6M caracteres)
MIN_ARTICLE_CHARS = 500  # descarta esboços (stubs) curtos demais para ensinar algo
SEED = 42  # mesma ordem de embaralhamento a cada execução, para um resultado reproduzível

# Marcas de que a extração do Hugging Face não limpou o wikitext deste artigo (infobox,
# tabela ou referência crua). Cerca de 1-2% dos artigos têm alguma dessas sobras; descartados.
DIRTY_MARKERS = ("{{", "{|", "<ref", "[[Ficheiro:", "[[File:", "[[Imagem:", "[[Categoria:")


def download_shard(destination: Path) -> None:
    print(f"baixando {SHARD_URL}")
    with urllib.request.urlopen(SHARD_URL, timeout=30) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            written += len(chunk)
            done = written / total if total else 0
            print(f"  {written / 1e6:,.0f} MB / {total / 1e6:,.0f} MB ({done:.0%})", end="\r")
    print()


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    while "\n\n\n" in text:  # listas e seções vazias deixam sequências longas de linhas em branco
        text = text.replace("\n\n\n", "\n\n")
    return unicodedata.normalize("NFC", text)


def extract_articles(shard_path: Path, target_chars: int) -> list[tuple[str, str]]:
    parquet_file = pq.ParquetFile(shard_path)
    group_order = list(range(parquet_file.metadata.num_row_groups))
    random.Random(SEED).shuffle(group_order)

    articles: list[tuple[str, str]] = []
    total_chars = 0
    for group_index in group_order:
        rows = parquet_file.read_row_group(group_index, columns=["title", "text"]).to_pydict()
        for title, raw_text in zip(rows["title"], rows["text"], strict=True):
            if any(marker in raw_text for marker in DIRTY_MARKERS):
                continue
            text = clean(raw_text)
            if len(text) < MIN_ARTICLE_CHARS:
                continue
            articles.append((title, text))
            total_chars += len(text)
            if total_chars >= target_chars:
                print(f"  {len(articles)} artigos, {total_chars:,} caracteres")
                return articles
        print(f"  {len(articles)} artigos, {total_chars:,} caracteres", end="\r")

    raise RuntimeError(
        f"o shard acabou com só {total_chars:,} caracteres, menos que os {target_chars:,} pedidos"
    )


def main() -> None:
    CORPUS_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = Path(tmp) / "wikipedia_pt_shard.parquet"
        download_shard(shard_path)

        print(f"extraindo artigos até {TARGET_CHARS:,} caracteres...")
        articles = extract_articles(shard_path, TARGET_CHARS)

    # Cada artigo começa com o título: marca a fronteira entre tópicos, do mesmo jeito que os
    # títulos de capítulo marcam fronteiras no corpus literário (scripts/prepare_corpus.py).
    wikipedia_text = "\n\n".join(f"{title}\n\n{text}" for title, text in articles) + "\n"
    wikipedia_path = CORPUS_DIR / "wikipedia_pt.txt"
    wikipedia_path.write_text(wikipedia_text, encoding="utf-8")
    print(f"{wikipedia_path} ({len(wikipedia_text):,} caracteres, {len(articles)} artigos)")

    machado_path = CORPUS_DIR / "machado_de_assis.txt"
    if not machado_path.is_file():
        raise SystemExit(
            f"{machado_path} não existe. Rode primeiro: uv run python scripts/prepare_corpus.py"
        )
    machado_text = machado_path.read_text(encoding="utf-8")

    merged = machado_text + "\n" + wikipedia_text
    merged_path = CORPUS_DIR / "machado_e_wikipedia.txt"
    merged_path.write_text(merged, encoding="utf-8")
    print(f"{merged_path} ({len(merged):,} caracteres)")


if __name__ == "__main__":
    main()
