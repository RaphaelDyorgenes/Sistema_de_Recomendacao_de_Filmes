"""Faz download do MovieLens-20M via kagglehub e salva em ``data/raw/``.

Opcionalmente amostra ``DATA_SAMPLE_SIZE`` interações para viabilizar
desenvolvimento local rápido, mantendo o pipeline compatível com o
dataset completo.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from recsys.utils.config import settings


def download_movielens(dest: Path | None = None) -> Path:
    """Baixa o dataset e copia os CSVs relevantes para ``dest``.

    Args:
        dest: Diretório de destino. Se ``None``, usa ``settings.data_raw_dir``.

    Returns:
        Caminho do diretório onde os arquivos foram salvos.
    """
    dest = dest or settings.data_raw_dir
    dest.mkdir(parents=True, exist_ok=True)

    print("⬇️  Baixando MovieLens-20M via kagglehub...")
    source = Path(kagglehub.dataset_download("grouplens/movielens-20m-dataset"))

    # kagglehub pode retornar o diretório pai; precisamos do subdir real.
    csv_dir = source
    if not list(csv_dir.glob("*.csv")):
        candidates = list(csv_dir.rglob("rating*.csv"))
        if candidates:
            csv_dir = candidates[0].parent

    relevant_files = ["rating.csv", "movie.csv", "tag.csv", "link.csv"]
    copied = 0
    for name in relevant_files:
        src_file = csv_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dest / name)
            print(f"  ✅ {name} → {dest / name}")
            copied += 1
        else:
            print(f"  ⚠️  {name} não encontrado em {csv_dir}")

    if copied == 0:
        # Listar o que existe para debug
        print(f"  ℹ️  Arquivos disponíveis em {csv_dir}:")
        for f in sorted(csv_dir.iterdir()):
            print(f"      {f.name}")

    return dest


def sample_ratings(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    n: int | None = None,
    seed: int | None = None,
) -> Path:
    """Amostra ``n`` interações de ``rating.csv`` de forma determinística.

    Args:
        raw_dir: Diretório com o ``rating.csv`` completo.
        processed_dir: Diretório de saída para a amostra.
        n: Número de linhas. Se ``0`` ou ``None``, copia tudo.
        seed: Semente para reprodutibilidade.

    Returns:
        Caminho do arquivo ``ratings_sample.csv`` gerado.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    processed_dir = processed_dir or settings.data_processed_dir
    n = n if n is not None else settings.data_sample_size
    seed = seed if seed is not None else settings.random_seed

    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = raw_dir / "rating.csv"
    if not ratings_path.exists():
        msg = f"rating.csv não encontrado em {raw_dir}"
        raise FileNotFoundError(msg)

    print(f"📄 Lendo {ratings_path}...")
    df = pd.read_csv(ratings_path)
    print(f"   Total de interações: {len(df):,}")

    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
        print(f"   Amostra reduzida para: {len(df):,} interações (seed={seed})")

    out_path = processed_dir / "ratings_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"   ✅ Salvo em {out_path}")

    return out_path


def main() -> None:
    """Fluxo completo: download + amostragem."""
    download_movielens()
    sample_ratings()


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    main()
