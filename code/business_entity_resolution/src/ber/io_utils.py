from pathlib import Path

import pandas as pd

CHUNK = 1_000_000


def read_source(path):
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK
    )


def write_parquet(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
