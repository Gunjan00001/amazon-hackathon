from pathlib import Path

import pandas as pd

from ber.colab_io import export_for_colab, import_cosine
from ber.config import Config


def _entity_frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "name_norm", "name_roman", "addr_norm"])


def _build_cfg(tmp_path):
    data = tmp_path / "data"
    processed = data / "processed"
    pairs = data / "pairs"
    processed.mkdir(parents=True)
    pairs.mkdir(parents=True)

    _entity_frame(
        [
            ("S1-1", "best bakery inc", "best bakery", "10 main st"),
            ("S1-2", "pizza palace", "pizza palace", "99 oak rd"),
            ("S1-1", "best bakery inc", "best bakery", "10 main st"),
        ]
    ).to_parquet(processed / "train_source1.parquet", index=False)
    _entity_frame(
        [
            ("S2-1", "best bakery", "best bakery", "10 main street"),
            ("S3-1", "pizza palace ltd", "pizza palace", "99 oak road"),
        ]
    ).to_parquet(processed / "train_source2.parquet", index=False)

    pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-1", "S3-1"],
            "label": [1, 0, 1],
        }
    ).to_parquet(pairs / "train_pairs.parquet", index=False)

    return Config(
        root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        data_dir=data,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "output",
        seed=42,
    )


def test_colab_export_roundtrip_attaches_cosine(tmp_path):
    cfg = _build_cfg(tmp_path)
    exported = export_for_colab(cfg, splits=("train",))

    entities = pd.read_parquet(exported["entities"])
    assert list(entities.columns) == ["entity_id", "name_norm", "name_roman", "addr_norm"]
    assert entities["entity_id"].nunique() == len(entities)

    exported_pairs = pd.read_parquet(exported["pairs"]["train"])
    assert list(exported_pairs.columns) == ["s1_id", "cand_id"]

    out_dir = Path(cfg.data_dir) / "colab_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-1", "S3-1"],
            "emb_name_cos": [0.9, 0.1, 0.8],
            "emb_addr_cos": [0.7, 0.2, 0.6],
        }
    ).to_parquet(out_dir / "cosine_train.parquet", index=False)

    cosine = import_cosine(cfg, "train")
    assert set(["s1_id", "cand_id", "emb_name_cos", "emb_addr_cos"]).issubset(cosine.columns)

    merged = exported_pairs.merge(cosine, on=["s1_id", "cand_id"], how="left")
    assert merged["emb_name_cos"].notna().all()
    row = merged[(merged["s1_id"] == "S1-1") & (merged["cand_id"] == "S2-1")].iloc[0]
    assert row["emb_name_cos"] == 0.9
    assert row["emb_addr_cos"] == 0.7
