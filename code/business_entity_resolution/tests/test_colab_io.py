from pathlib import Path

import pandas as pd

from ber.colab_io import export_for_colab, import_cosine, merge_cosine
from ber.config import Config


def _entity_frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "name_norm", "name_roman", "addr_norm"])


def _build_cfg(tmp_path):
    data = tmp_path / "data"
    processed = data / "processed"
    pairs = data / "pairs"
    processed.mkdir(parents=True)
    pairs.mkdir(parents=True)
    dataset = tmp_path / "dataset" / "train"
    dataset.mkdir(parents=True)

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
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Best Bakery Inc", "Pizza Palace"],
            "business_address": ["10 Main St", "99 Oak Rd"],
            "country": ["US", "US"],
        }
    ).to_csv(dataset / "train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "entity_id": ["S2-1", "S3-1"],
            "business_name": ["Best Bakery", "Pizza Palace Ltd"],
            "business_address": ["10 Main Street", "99 Oak Road"],
            "country": ["US", "US"],
        }
    ).to_csv(dataset / "train_source2.tsv", sep="\t", index=False)

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
    assert list(entities.columns) == ["entity_id", "name_raw", "addr_raw", "name_norm", "name_roman", "addr_norm"]
    assert entities["entity_id"].nunique() == len(entities)
    raw = entities[entities["entity_id"] == "S1-1"].iloc[0]
    assert raw["name_raw"] == "Best Bakery Inc"
    assert raw["addr_raw"] == "10 Main St"
    assert raw["name_norm"] == "best bakery inc"

    exported_pairs = pd.read_parquet(exported["pairs"]["train"])
    assert list(exported_pairs.columns) == ["s1_id", "cand_id"]

    out_dir = Path(cfg.data_dir) / "colab_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-1", "S3-1"],
            "name_e5_cos": [0.9, 0.1, 0.8],
            "addr_e5_cos": [0.7, 0.2, 0.6],
            "entity_e5_cos": [0.85, 0.15, 0.75],
        }
    ).to_parquet(out_dir / "cosine_train.parquet", index=False)

    cosine = import_cosine(cfg, "train")
    assert set(["s1_id", "cand_id", "name_e5_cos", "addr_e5_cos", "entity_e5_cos"]).issubset(cosine.columns)

    merged = exported_pairs.merge(cosine, on=["s1_id", "cand_id"], how="left")
    assert merged["name_e5_cos"].notna().all()
    row = merged[(merged["s1_id"] == "S1-1") & (merged["cand_id"] == "S2-1")].iloc[0]
    assert row["name_e5_cos"] == 0.9
    assert row["addr_e5_cos"] == 0.7


def test_merge_cosine_attaches_and_defaults_missing(tmp_path):
    cfg = _build_cfg(tmp_path)
    data = Path(cfg.data_dir)
    (data / "features").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-1", "S3-1"],
            "name_char3_cos": [0.5, 0.1, 0.2],
            "name_e5_cos": [-1.0, -1.0, -1.0],
            "addr_e5_cos": [-1.0, -1.0, -1.0],
            "entity_e5_cos": [-1.0, -1.0, -1.0],
            "label": [1, 0, 1],
        }
    ).to_parquet(data / "features" / "train.parquet", index=False)
    out_dir = data / "colab_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "s1_id": ["S1-1"],
            "cand_id": ["S2-1"],
            "name_e5_cos": [0.9],
            "addr_e5_cos": [0.7],
            "entity_e5_cos": [0.8],
        }
    ).to_parquet(out_dir / "cosine_train.parquet", index=False)

    result = merge_cosine(cfg, "train")
    assert result["rows"] == 3

    frame = pd.read_parquet(data / "features" / "train.parquet")
    assert "label" in frame.columns
    assert list(frame.columns).count("name_e5_cos") == 1
    hit = frame[(frame["s1_id"] == "S1-1") & (frame["cand_id"] == "S2-1")].iloc[0]
    assert hit["name_e5_cos"] == 0.9
    assert hit["addr_e5_cos"] == 0.7
    miss = frame[(frame["s1_id"] == "S1-1") & (frame["cand_id"] == "S3-1")].iloc[0]
    assert miss["name_e5_cos"] == -1.0
