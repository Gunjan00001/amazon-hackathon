from pathlib import Path

import pandas as pd

from ber.config import Config
from ber.features import _COUNTRY_CACHE
from ber.predict import _write_tsv


def _cfg(tmp_path):
    data = tmp_path / "data"
    for sub in ("processed", "pairs", "tmp"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "dataset" / "train"
    dataset.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": ["a", "b", "c"],
            "business_address": ["x", "y", "z"],
            "country": ["US", "India", "France"],
        }
    ).to_csv(dataset / "train_source1.tsv", sep="\t", index=False)
    return Config(
        root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        data_dir=data,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "output",
        seed=42,
    )


def _stage(cfg, pred_rows):
    pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"]}).to_parquet(
        cfg.data_dir / "processed" / "train_source1.parquet", index=False
    )
    pd.DataFrame(
        {"s1_id": [r[0] for r in pred_rows], "cand_id": [r[1] for r in pred_rows]}
    ).to_parquet(cfg.data_dir / "pairs" / "train_pairs.parquet", index=False)
    pred_dir = cfg.data_dir / "tmp" / "train_pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "s1_id": [r[0] for r in pred_rows],
            "cand_id": [r[1] for r in pred_rows],
            "prob": [r[2] for r in pred_rows],
        }
    ).to_parquet(pred_dir / "part_0000.parquet", index=False)
    _COUNTRY_CACHE.clear()


def _matches(cfg):
    frame = pd.read_csv(
        cfg.output_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    return {row["source1_entity_id"]: row["matched_entity_ids"] for _, row in frame.iterrows()}


def test_write_tsv_applies_per_country_thresholds(tmp_path):
    cfg = _cfg(tmp_path)
    _stage(
        cfg,
        [
            ("S1-1", "S2-1", 0.96),
            ("S1-1", "S2-2", 0.93),
            ("S1-2", "S3-1", 0.93),
            ("S1-3", "S2-3", 0.96),
        ],
    )
    _write_tsv(
        cfg,
        "train",
        use_one_to_one=False,
        global_threshold=0.95,
        by_country={"us": 0.95, "india": 0.925},
        singleton_tau=None,
        n_buckets=1,
    )
    matches = _matches(cfg)
    assert matches["S1-1"] == "S2-1"
    assert matches["S1-2"] == "S3-1"
    assert matches["S1-3"] == "S2-3"


def test_write_tsv_applies_singleton_tau(tmp_path):
    cfg = _cfg(tmp_path)
    _stage(cfg, [("S1-1", "S2-1", 0.4), ("S1-2", "S2-2", 0.6)])
    _write_tsv(
        cfg,
        "train",
        use_one_to_one=False,
        global_threshold=0.3,
        by_country={},
        singleton_tau=0.5,
        n_buckets=1,
    )
    matches = _matches(cfg)
    assert matches["S1-1"] == ""
    assert matches["S1-2"] == "S2-2"
