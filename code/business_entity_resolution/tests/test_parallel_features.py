from pathlib import Path

import pandas as pd

from ber.config import Config
from ber.features import run_features
from ber.prepare import prepare_frame


def _prepared(entity_id, name, address, country):
    return prepare_frame(
        pd.DataFrame(
            {
                "entity_id": [entity_id],
                "business_name": [name],
                "business_address": [address],
                "country": [country],
            }
        )
    )


def _make_dataset(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "dataset"
    processed = root / "processed"
    pairs_dir = root / "pairs"
    for path in (dataset / "train", processed, pairs_dir, root / "tmp"):
        path.mkdir(parents=True, exist_ok=True)

    s1 = pd.concat(
        [
            _prepared("S1-1", "Best Bakery Inc", "10 Main St, Austin, TX", "US"),
            _prepared("S1-2", "Pizza Palace", "99 Oak Rd, Dallas, TX", "US"),
        ],
        ignore_index=True,
    )
    s2 = pd.concat(
        [
            _prepared("S2-1", "Best Bakery", "10 Main Street, Austin, TX", "US"),
            _prepared("S2-2", "Pizza Palace Ltd", "99 Oak Road, Dallas, TX", "US"),
        ],
        ignore_index=True,
    )
    s3 = _prepared("S3-1", "Best Bakery Co", "10 Main, Austin, TX", "US")

    s1.to_parquet(processed / "train_source1.parquet", index=False)
    s2.to_parquet(processed / "train_source2.parquet", index=False)
    s3.to_parquet(processed / "train_source3.parquet", index=False)

    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "cand_id": ["S2-1", "S3-1", "S2-2", "S2-1"],
            "is_s2": [True, False, True, True],
            "pass_id": [1, 3, 1, 9],
            "block_score": [1.0, 0.6, 0.9, 0.4],
            "label": [1, 1, 1, 0],
        }
    )
    pairs.to_parquet(pairs_dir / "train_pairs.parquet", index=False)

    for source, frame in ((1, s1), (2, s2), (3, s3)):
        raw = pd.DataFrame(
            {
                "entity_id": frame["entity_id"],
                "business_name": frame["entity_id"],
                "business_address": frame["entity_id"],
                "country": "US",
            }
        )
        raw.to_csv(dataset / "train" / f"train_source{source}.tsv", sep="\t", index=False)

    return dataset


def _config(root: Path, dataset: Path):
    return Config(
        root=root,
        dataset_dir=dataset,
        data_dir=root,
        models_dir=root / "models",
        output_dir=root / "output",
        seed=42,
    )


def test_parallel_features_match_single_process(tmp_path):
    dataset_a = _make_dataset(tmp_path / "a")
    dataset_b = _make_dataset(tmp_path / "b")
    cfg_single = _config(tmp_path / "a", dataset_a)
    cfg_parallel = _config(tmp_path / "b", dataset_b)

    single = run_features(cfg_single, "train", workers=1, combine=True)
    parallel = run_features(cfg_parallel, "train", workers=2, combine=True)
    assert single["rows"] == parallel["rows"] == 4

    a = pd.read_parquet(cfg_single.data_dir / "features" / "train.parquet")
    b = pd.read_parquet(cfg_parallel.data_dir / "features" / "train.parquet")
    a = a.sort_values(["s1_id", "cand_id"]).reset_index(drop=True)
    b = b.sort_values(["s1_id", "cand_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
