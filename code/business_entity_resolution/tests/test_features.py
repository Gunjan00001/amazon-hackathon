import pandas as pd

from ber.features import compute_features
from ber.prepare import prepare_frame


def _record(eid, name, addr, country):
    frame = prepare_frame(
        pd.DataFrame(
            {
                "entity_id": [eid],
                "business_name": [name],
                "business_address": [addr],
                "country": [country],
            }
        )
    )
    frame["country"] = country
    return frame


def test_compute_features_golden():
    s1 = _record("S1-1", "Best Bakery Inc", "10 Main St, Austin, TX", "US")
    cand = pd.concat(
        [
            _record("S2-1", "Best Bakery", "10 Main Street, Austin, TX", "US"),
            _record("S3-1", "Pizza Palace", "99 Oak Rd, Dallas, TX", "US"),
        ]
    )
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1"],
            "cand_id": ["S2-1", "S3-1"],
            "is_s2": [True, False],
            "pass_id": [1, 4],
            "block_score": [1.0, 0.8],
        }
    )
    cfg = type("C", (), {"seed": 42, "cap": 200, "dataset_dir": None})()
    feats = compute_features(pairs, s1, cand, cfg)
    good = feats[feats["cand_id"] == "S2-1"].iloc[0]
    bad = feats[feats["cand_id"] == "S3-1"].iloc[0]
    assert good["name_exact"] == 0.0
    assert good["name_jaccard"] > bad["name_jaccard"]
    assert good["same_country"] == 1.0
    assert good["house_match"] == 1.0


def test_attach_country_maps_all_sources(tmp_path):
    from ber.features import _COUNTRY_CACHE, attach_country

    _COUNTRY_CACHE.clear()
    dataset = tmp_path / "ds" / "train"
    dataset.mkdir(parents=True)
    rows = {1: [("S1-1", "US")], 2: [("S2-1", "India")], 3: [("S3-1", "US")]}
    for source, entries in rows.items():
        body = "".join(f"{eid}\tx\tx\t{c}\n" for eid, c in entries)
        (dataset / f"train_source{source}.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + body, encoding="utf-8"
        )
    cfg = type("C", (), {"dataset_dir": tmp_path / "ds"})()
    frame = pd.DataFrame({"entity_id": ["S1-1", "S2-1", "S3-1"]})
    out = attach_country(frame, cfg, "train")
    assert out["country"].tolist() == ["us", "india", "us"]
