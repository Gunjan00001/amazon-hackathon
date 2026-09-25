import pandas as pd

from ber.blocking import generate_candidates


def _frame(ids, names, addresses, countries):
    return pd.DataFrame(
        {
            "entity_id": ids,
            "name_norm": names,
            "name_tokens": [n.split() for n in names],
            "name_script": ["latin"] * len(ids),
            "house_no": [a.split()[0] for a in addresses],
            "street_tokens": [a.split()[1:] for a in addresses],
            "postal": [""] * len(ids),
            "state_key": [""] * len(ids),
        }
    )


def test_generate_candidates_finds_exact_and_near():
    s1 = _frame(["S1-1"], ["best bakery"], ["10 main st"], ["US"])
    s2 = _frame(["S2-1"], ["best bakery"], ["10 main street"], ["US"])
    s3 = _frame(["S3-1"], ["best b akery"], ["11 side road"], ["US"])
    cfg = type("C", (), {"cap": 200, "max_block": 5000, "idf_min": 4.0, "seed": 42, "rare": None})()
    out = generate_candidates(s1, pd.concat([s2, s3]), cfg)
    assert set(out["cand_id"]) >= {"S2-1"}
    assert out.loc[out["cand_id"] == "S2-1", "pass_id"].min() == 1


def test_cap_is_enforced():
    s1 = _frame(["S1-1"], ["common name"], ["1 a st"], ["US"])
    cands = _frame(
        [f"S2-{i}" for i in range(300)],
        ["common name"] * 300,
        ["1 a st"] * 300,
        ["US"] * 300,
    )
    cfg = type("C", (), {"cap": 50, "max_block": 5000, "idf_min": 4.0, "seed": 42, "rare": None})()
    out = generate_candidates(s1, cands, cfg)
    assert out.groupby("s1_id").size().max() <= 50
