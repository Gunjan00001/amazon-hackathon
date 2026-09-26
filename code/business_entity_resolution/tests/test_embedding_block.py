import numpy as np
import pandas as pd
import pytest

from ber.embedding_block import (
    AnnIndex,
    build_ann_index,
    channel_attribution,
    l2_normalize,
    retrieve,
    union_with_lexical,
)


def _toy():
    emb = np.array(
        [
            [1.0, 0.0, 0.0],   # c0
            [0.0, 1.0, 0.0],   # c1
            [1.0, 1.0, 0.0],   # c2
            [-1.0, 0.0, 0.0],  # c3
        ],
        dtype=np.float32,
    )
    ids = np.array(["c0", "c1", "c2", "c3"], dtype=object)
    return emb, ids


def test_l2_normalize_rows_unit_norm():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(np.linalg.norm(out[1]), 0.0)


def test_flat_search_orders_by_cosine():
    emb, ids = _toy()
    index = AnnIndex(emb, ids, kind="flat")
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores, neighbors = index.search(query, k=3)
    assert list(neighbors[0]) == ["c0", "c2", "c1"]
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)
    assert scores[0][1] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-5)


def test_retrieve_frame_columns_and_channel():
    emb, ids = _toy()
    index = build_ann_index(emb, ids, kind="flat")
    frame = retrieve(index, emb, ["S1-a", "S1-b", "S1-c", "S1-d"], k=2, channel="entity")
    assert list(frame.columns) == ["s1_id", "cand_id", "score", "rank", "channel"]
    assert (frame["channel"] == "entity").all()
    assert len(frame) == 8
    top = frame[frame["s1_id"] == "S1-a"].sort_values("rank").iloc[0]
    assert top["cand_id"] == "c0"


def test_union_prefers_lexical_and_dedups():
    ann = pd.DataFrame(
        {
            "s1_id": ["a", "a", "a"],
            "cand_id": ["x", "y", "z"],
            "score": [0.9, 0.8, 0.7],
            "channel": ["name", "name", "addr"],
        }
    )
    lex = pd.DataFrame(
        {
            "s1_id": ["a", "a"],
            "cand_id": ["y", "w"],
            "pass_id": [3, 4],
            "block_score": [0.6, 0.7],
        }
    )
    out = union_with_lexical(ann, lex)
    assert not out.duplicated(["s1_id", "cand_id"]).any()
    assert set(out["cand_id"]) == {"w", "x", "y", "z"}
    y = out[out["cand_id"] == "y"].iloc[0]
    assert int(y["pass_id"]) == 3            # lexical wins over ANN pass 11
    addr = out[out["cand_id"] == "z"].iloc[0]
    assert int(addr["pass_id"]) == 12        # addr channel -> pass 12
    assert bool(addr["is_s2"]) is False


def test_union_without_lexical_assigns_channel_passes():
    ann = pd.DataFrame(
        {"s1_id": ["a", "a"], "cand_id": ["S2-1", "S3-2"], "score": [0.95, 0.4],
         "channel": ["entity", "entity"]}
    )
    out = union_with_lexical(ann, None)
    assert (out["pass_id"] == 13).all()
    assert out["is_s2"].tolist() == [True, False]


def test_channel_attribution_recovery_counts():
    truth = pd.DataFrame({"s1_id": ["a", "a", "a"], "cand_id": ["t1", "t2", "t3"]})
    lexical = pd.DataFrame({"s1_id": ["a"], "cand_id": ["t1"]})
    channels = {
        "name": pd.DataFrame({"s1_id": ["a"], "cand_id": ["t2"]}),
        "addr": pd.DataFrame({"s1_id": ["a", "a"], "cand_id": ["t2", "t3"]}),
    }
    report = channel_attribution(truth, lexical, channels)
    assert report["truth_pairs"] == 3
    assert report["lexical_found"] == 1
    assert report["lexical_missed"] == 2
    assert report["per_channel"] == {"name": 1, "addr": 2}
    assert report["any_channel"] == 2
    assert report["multiple_channels"] == 1
    assert report["none"] == 0


def test_channel_attribution_none_when_no_hits():
    truth = pd.DataFrame({"s1_id": ["a"], "cand_id": ["t9"]})
    report = channel_attribution(truth, None, {"name": pd.DataFrame(columns=["s1_id", "cand_id"])})
    assert report["lexical_missed"] == 1
    assert report["any_channel"] == 0
    assert report["none"] == 1
