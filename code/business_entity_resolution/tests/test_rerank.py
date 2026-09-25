import numpy as np
from ber.rerank import score_texts, serialize_pair


def test_serialize_pair_handles_empty_address_and_accents():
    s = serialize_pair("École Sainte", "", "France", "Ecole Sainte", "1 Rue A", "France")
    assert s == "École Sainte | none | france [SEP] Ecole Sainte | 1 Rue A | france"


def test_score_texts_uses_injected_fn():
    out = score_texts(["a", "b"], "any", "rev", 2, fn=lambda texts: np.array([0.1, 0.9], dtype="float32"))
    assert np.allclose(out, [0.1, 0.9], atol=1e-6)
