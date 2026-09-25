import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .text import fold_ascii, normalize_address, normalize_name, phonetic_key, postal_key

CLASSICAL_FEATURES = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaro",
    "name_exact", "name_jaccard", "name_len_diff",
    "addr_ratio", "addr_token_sort", "addr_jaccard", "addr_missing", "addr_len_diff",
    "same_country", "is_s2",
]

STRUCTURAL_FEATURES = [
    "postal_match", "phonetic_match", "addr_token_overlap", "name_token_overlap", "name_first_token_match",
]


def _cpdist(a, b, scorer):
    a = ["" if x is None else str(x) for x in a]
    b = ["" if x is None else str(x) for x in b]
    if not a:
        return np.zeros(0, dtype="float32")
    return np.asarray(process.cpdist(a, b, scorer=scorer, workers=-1), dtype="float32")


def _jaccard_sets(a, b):
    out = np.zeros(len(a), dtype="float32")
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(str(x).split()), set(str(y).split())
        union = len(sx | sy)
        out[i] = len(sx & sy) / union if union else 0.0
    return out


def _lens(arr):
    return np.fromiter((len(str(x)) for x in arr), dtype="float32", count=len(arr))


def classical_features(frame) -> np.ndarray:
    n1 = frame["n1"].astype(str).to_numpy(dtype=object)
    n2 = frame["n2"].astype(str).to_numpy(dtype=object)
    a1 = frame["a1"].astype(str).to_numpy(dtype=object)
    a2 = frame["a2"].astype(str).to_numpy(dtype=object)
    n1f = frame["n1_fold"].to_numpy(dtype=object) if "n1_fold" in frame.columns else np.array([fold_ascii(x) for x in n1])
    n2f = frame["n2_fold"].to_numpy(dtype=object) if "n2_fold" in frame.columns else np.array([fold_ascii(x) for x in n2])
    a1f = frame["a1_fold"].to_numpy(dtype=object) if "a1_fold" in frame.columns else np.array([fold_ascii(x) for x in a1])
    a2f = frame["a2_fold"].to_numpy(dtype=object) if "a2_fold" in frame.columns else np.array([fold_ascii(x) for x in a2])

    addr_missing = ((a1 == "") | (a2 == "")).astype("float32")
    addr_ratio = _cpdist(a1f, a2f, fuzz.ratio)
    addr_token_sort = _cpdist(a1f, a2f, fuzz.token_sort_ratio)
    addr_jaccard = _jaccard_sets(a1, a2)
    addr_ratio = np.where(addr_missing > 0, -1.0, addr_ratio).astype("float32")
    addr_token_sort = np.where(addr_missing > 0, -1.0, addr_token_sort).astype("float32")
    addr_jaccard = np.where(addr_missing > 0, -1.0, addr_jaccard).astype("float32")

    cols = {
        "name_ratio": _cpdist(n1f, n2f, fuzz.ratio),
        "name_partial": _cpdist(n1f, n2f, fuzz.partial_ratio),
        "name_token_sort": _cpdist(n1f, n2f, fuzz.token_sort_ratio),
        "name_token_set": _cpdist(n1f, n2f, fuzz.token_set_ratio),
        "name_jaro": _cpdist(n1f, n2f, JaroWinkler.similarity),
        "name_exact": ((n1 == n2) & (n1 != "")).astype("float32"),
        "name_jaccard": _jaccard_sets(n1, n2),
        "name_len_diff": np.abs(_lens(n1) - _lens(n2)),
        "addr_ratio": addr_ratio,
        "addr_token_sort": addr_token_sort,
        "addr_jaccard": addr_jaccard,
        "addr_missing": addr_missing,
        "addr_len_diff": np.abs(_lens(a1) - _lens(a2)),
        "same_country": (frame["c1"].astype(str) == frame["c2"].astype(str)).to_numpy().astype("float32"),
        "is_s2": frame["is_s2"].to_numpy().astype("float32"),
    }
    return np.column_stack([cols[c] for c in CLASSICAL_FEATURES]).astype("float32")


def prepare_records(df: pd.DataFrame) -> dict:
    name = df["business_name"].map(normalize_name).to_numpy(dtype=object)
    addr = df["business_address"].map(normalize_address).to_numpy(dtype=object)
    return {
        "name": name,
        "fold_name": np.array([fold_ascii(x) for x in name], dtype=object),
        "addr": addr,
        "fold_addr": np.array([fold_ascii(x) for x in addr], dtype=object),
        "country": df["country"].str.strip().str.lower().to_numpy(dtype=object),
        "postal": np.array([postal_key(x) or "" for x in addr], dtype=object),
        "is_s2": df["entity_id"].str.startswith("S2-").to_numpy(),
    }


def build_pair_frame_prepared(ps1: dict, pmid: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    i = pairs["s1_idx"].to_numpy()
    j = pairs["mid_idx"].to_numpy()
    return pd.DataFrame({
        "s1_idx": i, "mid_idx": j, "pass": pairs["pass"].to_numpy(),
        "n1": ps1["name"][i], "n1_fold": ps1["fold_name"][i],
        "n2": pmid["name"][j], "n2_fold": pmid["fold_name"][j],
        "a1": ps1["addr"][i], "a1_fold": ps1["fold_addr"][i],
        "a2": pmid["addr"][j], "a2_fold": pmid["fold_addr"][j],
        "c1": ps1["country"][i], "c2": pmid["country"][j],
        "p1": ps1["postal"][i], "p2": pmid["postal"][j],
        "is_s2": pmid["is_s2"][j],
    })


def build_pair_frame(s1_df: pd.DataFrame, mid_df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    return build_pair_frame_prepared(prepare_records(s1_df), prepare_records(mid_df), pairs)


def structural_features(frame) -> np.ndarray:
    n1f = frame["n1_fold"].to_numpy(dtype=object) if "n1_fold" in frame.columns else frame["n1"].astype(str).to_numpy(dtype=object)
    n2f = frame["n2_fold"].to_numpy(dtype=object) if "n2_fold" in frame.columns else frame["n2"].astype(str).to_numpy(dtype=object)
    a1 = frame["a1"].astype(str).to_numpy(dtype=object)
    a2 = frame["a2"].astype(str).to_numpy(dtype=object)
    if "p1" in frame.columns:
        p1, p2 = frame["p1"].astype(str).to_numpy(dtype=object), frame["p2"].astype(str).to_numpy(dtype=object)
    else:
        p1 = np.array([postal_key(x) or "" for x in a1], dtype=object)
        p2 = np.array([postal_key(x) or "" for x in a2], dtype=object)
    f1 = np.array([phonetic_key(str(x).split()[0]) if str(x).split() else "" for x in n1f], dtype=object)
    f2 = np.array([phonetic_key(str(x).split()[0]) if str(x).split() else "" for x in n2f], dtype=object)
    t1 = np.array([" ".join(t for t in str(x).split() if not t.isdigit()) for x in a1], dtype=object)
    t2 = np.array([" ".join(t for t in str(x).split() if not t.isdigit()) for x in a2], dtype=object)
    first1 = np.array([str(x).split()[0] if str(x).split() else "" for x in n1f], dtype=object)
    first2 = np.array([str(x).split()[0] if str(x).split() else "" for x in n2f], dtype=object)

    cols = {
        "postal_match": ((p1 == p2) & (p1 != "")).astype("float32"),
        "phonetic_match": ((f1 == f2) & (f1 != "")).astype("float32"),
        "addr_token_overlap": _jaccard_sets(t1, t2),
        "name_token_overlap": _jaccard_sets(n1f, n2f),
        "name_first_token_match": ((first1 == first2) & (first1 != "")).astype("float32"),
    }
    return np.column_stack([cols[c] for c in STRUCTURAL_FEATURES]).astype("float32")
