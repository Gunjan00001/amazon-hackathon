from array import array
from collections import defaultdict, namedtuple
import heapq
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .text import fold_ascii, normalize_address, normalize_name, phonetic_key, postal_key

STOPWORDS = {"private", "limited", "company", "corporation", "incorporated", "and", "the", "of", "llc", "llp"}

PASS_NAMES = ("postal", "rare_token", "phonetic", "addr_token")

RowKeys = namedtuple("RowKeys", "country tokens phonetics postal addr_tokens")


def _significant(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]


def _name_tokens(name):
    return _significant(fold_ascii(normalize_name(name)).split())


def _addr_tokens(address):
    norm = normalize_address(address)
    return [fold_ascii(t) for t in norm.split() if not t.isdigit()]


def _row_keys(name, address, country):
    tokens = _name_tokens(name)
    phonetics = [p for p in (phonetic_key(t) for t in tokens) if p]
    return RowKeys(
        (country or "").strip().lower(),
        tokens,
        phonetics,
        postal_key(address),
        _addr_tokens(address),
    )


def prepare_keys(df):
    return [
        _row_keys(r.business_name, r.business_address, r.country)
        for r in df.itertuples(index=False)
    ]


_SEP = "\x1f"


def save_keys(keys, path):
    df = pd.DataFrame({
        "country": [k.country for k in keys],
        "tokens": [_SEP.join(k.tokens) for k in keys],
        "phonetics": [_SEP.join(k.phonetics) for k in keys],
        "postal": [k.postal or "" for k in keys],
        "addr_tokens": [_SEP.join(k.addr_tokens) for k in keys],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_keys(path):
    df = pd.read_parquet(path)
    out = []
    for r in df.itertuples(index=False):
        out.append(RowKeys(
            r.country,
            r.tokens.split(_SEP) if r.tokens else [],
            r.phonetics.split(_SEP) if r.phonetics else [],
            r.postal or None,
            r.addr_tokens.split(_SEP) if r.addr_tokens else [],
        ))
    return out


def s1_blocking_tokens(row):
    return _row_keys(row["business_name"], row["business_address"], row["country"])


def mid_blocking_tokens(row):
    return _row_keys(row["business_name"], row["business_address"], row["country"])


def blocking_recall(pairs, labels, offset=10 ** 9):
    if not labels:
        return 0.0
    a = pairs["s1_idx"].to_numpy(dtype="int64")
    b = pairs["mid_idx"].to_numpy(dtype="int64")
    lk = np.fromiter((int(i) * offset + int(j) for i, j in labels), dtype="int64", count=len(labels))
    hits = int(np.isin(a * offset + b, lk).sum())
    return hits / len(labels)


def _idf(count, total):
    return math.log((total + 1.0) / (count + 1.0)) + 1.0


def build_index(mid_keys, max_postings=20000):
    total = max(1, len(mid_keys))
    token_df = defaultdict(int)
    addr_df = defaultdict(int)
    phonetic_df = defaultdict(int)
    for k in mid_keys:
        c = k.country
        for t in k.tokens:
            token_df[(c, t)] += 1
        for t in k.addr_tokens:
            addr_df[(c, t)] += 1
        for p in k.phonetics:
            phonetic_df[(c, p)] += 1

    postal_index = defaultdict(list)
    phonetic_index = defaultdict(list)
    token_index = defaultdict(list)
    addr_index = defaultdict(list)
    for i, k in enumerate(mid_keys):
        c = k.country
        if k.postal:
            postal_index[(c, k.postal)].append(i)
        for p in k.phonetics:
            if phonetic_df[(c, p)] <= max_postings:
                phonetic_index[(c, p)].append(i)
        for t in k.tokens:
            if token_df[(c, t)] <= max_postings:
                token_index[(c, t)].append(i)
        for t in k.addr_tokens:
            if addr_df[(c, t)] <= max_postings:
                addr_index[(c, t)].append(i)
    return {
        "total": total,
        "token_df": token_df,
        "addr_df": addr_df,
        "phonetic_df": phonetic_df,
        "postal": postal_index,
        "phonetic": phonetic_index,
        "token": token_index,
        "addr": addr_index,
    }


def score_keys(s1_keys, index, max_candidates_per_s1=200):
    total = index["total"]
    token_df, addr_df, phonetic_df = index["token_df"], index["addr_df"], index["phonetic_df"]
    postal_index, phonetic_index = index["postal"], index["phonetic"]
    token_index, addr_index = index["token"], index["addr"]

    rows_i = array("i")
    rows_j = array("i")
    rows_p = array("b")
    for i, k in enumerate(s1_keys):
        c = k.country
        score = {}
        pcode = {}
        if k.postal:
            for j in postal_index.get((c, k.postal), ()):
                score[j] = score.get(j, 0.0) + 10.0
                pcode.setdefault(j, 0)
        for t in k.tokens:
            w = _idf(token_df[(c, t)], total)
            for j in token_index.get((c, t), ()):
                score[j] = score.get(j, 0.0) + w
                pcode.setdefault(j, 1)
        for p in k.phonetics:
            w = 0.5 * _idf(phonetic_df[(c, p)], total)
            for j in phonetic_index.get((c, p), ()):
                score[j] = score.get(j, 0.0) + w
                pcode.setdefault(j, 2)
        addr_sorted = sorted({t for t in k.addr_tokens if addr_df[(c, t)]}, key=lambda t: addr_df[(c, t)])
        for t in addr_sorted[:2]:
            w = _idf(addr_df[(c, t)], total)
            for j in addr_index.get((c, t), ()):
                score[j] = score.get(j, 0.0) + w
                pcode.setdefault(j, 3)
        if len(score) > max_candidates_per_s1:
            top = heapq.nlargest(max_candidates_per_s1, score.items(), key=lambda kv: (kv[1], -kv[0]))
        else:
            top = score.items()
        for mid, _s in top:
            rows_i.append(i)
            rows_j.append(mid)
            rows_p.append(pcode[mid])

    pairs = pd.DataFrame({
        "s1_idx": np.array(rows_i, dtype="int32"),
        "mid_idx": np.array(rows_j, dtype="int32"),
        "pass": pd.Categorical.from_codes(np.array(rows_p, dtype="int8"), categories=list(PASS_NAMES)),
    })
    return pairs


def generate_candidates_from_keys(s1_keys, mid_keys, max_candidates_per_s1=200, max_postings=20000):
    return score_keys(s1_keys, build_index(mid_keys, max_postings), max_candidates_per_s1)


def generate_candidates(s1_df, mid_df, max_candidates_per_s1=200, max_postings=20000):
    return generate_candidates_from_keys(
        prepare_keys(s1_df), prepare_keys(mid_df), max_candidates_per_s1, max_postings
    )
