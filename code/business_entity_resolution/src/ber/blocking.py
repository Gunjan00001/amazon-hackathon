from array import array
from collections import defaultdict

import numpy as np
import pandas as pd

from .text import address_tokens, fold_ascii, normalize_name, phonetic_key, postal_key

STOPWORDS = {"private", "limited", "company", "corporation", "incorporated", "and", "the", "of", "llc", "llp"}

PASS_NAMES = ("postal", "rare_token", "phonetic", "addr_token")


def _significant(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]


def _row_keys(name, address, country):
    norm_name = normalize_name(name)
    tokens = _significant(norm_name.split())
    if tokens:
        first = fold_ascii(tokens[0])
    else:
        parts = fold_ascii(norm_name).split()
        first = parts[0] if parts else ""
    return {
        "country": country.strip().lower(),
        "tokens": tokens,
        "phonetic": phonetic_key(first) if first else "",
        "postal": postal_key(address),
        "addr_tokens": address_tokens(address),
    }


def s1_blocking_tokens(row):
    return _row_keys(row["business_name"], row["business_address"], row["country"])


def mid_blocking_tokens(row):
    return _row_keys(row["business_name"], row["business_address"], row["country"])


def blocking_recall(pairs, labels):
    got = {(int(a), int(b)) for a, b in zip(pairs["s1_idx"], pairs["mid_idx"])}
    return len(got & labels) / max(1, len(labels))


def _counts(keys, field):
    counts = defaultdict(int)
    for k in keys:
        for t in k[field]:
            counts[(k["country"], t)] += 1
    return counts


def _iter_keys(df):
    for r in df.itertuples(index=False):
        yield _row_keys(r.business_name, r.business_address, r.country)


def generate_candidates(s1_df, mid_df, max_candidates_per_s1=200, max_postings=20000):
    token_df = defaultdict(int)
    addr_df = defaultdict(int)
    for k in _iter_keys(mid_df):
        c = k["country"]
        for t in k["tokens"]:
            token_df[(c, t)] += 1
        for t in k["addr_tokens"]:
            addr_df[(c, t)] += 1

    postal_index = defaultdict(list)
    phonetic_index = defaultdict(list)
    token_index = defaultdict(list)
    addr_index = defaultdict(list)
    for i, k in enumerate(_iter_keys(mid_df)):
        c = k["country"]
        if k["postal"]:
            postal_index[(c, k["postal"])].append(i)
        if k["phonetic"]:
            phonetic_index[(c, k["phonetic"])].append(i)
        for t in k["tokens"]:
            if token_df[(c, t)] <= max_postings:
                token_index[(c, t)].append(i)
        for t in k["addr_tokens"]:
            if addr_df[(c, t)] <= max_postings:
                addr_index[(c, t)].append(i)

    rows_i = array("i")
    rows_j = array("i")
    rows_p = array("b")
    for i, k in enumerate(_iter_keys(s1_df)):
        c = k["country"]
        seen = set()
        budget = max_candidates_per_s1
        addr_sorted = sorted({t for t in k["addr_tokens"] if addr_df[(c, t)]}, key=lambda t: addr_df[(c, t)])
        passes = (
            (0, [postal_index.get((c, k["postal"]))] if k["postal"] else []),
            (1, [token_index.get((c, t)) for t in k["tokens"]]),
            (2, [phonetic_index.get((c, k["phonetic"]))] if k["phonetic"] else []),
            (3, [addr_index.get((c, t)) for t in addr_sorted[:1]]),
        )
        for pass_code, posting_lists in passes:
            done = False
            for postings in posting_lists:
                if not postings:
                    continue
                for j in postings:
                    if j in seen:
                        continue
                    seen.add(j)
                    rows_i.append(i)
                    rows_j.append(j)
                    rows_p.append(pass_code)
                    budget -= 1
                    if budget <= 0:
                        done = True
                        break
                if done:
                    break
            if done:
                break

    pairs = pd.DataFrame({
        "s1_idx": np.array(rows_i, dtype="int32"),
        "mid_idx": np.array(rows_j, dtype="int32"),
        "pass": pd.Categorical.from_codes(np.array(rows_p, dtype="int8"), categories=list(PASS_NAMES)),
    })
    return pairs
