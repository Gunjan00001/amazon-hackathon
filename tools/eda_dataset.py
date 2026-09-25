import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\Amazon project\DATA\student_resource\dataset")
OUT = Path(r"D:\Amazon project\tools\eda_stats.json")
CACHE = Path(r"D:\Amazon project\tools\eda_cache")
CHUNK = 1_000_000
SAMPLE = 200_000
QMARK_RE = r"\?{2,}"
NONASCII_RE = r"[^\x00-\x7F]"
LEN_BUCKETS = [0, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 10**9]


def hist(values, buckets):
    if len(values) == 0:
        return {}
    counts, edges = np.histogram(values, bins=buckets)
    return {f"{int(edges[i])}-{int(edges[i+1]) if edges[i+1] < 10**9 else 'inf'}": int(counts[i]) for i in range(len(counts))}


def pct(sample, q):
    return None if len(sample) == 0 else float(np.percentile(sample, q))


def scan_source(path, name):
    print(f"scanning {name} ...", flush=True)
    stats = {
        "file": name,
        "rows": 0,
        "dup_entity_ids": 0,
        "country": {},
        "empty_name": 0,
        "empty_address": 0,
        "empty_country": 0,
        "qmark_name": 0,
        "qmark_address": 0,
        "nonascii_name": 0,
        "nonascii_address": 0,
        "addr_starts_digit": 0,
        "name_len_sum": 0,
        "addr_len_sum": 0,
    }
    seen = set()
    name_sample, addr_sample = [], []
    examples = {"qmark_name": [], "nonascii_name": [], "empty_address": []}
    columns = None
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK, keep_default_na=False):
        if columns is None:
            columns = list(chunk.columns)
        stats["rows"] += len(chunk)
        ids = chunk["entity_id"]
        before = len(seen)
        seen.update(ids.tolist())
        stats["dup_entity_ids"] += len(ids) - (len(seen) - before)

        vc = chunk["country"].value_counts()
        for k, v in vc.items():
            stats["country"][k] = stats["country"].get(k, 0) + int(v)

        nm, ad = chunk["business_name"], chunk["business_address"]
        stats["empty_name"] += int((nm == "").sum())
        stats["empty_address"] += int((ad == "").sum())
        stats["empty_country"] += int((chunk["country"] == "").sum())

        qm = nm.str.contains(QMARK_RE, regex=True, na=False)
        qa = ad.str.contains(QMARK_RE, regex=True, na=False)
        stats["qmark_name"] += int(qm.sum())
        stats["qmark_address"] += int(qa.sum())
        stats["nonascii_name"] += int(nm.str.contains(NONASCII_RE, regex=True, na=False).sum())
        stats["nonascii_address"] += int(ad.str.contains(NONASCII_RE, regex=True, na=False).sum())
        stats["addr_starts_digit"] += int(ad.str.match(r"^\d", na=False).sum())

        nl, al = nm.str.len(), ad.str.len()
        stats["name_len_sum"] += int(nl.sum())
        stats["addr_len_sum"] += int(al.sum())

        if len(name_sample) < SAMPLE:
            take = min(SAMPLE - len(name_sample), len(nl))
            idx = np.random.default_rng(0).choice(len(nl), size=take, replace=False)
            name_sample.extend(nl.to_numpy()[idx].tolist())
            addr_sample.extend(al.to_numpy()[idx].tolist())

        for key, mask, col in (
            ("qmark_name", qm, nm),
            ("nonascii_name", nm.str.contains(NONASCII_RE, regex=True, na=False), nm),
            ("empty_address", ad == "", ad),
        ):
            if len(examples[key]) < 5:
                hits = col[mask].head(5 - len(examples[key])).tolist()
                examples[key].extend(hits)

    stats["columns"] = columns
    stats["unique_entity_ids"] = len(seen)
    stats["name_len_mean"] = round(stats["name_len_sum"] / max(stats["rows"], 1), 2)
    stats["addr_len_mean"] = round(stats["addr_len_sum"] / max(stats["rows"], 1), 2)
    stats["name_len_p50"] = pct(name_sample, 50)
    stats["name_len_p95"] = pct(name_sample, 95)
    stats["addr_len_p50"] = pct(addr_sample, 50)
    stats["addr_len_p95"] = pct(addr_sample, 95)
    stats["examples"] = examples
    stats["country"] = dict(sorted(stats["country"].items(), key=lambda kv: -kv[1]))
    for k in ("name_len_sum", "addr_len_sum"):
        del stats[k]
    print(f"  rows={stats['rows']:,} dup_ids={stats['dup_entity_ids']:,}", flush=True)
    return stats


def scan_ground_truth(path):
    print("scanning train_ground_truth ...", flush=True)
    per_s1_counts = []
    id_counts = {}
    total_ids = 0
    sample_rows = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK, keep_default_na=False):
        for s1, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            ids = [x for x in mids.split(",") if x]
            per_s1_counts.append(len(ids))
            total_ids += len(ids)
            for i in ids:
                id_counts[i] = id_counts.get(i, 0) + 1
            if len(sample_rows) < 3 and ids:
                sample_rows.append({"source1_entity_id": s1, "matched_entity_ids": mids, "n": len(ids)})
    arr = np.array(per_s1_counts)
    dist = pd.Series(arr).value_counts().sort_index()
    multi = sum(1 for v in id_counts.values() if v > 1)
    return {
        "rows": len(per_s1_counts),
        "singletons": int((arr == 0).sum()),
        "singleton_pct": round(float((arr == 0).mean() * 100), 3),
        "total_matches": total_ids,
        "mean_matches_per_s1": round(float(arr.mean()), 3),
        "max_matches_per_s1": int(arr.max()),
        "match_count_distribution": {int(k): int(v) for k, v in dist.items()},
        "unique_matched_ids": len(id_counts),
        "matched_ids_used_by_multiple_s1": multi,
        "sample_rows": sample_rows,
        "_ids": list(id_counts.keys()),
    }


def check_membership(gt_ids):
    print("checking GT ids exist in source files ...", flush=True)
    result = {}
    gt_set = set(gt_ids)
    for name in ("train_source2.tsv", "train_source3.tsv"):
        found = set()
        for chunk in pd.read_csv(BASE / "train" / name, sep="\t", dtype=str, chunksize=CHUNK, keep_default_na=False, usecols=["entity_id"]):
            found |= gt_set.intersection(chunk["entity_id"].tolist())
            if len(found) == len(gt_set):
                break
        prefix = "S2-" if "source2" in name else "S3-"
        expected = {i for i in gt_set if i.startswith(prefix)}
        result[name] = {
            "gt_ids_for_source": len(expected),
            "missing_from_source": len(expected - found),
            "examples_missing": list(sorted(expected - found))[:5],
        }
    return result


def cached(key, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    if path.exists():
        print(f"cache hit {key}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


def main():
    stats = {"train": {}, "test": {}}
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        stats["train"][name] = cached(name, lambda name=name: scan_source(BASE / "train" / name, name))
    for name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        stats["test"][name] = cached(name, lambda name=name: scan_source(BASE / "test" / name, name))

    gt_raw = cached("ground_truth_raw", lambda: scan_ground_truth(BASE / "train" / "train_ground_truth.tsv"))
    ids = gt_raw.pop("_ids", [])
    gt = dict(gt_raw)
    gt["membership"] = cached("gt_membership", lambda: check_membership(ids))
    stats["ground_truth"] = gt

    OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
