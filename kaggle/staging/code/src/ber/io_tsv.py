from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def s1_sort_key(eid: str):
    return _id_sort_key(eid)


def _id_sort_key(eid: str):
    prefix, _, num = eid.partition("-")
    return (prefix, int(num) if num.isdigit() else 0)


def load_records(path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols)


def load_ground_truth(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def split_id_list(s: str) -> list:
    return [x for x in (s or "").split(",") if x]


def format_id_list(ids) -> str:
    return ",".join(sorted(set(ids), key=_id_sort_key))


def _write(path, header, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def write_matching_results(path, s1_ids, matches: dict) -> None:
    rows = [(sid, format_id_list(matches.get(sid, []))) for sid in s1_ids]
    _write(path, ["source1_entity_id", "matched_entity_ids"], rows)


def write_candidate_pairs(path, s1_ids, candidates: dict) -> None:
    rows = [(sid, format_id_list(candidates.get(sid, []))) for sid in s1_ids]
    _write(path, ["source1_entity_id", "candidate_entity_ids"], rows)
