from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.address import parse_address
from ber.io_utils import read_source
from ber.normalize import detect_script, fold_name, name_tokens, normalize_name, strip_legal_suffix
from ber.translit import romanize


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    names = df["business_name"].map(normalize_name)
    stripped = names.map(strip_legal_suffix)
    parts = [
        parse_address(a, c)
        for a, c in zip(df["business_address"].tolist(), df["country"].tolist())
    ]
    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(),
            "name_norm": names.to_numpy(),
            "name_fold": names.map(fold_name).to_numpy(),
            "name_roman": df["business_name"].map(romanize).to_numpy(),
            "name_tokens": names.map(name_tokens).to_numpy(),
            "name_idf_tokens": names.map(lambda s: sorted(set(s.split()))).to_numpy(),
            "name_script": df["business_name"].map(detect_script).to_numpy(),
            "name_stripped": [s[0] for s in stripped],
            "suffix_class": [s[1] for s in stripped],
            "addr_norm": df["business_address"].map(normalize_name).to_numpy(),
            "addr_raw_missing": df["business_address"].str.strip().eq("").to_numpy(),
            "house_no": [p.house_no for p in parts],
            "street_tokens": [list(p.street_tokens) for p in parts],
            "postal": [p.postal for p in parts],
            "state_key": [p.state_key for p in parts],
            "landmark_flag": [p.landmark_flag for p in parts],
        }
    )
    return out


def _write_stream(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    for chunk in read_source(src):
        out = prepare_frame(chunk)
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema)
        writer.write_table(table)
        rows += len(out)
    if writer is not None:
        writer.close()
    return rows


def run_prepare(cfg) -> dict:
    counts = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            src = Path(cfg.dataset_dir) / split / f"{split}_source{source}.tsv"
            dst = Path(cfg.data_dir) / "processed" / f"{split}_source{source}.parquet"
            counts[f"{split}_source{source}"] = _write_stream(src, dst)
    gt_src = Path(cfg.dataset_dir) / "train" / "train_ground_truth.tsv"
    gt = pd.read_csv(gt_src, sep="\t", dtype=str, keep_default_na=False)
    gt_dst = Path(cfg.data_dir) / "processed" / "train_ground_truth.parquet"
    gt_dst.parent.mkdir(parents=True, exist_ok=True)
    gt.to_parquet(gt_dst, index=False)
    counts["train_ground_truth"] = len(gt)
    return counts
