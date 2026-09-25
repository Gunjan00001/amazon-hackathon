import argparse
import json
from pathlib import Path

from .. import config
from ..io_tsv import load_ground_truth, load_records
from ..text import fold_ascii, normalize_address, normalize_name, postal_key


def resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if (data_dir / "train" / "train_source1.tsv").exists():
        return data_dir
    hits = sorted(data_dir.glob("*/train/train_source1.tsv"))
    if hits:
        return hits[0].parent.parent
    raise FileNotFoundError(f"no train/train_source1.tsv under {data_dir}")


def enrich(df, sample=0):
    if sample:
        df = df.head(sample).copy()
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_addr"] = df["business_address"].map(normalize_address)
    df["fold_name"] = df["norm_name"].map(fold_ascii)
    df["fold_addr"] = df["norm_addr"].map(fold_ascii)
    df["postal"] = df["business_address"].map(postal_key)
    df["country_n"] = df["country"].str.strip().str.lower()
    return df


def stats(df):
    return {
        "rows": int(len(df)),
        "empty_addr": int((df["business_address"] == "").sum()),
        "non_ascii_name": int((~df["business_name"].map(str.isascii)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {"data_dir": str(data_dir)}
    for split in ("train", "test"):
        for n in (1, 2, 3):
            path = data_dir / split / f"{split}_source{n}.tsv"
            df = enrich(load_records(path), args.sample)
            df.to_parquet(out_dir / f"{split}_s{n}.parquet", index=False)
            all_stats[f"{split}_s{n}"] = stats(df)

    gt = load_ground_truth(data_dir / "train" / "train_ground_truth.tsv")
    if args.sample:
        gt = gt.head(args.sample)
    gt.to_parquet(out_dir / "labels.parquet", index=False)
    all_stats["labels"] = {"rows": int(len(gt))}

    (out_dir / "stats.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
