"""Drive the Kaggle pipeline from the CLI (datasets + kernel chain).

Requires ~/.kaggle/kaggle.json: Kaggle -> Settings -> API -> Create New Token.

Usage:
    python kaggle/push_all.py datasets
    python kaggle/push_all.py push --wait
    python kaggle/push_all.py push --only N1 N2
    python kaggle/push_all.py status
    python kaggle/push_all.py fetch --collect
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "gunjanpal001"
STAGE = REPO / "kaggle" / "staging"
NBS = REPO / "kaggle" / "notebooks"
CODE_SRC = REPO / "code" / "business_entity_resolution" / "src"
DEFAULT_RAW = Path(r"D:\Amazon project\DATA\student_resource\dataset")
OUTPUT_DIR = REPO / "output"

RAW_ID = f"{USER}/amz-er-2026-raw"
CODE_ID = f"{USER}/amz-er-2026-code"
WHEELS_ID = f"{USER}/amz-er-2026-wheels"

KERNELS = [
    {"nb": "N1", "file": "N1_clean", "slug": "amz-er-n1-clean", "gpu": False},
    {"nb": "N2", "file": "N2_block", "slug": "amz-er-n2-block", "gpu": False},
    {"nb": "N3", "file": "N3_embed", "slug": "amz-er-n3-embed", "gpu": True},
    {"nb": "N4", "file": "N4_train_gbdt", "slug": "amz-er-n4-train-gbdt", "gpu": False},
    {"nb": "N5", "file": "N5_rerank", "slug": "amz-er-n5-rerank", "gpu": True},
    {"nb": "N6", "file": "N6_decide", "slug": "amz-er-n6-decide", "gpu": False},
]

PLANS = {
    "v1": [
        ("N1", []),
        ("N2", ["amz-er-n1-clean"]),
        ("N4", ["amz-er-n1-clean", "amz-er-n2-block"]),
        ("N6", ["amz-er-n2-block", "amz-er-n4-train-gbdt"]),
    ],
    "full": [
        ("N1", []),
        ("N2", ["amz-er-n1-clean"]),
        ("N3", ["amz-er-n1-clean"]),
        ("N4", ["amz-er-n1-clean", "amz-er-n2-block", "amz-er-n3-embed"]),
        ("N5", ["amz-er-n4-train-gbdt"]),
        ("N6", ["amz-er-n2-block", "amz-er-n4-train-gbdt", "amz-er-n5-rerank"]),
    ],
}


def run(cmd, check=True, quiet=False):
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if not quiet:
        print(r.stdout.strip())
        if r.stderr.strip():
            print(r.stderr.strip())
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r


def dataset_exists(ds_id):
    r = subprocess.run(["kaggle", "datasets", "files", ds_id], capture_output=True, text=True)
    return r.returncode == 0


def sync_dataset(ds_id, title, folder, message):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / "dataset-metadata.json"
    wrote_meta = not meta.exists()
    if wrote_meta:
        meta.write_text(json.dumps({
            "title": title, "id": ds_id,
            "licenses": [{"name": "unknown"}], "isPrivate": True,
        }, indent=2), encoding="utf-8")
    try:
        if dataset_exists(ds_id):
            print(f"updating dataset {ds_id}")
            run(["kaggle", "datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "zip"])
        else:
            run(["kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "zip"])
            print(f"created dataset {ds_id}")
    finally:
        if wrote_meta:
            meta.unlink(missing_ok=True)


def stage_raw(raw_dir):
    src = Path(raw_dir)
    dst = STAGE / "raw"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for split in ("train", "test"):
        (dst / split).mkdir()
        for f in sorted((src / split).glob("*.tsv")):
            try:
                os.link(f, dst / split / f.name)
            except OSError:
                shutil.copy2(f, dst / split / f.name)
    return dst


def stage_wheels():
    dst = STAGE / "wheels"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    run(["python", "-m", "pip", "download", "anyascii", "jellyfish", "rapidfuzz",
         "--only-binary=:all:", "--python-version", "3.12", "--platform", "manylinux2014_x86_64",
         "--implementation", "cp", "--abi", "cp312", "-d", str(dst), "--no-deps"])
    return dst


def cmd_datasets(args):
    only = set(args.only) if args.only else {"raw", "code", "wheels"}
    if "raw" in only:
        sync_dataset(RAW_ID, "amz-er-2026-raw", stage_raw(args.raw_dir), "raw challenge TSVs")
    if "code" in only:
        code_dir = STAGE / "code"
        if (code_dir / "src").exists():
            shutil.rmtree(code_dir / "src")
        code_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CODE_SRC, code_dir / "src")
        sync_dataset(CODE_ID, "amz-er-2026-code", code_dir, "pipeline source")
    if "wheels" in only:
        sync_dataset(WHEELS_ID, "amz-er-2026-wheels", stage_wheels(), "offline dependency wheels")


def kernel_dir(spec, needs):
    kdir = STAGE / "kernels" / spec["slug"]
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NBS / f"{spec['file']}.ipynb", kdir / f"{spec['file']}.ipynb")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{spec['slug']}",
        "title": spec["slug"],
        "code_file": f"{spec['file']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": spec["gpu"],
        "enable_internet": True,
        "enable_tpu": False,
        "dataset_sources": [RAW_ID, CODE_ID, WHEELS_ID],
        "kernel_sources": [f"{USER}/{s}" for s in needs],
        "competition_sources": [],
    }, indent=2), encoding="utf-8")
    return kdir


def kernel_status(slug):
    r = run(["kaggle", "kernels", "status", f"{USER}/{slug}"], check=False, quiet=True)
    return (r.stdout + r.stderr).strip()


def wait_kernel(slug, timeout_s=21600, poll_s=60):
    start = time.time()
    while time.time() - start < timeout_s:
        status = kernel_status(slug)
        print(f"[{slug}] {status}", flush=True)
        low = status.lower()
        if "complete" in low:
            return True
        if "error" in low or "cancel" in low:
            return False
        time.sleep(poll_s)
    return False


def cmd_push(args):
    by_nb = {s["nb"]: s for s in KERNELS}
    only = set(args.only) if args.only else None
    for nb, needs in PLANS[args.plan]:
        if only and nb not in only:
            continue
        spec = by_nb[nb]
        run(["kaggle", "kernels", "push", "-p", str(kernel_dir(spec, needs))])
        if args.wait:
            ok = wait_kernel(spec["slug"])
            if not ok:
                print(f"kernel {spec['slug']} did not complete; stopping")
                raise SystemExit(1)


def cmd_status(args):
    for spec in KERNELS:
        print(spec["nb"], kernel_status(spec["slug"]))


def cmd_fetch(args):
    for spec in KERNELS:
        out = STAGE / "outputs" / spec["slug"]
        out.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "kernels", "output", f"{USER}/{spec['slug']}", "-p", str(out)], check=False)
    if args.collect:
        found = list((STAGE / "outputs").glob("**/matching_results.tsv"))
        if not found:
            print("no matching_results.tsv found in fetched outputs")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            hits = sorted((STAGE / "outputs").glob(f"**/{name}"))
            if hits:
                shutil.copy2(hits[-1], OUTPUT_DIR / name)
                print("collected", hits[-1], "->", OUTPUT_DIR / name)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("datasets")
    d.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    d.add_argument("--only", nargs="*", choices=["raw", "code", "wheels"])
    d.set_defaults(func=cmd_datasets)
    p = sub.add_parser("push")
    p.add_argument("--plan", choices=["v1", "full"], default="full")
    p.add_argument("--only", nargs="*")
    p.add_argument("--wait", action="store_true")
    p.set_defaults(func=cmd_push)
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    f = sub.add_parser("fetch")
    f.add_argument("--collect", action="store_true")
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
