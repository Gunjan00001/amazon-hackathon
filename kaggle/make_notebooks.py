import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks"

CODE_PATH = "/kaggle/input/amz-er-2026-code/src"
RAW_PATH = "/kaggle/input/amz-er-2026-raw"
ARTIFACT_PATH = "/kaggle/working/artifacts"

STAGES = [
    ("N1", "clean", "Clean and normalize all record files", False, "ber.stages.clean", []),
    ("N2", "block", "Generate candidate pairs", False, "ber.stages.block", ["--clean-dir", "CLEAN"]),
    ("N3", "embed", "Encode unique records with multilingual-e5-small (GPU)", True, "ber.stages.embed", ["--clean-dir", "CLEAN"]),
    ("N4", "train_gbdt", "Train LightGBM and score test candidates", False, "ber.stages.train_gbdt",
     ["--clean-dir", "CLEAN", "--block-dir", "BLOCK", "--out-dir", 'ART + "/gbdt"', "--embed-opt"]),
    ("N5", "rerank", "Cross-encoder rerank top candidates (GPU)", True, "ber.stages.rerank",
     ["--clean-dir", "CLEAN", "--gbdt-dir", "GBDT", "--out-dir", 'ART + "/rerank"']),
    ("N6", "decide", "Threshold, write submission TSVs", False, "ber.stages.decide",
     ["--clean-dir", "CLEAN", "--block-dir", "BLOCK", "--gbdt-dir", "GBDT", "--out-dir", 'ART + "/out"', "--ce-opt"]),
]

PIP = {
    "N1": "anyascii jellyfish rapidfuzz pyarrow",
    "N2": "anyascii jellyfish rapidfuzz pyarrow",
    "N3": "sentence-transformers",
    "N4": "lightgbm rapidfuzz pyarrow",
    "N5": "sentence-transformers",
    "N6": "rapidfuzz pyarrow",
}

FIND_INPUT = '''
import glob
import os
import subprocess
import sys

CODE = "{code}"
RAW = "{raw}"
ART = "{art}"
os.environ["BER_DATA_DIR"] = RAW
os.environ["BER_ARTIFACT_DIR"] = ART
sys.path.insert(0, CODE)


def find(sub):
    hits = sorted(glob.glob(f"/kaggle/input/**/artifacts/{{sub}}", recursive=True))
    print(sub, "->", hits[:3])
    return hits[0] if hits else ""


CLEAN = find("clean")
BLOCK = find("block")
EMBED = find("embed")
GBDT = find("gbdt")
RERANK = find("rerank")
print("inputs located")


def run(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], check=True)
'''

PRINT = '''
import json
from pathlib import Path

for sub in ("clean", "block", "embed", "gbdt", "rerank", "out"):
    for name in ("metrics.json", "stats.json"):
        p = Path(ART) / sub / name
        if p.exists():
            print("==", sub, name, "==")
            print(p.read_text(encoding="utf-8"))

out_dir = Path(ART) / "out"
if out_dir.exists():
    print("outputs:", sorted(x.name for x in out_dir.glob("*.tsv")))
'''


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def run_expression(module, args):
    parts = [f'"{module}"']
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--embed-opt":
            parts.append('*(["--embed-dir", EMBED] if EMBED else [])')
        elif a == "--ce-opt":
            parts.append('*(["--ce-dir", RERANK] if RERANK else [])')
        elif a in ("CLEAN", "BLOCK", "GBDT"):
            parts.append(a)
        elif a in ('ART + "/gbdt"', 'ART + "/rerank"', 'ART + "/out"'):
            parts.append(a)
        else:
            parts.append(f'"{a}"')
    return "run(" + ", ".join(parts) + ")"


def make_notebook(nb_id, slug, title, gpu, module, args):
    settings = "GPU T4 x2" if gpu else "CPU (no accelerator)"
    header = (
        f"# {nb_id} - {title}\n\n"
        f"Kaggle settings: **Internet ON**, accelerator **{settings}**. "
        "Add inputs: `amz-er-2026-raw` + `amz-er-2026-code` + previous stage outputs. "
        "See `kaggle/PUSH_INSTRUCTIONS.md` in the repo."
    )
    cells = [
        md_cell(header),
        code_cell(f"!pip -q install {PIP[nb_id]}"),
        code_cell(FIND_INPUT.format(code=CODE_PATH, raw=RAW_PATH, art=ARTIFACT_PATH)),
        code_cell(run_expression(module, args)),
        code_cell(PRINT),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{nb_id}_{slug}.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


def main():
    for nb_id, slug, title, gpu, module, args in STAGES:
        path = make_notebook(nb_id, slug, title, gpu, module, args)
        print("wrote", path)


if __name__ == "__main__":
    main()
