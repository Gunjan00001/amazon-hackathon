# Business Entity Resolution Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Kaggle-only cascade (C+B) entity-resolution pipeline — classical blocking + LightGBM (submission v1), multilingual embedding features (v2), cross-encoder rerank (v3) — validated and packaged within 24 hours.

**Architecture:** Pure-Python core modules under `code/business_entity_resolution/src/ber/` are unit-tested locally (CPU) and wrapped by thin CLI stages. Kaggle notebooks N1–N6 run the stages in order, passing sharded parquet/npy artifacts through versioned Kaggle Datasets. Final outputs are the two required TSVs, validated locally before upload.

**Tech Stack:** Python 3.12 (local venv via uv; Kaggle image at runtime), pandas, pyarrow, numpy, scikit-learn, LightGBM, rapidfuzz, jellyfish, anyascii, pytest (local); sentence-transformers + torch (Kaggle GPU, N3/N5 only); Kaggle Notebooks + private Datasets.

## Global Constraints

- Local ML env: `uv venv .venv --python 3.12`; all local commands use `.venv\Scripts\python.exe`. No training or GPU work locally (RULES §6).
- TSV I/O: `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`; write UTF-8, tab-separated, commas are data, no quoting.
- Data paths: `BER_DATA_DIR` env var, local default `D:\Amazon project\DATA\student_resource\dataset`; Kaggle raw input `/kaggle/input/amz-er-2026-raw`, artifacts root `BER_ARTIFACT_DIR` (`/kaggle/working/artifacts` on Kaggle, `artifacts/` locally).
- Country is an open set: never filter or one-hot to `US`/`India`; France must always flow through.
- The same cleaning + feature code runs for train/validation/test; splits grouped by `source1_entity_id`; no label-derived features; held-out country for France-proxy validation.
- Models must be MIT/Apache-2.0 and <=8B parameters; pin revisions.
- No comments in code unless they explain a non-obvious constraint.
- Commit after every task; push to `origin Approach-2` (`https://github.com/Gunjan00001/amazon-hackathon.git`); tags use `major.minor.bugs`.
- Unit tests must pass before each commit.

## File Structure

```
code/business_entity_resolution/
  src/ber/
    __init__.py            # version string
    config.py              # env-driven paths, seed, shared constants
    score.py               # macro F0.5 (singleton rule) + threshold search
    text.py                # cleaning: normalize, translit, suffixes, address parse
    io_tsv.py              # readers/writers, id-list formatting
    blocking.py            # blocking keys, indexes, candidate generation
    features.py            # classical + structural pair features (shared spec)
    vector_features.py     # embedding feature assembly (torch imported lazily)
    gbdt.py                # LightGBM train/predict/tune wrappers
    decision.py            # top-K pruning, thresholding, one-to-one assignment
    rerank.py              # cross-encoder serialization + batch scoring (lazy torch)
    stages/
      __init__.py
      clean.py             # N1 CLI
      block.py             # N2 CLI
      embed.py             # N3 CLI (GPU)
      train_gbdt.py        # N4 CLI
      rerank.py            # N5 CLI (GPU)
      decide.py            # N6 CLI
  tests/                   # local pytest suite (fixtures, no full data)
  README.md                # reproduce end-to-end (Kaggle run order + local tests)
  requirements.txt         # pinned
kaggle/
  make_notebooks.py        # generates N1..N6 .ipynb from cell lists
  notebooks/N1_clean.ipynb ... N6_decide.ipynb
  PUSH_INSTRUCTIONS.md     # dataset upload + run + paste-back checklist
```

Artifact layout (versioned Kaggle Dataset `amz-er-2026-artifacts`):

```
artifacts/clean/{s1.parquet,s2.parquet,s3.parquet,labels.parquet,stats.json}
artifacts/block/{train_candidates.parquet,test_candidates.parquet,stats.json}
artifacts/embed/{ids.parquet,vecs_s1.npy,vecs_s2.npy,vecs_s3.npy,stats.json}
artifacts/gbdt/{model.txt,test_scores.parquet,pruned_test.parquet,metrics.json}
artifacts/rerank/{test_ce_scores.parquet,metrics.json}
artifacts/out/{candidate_pairs.tsv,matching_results.tsv,metrics.json}
```

---

### Task 0: Local env, package scaffold, macro-F0.5 scorer

**Files:**
- Create: `code/business_entity_resolution/src/ber/__init__.py`
- Create: `code/business_entity_resolution/src/ber/config.py`
- Create: `code/business_entity_resolution/src/ber/score.py`
- Create: `code/business_entity_resolution/tests/test_score.py`
- Create: `code/business_entity_resolution/requirements.txt`
- Create: `.gitignore` entry for `artifacts/` (append)

**Interfaces:**
- Produces: `entity_f05(labels: np.ndarray, preds: np.ndarray) -> float`, `macro_f05(groups: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float) -> float`, `best_threshold(groups, labels, scores, grid=np.arange(0.05,0.96,0.025)) -> tuple[float, float]`, `config.DATA_DIR`, `config.ARTIFACT_DIR`, `config.SEED`, `config.S2_PREFIXES`.

- [ ] **Step 1: Create the venv and install deps**

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe pandas pyarrow numpy scikit-learn lightgbm rapidfuzz jellyfish anyascii pytest
```

- [ ] **Step 2: Write the failing scorer tests**

`code/business_entity_resolution/tests/test_score.py`:

```python
import numpy as np
from ber.score import best_threshold, entity_f05, macro_f05


def test_entity_f05_worked_example():
    labels = np.array([1, 1, 0])
    preds = np.array([1, 1, 1])
    assert abs(entity_f05(labels, preds) - 0.714285) < 1e-5


def test_entity_f05_singleton_rule():
    empty = np.array([], dtype=int)
    assert entity_f05(empty, np.array([], dtype=int)) == 1.0
    assert entity_f05(empty, np.array([1], dtype=int)) == 0.0


def test_macro_f05_groups_and_thresholds():
    groups = np.array([0, 0, 1, 1, 2])
    labels = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.1])
    assert abs(macro_f05(groups, labels, scores, 0.5) - 1.0) < 1e-9
    f, th = best_threshold(groups, labels, scores, grid=np.array([0.5]))
    assert (f, th) == (1.0, 0.5)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber'`

- [ ] **Step 4: Implement config + scorer**

`config.py`:

```python
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", r"D:\Amazon project\DATA\student_resource\dataset"))
ARTIFACT_DIR = Path(os.environ.get("BER_ARTIFACT_DIR", "artifacts"))
SEED = 42
S2_PREFIX = "S2-"
S3_PREFIX = "S3-"
```

`score.py`:

```python
import numpy as np


def entity_f05(labels: np.ndarray, preds: np.ndarray) -> float:
    n_true = int(labels.sum())
    n_pred = int(preds.sum())
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    tp = int((labels & preds).sum())
    precision = tp / n_pred
    recall = tp / n_true
    if precision == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(groups: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    preds = scores >= threshold
    return float(np.mean([entity_f05(labels[groups == g], preds[groups == g]) for g in np.unique(groups)]))


def best_threshold(groups, labels, scores, grid=np.arange(0.05, 0.96, 0.025)):
    best = (0.0, float(grid[0]))
    for th in grid:
        f = macro_f05(groups, labels, scores, float(th))
        if f > best[0]:
            best = (f, float(th))
    return best
```

`__init__.py`: `__version__ = "0.3.0"`

Add `code/business_entity_resolution/tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

`requirements.txt` (pinned at install time; fill exact versions from `uv pip freeze`):

```
pandas
pyarrow
numpy
scikit-learn
lightgbm
rapidfuzz
jellyfish
anyascii
pytest
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_score.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add .gitignore code/business_entity_resolution
git commit -m "feat: pipeline scaffold and macro F0.5 scorer"
```

---

### Task 1: Cleaning module `text.py`

**Files:**
- Create: `code/business_entity_resolution/src/ber/text.py`
- Test: `code/business_entity_resolution/tests/test_text.py`

**Interfaces:**
- Produces: `normalize(s) -> str`, `fold_ascii(s) -> str`, `normalize_name(s) -> str`, `normalize_address(s) -> str`, `address_tokens(s) -> list[str]`, `postal_key(s) -> str | None`, `phonetic_key(token) -> str`, `LEGAL_SUFFIXES`, `ADDRESS_ABBREV`.

- [ ] **Step 1: Write the failing tests**

`tests/test_text.py`:

```python
from ber.text import fold_ascii, normalize_address, normalize_name, phonetic_key, postal_key


def test_normalize_name_expands_legal_suffixes():
    assert normalize_name("Acme Pvt. Ltd.") == "acme private limited"


def test_normalize_name_handles_ampersand_and_accents():
    assert normalize_name("École & Fils") == "ecole and fils"


def test_fold_ascii_removes_non_ascii():
    out = fold_ascii("Sun पावर Provision")
    assert out.isascii()
    assert "sun" in out and "provision" in out


def test_normalize_address_keeps_digits():
    assert normalize_address("12, MG Rd., Bengaluru 560001") == "12 mg road bengaluru 560001"


def test_postal_key_detects_pin_and_zip():
    assert postal_key("12 MG Road Bengaluru 560001") == "560001"
    assert postal_key("1 Main St, Austin, TX 78701") == "78701"
    assert postal_key("no digits here") is None


def test_phonetic_key_is_ascii_and_stable():
    a = phonetic_key("Bengaluru")
    b = phonetic_key("Bengaluru")
    assert a and a == b and a.isascii()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_text.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'ber.text'`

- [ ] **Step 3: Implement `text.py`**

```python
import re
import unicodedata

import jellyfish
from anyascii import anyascii

LEGAL_SUFFIXES = {
    "pvt": "private", "ltd": "limited", "co": "company", "corp": "corporation",
    "inc": "incorporated", "llp": "limited liability partnership", "llc": "limited liability company",
    "plc": "public limited company", "srl": "societe a responsabilite limitee", "sarl": "societe a responsabilite limitee",
}

ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "opp": "opposite", "nr": "near", "hse": "house", "bldg": "building", "flr": "floor",
    "apt": "apartment", "dist": "district", "pin": "postal", "po": "post office",
}

POSTAL_RE = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def fold_ascii(s: str) -> str:
    s = anyascii(s or "")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if ord(c) < 128).lower()


def _expand(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


def normalize_name(s: str) -> str:
    return " ".join(_expand(normalize(s).split(), LEGAL_SUFFIXES))


def normalize_address(s: str) -> str:
    return " ".join(_expand(normalize(s).split(), ADDRESS_ABBREV))


def address_tokens(s: str) -> list[str]:
    return [t for t in normalize_address(s).split() if not t.isdigit()]


def postal_key(s: str) -> str | None:
    m = POSTAL_RE.search(normalize(s))
    return m.group(1) if m else None


def phonetic_key(token: str) -> str:
    return jellyfish.metaphone(fold_ascii(token)) or ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_text.py -v`
Expected: 6 passed (adjust expected strings if `metaphone` behavior differs; keep assertions semantic, not exact codes)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: text cleaning and address parsing"
```

---

### Task 2: TSV I/O module `io_tsv.py`

**Files:**
- Create: `code/business_entity_resolution/src/ber/io_tsv.py`
- Test: `code/business_entity_resolution/tests/test_io_tsv.py`

**Interfaces:**
- Produces: `load_records(path, usecols=None) -> pd.DataFrame`, `load_ground_truth(path) -> pd.DataFrame`, `split_id_list(s) -> list[str]`, `format_id_list(ids) -> str`, `write_matching_results(path, s1_ids, matches) -> None`, `write_candidate_pairs(path, s1_ids, candidates) -> None`, `s1_sort_key(eid)`.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
from ber.io_tsv import format_id_list, split_id_list, write_matching_results


def test_id_list_roundtrip_and_empty():
    ids = ["S2-00047", "S3-00812", "S2-00193"]
    assert format_id_list(ids) == "S2-00047,S2-00193,S3-00812"
    assert format_id_list([]) == ""
    assert split_id_list("S2-00047,S3-00812") == ["S2-00047", "S3-00812"]
    assert split_id_list("") == []


def test_write_matching_results_exact_format(tmp_path):
    p = tmp_path / "matching_results.tsv"
    write_matching_results(p, ["S1-00002", "S1-00001"], {"S1-00001": ["S2-00047"], "S1-00002": []})
    text = p.read_text(encoding="utf-8")
    assert text == "source1_entity_id\tmatched_entity_ids\nS1-00002\t\nS1-00001\tS2-00047\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_io_tsv.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `io_tsv.py`**

```python
import re
from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
_NUM_RE = re.compile(r"(\d+)")


def s1_sort_key(eid: str):
    return (eid[:3], int(_NUM_RE.search(eid).group(1)))


def load_records(path, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols)
    return df


def load_ground_truth(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def split_id_list(s: str) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def _id_sort_key(eid: str):
    return (eid[:3], int(_NUM_RE.search(eid).group(1)))


def format_id_list(ids) -> str:
    return ",".join(sorted(set(ids), key=_id_sort_key))


def _write(path, header: list[str], rows) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_io_tsv.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: TSV readers and submission writers"
```

---

### Task 3: Blocking module `blocking.py`

**Files:**
- Create: `code/business_entity_resolution/src/ber/blocking.py`
- Test: `code/business_entity_resolution/tests/test_blocking.py`

**Interfaces:**
- Produces: `s1_blocking_tokens(row) -> list[str]`, `mid_blocking_tokens(row) -> list[str]`, `build_rare_token_index(df, token_col, max_postings) -> dict[str, list[int]]`, `generate_candidates(s1_df, mid_df, max_candidates_per_s1, max_postings) -> pd.DataFrame` with columns `s1_idx:int32, mid_idx:int32, pass:str`, and `blocking_recall(pairs, labels) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
from ber.blocking import blocking_recall, generate_candidates


def _s1():
    return pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Zenith Trading Company", "Orchid Diagnostics"],
        "business_address": ["12 MG Road Bengaluru 560001", "44 Park Ave Austin 78701"],
        "country": ["India", "US"],
    })


def _mid():
    return pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S3-3"],
        "business_name": ["Zenith Trading Pvt Ltd", "Orchid Diagnostics LLC", "Zebra Foods"],
        "business_address": ["12 MG Rd Bengaluru 560001", "44 Park Avenue Austin 78701", "9 Beach Rd Goa 403001"],
        "country": ["India", "US", "India"],
    })


def test_generates_expected_pairs_and_country_gate():
    pairs = generate_candidates(_s1(), _mid(), max_candidates_per_s1=50, max_postings=100)
    got = set(zip(pairs["s1_idx"], pairs["mid_idx"]))
    assert (0, 0) in got and (1, 1) in got
    assert (0, 2) not in got
    assert pairs["pass"].notna().all()


def test_recall_helper_counts_hits():
    labels = {(0, 0), (1, 1)}
    pairs = pd.DataFrame({"s1_idx": [0, 1], "mid_idx": [0, 2]})
    assert blocking_recall(pairs, labels) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_blocking.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `blocking.py`**

Key logic (implement exactly; pass priorities in this order: `postal`, `rare_token`, `phonetic`, `addr_token`):

```python
import numpy as np
import pandas as pd

from .text import address_tokens, fold_ascii, normalize_address, normalize_name, phonetic_key, postal_key

STOPWORDS = {"private", "limited", "company", "corporation", "incorporated", "and", "the", "of", "llc", "llp"}


def _significant(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]


def s1_blocking_tokens(row) -> dict:
    norm_name = normalize_name(row["business_name"])
    tokens = _significant(norm_name.split())
    first = fold_ascii(tokens[0]) if tokens else fold_ascii(norm_name.split()[0])
    return {
        "country": row["country"].strip().lower(),
        "tokens": tokens,
        "phonetic": phonetic_key(first),
        "postal": postal_key(row["business_address"]),
    }


def mid_blocking_tokens(row) -> dict:
    norm_name = normalize_name(row["business_name"])
    tokens = _significant(norm_name.split())
    first = fold_ascii(tokens[0]) if tokens else fold_ascii(norm_name.split()[0])
    return {
        "country": row["country"].strip().lower(),
        "tokens": tokens,
        "phonetic": phonetic_key(first),
        "postal": postal_key(row["business_address"]),
    }
```

`generate_candidates(s1_df, mid_df, max_candidates_per_s1, max_postings)`: precompute per-row token dicts; build three in-memory indexes keyed by `(country, key)` — `postal`, `phonetic`, and rare-token inverted index (tokens whose postings <= `max_postings`); also an address-token index on the rarest token. Iterate S1 rows in pass-priority order, collecting mid indexes, dedup with a per-S1 seen set, stop at `max_candidates_per_s1`, and emit `(s1_idx, mid_idx, pass)`. Never drop the country gate.

`blocking_recall(pairs, labels)`: `hits = len({(int(a), int(b)) for a, b in zip(pairs.s1_idx, pairs.mid_idx)} & labels)`; return `hits / max(1, len(labels))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_blocking.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: blocking keys and candidate generation"
```

---

### Task 4: Pair features `features.py`

**Files:**
- Create: `code/business_entity_resolution/src/ber/features.py`
- Test: `code/business_entity_resolution/tests/test_features.py`

**Interfaces:**
- Produces: `CLASSICAL_FEATURES: list[str]`, `build_pair_frame(s1_df, mid_df, pairs) -> pd.DataFrame` (columns: the 15 classical + `s1_idx`, `mid_idx`, `pass`, `is_s2`), `classical_features(frame) -> np.ndarray`, `structural_features(frame) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from ber.features import build_pair_frame, classical_features


def test_classical_features_identical_strings_are_max():
    frame = pd.DataFrame({
        "n1": ["acme trading"], "n2": ["acme trading"],
        "a1": ["12 mg road"], "a2": ["12 mg road"],
        "c1": ["india"], "c2": ["india"],
        "is_s2": [True], "s1_idx": [0], "mid_idx": [0], "pass": ["rare_token"],
    })
    X = classical_features(frame)
    assert X.shape == (1, 15)
    assert X[0, 0] == 100.0  # fuzz.ratio
    assert X[0, 5] == 1.0    # name_exact


def test_build_pair_frame_joins_columns():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"], "business_name": ["Acme Pvt Ltd"],
        "business_address": ["12 MG Rd"], "country": ["India"],
    })
    mid = pd.DataFrame({
        "entity_id": ["S2-9"], "business_name": ["Acme Private Limited"],
        "business_address": ["12 MG Road"], "country": ["India"],
    })
    pairs = pd.DataFrame({"s1_idx": np.array([0], dtype="int32"), "mid_idx": np.array([0], dtype="int32"), "pass": ["rare_token"]})
    frame = build_pair_frame(s1, mid, pairs)
    assert frame.loc[0, "n1"] == "acme private limited"
    assert frame.loc[0, "n2"] == "acme private limited"
    assert set(["s1_idx", "mid_idx", "pass", "is_s2"]) <= set(frame.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_features.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `features.py`**

Port the 15 features from `tools/bench_gbdt.py` (same names, same 0–100 rapidfuzz scales, `-1.0` sentinel for missing addresses) but compute them vectorized with `rapidfuzz.process.cpdist(..., workers=-1)`; keep `name_jaccard` / `addr_jaccard` as vectorized set ops. `build_pair_frame` normalizes S1/S2/S3 text with `text.py`, joins name/address/country columns for the pair indexes, and derives `is_s2` from `S2-` prefix.

```python
CLASSICAL_FEATURES = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaro",
    "name_exact", "name_jaccard", "name_len_diff",
    "addr_ratio", "addr_token_sort", "addr_jaccard", "addr_missing", "addr_len_diff",
    "same_country", "is_s2",
]
```

`structural_features(frame)` adds `postal_match`, `phonetic_match`, `addr_token_overlap`, `name_token_overlap`, `name_first_token_match` (0/1 floats), also all vectorized.

- [ ] **Step 4: Run tests to verify they pass + compare sanity against old bench on a fixture**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_features.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: vectorized pair features"
```

---

### Task 5: GBDT wrapper `gbdt.py`

**Files:**
- Create: `code/business_entity_resolution/src/ber/gbdt.py`
- Test: `code/business_entity_resolution/tests/test_gbdt.py`

**Interfaces:**
- Produces: `PARAMS`, `split_groups(groups, val_frac=0.2, seed=42) -> tuple[np.ndarray, np.ndarray]`, `train_model(X, y, groups, params=None) -> tuple[LGBMClassifier, dict]` (metrics: `pair_auc`, `average_precision`, `best_macro_f05`, `best_threshold`, `best_iteration`), `predict_scores(model, X) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from ber.gbdt import predict_scores, split_groups, train_model


def test_split_groups_is_disjoint_and_grouped():
    groups = np.repeat(np.arange(100), 5)
    tr, va = split_groups(groups, val_frac=0.2, seed=0)
    assert not (set(groups[tr]) & set(groups[va]))
    assert len(set(groups[va])) == 20


def test_train_model_separates_easy_signal():
    rng = np.random.default_rng(0)
    n = 4000
    groups = np.repeat(np.arange(400), 10)
    X = rng.normal(size=(n, 4)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model, metrics = train_model(X, y, groups)
    assert metrics["pair_auc"] > 0.9
    assert 0.0 <= metrics["best_threshold"] <= 1.0
    scores = predict_scores(model, X)
    assert scores.shape == (n,) and scores.min() >= 0 and scores.max() <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_gbdt.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `gbdt.py`**

```python
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score

from .score import best_threshold

PARAMS = dict(objective="binary", n_estimators=1000, learning_rate=0.05, num_leaves=63,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, random_state=42, verbosity=-1)


def split_groups(groups, val_frac=0.2, seed=42):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_groups = set(uniq[: max(1, int(val_frac * len(uniq)))].tolist())
    val_mask = np.isin(groups, list(val_groups))
    return ~val_mask, val_mask


def train_model(X, y, groups, params=None):
    model = LGBMClassifier(**(params or PARAMS))
    tr, va = split_groups(groups)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
              callbacks=[early_stopping(50, verbose=False)])
    scores = model.predict_proba(X[va])[:, 1]
    f, th = best_threshold(groups[va], y[va], scores)
    metrics = {
        "pair_auc": float(roc_auc_score(y[va], scores)),
        "average_precision": float(average_precision_score(y[va], scores)),
        "best_macro_f05": float(f), "best_threshold": float(th),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    return model, metrics


def predict_scores(model, X):
    return model.predict_proba(X)[:, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_gbdt.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: LightGBM train/eval wrapper"
```

---

### Task 6: Decision logic `decision.py`

**Files:**
- Create: `code/business_entity_resolution/src/ber/decision.py`
- Test: `code/business_entity_resolution/tests/test_decision.py`

**Interfaces:**
- Produces: `prune_top_k(df, k, score_col="score") -> pd.DataFrame`, `matches_from_scores(df, threshold, score_col="score") -> dict[int, list[int]]`, `one_to_one_assign(df, threshold, score_col="score") -> dict[int, list[int]]`, `to_entity_ids(matches, s1_ids, mid_ids) -> dict[str, list[str]]`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from ber.decision import matches_from_scores, one_to_one_assign, prune_top_k, to_entity_ids


def _df():
    return pd.DataFrame({
        "s1_idx": np.array([0, 0, 0, 1, 1], dtype="int32"),
        "mid_idx": np.array([10, 11, 12, 10, 20], dtype="int32"),
        "score": np.array([0.9, 0.4, 0.8, 0.85, 0.7]),
    })


def test_prune_top_k_per_s1():
    out = prune_top_k(_df(), k=2)
    assert len(out) == 4
    assert set(out.loc[out.s1_idx == 0, "mid_idx"]) == {10, 12}


def test_matches_from_scores_threshold():
    got = matches_from_scores(_df(), 0.75)
    assert got == {0: [10, 12], 1: [10]}


def test_one_to_one_assign_resolves_conflict():
    got = one_to_one_assign(_df(), 0.5)
    assert got[0] == [10, 12]
    assert got[1] == [20]


def test_to_entity_ids_maps_indexes():
    got = to_entity_ids({0: [10]}, ["S1-1", "S1-2"], {10: "S2-7", 11: "S3-3"})
    assert got == {"S1-1": ["S2-7"], "S1-2": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_decision.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `decision.py`**

```python
import pandas as pd


def prune_top_k(df, k, score_col="score"):
    return (df.sort_values(score_col, ascending=False)
              .groupby("s1_idx", sort=False, as_index=False)
              .head(k)
              .reset_index(drop=True))


def matches_from_scores(df, threshold, score_col="score"):
    kept = df[df[score_col] >= threshold]
    out = {}
    for s1, mid in zip(kept["s1_idx"], kept["mid_idx"]):
        out.setdefault(int(s1), []).append(int(mid))
    return out


def one_to_one_assign(df, threshold, score_col="score"):
    kept = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)
    used_mid, used_s1, out = set(), set(), {}
    for s1, mid in zip(kept["s1_idx"], kept["mid_idx"]):
        s1, mid = int(s1), int(mid)
        if mid in used_mid:
            continue
        used_mid.add(mid)
        used_s1.add(s1)
        out.setdefault(s1, []).append(mid)
    return out


def to_entity_ids(matches, s1_ids, mid_ids):
    return {sid: [mid_ids[m] for m in matches.get(i, [])] for i, sid in enumerate(s1_ids)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_decision.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: top-k pruning, thresholding, one-to-one assignment"
```

---

### Task 7: Stages N1/N2 + Kaggle notebooks + push instructions (user can start running)

**Files:**
- Create: `code/business_entity_resolution/src/ber/stages/__init__.py`
- Create: `code/business_entity_resolution/src/ber/stages/clean.py`
- Create: `code/business_entity_resolution/src/ber/stages/block.py`
- Create: `kaggle/make_notebooks.py`
- Create: `kaggle/notebooks/N1_clean.ipynb`, `N2_block.ipynb` (generated)
- Create: `kaggle/PUSH_INSTRUCTIONS.md`

**Interfaces:**
- Consumes: `config`, `text`, `io_tsv`, `blocking`.
- Produces: `artifacts/clean/*.parquet` + `stats.json`; `artifacts/block/{train,test}_candidates.parquet` + `stats.json`; notebooks that run them.

- [ ] **Step 1: Implement `clean.py` stage**

`python -m ber.stages.clean --data-dir ... --out-dir ... [--sample N]` reads the 6 record TSVs + ground truth, applies `normalize_name` / `normalize_address` / `fold_ascii` / `postal_key` / `country_n`, and writes `s1.parquet`, `s2.parquet`, `s3.parquet` (train and test subdirs), `labels.parquet`, and `stats.json` (row counts, empty-address counts, non-ASCII name counts per file, sample rows).

- [ ] **Step 2: Implement `block.py` stage**

`python -m ber.stages.block --clean-dir ... --out-dir ... [--max-candidates 200] [--max-postings 20000]` loads clean parquet, calls `blocking.generate_candidates` for train and test, joins ground truth on train to compute `recall_ceiling`, writes parquet shards and `stats.json` (pairs, pairs/S1 mean/p95, recall ceiling, pairs by pass).

- [ ] **Step 3: Smoke-test both stages locally on a tiny sample**

```powershell
$env:BER_ARTIFACT_DIR="artifacts"
.venv\Scripts\python.exe -m ber.stages.clean --sample 2000
.venv\Scripts\python.exe -m ber.stages.block --clean-dir artifacts\clean
```
Run from `code\business_entity_resolution\src`. Expected: parquet + stats written; recall ceiling printed; no exceptions. (This is local CPU prep/validation only, allowed by RULES §6.)

- [ ] **Step 4: Write `kaggle/make_notebooks.py` + `PUSH_INSTRUCTIONS.md`**

`make_notebooks.py` emits one `.ipynb` per stage from a `CELLS` dict with cells: (1) markdown header, (2) `!pip -q install ...` for missing deps, (3) bash-run cell `!PYTHONPATH=/kaggle/input/amz-er-2026-code/src BER_DATA_DIR=/kaggle/input/amz-er-2026-raw BER_ARTIFACT_DIR=/kaggle/working/artifacts python -m ber.stages.<stage> ...`, (4) `print(Path(".../stats.json").read_text())` for paste-back. `PUSH_INSTRUCTIONS.md` documents: create private `amz-er-2026-raw` (7 TSVs from `D:\Amazon project\DATA\student_resource\dataset`, skip `.DS_Store`); create private `amz-er-2026-code` (contents of `code/business_entity_resolution/src`; update via New Version); per-notebook Add Input → raw + code + previous notebook's saved output; Internet ON; GPU only on N3/N5; Save Version after each run; download N6 output into repo `output/`.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution kaggle
git commit -m "feat: clean and block stages with Kaggle notebooks N1-N2"
```

---

### Task 8: Train/predict stages + N4/N6 → submission v1 (classical)

**Files:**
- Create: `code/business_entity_resolution/src/ber/stages/train_gbdt.py`
- Create: `code/business_entity_resolution/src/ber/stages/decide.py`
- Create: `kaggle/notebooks/N4_train_gbdt.ipynb`, `N6_decide.ipynb` (generated)

**Interfaces:**
- Consumes: `features`, `gbdt`, `decision`, `score`, `io_tsv`.
- Produces: `artifacts/gbdt/{model.txt,test_scores.parquet,metrics.json}`; `artifacts/out/{candidate_pairs.tsv,matching_results.tsv,metrics.json}`.

- [ ] **Step 1: Implement `train_gbdt.py`**

CLI `python -m ber.stages.train_gbdt --clean-dir ... --block-dir ... --out-dir ... [--embed-dir ...] [--top-k 10]`: build pair frame for train candidates, attach labels (1 if mid in the S1's ground-truth list), build features, `gbdt.train_model`, predict test candidates **in shards** (chunked parquet read), write `test_scores.parquet`, also write `pruned_test.parquet` (top-K per S1 by GBDT score, same K that N5 and N6 use), copy model with `model.booster_.save_model(out/"model.txt")`, write `metrics.json` (AUC, AP, macro-F0.5, threshold).

- [ ] **Step 2: Implement `decide.py` (v1 path)**

CLI `python -m ber.stages.decide --block-dir ... --gbdt-dir ... --clean-dir ... --out-dir ... --threshold <from metrics> [--top-k 10] [--one-to-one]`: load test scores, optional `prune_top_k`, choose threshold from `metrics.json` unless overridden, `matches_from_scores` (or `one_to_one_assign` if flag), `to_entity_ids`, write `candidate_pairs.tsv` from the pruned candidate set and `matching_results.tsv` from matches; write `metrics.json` (rows, matches/S1, empty rows, candidates ⊇ matches check).

- [ ] **Step 3: Local end-to-end rehearsal on the sample**

```powershell
$env:BER_ARTIFACT_DIR="artifacts"
.venv\Scripts\python.exe -m ber.stages.train_gbdt --clean-dir artifacts\clean --block-dir artifacts\block --out-dir artifacts\gbdt
.venv\Scripts\python.exe -m ber.stages.decide --block-dir artifacts\block --gbdt-dir artifacts\gbdt --clean-dir artifacts\clean --out-dir artifacts\out
```
Expected: metrics printed; TSVs written; local check `matches ⊆ candidates` passes. Then run the real validator against full test only after the Kaggle full run.

- [ ] **Step 4: Generate N4/N6 notebooks and commit; hand off to user**

```bash
git add code/business_entity_resolution kaggle
git commit -m "feat: GBDT training and decision stages, notebooks N4 and N6"
```

**Milestone:** user runs N1 → N2 → N4 → N6 on Kaggle CPU, downloads N6 output into `output/`, then locally runs the validator from `DATA/student_resource`; v1 submission ready with zero GPU used.

---

### Task 9: Embedding stage N3 + vector features → submission v2

**Files:**
- Create: `code/business_entity_resolution/src/ber/vector_features.py`
- Create: `code/business_entity_resolution/src/ber/stages/embed.py`
- Create: `kaggle/notebooks/N3_embed.ipynb` (generated)
- Modify: `code/business_entity_resolution/src/ber/stages/train_gbdt.py` (accept `--embed-dir`)
- Test: `code/business_entity_resolution/tests/test_vector_features.py`

**Interfaces:**
- Produces: `pair_vector_features(name_vecs1, addr_vecs1, name_vecs2, addr_vecs2, pair_idx) -> np.ndarray` (3 columns per field type — cosine, abs-diff mean, product mean — 6 total, rows selected by `pair_idx`)`, `load_vector_store(embed_dir) -> dict[source, tuple[ids, vecs]]`, `artifacts/embed/{ids.parquet,vecs_<source>.npy,stats.json}`.
- Model: `intfloat/multilingual-e5-small`, revision pinned in `vector_features.MODEL_ID` / `MODEL_REVISION`; e5 prefixes `query: ` for S1 text and `passage: ` for S2/S3 text; text = `business_name + " | " + business_address`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from ber.vector_features import cosine_matrix_rows, pair_vector_features


def test_pair_vector_features_identical_vectors_cosine_one():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    f = pair_vector_features(a, a, a, a, np.array([0, 1], dtype="int32"))
    assert f.shape[1] == 6
    assert np.allclose(f[:, 0], 1.0)


def test_cosine_matrix_rows_matches_manual():
    a = np.array([[1.0, 0.0]], dtype="float32")
    b = np.array([[1.0, 1.0]], dtype="float32")
    c = cosine_matrix_rows(a, b)
    assert abs(c[0] - 0.7071067) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_vector_features.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `vector_features.py` and `embed.py`**

`vector_features.py` holds the pure-numpy parts (unit-tested locally) plus `encode_texts(texts, model_id, revision, batch_size)` that imports `sentence_transformers` lazily and raises a clear error if unavailable. `embed.py` CLI: `python -m ber.stages.embed --clean-dir ... --out-dir ... [--batch-size 256] [--sample N]`; deduplicates `(entity_id, text)` per source, encodes on GPU with fp16, writes `ids.parquet` and one `vecs_<source>.npy` per source (float16), `stats.json` with counts and encode time.

- [ ] **Step 4: Extend `train_gbdt.py`**

When `--embed-dir` is given, load the vector store, join vectors by record index, append `pair_vector_features` output columns to the feature matrix for both train and test candidates; metric keys unchanged.

- [ ] **Step 5: Run local tests + sample rehearsal**

```powershell
.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests -v
```
Expected: all green. Full GPU run happens on Kaggle in N3/N4 (T4 x2, Internet ON).

- [ ] **Step 6: Commit**

```bash
git add code/business_entity_resolution kaggle
git commit -m "feat: multilingual embedding stage and vector pair features"
```

**Milestone:** user runs N3 (GPU) → N4 with `--embed-dir` → N6 again; v2 submission.

---

### Task 10: Cross-encoder stage N5 → submission v3

**Files:**
- Create: `code/business_entity_resolution/src/ber/rerank.py`
- Create: `code/business_entity_resolution/src/ber/stages/rerank.py`
- Create: `kaggle/notebooks/N5_rerank.ipynb` (generated)
- Modify: `code/business_entity_resolution/src/ber/stages/decide.py` (accept `--ce-dir`)
- Test: `code/business_entity_resolution/tests/test_rerank.py`

**Interfaces:**
- Produces: `serialize_pair(name1, addr1, country1, name2, addr2, country2) -> str`, `score_texts(texts, model_id, revision, batch_size, fn=None) -> np.ndarray` (`fn` injectable for local tests), `artifacts/rerank/{test_ce_scores.parquet,val_ce_scores.parquet,metrics.json}` with `best_threshold`, `best_macro_f05`, and `lift_over_gbdt`.
- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M), revision pinned; fallback `BAAI/bge-reranker-base` only if quota and time allow.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from ber.rerank import serialize_pair, score_texts


def test_serialize_pair_handles_empty_address_and_accents():
    s = serialize_pair("École Sainte", "", "France", "Ecole Sainte", "1 Rue A", "France")
    assert s == "École Sainte | none | france [SEP] Ecole Sainte | 1 Rue A | france"


def test_score_texts_uses_injected_fn():
    out = score_texts(["a", "b"], "any", "rev", 2, fn=lambda texts: np.array([0.1, 0.9], dtype="float32"))
    assert out.tolist() == [0.1, 0.9]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests\test_rerank.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `rerank.py` and the stage**

Stage CLI: `python -m ber.stages.rerank --clean-dir ... --gbdt-dir ... --out-dir ... --top-k 10 [--tune-sample 20000]`. It loads `pruned_test.parquet` (written by N4 as top-K by GBDT score — N5 and N6 must use the same K, default 10), serializes pairs in the exact format `name1 | addr1_or_none | country1 [SEP] name2 | addr2_or_none | country2` (empty address → `none`, country lowercased, names/addresses otherwise raw), CE-scores in shards, and writes `test_ce_scores.parquet`. With `--tune`, it samples `tune-sample` S1 entities from the N4 validation groups, CE-scores their candidates, and computes the best macro-F0.5 threshold plus lift vs the GBDT threshold on the same pairs; writes `val_ce_scores.parquet` and `metrics.json`.

- [ ] **Step 4: Extend `decide.py`**

With `--ce-dir`, use CE scores for the threshold decision (threshold from `rerank/metrics.json`); GBDT scores remain only for candidate pruning. Report both GBDT and CE metrics in `metrics.json`.

- [ ] **Step 5: Run local tests**

Run: `.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests -v`
Expected: all green (no torch needed locally).

- [ ] **Step 6: Commit**

```bash
git add code/business_entity_resolution kaggle
git commit -m "feat: cross-encoder rerank stage and CE-driven decisions"
```

**Milestone:** user runs N5 (GPU) → N6 with `--ce-dir`; v3 submission. If v3 drops validation macro F0.5 versus v2, keep the v2 leaderboard score and note the experiment in the methodology doc.

---

### Task 11: Submission packaging and documentation

**Files:**
- Create: `code/business_entity_resolution/README.md`
- Create: `code/business_entity_resolution/requirements.txt` (final pin from `uv pip freeze`)
- Create: `submission/make_zip.py`
- Create: `submission/Documentation_template.md` (filled from `DATA/student_resource/Documentation_template.md`)

**Interfaces:**
- Produces: `<team_name>_submission.zip` with `output/`, `code/business_entity_resolution/`, `Documentation_template.md`.

- [ ] **Step 1: Write the package README**

Sections: overview, environment (local venv + Kaggle image), run order N1–N6 with exact settings (accelerator, internet, inputs), artifact flow, reproduction from raw TSVs, local test + validator commands, model IDs/licenses/revisions.

- [ ] **Step 2: Fill the methodology template**

Cover: methodology (cascade), candidate generation/blocking strategy with measured recall ceiling, feature engineering, model architectures, threshold and one-to-one post-processing, validation protocol (grouped split + country holdout), and results per submission version. Reference actual `metrics.json` numbers from the Kaggle runs.

- [ ] **Step 3: Write `make_zip.py` and build the archive**

```python
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = os.environ.get("TEAM_NAME", "team")

def main():
    staging = ROOT / "submission" / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "output").mkdir(parents=True)
    (staging / "code").mkdir()
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(ROOT / "output" / name, staging / "output" / name)
    shutil.copytree(ROOT / "code" / "business_entity_resolution", staging / "code" / "business_entity_resolution")
    shutil.copy2(ROOT / "submission" / "Documentation_template.md", staging / "Documentation_template.md")
    shutil.make_archive(str(ROOT / "submission" / f"{TEAM}_submission"), "zip", staging)

if __name__ == "__main__":
    main()
```

Run: `.venv\Scripts\python.exe submission\make_zip.py`
Expected: `submission/<team>_submission.zip` contains exactly the three required top-level entries. Set `TEAM_NAME` to the portal team name before the final build.

- [ ] **Step 4: Commit**

```bash
git add code/business_entity_resolution submission
git commit -m "docs: submission package, README, methodology write-up"
```

---

### Task 12: Final validation, versioning, push

**Files:**
- Modify: `README.md` (version log + status + roadmap ticks)
- Modify: `RULES.md` §5.5 (version)

- [ ] **Step 1: Run the official validator against the final outputs**

From `DATA/student_resource`:

```powershell
python utils/validate_submission.py --matching D:\Amazon Proj Approach 3\amazon-hackathon\output\matching_results.tsv --candidate D:\Amazon Proj Approach 3\amazon-hackathon\output\candidate_pairs.tsv --test-dir dataset/test
```
Expected: `PASS`, exit 0. Fix and rerun until PASS.

- [ ] **Step 2: Update versions and logs**

`README.md`: status = submission-ready; version log lines for `0.3.0` (pipeline v1 capability) and `1.0.0` (validated submission); `RULES.md` §5.5 → `1.0.0`.

- [ ] **Step 3: Commit, tag, push**

```bash
git add README.md RULES.md
git commit -m "release: validated 1.0.0 submission outputs and docs"
git tag 1.0.0
git push origin Approach-2
git push origin 1.0.0
```

- [ ] **Step 4: Final package check**

Run `submission/make_zip.py` once more and confirm the archive matches the current outputs; upload `matching_results.tsv` to the portal and submit the zip per the statement requirements.
