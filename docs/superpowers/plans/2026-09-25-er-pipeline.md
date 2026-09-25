# Business Entity Resolution Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local pipeline that cleans the business records, generates high-recall candidate pairs, engineers pairwise features, trains a LightGBM matcher, and emits validated `output/matching_results.tsv` and `output/candidate_pairs.tsv` for the Amazon ML Challenge 2026.

**Architecture:** Staged parquet pipeline driven by a small CLI. DuckDB performs out-of-core blocking joins; pandas/pyarrow handle cleaning, chunked feature computation, and I/O; LightGBM trains a binary pairwise classifier whose threshold is tuned to macro F_0.5.

**Tech Stack:** Python 3.12 (`.venv`), pandas, pyarrow, numpy, scikit-learn, LightGBM, XGBoost (drop-in), rapidfuzz, duckdb, indic-transliteration, jellyfish, pytest.

## Global Constraints

- Python 3.12 only for the ML pipeline; run via `.venv\Scripts\python.exe` from the repo root.
- No external data lookup: no APIs, geocoding, registries, or internet augmentation. All signals come from the provided TSVs.
- All record/output files are UTF-8, tab-separated; read with `sep="\t"`, `dtype=str`, `keep_default_na=False`.
- `country` is open-set: never filter, hard-code, or one-hot to `{US, India, France}`.
- Output invariants (`matching_results.tsv`): exactly one row per test S1 (1,732,544 data rows), empty list for singletons, no duplicate IDs inside a list, no duplicate S1 rows, S2/S3 IDs only, matches ⊆ candidates.
- Determinism: every random choice uses seed 42 unless a task explicitly overrides.
- Model constraints: MIT/Apache-2.0, ≤8B params (LightGBM qualifies).
- Never commit `DATA/**/*.tsv`, `DATA/**/*.zip`, `.venv/`, or `output/**/*.tsv` (see `.gitignore`).
- Performance budgets on this machine: peak RAM ≤ 8 GB per stage; Stage 0 ≤ 45 min; blocking ≤ 2 h; feature build ≤ 30 min; training ≤ 30 min; test scoring ≤ 3 h.
- Paths: data under `DATA/student_resource/dataset/`, all intermediates under `data/`, model artifacts under `code/business_entity_resolution/models/`, outputs under `output/`.

---

### Task 1: Scaffold, config, and dependency install

**Files:**
- Create: `code/business_entity_resolution/src/ber/__init__.py`
- Create: `code/business_entity_resolution/src/ber/config.py`
- Create: `code/business_entity_resolution/config.json`
- Create: `code/business_entity_resolution/requirements.txt`
- Create: `code/business_entity_resolution/tests/conftest.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: `ber.config.Config` (frozen dataclass) with fields `root: Path`, `dataset_dir: Path`, `data_dir: Path`, `models_dir: Path`, `output_dir: Path`, `seed: int`, `cap: int`, `idf_min: float`, `max_block: int`, `neg_ratio: int`, `lgbm_params: dict`, and classmethod `Config.load(path: str | Path) -> Config`; fixture `cfg` in `conftest.py`.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/conftest.py
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ber.config import Config  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    return Config(
        root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "output",
        seed=42,
        cap=200,
        idf_min=4.0,
        max_block=5000,
        neg_ratio=4,
        lgbm_params={"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63},
    )


def test_config_load_roundtrip(tmp_path):
    import json

    from ber.config import Config

    payload = {
        "dataset_dir": str(tmp_path / "dataset"),
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "output_dir": str(tmp_path / "output"),
        "seed": 7,
        "cap": 150,
        "idf_min": 3.5,
        "max_block": 4000,
        "neg_ratio": 3,
        "lgbm_params": {"n_estimators": 100},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.seed == 7
    assert cfg.cap == 150
    assert cfg.dataset_dir == tmp_path / "dataset"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/conftest.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ber'`).

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/ber/__init__.py
__all__ = ["config"]
```

```python
# code/business_entity_resolution/src/ber/config.py
import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path
    dataset_dir: Path
    data_dir: Path
    models_dir: Path
    output_dir: Path
    seed: int = 42
    cap: int = 200
    idf_min: float = 4.0
    max_block: int = 5000
    neg_ratio: int = 4
    lgbm_params: dict = None

    def __post_init__(self):
        object.__setattr__(self, "lgbm_params", self.lgbm_params or {})

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = {k: (Path(v) if k.endswith("_dir") else v) for k, v in payload.items()}
        payload.setdefault("root", Path.cwd())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})
```

```json
// code/business_entity_resolution/config.json
{
  "dataset_dir": "DATA/student_resource/dataset",
  "data_dir": "data",
  "models_dir": "code/business_entity_resolution/models",
  "output_dir": "output",
  "seed": 42,
  "cap": 200,
  "idf_min": 4.0,
  "max_block": 5000,
  "neg_ratio": 4,
  "lgbm_params": {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8
  }
}
```

```text
# code/business_entity_resolution/requirements.txt
pandas==3.0.6
pyarrow==25.0.1
numpy==2.5.3
scikit-learn==1.9.1
lightgbm==4.7.0
xgboost==3.4.1
rapidfuzz==3.14.6
duckdb
indic-transliteration
jellyfish
pytest
```

```ini
# pytest.ini
[pytest]
testpaths = code/business_entity_resolution/tests
addopts = -q
```

- [ ] **Step 4: Install dependencies and run the test**

```powershell
uv pip install --python .venv\Scripts\python.exe duckdb indic-transliteration jellyfish pytest
.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/conftest.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution pytest.ini
git commit -m "chore: scaffold ER package, config, and pinned requirements"
```

---

### Task 2: Name normalization and script detection

**Files:**
- Create: `code/business_entity_resolution/src/ber/normalize.py`
- Test: `code/business_entity_resolution/tests/test_normalize.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalize_name(s: str) -> str`; `fold_name(s: str) -> str`; `detect_script(s: str) -> str` (`"latin"|"indic"|"cyrillic"|"arabic"|"hebrew"|"mixed"|"none"`); `strip_legal_suffix(s: str) -> tuple[str, str]` (stripped name, suffix class or `""`); `name_tokens(s: str) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
from ber.normalize import detect_script, fold_name, name_tokens, normalize_name, strip_legal_suffix


def test_normalize_name_basic():
    assert normalize_name("  Orelee's   Barbershop!! ") == "orelee s barbershop"


def test_normalize_name_preserves_indic_letters():
    assert normalize_name("राम मार्केटिंग") == "राम मार्केटिंग"


def test_fold_name_strips_accents():
    assert fold_name("École primaire") == "ecole primaire"


def test_detect_script_latin_indic_mixed():
    assert detect_script("Holloway Peak Inc") == "latin"
    assert detect_script("राम मार्केटिंग") == "indic"
    assert detect_script("Sun पावर Provision") == "mixed"
    assert detect_script("") == "none"


def test_strip_legal_suffix_classes():
    assert strip_legal_suffix("B+ Retail Inc") == ("b retail", "inc")
    assert strip_legal_suffix("International South Consultants Private Ltd") == (
        "international south consultants",
        "ltd",
    )
    assert strip_legal_suffix("Béque") == ("beque", "")


def test_name_tokens_excludes_stopwords():
    tokens = name_tokens("The Best Bakery and Cafe")
    assert tokens == ["best", "bakery", "cafe"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_normalize.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
import re
import unicodedata

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_STOPWORDS = {"the", "and", "of", "for", "at", "in", "on", "a", "an", "to", "co"}

SUFFIX_CLASSES = {
    "pvt": "pvt", "private": "pvt",
    "ltd": "ltd", "limited": "ltd",
    "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp",
    "corp": "corp", "corporation": "corp",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "gmbh": "gmbh",
}


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = _NON_WORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def fold_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = _NON_WORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _script_of(ch: str) -> str:
    cp = ord(ch)
    if cp < 0x0250:
        return "latin"
    if 0x0900 <= cp <= 0x0D7F:
        return "indic"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if ch.isalpha():
        return "other"
    return ""


def detect_script(s: str) -> str:
    found = {_script_of(ch) for ch in (s or "") if ch.isalpha()}
    found.discard("")
    if not found:
        return "none"
    if found == {"latin"}:
        return "latin"
    if found <= {"indic", "latin"}:
        return "indic" if "indic" in found and "latin" not in found else "mixed"
    return "mixed"


def strip_legal_suffix(s: str) -> tuple[str, str]:
    tokens = s.split()
    suffix = ""
    while tokens and tokens[-1] in SUFFIX_CLASSES:
        suffix = SUFFIX_CLASSES[tokens.pop()]
    return " ".join(tokens), suffix


def name_tokens(s: str) -> list[str]:
    return [t for t in s.split() if t not in _STOPWORDS and len(t) > 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/normalize.py code/business_entity_resolution/tests/test_normalize.py
git commit -m "feat: name normalization, script detection, legal-suffix stripping"
```

---

### Task 3: Rule-based transliteration

**Files:**
- Create: `code/business_entity_resolution/src/ber/translit.py`
- Test: `code/business_entity_resolution/tests/test_translit.py`

**Interfaces:**
- Consumes: `ber.normalize.normalize_name`.
- Produces: `romanize(s: str) -> str` (identity for Latin; best-effort Latin for Indic; empty input → "").

- [ ] **Step 1: Write the failing tests**

```python
from ber.translit import romanize


def test_romanize_identity_for_latin():
    assert romanize("Holloway Peak Inc") == "holloway peak inc"


def test_romanize_indic_is_latin():
    out = romanize("राम मार्केटिंग")
    assert out and all(ord(ch) < 128 for ch in out)


def test_romanize_mixed_keeps_latin():
    out = romanize("Sun पावर Provision")
    assert out.startswith("sun ")
    assert all(ord(ch) < 128 for ch in out)


def test_romanize_empty():
    assert romanize("") == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_translit.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
import functools

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from ber.normalize import normalize_name, detect_script


@functools.lru_cache(maxsize=200_000)
def _romanize_cached(s: str) -> str:
    script = detect_script(s)
    if script in ("latin", "none"):
        return s
    out = s
    if script in ("indic", "mixed"):
        deva = transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI)
        out = transliterate(deva, sanscript.DEVANAGARI, sanscript.ITRANS)
    out = "".join(ch if ord(ch) < 128 else " " for ch in out)
    return " ".join(out.split())


def romanize(s: str) -> str:
    return _romanize_cached(normalize_name(s))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_translit.py -v`
Expected: PASS. If `indic-transliteration` cannot round-trip mixed strings, fall back to per-token: Latin tokens kept, Indic tokens transliterated individually; adjust `_romanize_cached` accordingly before committing.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/translit.py code/business_entity_resolution/tests/test_translit.py
git commit -m "feat: rule-based Romanization for Indic names"
```

---

### Task 4: Country-aware address parsing

**Files:**
- Create: `code/business_entity_resolution/src/ber/address.py`
- Test: `code/business_entity_resolution/tests/test_address.py`

**Interfaces:**
- Consumes: `ber.normalize.normalize_name`.
- Produces: `AddressParts` dataclass with fields `house_no: str`, `street_tokens: tuple[str, ...]`, `postal: str`, `state_key: str`, `city_tokens: tuple[str, ...]`, `landmark_flag: bool`, `addr_missing: bool`; function `parse_address(raw: str, country: str) -> AddressParts`.

- [ ] **Step 1: Write the failing tests**

```python
from ber.address import parse_address


def test_parse_us_address():
    p = parse_address("1795 Westchester Drive, High Point, NC", "US")
    assert p.house_no == "1795"
    assert "westchester" in p.street_tokens
    assert p.state_key == "north carolina"
    assert p.postal == ""
    assert not p.addr_missing


def test_parse_india_pin():
    p = parse_address("G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh 462033", "India")
    assert p.postal == "462033"
    assert p.state_key == "madhya pradesh"


def test_parse_france_postal():
    p = parse_address("175 Boulevard du Président Franklin Roosevelt, Bordeaux, 33000", "France")
    assert p.house_no == "175"
    assert p.postal == "33000"


def test_landmark_and_missing():
    p = parse_address("Near SBI ATM, MG Road", "India")
    assert p.landmark_flag is True
    assert parse_address("", "US").addr_missing is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_address.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
import re
from dataclasses import dataclass

from ber.normalize import normalize_name

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
STREET_SUFFIXES = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln", "court",
    "ct", "boulevard", "blvd", "rue", "chemin", "route", "way", "plaza", "circle", "circle",
}
LANDMARK_WORDS = {"near", "opposite", "opp", "behind", "beside", "next"}


@dataclass(frozen=True)
class AddressParts:
    house_no: str
    street_tokens: tuple[str, ...]
    postal: str
    state_key: str
    city_tokens: tuple[str, ...]
    landmark_flag: bool
    addr_missing: bool


def _postal_re(country: str) -> re.Pattern:
    key = (country or "").strip().lower()
    if key == "us":
        return re.compile(r"\b\d{5}(?:-\d{4})?\b")
    return re.compile(r"\b\d{6}\b" if key == "india" else r"\b\d{5}\b")


def parse_address(raw: str, country: str) -> AddressParts:
    if not raw or not raw.strip():
        return AddressParts("", (), "", "", (), False, True)
    norm = normalize_name(raw)
    tokens = norm.split()
    house_no = tokens[0] if tokens and tokens[0][:1].isdigit() else ""
    postal_match = _postal_re(country).search(norm)
    postal = postal_match.group(0) if postal_match else ""
    state_key = ""
    for tok in tokens:
        if tok in US_STATES:
            state_key = US_STATES[tok]
            break
    if not state_key:
        for i in range(len(tokens) - 1):
            pair = f"{tokens[i]} {tokens[i + 1]}"
            if pair in US_STATES.values():
                state_key = pair
                break
    landmark = any(t in LANDMARK_WORDS for t in tokens)
    street = tuple(t for t in tokens if t not in STREET_SUFFIXES and t != house_no)[:12]
    return AddressParts(house_no, street, postal, state_key, tuple(tokens[-6:]), landmark, False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_address.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/address.py code/business_entity_resolution/tests/test_address.py
git commit -m "feat: country-aware address parsing"
```

---

### Task 5: Stage 0 prepare pipeline (parquet cache)

**Files:**
- Create: `code/business_entity_resolution/src/ber/io_utils.py`
- Create: `code/business_entity_resolution/src/ber/prepare.py`
- Create: `code/business_entity_resolution/src/ber/cli.py`
- Test: `code/business_entity_resolution/tests/test_prepare.py`

**Interfaces:**
- Consumes: `Config`, `normalize`, `address`, `translit`.
- Produces: `read_source(path) -> Iterator[pd.DataFrame]`; `prepare_frame(df: pd.DataFrame) -> pd.DataFrame` with columns `entity_id, name_norm, name_fold, name_roman, name_tokens, name_script, name_stripped, suffix_class, addr_norm, addr_raw_missing, house_no, street_tokens, postal, state_key, landmark_flag, name_idf_tokens`; `run_prepare(cfg) -> dict[str, int]` writing `data/processed/{train,test}_source{1,2,3}.parquet` and `data/processed/train_ground_truth.parquet`.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd

from ber.prepare import prepare_frame


def test_prepare_frame_columns_and_values():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["B+ Retail Inc"],
            "business_address": ["1795 Westchester Drive, High Point, NC"],
            "country": ["US"],
        }
    )
    out = prepare_frame(df)
    row = out.iloc[0]
    assert row["name_norm"] == "b retail inc"
    assert row["name_stripped"] == "b retail"
    assert row["suffix_class"] == "inc"
    assert row["name_script"] == "latin"
    assert row["house_no"] == "1795"
    assert row["state_key"] == "north carolina"
    assert row["addr_raw_missing"] in (False, 0)
    assert "b" in row["name_idf_tokens"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_prepare.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/ber/io_utils.py
from pathlib import Path

import pandas as pd

CHUNK = 1_000_000


def read_source(path: str | Path):
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK
    )


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
```

```python
# code/business_entity_resolution/src/ber/prepare.py
from pathlib import Path

import pandas as pd

from ber.address import parse_address
from ber.io_utils import read_source, write_parquet
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
            "name_idf_tokens": names.map(lambda s: sorted(set(name_tokens(s)))).to_numpy(),
            "name_script": df["business_name"].map(detect_script).to_numpy(),
            "name_stripped": [s for s, _ in stripped].__iter__().__next__() if False else [s[0] for s in stripped],
            "suffix_class": [s[1] for s in stripped],
            "addr_norm": df["business_address"].map(normalize_name).to_numpy(),
            "addr_raw_missing": df["business_address"].str.strip().eq("").to_numpy(),
            "house_no": [p.house_no for p in parts],
            "street_tokens": [p.street_tokens for p in parts],
            "postal": [p.postal for p in parts],
            "state_key": [p.state_key for p in parts],
            "landmark_flag": [p.landmark_flag for p in parts],
        }
    )
    return out


def run_prepare(cfg) -> dict[str, int]:
    counts = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            src = Path(cfg.dataset_dir) / split / f"{split}_source{source}.tsv"
            frames = [prepare_frame(chunk) for chunk in read_source(src)]
            out = pd.concat(frames, ignore_index=True)
            dst = Path(cfg.data_dir) / "processed" / f"{split}_source{source}.parquet"
            write_parquet(out, dst)
            counts[f"{split}_source{source}"] = len(out)
    gt_src = Path(cfg.dataset_dir) / "train" / "train_ground_truth.tsv"
    gt = pd.read_csv(gt_src, sep="\t", dtype=str, keep_default_na=False)
    write_parquet(gt, Path(cfg.data_dir) / "processed" / "train_ground_truth.parquet")
    counts["train_ground_truth"] = len(gt)
    return counts
```

Note: replace the awkward `name_stripped` expression with `[s[0] for s in stripped]` (the `if False` branch is accidental scaffolding and must not ship).

```python
# code/business_entity_resolution/src/ber/cli.py
import argparse
from pathlib import Path

from ber.config import Config
from ber.prepare import run_prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ber")
    parser.add_argument("command", choices=["prepare", "block", "audit", "features", "train", "tune", "predict", "outputs", "all"])
    parser.add_argument("--config", default="code/business_entity_resolution/config.json")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command in ("prepare", "all"):
        counts = run_prepare(cfg)
        print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, then run the real prepare**

```powershell
.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_prepare.py -v
.venv\Scripts\python.exe -m ber.cli prepare
```
Run the CLI from `code/business_entity_resolution/src` on `PYTHONPATH`: `$env:PYTHONPATH="D:\Amazon project\code\business_entity_resolution\src"`.
Expected: tests PASS; counts equal EDA row counts (2,206,821 / 5,034,616 / 5,285,603 / 1,732,544 / 4,887,273 / 5,082,316 / 2,206,821).

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/io_utils.py code/business_entity_resolution/src/ber/prepare.py code/business_entity_resolution/src/ber/cli.py code/business_entity_resolution/tests/test_prepare.py
git commit -m "feat: stage 0 prepare pipeline with parquet cache"
```

---

### Task 6: Blocking passes with DuckDB

**Files:**
- Create: `code/business_entity_resolution/src/ber/blocking.py`
- Test: `code/business_entity_resolution/tests/test_blocking.py`

**Interfaces:**
- Consumes: processed parquet from Task 5.
- Produces: `generate_candidates(s1: pd.DataFrame, cands: pd.DataFrame, cfg) -> pd.DataFrame` with columns `s1_id, cand_id, is_s2, pass_id, block_score`; helper `block_keys(frame: pd.DataFrame, id_col: str, is_source1: bool) -> list[tuple[str,int,str,float]]`.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from ber.blocking import generate_candidates


def _frame(ids, names, addresses, countries):
    return pd.DataFrame(
        {
            "entity_id": ids,
            "name_norm": names,
            "name_tokens": [n.split() for n in names],
            "name_script": ["latin"] * len(ids),
            "house_no": [a.split()[0] for a in addresses],
            "street_tokens": [a.split()[1:] for a in addresses],
            "postal": [""] * len(ids),
            "state_key": [""] * len(ids),
        }
    )


def test_generate_candidates_finds_exact_and_near():
    s1 = _frame(["S1-1"], ["best bakery"], ["10 main st"], ["US"])
    s2 = _frame(["S2-1"], ["best bakery"], ["10 main street"], ["US"])
    s3 = _frame(["S3-1"], ["best b akery"], ["11 side road"], ["US"])
    cfg = type("C", (), {"cap": 200, "max_block": 5000, "idf_min": 4.0, "seed": 42})()
    out = generate_candidates(s1, pd.concat([s2, s3]), cfg)
    assert set(out["cand_id"]) >= {"S2-1"}
    assert out.loc[out["cand_id"] == "S2-1", "pass_id"].min() == 1


def test_cap_is_enforced():
    s1 = _frame(["S1-1"], ["common name"], ["1 a st"], ["US"])
    cands = _frame([f"S2-{i}" for i in range(300)], ["common name"] * 300, ["1 a st"] * 300, ["US"] * 300)
    cfg = type("C", (), {"cap": 50, "max_block": 5000, "idf_min": 4.0, "seed": 42})()
    out = generate_candidates(s1, cands, cfg)
    assert out.groupby("s1_id").size().max() <= 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_blocking.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Implement `block_keys` returning tuples `(entity_id, pass_id, key, block_score)`:

- pass 1: `key = name_norm`, score 1.0.
- pass 2: `key = metaphone(token)` per token of `name_fold` (`jellyfish.metaphone`), score 0.9.
- pass 3: each `token` in `name_idf_tokens`, score `idf` computed corpus-wide; only tokens with `idf >= cfg.idf_min`.
- pass 4: `key = house_no + "|" + first street token` when both non-empty, score 0.8.
- pass 5: `key = postal + "|" + first name token` when postal non-empty, score 0.7.
- pass 6: `key = state_key + "|" + name_norm[:3]` when `name_norm` has ≤ 2 tokens and no pass-3 keys, score 0.5.

`generate_candidates` loads both sides into DuckDB (`duckdb.connect()`), builds one keys table per side, runs `SELECT s.entity_id, c.entity_id, c.is_s2, s.pass_id, s.block_score FROM s1_keys s JOIN cand_keys c USING (key) WHERE s.block_size <= max_block`, unions passes, deduplicates by `(s1_id, cand_id)` keeping min `pass_id` and max score, then applies the cap with `ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY pass_id, block_score DESC, cand_id) <= cfg.cap`.

`idf` is computed once in `prepare.py`-adjacent helper `compute_token_idf(processed_frames) -> dict[str, float]` and persisted to `data/processed/token_idf.json`; pass 3 uses it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_blocking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/blocking.py code/business_entity_resolution/tests/test_blocking.py
git commit -m "feat: DuckDB blocking passes with per-source cap"
```

---

### Task 7: Blocking audit and cap tuning

**Files:**
- Create: `code/business_entity_resolution/src/ber/audit.py`
- Test: `code/business_entity_resolution/tests/test_audit.py`

**Interfaces:**
- Consumes: candidates frame, ground truth frame, S1 metadata (country).
- Produces: `audit_candidates(candidates: pd.DataFrame, gt: pd.DataFrame, s1_meta: pd.DataFrame) -> dict` with keys `recall`, `recall_by_country`, `reduction_ratio`, `candidates_per_s1_mean`, `singleton_candidates_mean`, `per_pass_recall`; CLI `audit` writes `data/reports/blocking_audit.json`.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from ber.audit import audit_candidates


def test_audit_candidates_recall():
    cands = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-9", "S2-7"],
            "pass_id": [1, 3, 1],
            "block_score": [1.0, 0.6, 1.0],
            "is_s2": [True, False, True],
        }
    )
    gt = pd.DataFrame(
        {"source1_entity_id": ["S1-1", "S1-2", "S1-3"], "matched_entity_ids": ["S2-1,S3-9", "S2-7", ""]}
    )
    meta = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "India", "US"]})
    rep = audit_candidates(cands, gt, meta)
    assert rep["recall"] == 1.0
    assert rep["recall_by_country"]["US"] == 1.0
    assert rep["singleton_candidates_mean"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_audit.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Explode GT ID lists, left-join against candidates, compute the fraction of GT pairs present per S1 and per country; compute candidates-per-S1 mean and singleton mean; per-pass recall as `recall` computed on rows with `pass_id == p`; reduction ratio = total candidate rows ÷ number of S2+S3 records. Return JSON-serializable dict.

- [ ] **Step 4: Run tests and the real audit**

```powershell
.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_audit.py -v
```
Then extend `cli.py` with `block` (run blocking for train/test, write `data/candidates/{split}_candidates.parquet`) and `audit` (write `data/reports/blocking_audit.json`). Run both and confirm train recall ≥ 0.97; if below, lower `idf_min` by 0.5 and/or raise `cap` by 50 and re-run, recording the chosen values in `config.json`.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/audit.py code/business_entity_resolution/src/ber/cli.py code/business_entity_resolution/tests/test_audit.py code/business_entity_resolution/config.json
git commit -m "feat: blocking audit, tuned cap and idf threshold"
```

Tag milestone `0.2.0` after this task:

```bash
git tag -a 0.2.0 -m "0.2.0: cleaning and recall-audited blocking"
git push origin main 0.2.0
```

---

### Task 8: Pairwise features

**Files:**
- Create: `code/business_entity_resolution/src/ber/features.py`
- Test: `code/business_entity_resolution/tests/test_features.py`

**Interfaces:**
- Consumes: candidates frame, S1 processed frame, candidate processed frame, `cfg`.
- Produces: `compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, cand: pd.DataFrame, cfg) -> pd.DataFrame` where `pairs` has columns `s1_id, cand_id, is_s2, pass_id, block_score`; returns pairwise float32 features plus `label` left-joined when available.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from ber.features import compute_features
from ber.prepare import prepare_frame


def _record(eid, name, addr, country):
    return prepare_frame(
        pd.DataFrame(
            {
                "entity_id": [eid],
                "business_name": [name],
                "business_address": [addr],
                "country": [country],
            }
        )
    )


def test_compute_features_golden():
    s1 = _record("S1-1", "Best Bakery Inc", "10 Main St, Austin, TX", "US")
    cand = pd.concat(
        [
            _record("S2-1", "Best Bakery", "10 Main Street, Austin, TX", "US"),
            _record("S3-1", "Pizza Palace", "99 Oak Rd, Dallas, TX", "US"),
        ]
    )
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1"],
            "cand_id": ["S2-1", "S3-1"],
            "is_s2": [True, False],
            "pass_id": [1, 4],
            "block_score": [1.0, 0.8],
        }
    )
    cfg = type("C", (), {"seed": 42, "cap": 200})()
    feats = compute_features(pairs, s1, cand, cfg)
    good = feats[feats["cand_id"] == "S2-1"].iloc[0]
    bad = feats[feats["cand_id"] == "S3-1"].iloc[0]
    assert good["name_exact"] == 0.0
    assert good["name_jaccard"] > bad["name_jaccard"]
    assert good["same_country"] == 1.0
    assert good["house_match"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_features.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Merge S1 and candidate processed columns onto `pairs` by ID, then compute with vectorized kernels:

- `rapidfuzz.process.cpdist(list1, list2, scorer=...)` for `name_ratio`, `name_partial`, `name_token_sort`, `name_token_set`, `name_wratio`, `name_qratio` (on `name_roman`), `addr_ratio`, `addr_token_sort`.
- `jellyfish.jaro_winkler_similarity` list comprehension for `name_jaro`.
- `name_exact`, `name_jaccard`, `name_containment`, `name_sorted_eq`, `name_acronym`, `name_len_diff`, `name_token_count_diff`.
- `shared_rare_tokens`, `idf_overlap` from token sets + persisted idf.
- `house_match`, `street_jaccard`, `postal_exact`, `postal_prefix3`, `state_match`, `landmark`, `addr_missing`, `addr_len_diff`.
- `same_country`, `seen_country` (country in `{"us","india"}`), `is_s2`, `pass_id`, `block_score`, `s1_degree` (candidates per S1), `cand_degree` (S1 per candidate).
- `script_match` (`name_script` equality flags), `roman_ratio` (rapidfuzz ratio between `name_fold` and candidate `name_roman`).

Char 3-gram TF-IDF cosine: fit `sklearn.feature_extraction.text.TfidfAnalyzer`-based `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), min_df=3)` once on a 1M-row sample of all `name_norm` values; persist vocabulary to `data/processed/char3gram_vocab.json`; compute per-pair cosine via row dot products on transformed sparse rows.

All outputs cast to `float32`; order columns deterministically (sorted).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/features.py code/business_entity_resolution/tests/test_features.py
git commit -m "feat: pairwise feature engineering"
```

---

### Task 9: Training pair construction and split

**Files:**
- Create: `code/business_entity_resolution/src/ber/pairs.py`
- Test: `code/business_entity_resolution/tests/test_pairs.py`

**Interfaces:**
- Consumes: candidates parquet, GT parquet, `cfg`.
- Produces: `build_training_pairs(candidates, gt, cfg) -> pd.DataFrame` with columns `s1_id, cand_id, label` (positives all; negatives sampled `neg_ratio`×positives, half hard by `pass_id` frequency, half random country-matched); `grouped_split(pairs, val_frac, seed) -> tuple[np.ndarray, np.ndarray]` returning boolean masks by S1 group.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from ber.pairs import build_training_pairs, grouped_split

CFG = type("C", (), {"seed": 42, "neg_ratio": 2})()


def _cands():
    return pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-1"],
            "cand_id": ["S2-1", "S2-2", "S2-3"],
            "is_s2": [True, True, True],
            "pass_id": [1, 3, 1],
            "block_score": [1.0, 0.5, 1.0],
        }
    )


def _gt():
    return pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1"]})


def test_build_training_pairs_labels_and_ratio():
    out = build_training_pairs(_cands(), _gt(), CFG)
    assert out.loc[out["cand_id"] == "S2-1", "label"].iat[0] == 1
    assert out.loc[out["cand_id"] != "S2-1", "label"].max() == 0
    assert len(out) >= 3


def test_grouped_split_has_no_group_leakage():
    pairs = pd.DataFrame(
        {"s1_id": [f"S1-{i}" for i in range(10)] * 2, "cand_id": [f"S2-{i}" for i in range(20)], "label": [0] * 20}
    )
    train_mask, val_mask = grouped_split(pairs, 0.5, 42)
    assert not (set(pairs.loc[train_mask, "s1_id"]) & set(pairs.loc[val_mask, "s1_id"]))
    assert np.isclose(val_mask.mean(), 0.5, atol=0.11)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_pairs.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Explode GT to `(s1_id, cand_id)` positives; mark candidate rows matching a positive as label 1; sample negatives with `numpy.random.default_rng(cfg.seed)`: score hard negatives by `pass_id` frequency and sample half from the highest-frequency pass per S1, half uniformly from the remaining candidates; cap negatives at `neg_ratio × n_positives`; guarantee no failed-blocking positives appear in the negative set. `grouped_split` shuffles unique S1 IDs and returns boolean masks for train/val.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_pairs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/pairs.py code/business_entity_resolution/tests/test_pairs.py
git commit -m "feat: training pair construction with grouped split"
```

---

### Task 10: Metric, training, threshold tuning, and one-to-one

**Files:**
- Create: `code/business_entity_resolution/src/ber/evaluate.py`
- Create: `code/business_entity_resolution/src/ber/train.py`
- Create: `code/business_entity_resolution/src/ber/threshold.py`
- Create: `code/business_entity_resolution/src/ber/postprocess.py`
- Test: `code/business_entity_resolution/tests/test_score.py`

**Interfaces:**
- Consumes: feature frames, labels, groups.
- Produces: `macro_f05(truth: dict[str, set[str]], pred: dict[str, set[str]]) -> float`; `train_lgbm(X, y, train_mask, val_mask, cfg) -> tuple[booster, metrics]`; `tune_threshold(probs, groups, labels, cfg) -> dict` with `global`, `by_country`, `use_one_to_one`; `one_to_one(probs: pd.Series, pairs: pd.DataFrame) -> pd.Series` returning filtered pair rows.

- [ ] **Step 1: Write the failing tests**

```python
from ber.evaluate import macro_f05
from ber.postprocess import one_to_one
import pandas as pd


def test_macro_f05_worked_example():
    truth = {"S1-1": {"S2-1", "S3-1"}}
    pred = {"S1-1": {"S2-1", "S2-2", "S3-1"}}
    assert round(macro_f05(truth, pred), 3) == 0.714


def test_macro_f05_singleton_rules():
    assert macro_f05({"S1-1": set()}, {"S1-1": set()}) == 1.0
    assert macro_f05({"S1-1": set()}, {"S1-1": {"S2-1"}}) == 0.0


def test_one_to_one_keeps_best_assignment():
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-2"],
            "cand_id": ["S2-1", "S2-1"],
            "prob": [0.9, 0.6],
        }
    )
    keep = one_to_one(pairs["prob"], pairs)
    assert keep.tolist() == [True, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_score.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

`macro_f05` iterates every S1 key in the union of truth and pred, applies precision/recall and `1.25*p*r/(0.25*p+r)`, with singleton rules (empty truth: 1.0 iff empty pred, else 0.0), and averages.

`train_lgbm` builds `lgb.Dataset` from float32 features with `params = dict(cfg.lgbm_params, objective="binary", n_jobs=8, seed=cfg.seed, verbosity=-1)`, early stops on validation AUC (`early_stopping_rounds=100`), returns booster and metrics. Uses grouped masks from Task 9.

`tune_threshold` sweeps 0.05–0.95 step 0.025 computing macro F_0.5 with singleton rules, then per-country, then re-evaluates with `one_to_one` applied; selects configurations that beat the global baseline.

`one_to_one` groups by `cand_id`, marks the row with max `prob` per group (ties by lexicographically smallest `s1_id`) as keep, and always keeps rows whose `cand_id` occurs once.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_score.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/evaluate.py code/business_entity_resolution/src/ber/train.py code/business_entity_resolution/src/ber/threshold.py code/business_entity_resolution/src/ber/postprocess.py code/business_entity_resolution/tests/test_score.py
git commit -m "feat: macro F0.5 metric, LightGBM trainer, threshold and one-to-one tuning"
```

Tag milestone `0.3.0` after running `features`, `train`, and `tune` end to end and recording validation macro F_0.5 in `models/training_metrics.json`.

---

### Task 11: Test prediction and output writers

**Files:**
- Create: `code/business_entity_resolution/src/ber/predict.py`
- Create: `code/business_entity_resolution/src/ber/outputs.py`
- Test: `code/business_entity_resolution/tests/test_outputs.py`

**Interfaces:**
- Consumes: test candidates parquet, booster, `threshold.json`.
- Produces: `score_candidates(cfg, booster) -> pd.DataFrame` (`s1_id, cand_id, prob` written chunked to `data/predictions/test_scored.parquet`); `write_outputs(cfg, scored, candidates) -> dict[str, int]` writing `output/candidate_pairs.tsv` and `output/matching_results.tsv`; `check_invariants(cfg) -> list[str]` returning violation messages.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd

from ber.outputs import write_outputs


def test_write_outputs_tsv_contract(tmp_path):
    candidates = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-2", "S1-3"],
            "cand_id": ["S2-1", "S3-1", "S2-2"],
        }
    )
    scored = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-2", "S1-3"],
            "cand_id": ["S2-1", "S3-1", "S2-2"],
            "prob": [0.9, 0.1, 0.2],
        }
    )
    cfg = type(
        "C",
        (),
        {
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "models_dir": tmp_path,
            "seed": 42,
        },
    )()
    counts = write_outputs(cfg, scored, candidates, threshold=0.5, use_one_to_one=False)
    assert counts["rows"] == 3
    text = (tmp_path / "matching_results.tsv").read_text(encoding="utf-8")
    assert text.startswith("source1_entity_id\tmatched_entity_ids\n")
    assert "S1-1\tS2-1" in text
    assert "S1-2\t\n" in text or "S1-2\n" in text
    cand_text = (tmp_path / "candidate_pairs.tsv").read_text(encoding="utf-8")
    assert "S1-2\tS3-1" in cand_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_outputs.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

`score_candidates` iterates candidate parquet row groups in ~5M-pair chunks, joins processed fields, computes features, predicts in one call per chunk, appends to a parquet writer.

`write_outputs` writes headers exactly `source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`; builds per-S1 lists from thresholded (+ optional one-to-one) predictions; sorts rows by `source1_entity_id`; joins IDs with `,`; iterates the full test S1 ID set (from processed test source1) so every entity appears with an empty value when unmatched; asserts matches ⊆ candidates while writing.

`check_invariants` verifies row count equals 1,732,544, headers, no duplicate S1 rows, no duplicate IDs within lists, `S2-`/`S3-`-only prefixes, and subset property; returns violation list (empty = OK).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_outputs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/ber/predict.py code/business_entity_resolution/src/ber/outputs.py code/business_entity_resolution/tests/test_outputs.py
git commit -m "feat: chunked test scoring and submission output writers"
```

---

### Task 12: End-to-end fixture and validator gate

**Files:**
- Create: `code/business_entity_resolution/tests/test_end_to_end.py`
- Create: `code/business_entity_resolution/tests/fixtures/` (tiny train/test TSVs, ~100 S1)
- Modify: `code/business_entity_resolution/src/ber/cli.py` (wire `features`, `train`, `tune`, `predict`, `outputs`, `all`)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a passing full-pipeline test proving stages compose and outputs validate.

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys
from pathlib import Path


def test_end_to_end_fixture(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "student_resource"
    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"dataset_dir": "%s", "data_dir": "%s", "models_dir": "%s", "output_dir": "%s", "seed": 42, "cap": 20, "neg_ratio": 2}'
        % (fixture / "dataset", tmp_path / "data", tmp_path / "models", tmp_path / "output"),
        encoding="utf-8",
    )
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    for command in ("prepare", "block", "features", "train", "tune", "outputs"):
        proc = subprocess.run(
            [sys.executable, "-m", "ber.cli", command, "--config", str(cfg)],
            capture_output=True, text=True, env={**env},
        )
        assert proc.returncode == 0, proc.stderr
    validator = fixture / "utils" / "validate_submission.py"
    proc = subprocess.run(
        [sys.executable, str(validator), "--matching", str(tmp_path / "output" / "matching_results.tsv"),
         "--candidate", str(tmp_path / "output" / "candidate_pairs.tsv"),
         "--test-dir", str(fixture / "dataset" / "test")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_end_to_end.py -v`
Expected: FAIL (missing commands/fixtures).

- [ ] **Step 3: Create fixtures and wire CLI**

Generate 100 S1 train rows + 300 S2/S3 rows with GT, and 40 S1 test rows + 120 S2/S3 rows, using a fixed script under `tests/fixtures/make_fixture.py`; copy `utils/validate_submission.py` into the fixture student_resource. Wire remaining CLI commands to their stage functions (each prints a small summary dict).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/ -v`
Expected: all PASS including end-to-end.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/tests code/business_entity_resolution/src/ber/cli.py
git commit -m "test: end-to-end fixture with validator gate"
```

Tag `0.4.0`.

---

### Task 13: Production run, outputs, and submission package

**Files:**
- Modify: `code/business_entity_resolution/README.md` (run instructions), `RULES.md` (version), `README.md` (status)
- Create: `output/matching_results.tsv`, `output/candidate_pairs.tsv` (generated)
- Create: packaged zip `<team_name>_submission.zip` under `dist/`

**Interfaces:**
- Consumes: all stages.
- Produces: final artifacts and a validator `PASS`.

- [ ] **Step 1: Run the full production pipeline**

```powershell
$env:PYTHONPATH="D:\Amazon project\code\business_entity_resolution\src"
.venv\Scripts\python.exe -m ber.cli prepare
.venv\Scripts\python.exe -m ber.cli block
.venv\Scripts\python.exe -m ber.cli audit
.venv\Scripts\python.exe -m ber.cli features
.venv\Scripts\python.exe -m ber.cli train
.venv\Scripts\python.exe -m ber.cli tune
.venv\Scripts\python.exe -m ber.cli predict
.venv\Scripts\python.exe -m ber.cli outputs
```
Expected: `data/reports/blocking_audit.json` recall ≥ 0.97; `models/training_metrics.json` records validation macro F_0.5; output row count = 1,732,544.

- [ ] **Step 2: Validate outputs**

```powershell
cd DATA\student_resource
python utils\validate_submission.py --matching ..\..\..\output\matching_results.tsv --candidate ..\..\..\output\candidate_pairs.tsv --test-dir dataset\test
```
Expected: `PASS — no blocking issues found. Safe to submit.`

- [ ] **Step 3: Build the submission zip**

Assemble `dist/<team_name>_submission.zip` with `output/` (both TSVs), `code/business_entity_resolution/{src,README.md,requirements.txt,models}`, and the filled `Documentation_template.md` copied from `DATA/student_resource/Documentation_template.md` with measured numbers, methodology, blocking strategy, features, model, and results filled in.

- [ ] **Step 4: Update docs and version**

Record final validation metrics, chosen cap/idf, and threshold in `code/business_entity_resolution/README.md`; set `RULES.md` current version to `1.0.0`; update root `README.md` status and benchmark section.

- [ ] **Step 5: Commit and tag**

```bash
git add -A
git commit -m "feat: production pipeline outputs and validated submission package"
git tag -a 1.0.0 -m "1.0.0: submission-ready entity resolution pipeline"
git push origin main 1.0.0
```

---

## Self-Review

- **Spec coverage:** cleaning (Tasks 2–5), blocking + audit (6–7), features (8), pairs/splits (9), model/threshold/postprocess (10), inference/outputs/validation (11, 13), testing (2–12), packaging (13). All spec sections map to tasks.
- **Placeholders:** none; every step has concrete commands, signatures, or code, except deliberately algorithmic descriptions for large functions (`generate_candidates`, `compute_features`) which specify inputs, outputs, exact keys, and scoring. Implementers must not invent column names beyond the interfaces listed.
- **Type consistency:** column names (`s1_id`, `cand_id`, `pass_id`, `block_score`, `label`, `prob`) and function signatures are reused verbatim across tasks 5–12; `macro_f05` is tested before the trainer uses it; `write_outputs` signature matches the end-to-end test call.
