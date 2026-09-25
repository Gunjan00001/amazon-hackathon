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
    pass_caps: dict = None

    def __post_init__(self):
        object.__setattr__(self, "lgbm_params", self.lgbm_params or {})
        object.__setattr__(self, "pass_caps", self.pass_caps or {})

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = {k: (Path(v) if k.endswith("_dir") else v) for k, v in payload.items()}
        payload.setdefault("root", Path.cwd())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})
