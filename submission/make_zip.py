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

    code_src = ROOT / "code" / "business_entity_resolution"
    shutil.copytree(
        code_src, staging / "code" / "business_entity_resolution",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    shutil.copy2(ROOT / "submission" / "Documentation_template.md", staging / "Documentation_template.md")

    archive = ROOT / "submission" / f"{TEAM}_submission"
    if Path(str(archive) + ".zip").exists():
        Path(str(archive) + ".zip").unlink()
    shutil.make_archive(str(archive), "zip", staging)
    print("wrote", str(archive) + ".zip")


if __name__ == "__main__":
    main()
