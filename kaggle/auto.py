"""One-command driver: update code dataset, push a search kernel, wait, fetch, collect, validate.

Usage:
    python kaggle/auto.py --plan auto --update-code
    python kaggle/auto.py --plan auto --no-wait
    python kaggle/auto.py --validate-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "output"
STUDENT_CANDIDATES = [
    Path(r"D:\Amazon project\DATA\student_resource"),
    REPO / "DATA" / "student_resource",
]


def find_student_resource():
    for c in STUDENT_CANDIDATES:
        if (c / "utils" / "validate_submission.py").exists() and (c / "dataset" / "test").exists():
            return c
    return None


def run(cmd, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    return r.returncode


def validate():
    student = find_student_resource()
    if student is None:
        print("validator not found; skipping (set paths under DATA/student_resource)")
        return 1
    matching = OUTPUT_DIR / "matching_results.tsv"
    candidate = OUTPUT_DIR / "candidate_pairs.tsv"
    if not matching.exists():
        print("missing", matching)
        return 1
    code = run([
        sys.executable, "utils/validate_submission.py",
        "--matching", str(matching),
        "--candidate", str(candidate),
        "--test-dir", "dataset/test",
    ], cwd=student)
    print("validator exit", code)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="auto")
    ap.add_argument("--update-code", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        sys.exit(validate())

    if args.update_code:
        run([sys.executable, str(REPO / "kaggle" / "push_all.py"), "datasets", "--only", "code"])
    push = [sys.executable, str(REPO / "kaggle" / "push_all.py"), "push", "--plan", args.plan]
    if args.only:
        push += ["--only", *args.only]
    if not args.no_wait:
        push.append("--wait")
    rc = run(push)
    if rc != 0:
        sys.exit(rc)
    run([sys.executable, str(REPO / "kaggle" / "push_all.py"), "fetch", "--collect"])
    sys.exit(validate())


if __name__ == "__main__":
    main()
