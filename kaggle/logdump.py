import json
import os
import subprocess
import sys

slug = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
r = subprocess.run(["kaggle", "kernels", "logs", slug], capture_output=True, env=env)
raw = r.stdout.decode("utf-8", "replace")
if not raw.strip():
    print(r.stderr.decode("utf-8", "replace")[:1000])
    sys.exit(r.returncode)
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print(raw[-n:])
    sys.exit(0)
out = "".join(e.get("data", "") for e in data)
print(out[-n:])
