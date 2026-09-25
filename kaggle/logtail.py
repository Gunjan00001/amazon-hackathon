import json
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
data = json.load(open(path, encoding="utf-8"))
out = "".join(e.get("data", "") for e in data)
print(out[-n:])
