#!/usr/bin/env python3
"""Write MANIFEST.sha256 covering every file under the given roots (default: the repository and data/archive)."""
import hashlib, os, sys
roots = sys.argv[1:] or ["code", "results", "configs", "docs", "data/archive"]
skip = {".git", "__pycache__", ".venv", "node_modules"}
out = []
for root in roots:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip]
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out.append(f"{h.hexdigest()}  {p}")
with open("MANIFEST.sha256", "w") as f:
    f.write("\n".join(out) + "\n")
print(f"{len(out)} files listed in MANIFEST.sha256")
