#!/usr/bin/env python3
"""Verify local files against MANIFEST.sha256. Exit code 1 if any file is missing or altered."""
import hashlib, os, sys
bad = missing = ok = 0
for line in open("MANIFEST.sha256"):
    digest, path = line.rstrip("\n").split("  ", 1)
    if not os.path.exists(path):
        missing += 1; print("MISSING ", path); continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != digest:
        bad += 1; print("ALTERED ", path)
    else:
        ok += 1
print(f"ok {ok}  altered {bad}  missing {missing}")
sys.exit(1 if (bad or missing) else 0)
