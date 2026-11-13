#!/usr/bin/env python3
"""Download the archive files listed in data/README.md from the Zenodo record and verify them.
Set ZENODO_RECORD to the numeric record id once the record is published."""
import os, sys, json, urllib.request, subprocess
REC = os.environ.get("ZENODO_RECORD", "")
if not REC:
    sys.exit("Set ZENODO_RECORD=<record id> (see data/README.md).")
os.makedirs("data/archive", exist_ok=True)
meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{REC}"))
for f in meta["files"]:
    dest = os.path.join("data/archive", f["key"])
    if os.path.exists(dest):
        print("have", dest); continue
    print("fetching", f["key"], f"{f['size']/1e9:.2f} GB")
    urllib.request.urlretrieve(f["links"]["self"], dest)
subprocess.call([sys.executable, "tools/verify_manifest.py"])
