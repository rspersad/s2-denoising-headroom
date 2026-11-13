"""Add --imgsz to evaluate_factorial.py so the frozen detector can be run on the same
640 native tiles at a larger inference size (test-time upscaling). Idempotent; .bak copy first.
Usage (Astraea):  python scripts/vessel/patch_imgsz.py scripts/evaluate_factorial.py
"""
import sys, shutil
p = sys.argv[1]; s = open(p).read()
if "--imgsz" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak-imgsz")
old_arg = '    ap.add_argument("--device", default="0")\n'
assert old_arg in s, "device argument line not found"
s = s.replace(old_arg, old_arg + '    ap.add_argument("--imgsz", type=int, default=TILE, help="inference size; tiles stay 640 native and are resized by Ultralytics val (default TILE)")\n')
old_val = 'res = model.val(data=data_yaml, imgsz=TILE,'
assert old_val in s, "model.val imgsz=TILE not found"
s = s.replace(old_val, 'res = model.val(data=data_yaml, imgsz=args.imgsz,')
open(p, "w").write(s)
print("patched", p, "(backup at", p + ".bak-imgsz)")
