"""Add --imgsz to train_p2.py (default 640 reproduces tiled_p2). Idempotent; .bak copy first."""
import sys, shutil
p = sys.argv[1]; s = open(p).read()
if "--imgsz" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak-imgsz")
old = 'ap.add_argument("--seed", type=int, default=0, help="training seed (0 = the original runs)")\n'
assert old in s
s = s.replace(old, old + 'ap.add_argument("--imgsz", type=int, default=640, help="training image size (640 = the original runs)")\n')
assert "imgsz=640," in s
s = s.replace("imgsz=640,", "imgsz=args.imgsz,")
open(p, "w").write(s); print("patched", p)
