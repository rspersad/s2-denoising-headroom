"""Add --save-pred to evaluate_factorial.py: keep each condition's predictions.json and tile labels.
Usage (Astraea):  python patch_evaluate_factorial.py ~/praxis/scripts/evaluate_factorial.py
Idempotent: refuses to patch twice. Makes a .bak copy first.
"""
import sys, shutil, re
p = sys.argv[1]
s = open(p).read()
if "--save-pred" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak-savepred")

# 1. argument
old_arg = '    ap.add_argument("--device", default="0")\n'
assert old_arg in s, "device argument line not found"
s = s.replace(old_arg, old_arg + '    ap.add_argument("--save-pred", default=None, help="directory to keep per-condition predictions.json and tile labels")\n')

# 2. save_json on the val call
old_val = 'verbose=False, save=False, plots=False, workers=0,\n                                                    project=tmp, name="val", exist_ok=True)'
assert old_val in s, "model.val call not found in expected form"
s = s.replace(old_val, 'verbose=False, save=False, plots=False, workers=0,\n                                                    save_json=bool(args.save_pred),\n                                                    project=tmp, name="val", exist_ok=True)')

# 3. copy out before the temp dir is removed
old_tail = '                                map50 = float(res.box.map50)\n                                map5095 = float(res.box.map)\n                            finally:\n'
assert old_tail in s, "map50 / finally block not found"
new_tail = ('                                map50 = float(res.box.map50)\n                                map5095 = float(res.box.map)\n'
            '                                if args.save_pred:\n'
            '                                    dest = os.path.join(args.save_pred, method, scene, band, f"{rec_sev}_r{rec_r}")\n'
            '                                    os.makedirs(dest, exist_ok=True)\n'
            '                                    pj = os.path.join(tmp, "val", "predictions.json")\n'
            '                                    if os.path.exists(pj): shutil.copy(pj, os.path.join(dest, "predictions.json"))\n'
            '                                    else: print(f"[warn] no predictions.json for {method}/{scene}/{band}/{rec_sev}/r{rec_r}", flush=True)\n'
            '                                    lab = os.path.join(tmp, "labels", "val")\n'
            '                                    if os.path.isdir(lab): shutil.copytree(lab, os.path.join(dest, "labels"), dirs_exist_ok=True)\n'
            '                                    imgs = os.path.join(tmp, "images", "val")\n'
            '                                    with open(os.path.join(dest, "tiles.txt"), "w") as tf:\n'
            '                                        tf.write("\\n".join(sorted(os.listdir(imgs))) + "\\n")\n'
            '                            finally:\n')
s = s.replace(old_tail, new_tail)
open(p, "w").write(s)
print("patched", p, "(backup at", p + ".bak-savepred)")
