#!/usr/bin/env python3
"""
train_p2.py — Train a YOLOv9 + P2 small-object head on the tiled S2-SHIPS dataset.

Builds the P2 architecture from a yaml beside this script and transfers the compatible
(backbone) weights from a pretrained checkpoint; the new P2 head and reshaped head
layers initialize fresh. Expect an Ultralytics "Transferred X/Y items" message —
partial transfer of the backbone is intended.

Defaults reproduce the tiled_p2 run (yolov9m-p2.yaml, yolov9m.pt, batch 16) exactly.
The sensitivity arm (Section 3.9.7) uses --model yolov9c-p2.yaml --pretrained yolov9c.pt
--name tiled_p2_c with every other setting unchanged.

If you hit CUDA out-of-memory: the P2 head adds a 160x160 (stride-4) feature map at
imgsz 640, which is memory-heavy. Drop --batch to 8 (or 4) and record the change.
"""
import argparse, os
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov9m-p2.yaml", help="architecture yaml beside this script")
ap.add_argument("--pretrained", default="yolov9m.pt", help="checkpoint to transfer backbone weights from")
ap.add_argument("--name", default="tiled_p2", help="run name under --project")
ap.add_argument("--project", default="/data/praxis/models/yolov9m-detection")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--seed", type=int, default=0, help="training seed (0 = the original runs)")
ap.add_argument("--imgsz", type=int, default=640, help="training image size (640 = the original runs)")
args = ap.parse_args()

model = YOLO(os.path.join(HERE, args.model)).load(args.pretrained)
model.train(
    data="/data/praxis/detection_tiled/data.yaml",
    epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
    optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=0.0005, cos_lr=True,
    mosaic=1.0, fliplr=0.5,
    project=args.project, name=args.name, device=0, seed=args.seed,
)
