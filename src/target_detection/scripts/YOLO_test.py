#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
from pathlib import Path

# Load a pretrained YOLO11n model
ROOT = Path(__file__).resolve().parent
weights_path = ROOT / "../ultralytics/runs/detect/train/weights/yolo11n.pt" 
model = YOLO(weights_path)
names = model.names if hasattr(model, 'names') else model.get('names', [])

# 打印纯名称
if isinstance(names, dict):
    for name in names.values():
        print(name)
elif isinstance(names, list):
    for name in names:
        print(name)


