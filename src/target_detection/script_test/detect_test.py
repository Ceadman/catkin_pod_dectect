#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
from pathlib import Path

# Load a pretrained YOLO11n model
ROOT = Path(__file__).resolve().parent
weights_path = ROOT / "../model/balloon.pt" 
model = YOLO(weights_path)

# Run inference on with arguments
model.predict("/home/ceadman111/catkin_Attack-target/src/target_detection/test_picture/balloon1.jpg", save=True, imgsz=640, conf=0.5)

