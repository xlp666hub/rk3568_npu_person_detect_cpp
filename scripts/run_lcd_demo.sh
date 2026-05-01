#!/bin/bash

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export DISPLAY=:0
export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/rknn/lib:${LD_LIBRARY_PATH}

cd "$PROJECT_DIR"

pkill -f web_rknn_camera.py || true
pkill -f lcd_rknn_camera.py || true
pkill -f lcd_camera_preview.py || true

./build/lcd_camera_detect_demo /dev/video9 models/yolov5n.rknn
