#!/bin/bash

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/rknn/lib:${LD_LIBRARY_PATH}

cd "$PROJECT_DIR"

IMAGE_PATH=${1:-assets/test.jpg}
MODEL_PATH=${2:-models/yolov5n.rknn}

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: image not found: $IMAGE_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: model not found: $MODEL_PATH"
    exit 1
fi

./build/image_detect_demo "$IMAGE_PATH" "$MODEL_PATH"

echo
echo "Result saved to:"
echo "  outputs/cpp_detect_result.jpg"
