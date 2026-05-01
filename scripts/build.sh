#!/bin/bash

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

mkdir -p build

COMMON_INC="-Ithird_party/rknn/include"
COMMON_LIB="-Lthird_party/rknn/lib -lrknnrt"
OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
CXXFLAGS="-std=c++11"

echo "Building test_rknn_runtime..."
g++ src/test_rknn_runtime.cpp \
  -o build/test_rknn_runtime \
  ${COMMON_INC} \
  ${COMMON_LIB} \
  ${CXXFLAGS}

echo "Building image_infer_demo..."
g++ src/image_infer_demo.cpp \
  -o build/image_infer_demo \
  ${COMMON_INC} \
  ${COMMON_LIB} \
  ${OPENCV_FLAGS} \
  ${CXXFLAGS}

echo "Building image_detect_demo..."
g++ src/image_detect_demo.cpp \
  -o build/image_detect_demo \
  ${COMMON_INC} \
  ${COMMON_LIB} \
  ${OPENCV_FLAGS} \
  ${CXXFLAGS}

echo "Building lcd_camera_detect_demo..."
g++ src/lcd_camera_detect_demo.cpp \
  -o build/lcd_camera_detect_demo \
  ${COMMON_INC} \
  ${COMMON_LIB} \
  ${OPENCV_FLAGS} \
  ${CXXFLAGS}

echo
echo "Build finished."
ls -lh build/
