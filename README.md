# RK3568 NPU Person Detection C++ Demo

本项目基于 RK3568 平台实现边缘端人体检测。系统使用 USB 摄像头采集图像，调用 RKNN Runtime C API 在 RK3568 NPU 上运行 YOLOv5n 模型，并通过 MIPI 屏幕进行本地实时显示。

项目前期使用 Python 完成摄像头采集、RKNN 模型转换、Web 显示和推理链路验证；当前目录为 C++ 主线版本，主要用于板端部署运行。

---

## 1. 功能特性

- 支持 USB UVC 摄像头采集
- 支持 MJPG 1280x720 视频输入
- 基于 RKNN Runtime C API 调用 RK3568 NPU
- 支持 YOLOv5n RKNN 模型推理
- C++ 实现 YOLOv5 输出解析、person 类别筛选和 NMS
- 支持单张图片检测并保存结果图
- 支持摄像头实时检测并显示到 MIPI 屏幕
- 显示 camera_fps、infer_ms、person_count 等运行状态

---

## 2. 硬件平台

- 开发板：LubanCat 2 / RK3568
- 摄像头：USB UVC 摄像头
- 显示屏：野火 MIPI 屏幕
- NPU：RK3568 内置 RKNPU

---

## 3. 软件环境

- Debian 10
- Linux Kernel 4.19.232
- OpenCV 3.2.0
- RKNN Runtime 1.5.0
- RKNPU Driver 0.8.2
- g++ / pkg-config

---

## 4. 项目目录

    rk3568_npu_person_detect_cpp/
    ├── src/
    │   ├── test_rknn_runtime.cpp
    │   ├── image_infer_demo.cpp
    │   ├── image_detect_demo.cpp
    │   └── lcd_camera_detect_demo.cpp
    ├── models/
    │   └── yolov5n.rknn
    ├── assets/
    │   └── test.jpg
    ├── outputs/
    ├── third_party/
    │   └── rknn/
    │       ├── include/rknn_api.h
    │       └── lib/librknnrt.so
    └── scripts/
        ├── build.sh
        ├── run_image_demo.sh
        └── run_lcd_demo.sh

---

## 5. 编译

执行：

    ./scripts/build.sh

编译成功后，会在 build/ 目录下生成：

    test_rknn_runtime
    image_infer_demo
    image_detect_demo
    lcd_camera_detect_demo

---

## 6. 运行 RKNN Runtime 测试

执行：

    export LD_LIBRARY_PATH=$(pwd)/third_party/rknn/lib:$LD_LIBRARY_PATH
    ./build/test_rknn_runtime models/yolov5n.rknn

正常输出应包含：

    rknn_init success
    api version
    driver version
    input num : 1
    output num: 1
    C++ RKNN runtime test OK

该程序用于验证 C++ 是否能正常调用 RKNN Runtime C API，并查询模型输入输出信息。

---

## 7. 运行单图检测

执行：

    ./scripts/run_image_demo.sh

默认输入图片：

    assets/test.jpg

默认输出图片：

    outputs/cpp_detect_result.jpg

也可以指定输入图片：

    ./scripts/run_image_demo.sh assets/test.jpg

该程序会读取图片，完成 letterbox 预处理，调用 RKNN C API 执行 NPU 推理，然后解析 YOLOv5 输出并绘制人体检测框。

---

## 8. 运行 MIPI 屏幕实时检测

运行前请确认：

- 摄像头节点为 /dev/video9
- MIPI 屏幕桌面环境已启动
- 模型文件 models/yolov5n.rknn 存在

执行：

    ./scripts/run_lcd_demo.sh

程序会打开 USB 摄像头，并在 MIPI 屏幕上显示实时检测结果，包括：

    camera_fps
    infer_ms
    person_count
    STATUS: PERSON DETECTED / NO PERSON

---

## 9. 当前性能表现

在 RK3568 平台上，使用 YOLOv5n FP RKNN 模型：

    摄像头输入：1280x720 MJPG
    模型输入：640x640
    NPU 推理耗时：约 340~380 ms
    摄像头显示帧率：单线程版本约 16 FPS

当前 C++ 版本为单线程实现，推理阶段会阻塞摄像头显示循环。后续可以优化为双线程结构：

    线程 1：摄像头采集与显示
    线程 2：周期性取最新帧做 RKNN 推理

---

## 10. 已完成内容

- C++ 加载 RKNN 模型文件
- rknn_init 初始化 RKNN Runtime
- rknn_query 查询模型输入输出信息
- rknn_inputs_set 设置 NHWC uint8 输入
- rknn_run 执行 NPU 推理
- rknn_outputs_get 获取模型输出
- C++ 解析 YOLOv5 输出
- C++ 实现 person 类别筛选和 NMS
- OpenCV C++ 采集 USB 摄像头
- OpenCV C++ 将检测结果显示到 MIPI 屏幕

---

## 11. 已知问题

### 11.1 VIDIOC_QUERYCTRL: Input/output error

该提示来自 OpenCV/V4L2 查询摄像头控制项时，摄像头不支持部分 control。当前不影响摄像头采集和显示。

### 11.2 dbind-WARNING

该提示来自桌面环境辅助功能 DBus，不影响 OpenCV 窗口显示。

### 11.3 INT8 模型暂未作为主线

前期测试中，INT8 RKNN 模型推理速度更快，但单输出 YOLOv5n 量化后输出置信度异常为 0。因此当前主线使用 FP RKNN 模型。

---

## 12. 后续优化方向

- 将实时检测版本改为采集线程和推理线程分离
- 封装 RKNNDetector 类，拆分当前大文件
- 将 YOLO 后处理单独封装为模块
- 优化 YOLO 后处理，降低 CPU 占用
- 尝试三输出 YOLOv5 结构以改善 INT8 量化效果
- 增加检测截图保存功能
- 增加 C++ Web 远程显示版本

---

## 13. 项目定位

本项目不是单纯的模型调用 Demo，而是一个面向 RK3568 边缘端部署的 C++ 视觉检测系统验证项目。

项目覆盖了：

- 摄像头采集
- RKNN 模型部署
- RK3568 NPU 推理
- YOLOv5 后处理
- MIPI 本地显示
- 性能测试与问题排查

该项目可作为 RK3568 边缘 AI 部署、嵌入式 Linux 应用开发、NPU 推理部署方向的学习和展示项目。

---

## 14. 单线程阻塞问题与双线程优化

### 14.1 问题现象

在初版 C++ 实时检测程序 `lcd_camera_detect_demo.cpp` 中，摄像头采集、RKNN 推理、YOLO 后处理和 MIPI 显示都运行在同一个主循环中。

程序流程如下：

    cap.read(frame)
    rknn_inputs_set()
    rknn_run()
    rknn_outputs_get()
    YOLO postprocess
    cv::imshow()

实测发现，RKNN 推理单次耗时约 340~380 ms。当主循环执行 `rknn_run()` 时，摄像头显示流程会被阻塞，因此 MIPI 屏幕画面会出现周期性卡顿。

### 14.2 原因分析

虽然模型推理由 RK3568 NPU 执行，但主线程仍然需要等待推理完成后才能继续执行后续显示逻辑。也就是说，NPU 加速只降低了模型计算对 CPU 的压力，并不自动解决主循环阻塞问题。

单线程结构下，显示帧率会受到推理耗时影响：

    显示循环等待推理完成
    推理耗时越长，画面卡顿越明显

### 14.3 优化方案

为了解决该问题，新增双线程版本：

    src/lcd_camera_detect_threaded.cpp

优化后的结构如下：

    主线程：
        持续读取 USB 摄像头画面
        读取最近一次检测结果
        绘制检测框和状态信息
        通过 MIPI 屏幕显示

    推理线程：
        周期性复制最新一帧
        调用 RKNN Runtime C API 执行 NPU 推理
        完成 YOLOv5 后处理
        更新最新检测结果

主线程和推理线程之间通过互斥锁保护共享数据，包括最新帧、检测框、推理耗时和人数统计。

### 14.4 优化效果

双线程优化后，RKNN 推理耗时基本不变，仍然约 340~380 ms，但推理过程不再阻塞 MIPI 显示主循环。实测 MIPI 屏幕画面连续性明显改善，周期性卡顿现象基本消失。

该优化说明：在边缘 AI 部署中，NPU 只负责加速模型推理，摄像头采集、预处理、后处理和显示链路仍需要合理的软件架构设计。对于实时视觉系统，仅关注模型推理速度是不够的，还需要关注数据流和线程模型。
