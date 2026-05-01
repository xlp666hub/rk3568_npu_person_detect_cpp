#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

static const int INPUT_SIZE = 640;
static const int NUM_CLASSES = 80;
static const int PERSON_CLASS_ID = 0;

struct LetterboxInfo {
    float scale;
    int pad_left;
    int pad_top;
};

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

struct SharedState {
    std::mutex mutex;
    cv::Mat latest_frame;
    std::vector<Detection> latest_detections;

    int frame_count = 0;
    double camera_fps = 0.0;
    double infer_ms = 0.0;

    std::atomic<bool> running{true};
};

static std::vector<unsigned char> read_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + path);
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("invalid file size: " + path);
    }

    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("failed to read file: " + path);
    }

    return buffer;
}

static cv::Mat letterbox(const cv::Mat& image, int dst_w, int dst_h, LetterboxInfo& info)
{
    int src_w = image.cols;
    int src_h = image.rows;

    float scale = std::min(static_cast<float>(dst_w) / src_w,
                           static_cast<float>(dst_h) / src_h);

    int new_w = static_cast<int>(std::round(src_w * scale));
    int new_h = static_cast<int>(std::round(src_h * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    int pad_w = dst_w - new_w;
    int pad_h = dst_h - new_h;

    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    info.scale = scale;
    info.pad_left = pad_left;
    info.pad_top = pad_top;

    return padded;
}

static float calc_iou(const cv::Rect& a, const cv::Rect& b)
{
    int inter_x1 = std::max(a.x, b.x);
    int inter_y1 = std::max(a.y, b.y);
    int inter_x2 = std::min(a.x + a.width, b.x + b.width);
    int inter_y2 = std::min(a.y + a.height, b.y + b.height);

    int inter_w = std::max(0, inter_x2 - inter_x1);
    int inter_h = std::max(0, inter_y2 - inter_y1);

    float inter_area = static_cast<float>(inter_w * inter_h);
    float union_area = static_cast<float>(a.area() + b.area()) - inter_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

static std::vector<Detection> nms(const std::vector<Detection>& detections, float nms_thresh)
{
    std::vector<Detection> sorted = detections;

    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<Detection> results;
    std::vector<bool> removed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) {
            continue;
        }

        results.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (removed[j]) {
                continue;
            }

            float iou = calc_iou(sorted[i].box, sorted[j].box);
            if (iou > nms_thresh) {
                removed[j] = true;
            }
        }
    }

    return results;
}

static std::vector<Detection> postprocess_yolov5(
    const float* output_data,
    int num_boxes,
    int element_per_box,
    int orig_w,
    int orig_h,
    const LetterboxInfo& lb,
    float conf_thresh,
    float nms_thresh
)
{
    std::vector<Detection> candidates;

    for (int i = 0; i < num_boxes; ++i) {
        const float* pred = output_data + i * element_per_box;

        float cx = pred[0];
        float cy = pred[1];
        float w = pred[2];
        float h = pred[3];
        float obj_conf = pred[4];

        if (obj_conf < conf_thresh) {
            continue;
        }

        int best_class_id = -1;
        float best_class_score = 0.0f;

        for (int c = 0; c < NUM_CLASSES; ++c) {
            float cls_score = pred[5 + c];
            if (cls_score > best_class_score) {
                best_class_score = cls_score;
                best_class_id = c;
            }
        }

        if (best_class_id != PERSON_CLASS_ID) {
            continue;
        }

        float final_score = obj_conf * best_class_score;

        if (final_score < conf_thresh) {
            continue;
        }

        float x1 = (cx - w / 2.0f - lb.pad_left) / lb.scale;
        float y1 = (cy - h / 2.0f - lb.pad_top) / lb.scale;
        float x2 = (cx + w / 2.0f - lb.pad_left) / lb.scale;
        float y2 = (cy + h / 2.0f - lb.pad_top) / lb.scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h - 1)));

        int box_x = static_cast<int>(x1);
        int box_y = static_cast<int>(y1);
        int box_w = static_cast<int>(x2 - x1);
        int box_h = static_cast<int>(y2 - y1);

        if (box_w <= 0 || box_h <= 0) {
            continue;
        }

        Detection det;
        det.box = cv::Rect(box_x, box_y, box_w, box_h);
        det.score = final_score;
        det.class_id = best_class_id;

        candidates.push_back(det);
    }

    return nms(candidates, nms_thresh);
}

static bool run_rknn_inference(
    rknn_context ctx,
    const cv::Mat& frame,
    std::vector<Detection>& detections,
    double& infer_ms,
    float conf_thresh,
    float nms_thresh
)
{
    int orig_w = frame.cols;
    int orig_h = frame.rows;

    LetterboxInfo lb_info;
    cv::Mat input_bgr = letterbox(frame, INPUT_SIZE, INPUT_SIZE, lb_info);

    cv::Mat input_rgb;
    cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);

    if (!input_rgb.isContinuous()) {
        input_rgb = input_rgb.clone();
    }

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = INPUT_SIZE * INPUT_SIZE * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_rgb.data;

    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed, ret=" << ret << std::endl;
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_run failed, ret=" << ret << std::endl;
        return false;
    }

    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed, ret=" << ret << std::endl;
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float* output_data = reinterpret_cast<float*>(outputs[0].buf);

    detections = postprocess_yolov5(
        output_data,
        25200,
        85,
        orig_w,
        orig_h,
        lb_info,
        conf_thresh,
        nms_thresh
    );

    rknn_outputs_release(ctx, 1, outputs);

    return true;
}

static void draw_overlay(
    cv::Mat& frame,
    const std::vector<Detection>& detections,
    double camera_fps,
    double infer_ms,
    int frame_count
)
{
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        char label[64];
        std::snprintf(label, sizeof(label), "person %.2f", det.score);

        int text_y = std::max(20, det.box.y - 8);

        cv::putText(
            frame,
            label,
            cv::Point(det.box.x, text_y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2
        );
    }

    std::string status = detections.empty() ? "STATUS: NO PERSON" : "STATUS: PERSON DETECTED";

    std::vector<std::string> lines;
    lines.push_back("RK3568 C++ RKNN Threaded Detection");
    lines.push_back("camera_fps: " + std::to_string(camera_fps));
    lines.push_back("infer_ms: " + std::to_string(infer_ms));
    lines.push_back("person_count: " + std::to_string(detections.size()));
    lines.push_back(status);
    lines.push_back("frame: " + std::to_string(frame_count));

    int y = 35;
    for (const auto& line : lines) {
        cv::putText(
            frame,
            line,
            cv::Point(20, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 255, 0),
            2
        );
        y += 30;
    }
}

static void inference_thread_func(
    SharedState* state,
    const std::string& model_path,
    int detect_interval_ms,
    float conf_thresh,
    float nms_thresh
)
{
    rknn_context ctx = 0;

    try {
        std::vector<unsigned char> model_data = read_file(model_path);

        int ret = rknn_init(
            &ctx,
            model_data.data(),
            model_data.size(),
            0,
            nullptr
        );

        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_init failed in inference thread, ret=" << ret << std::endl;
            state->running = false;
            return;
        }

        std::cout << "rknn_init success in inference thread" << std::endl;

        int last_processed_frame = -1;

        while (state->running) {
            cv::Mat frame_for_infer;
            int current_frame_count = 0;

            {
                std::lock_guard<std::mutex> lock(state->mutex);

                if (!state->latest_frame.empty()) {
                    current_frame_count = state->frame_count;

                    if (current_frame_count != last_processed_frame) {
                        frame_for_infer = state->latest_frame.clone();
                        last_processed_frame = current_frame_count;
                    }
                }
            }

            if (frame_for_infer.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            std::vector<Detection> detections;
            double infer_ms = 0.0;

            bool ok = run_rknn_inference(
                ctx,
                frame_for_infer,
                detections,
                infer_ms,
                conf_thresh,
                nms_thresh
            );

            if (ok) {
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->latest_detections = detections;
                    state->infer_ms = infer_ms;
                }

                std::cout << "detect: person=" << detections.size()
                          << ", infer=" << infer_ms << " ms" << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(detect_interval_ms));
        }

        rknn_destroy(ctx);
        std::cout << "rknn destroyed in inference thread" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "inference thread exception: " << e.what() << std::endl;

        if (ctx != 0) {
            rknn_destroy(ctx);
        }

        state->running = false;
    }
}

int main(int argc, char** argv)
{
    std::string device = "/dev/video9";
    std::string model_path = "models/yolov5n.rknn";

    int width = 1280;
    int height = 720;
    int fps = 30;

    int detect_interval_ms = 500;
    float conf_thresh = 0.25f;
    float nms_thresh = 0.30f;

    if (argc >= 2) {
        device = argv[1];
    }

    if (argc >= 3) {
        model_path = argv[2];
    }

    std::cout << "device             : " << device << std::endl;
    std::cout << "model path         : " << model_path << std::endl;
    std::cout << "resolution         : " << width << "x" << height << std::endl;
    std::cout << "detect interval ms : " << detect_interval_ms << std::endl;
    std::cout << "mode               : threaded capture/display + inference" << std::endl;

    setenv("DISPLAY", ":0", 0);

    SharedState state;

    std::thread infer_thread(
        inference_thread_func,
        &state,
        model_path,
        detect_interval_ms,
        conf_thresh,
        nms_thresh
    );

    cv::VideoCapture cap(device, cv::CAP_V4L2);

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FPS, fps);

    if (!cap.isOpened()) {
        std::cerr << "failed to open camera: " << device << std::endl;
        state.running = false;
        infer_thread.join();
        return -1;
    }

    std::cout << "camera opened" << std::endl;

    std::string window_name = "RK3568 C++ RKNN Threaded LCD Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    int frame_count = 0;
    int fps_count = 0;
    double camera_fps = 0.0;

    auto fps_time = std::chrono::high_resolution_clock::now();

    try {
        while (state.running) {
            cv::Mat frame;

            if (!cap.read(frame)) {
                std::cerr << "read frame failed" << std::endl;
                continue;
            }

            frame_count++;
            fps_count++;

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - fps_time).count();

            if (elapsed_sec >= 1.0) {
                camera_fps = fps_count / elapsed_sec;
                fps_count = 0;
                fps_time = now;

                std::cout << "camera_fps: " << camera_fps << std::endl;
            }

            std::vector<Detection> detections;
            double infer_ms = 0.0;

            {
                std::lock_guard<std::mutex> lock(state.mutex);

                state.latest_frame = frame.clone();
                state.frame_count = frame_count;
                state.camera_fps = camera_fps;

                detections = state.latest_detections;
                infer_ms = state.infer_ms;
            }

            draw_overlay(frame, detections, camera_fps, infer_ms, frame_count);

            cv::imshow(window_name, frame);

            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "main loop exception: " << e.what() << std::endl;
    }

    state.running = false;

    if (infer_thread.joinable()) {
        infer_thread.join();
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "exit OK" << std::endl;

    return 0;
}
