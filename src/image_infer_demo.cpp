#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

static const int INPUT_SIZE = 640;

struct LetterboxInfo {
    float scale;
    int pad_left;
    int pad_top;
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

int main(int argc, char** argv)
{
    std::string model_path = "models/yolov5n.rknn";
    std::string image_path = "assets/test.jpg";
    std::string output_path = "outputs/cpp_infer_test.jpg";

    if (argc >= 2) {
        image_path = argv[1];
    }

    if (argc >= 3) {
        model_path = argv[2];
    }

    std::cout << "image path : " << image_path << std::endl;
    std::cout << "model path : " << model_path << std::endl;
    std::cout << "output path: " << output_path << std::endl;

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
            std::cerr << "rknn_init failed, ret=" << ret << std::endl;
            return -1;
        }

        std::cout << "rknn_init success" << std::endl;

        rknn_input_output_num io_num;
        std::memset(&io_num, 0, sizeof(io_num));

        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret != RKNN_SUCC) {
            std::cerr << "query io num failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        std::cout << "input num : " << io_num.n_input << std::endl;
        std::cout << "output num: " << io_num.n_output << std::endl;

        rknn_tensor_attr output_attr;
        std::memset(&output_attr, 0, sizeof(output_attr));
        output_attr.index = 0;

        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "query output attr failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        std::cout << "output n_dims: " << output_attr.n_dims << std::endl;
        std::cout << "output dims  : [";
        for (uint32_t i = 0; i < output_attr.n_dims; ++i) {
            std::cout << output_attr.dims[i];
            if (i + 1 < output_attr.n_dims) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        std::cout << "output n_elems: " << output_attr.n_elems << std::endl;

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "failed to read image: " << image_path << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        std::cout << "original image: " << image.cols << "x" << image.rows << std::endl;

        LetterboxInfo lb_info;
        cv::Mat input_bgr = letterbox(image, INPUT_SIZE, INPUT_SIZE, lb_info);

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

        ret = rknn_inputs_set(ctx, 1, inputs);
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_inputs_set failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        ret = rknn_run(ctx, nullptr);
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_run failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        rknn_output outputs[1];
        std::memset(outputs, 0, sizeof(outputs));

        outputs[0].want_float = 1;

        ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_outputs_get failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "inference time: " << infer_ms << " ms" << std::endl;

        float* output_data = reinterpret_cast<float*>(outputs[0].buf);

        int num_boxes = 25200;
        int element_per_box = 85;

        float max_obj = 0.0f;
        float max_person_score = 0.0f;
        int max_obj_index = -1;

        for (int i = 0; i < num_boxes; ++i) {
            float obj_conf = output_data[i * element_per_box + 4];
            float person_cls = output_data[i * element_per_box + 5];
            float person_score = obj_conf * person_cls;

            if (obj_conf > max_obj) {
                max_obj = obj_conf;
                max_obj_index = i;
            }

            if (person_score > max_person_score) {
                max_person_score = person_score;
            }
        }

        std::cout << "max obj_conf     : " << max_obj << std::endl;
        std::cout << "max obj index    : " << max_obj_index << std::endl;
        std::cout << "max person_score : " << max_person_score << std::endl;

        cv::Mat result = image.clone();

        cv::putText(
            result,
            "C++ RKNN inference OK",
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::putText(
            result,
            "infer_ms: " + std::to_string(infer_ms),
            cv::Point(20, 80),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::putText(
            result,
            "max_person_score: " + std::to_string(max_person_score),
            cv::Point(20, 120),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::imwrite(output_path, result);
        std::cout << "saved result: " << output_path << std::endl;

        rknn_outputs_release(ctx, 1, outputs);
        rknn_destroy(ctx);

        std::cout << "C++ image inference demo OK" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;

        if (ctx != 0) {
            rknn_destroy(ctx);
        }

        return -1;
    }

    return 0;
}
