#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

#include "rknn_api.h"

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

static void print_tensor_attr(const rknn_tensor_attr& attr)
{
    std::cout << "index=" << attr.index
              << ", name=" << attr.name
              << ", n_dims=" << attr.n_dims
              << ", dims=[";

    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        std::cout << attr.dims[i];
        if (i + 1 < attr.n_dims) {
            std::cout << ", ";
        }
    }

    std::cout << "], n_elems=" << attr.n_elems
              << ", size=" << attr.size
              << ", fmt=" << attr.fmt
              << ", type=" << attr.type
              << ", qnt_type=" << attr.qnt_type
              << std::endl;
}

int main(int argc, char** argv)
{
    std::string model_path = "models/yolov5n.rknn";

    if (argc >= 2) {
        model_path = argv[1];
    }

    std::cout << "model path: " << model_path << std::endl;

    rknn_context ctx = 0;

    try {
        std::vector<unsigned char> model_data = read_file(model_path);
        std::cout << "model size: " << model_data.size() << " bytes" << std::endl;

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

        rknn_sdk_version sdk_ver;
        std::memset(&sdk_ver, 0, sizeof(sdk_ver));

        ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
        if (ret == RKNN_SUCC) {
            std::cout << "api version   : " << sdk_ver.api_version << std::endl;
            std::cout << "driver version: " << sdk_ver.drv_version << std::endl;
        } else {
            std::cerr << "query sdk version failed, ret=" << ret << std::endl;
        }

        rknn_input_output_num io_num;
        std::memset(&io_num, 0, sizeof(io_num));

        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret != RKNN_SUCC) {
            std::cerr << "query input/output num failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        std::cout << "input num : " << io_num.n_input << std::endl;
        std::cout << "output num: " << io_num.n_output << std::endl;

        std::cout << "\n===== Input Tensor Attr =====" << std::endl;
        for (uint32_t i = 0; i < io_num.n_input; ++i) {
            rknn_tensor_attr attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.index = i;

            ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
            if (ret == RKNN_SUCC) {
                print_tensor_attr(attr);
            } else {
                std::cerr << "query input attr failed, index=" << i
                          << ", ret=" << ret << std::endl;
            }
        }

        std::cout << "\n===== Output Tensor Attr =====" << std::endl;
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_tensor_attr attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.index = i;

            ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
            if (ret == RKNN_SUCC) {
                print_tensor_attr(attr);
            } else {
                std::cerr << "query output attr failed, index=" << i
                          << ", ret=" << ret << std::endl;
            }
        }

        rknn_destroy(ctx);
        std::cout << "\nrknn_destroy success" << std::endl;
        std::cout << "C++ RKNN runtime test OK" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;

        if (ctx != 0) {
            rknn_destroy(ctx);
        }

        return -1;
    }

    return 0;
}
