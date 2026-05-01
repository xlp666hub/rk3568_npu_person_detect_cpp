cmake_minimum_required(VERSION 3.10)

project(rk3568_npu_person_detect_cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RKNN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/rknn/include)
set(RKNN_LIB_DIR ${CMAKE_SOURCE_DIR}/third_party/rknn/lib)

include_directories(${RKNN_INCLUDE_DIR})
link_directories(${RKNN_LIB_DIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

add_executable(test_rknn_runtime
    src/test_rknn_runtime.cpp
)

target_link_libraries(test_rknn_runtime
    rknnrt
)

add_executable(image_infer_demo
    src/image_infer_demo.cpp
)

target_link_libraries(image_infer_demo
    rknnrt
    ${OPENCV_LIBRARIES}
)

add_executable(image_detect_demo
    src/image_detect_demo.cpp
)

target_link_libraries(image_detect_demo
    rknnrt
    ${OPENCV_LIBRARIES}
)

add_executable(lcd_camera_detect_demo
    src/lcd_camera_detect_demo.cpp
)

target_link_libraries(lcd_camera_detect_demo
    rknnrt
    ${OPENCV_LIBRARIES}
)
