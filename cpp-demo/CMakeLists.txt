
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mobilevlm)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")

add_definitions(-w)

message( "${TARGET_ARCH} mode, starting......")
check_and_replace_path(SDK)
check_and_replace_path(SAIL_PATH)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(BM_LIBS bmlib bmrt bmcv)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
include_directories("${SDK}/include/")
include_directories("${SDK}/include/opencv4")

link_directories("${SDK}/lib/")
# 使用头文件
include_directories("${PROJECT_SOURCE_DIR}/include/")
# 使用第三方库
include_directories("${PROJECT_SOURCE_DIR}/third_part/include/")
link_directories("${PROJECT_SOURCE_DIR}/third_part/lib/")
set(CNPY_LIB ${PROJECT_SOURCE_DIR}/third_part/lib/libcnpy.a)
set(Z_LIB z)
# use sail headers
include_directories(${SAIL_PATH}/include)
include_directories(${SAIL_PATH}/include/sail)
# use sail
set(SAIL_LIBS  ${SAIL_PATH}/lib/libsail.so)
# sail为需要链接的库
message("SDK: " ${SDK})

set(SOURCE_FILES 
    demo.cpp)

add_executable(mobilevlm.soc ${SOURCE_FILES})
target_link_libraries(mobilevlm.soc ${BM_LIBS} ${OPENCV_LIBS}  ${JPU_LIBS} ${SAIL_LIBS})

target_link_libraries(mobilevlm.soc ${TORCH_LIBRARIES})

target_link_libraries(mobilevlm.soc ${CNPY_LIB} ${Z_LIB})

