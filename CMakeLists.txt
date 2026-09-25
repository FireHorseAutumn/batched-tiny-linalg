cmake_minimum_required(VERSION 3.20)
# nvcc 13.2 rejects GCC 16; must be set before project() so the
# compiler-ID probe inherits it.
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
project(batched-tiny-linalg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(btl INTERFACE)
target_include_directories(btl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(btl INTERFACE
    CUDA::cudart CUDA::cusolver CUDA::cublas Eigen3::Eigen
)

# TODO: add targets as sources land
# add_executable(test_cholesky tests/test_cholesky.cpp)
# target_link_libraries(test_cholesky PRIVATE btl)
# add_executable(bench_cholesky benchmarks/bench_cholesky.cpp)
# target_link_libraries(bench_cholesky PRIVATE btl)

enable_testing()
