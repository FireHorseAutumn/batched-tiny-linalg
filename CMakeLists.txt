cmake_minimum_required(VERSION 3.20)

# Host compiler: when the default g++ is newer than nvcc supports, prefer
# g++-15; otherwise fall back to the override flag with -std=c++20 (newer
# libstdc++ headers do not parse under nvcc's c++17 mode). Runs before
# project(), where CMAKE_CXX_COMPILER* are not yet known.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(batched-tiny-linalg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
# No version pin: Eigen 5.0.1's config file rejects a 3.4 request
# (major-version mismatch policy). Any installed Eigen3 is accepted.
find_package(Eigen3 REQUIRED NO_MODULE)

# nlohmann/json via FetchContent — no system package needed; pinned tag,
# fetched at configure time. Link: nlohmann_json::nlohmann_json.
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(btl INTERFACE)
target_include_directories(btl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(btl INTERFACE
    CUDA::cudart CUDA::cusolver CUDA::cublas Eigen3::Eigen
)

# Tests and benchmarks are .cpp files compiled as CUDA (they launch the
# header-only kernels directly).
add_executable(test_cholesky tests/test_cholesky.cpp)
set_source_files_properties(tests/test_cholesky.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_cholesky PRIVATE btl)

add_executable(bench_cholesky benchmarks/bench_cholesky.cpp)
set_source_files_properties(benchmarks/bench_cholesky.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(bench_cholesky PRIVATE btl nlohmann_json::nlohmann_json)

enable_testing()
add_test(NAME test_cholesky COMMAND test_cholesky)
