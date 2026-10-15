cmake_minimum_required(VERSION 3.20)
project(signal_pipeline LANGUAGES CXX)

option(ENABLE_OPENMP "Enable OpenMP CPU parallelism" ON)
option(ENABLE_CUDA "Build optional CUDA implementation" OFF)
option(BUILD_TESTING "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3f)

if(ENABLE_OPENMP)
    find_package(OpenMP)
endif()

add_library(signal_core
    src/signal.cpp
    src/pipeline.cpp
)

target_include_directories(signal_core PUBLIC include)
target_link_libraries(signal_core PUBLIC PkgConfig::FFTW3)

if(OpenMP_CXX_FOUND AND ENABLE_OPENMP)
    target_link_libraries(signal_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(signal_core PUBLIC SIGNAL_HAS_OPENMP=1)
endif()

add_executable(signal_bench src/main.cpp)
target_link_libraries(signal_bench PRIVATE signal_core)

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(signal_cuda cuda/pipeline_cuda.cu)
    target_include_directories(signal_cuda PUBLIC include)
    target_link_libraries(signal_cuda
        PUBLIC signal_core CUDA::cudart CUDA::cufft
    )
    target_compile_definitions(signal_bench PRIVATE SIGNAL_ENABLE_CUDA=1)
    target_link_libraries(signal_bench PRIVATE signal_cuda)
    set_target_properties(signal_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(BUILD_TESTING)
    enable_testing()
    add_executable(signal_tests tests/test_pipeline.cpp)
    target_link_libraries(signal_tests PRIVATE signal_core)
    add_test(NAME signal_tests COMMAND signal_tests)
endif()