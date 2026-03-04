cmake_minimum_required(VERSION 3.10)
project(protein-align LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(include)

set(SHARED_SOURCES 
    src/parse.cpp
    src/sw/sw_cpu.cpp
    src/sw/sw_optimised.cu
    src/nw/nw_cpu.cpp
)

add_executable(benchmark 
    src/benchmark.cpp 
    ${SHARED_SOURCES}
)

add_executable(run_tests 
    test/test_main.cpp
    test/test_parser.cpp
    test/test_sw.cpp
    test/test_nw.cpp
    ${SHARED_SOURCES}
)