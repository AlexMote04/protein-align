cmake_minimum_required(VERSION 3.10)
project(protein-align LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

include_directories(include)

set(SHARED_SOURCES 
    src/parse.cpp
    src/sw_cpu.cpp
    src/sw_gpu.cu
)

# add_executable(align 
#     src/main.cpp 
#     ${SHARED_SOURCES}
# )

# target_link_libraries(align PRIVATE cuda_cudart)

add_executable(run_tests 
    test/test_main.cpp
    test/test_parser.cpp
    test/test_cpu.cpp 
    test/test_gpu.cpp
    ${SHARED_SOURCES}
)