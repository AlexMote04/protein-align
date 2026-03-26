cmake_minimum_required(VERSION 3.14)
project(protein-align LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(include)

# ==========================================
# Fetch and Build Parasail
# ==========================================
include(FetchContent)

# Disable Parasail's extra binaries and tests to speed up our compile time
set(PARASAIL_BUILD_APPS OFF CACHE BOOL "Disable parasail apps" FORCE)
set(PARASAIL_BUILD_TESTS OFF CACHE BOOL "Disable parasail tests" FORCE)

FetchContent_Declare(
    parasail
    GIT_REPOSITORY https://github.com/jeffdaily/parasail.git
    GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(parasail)
# ==========================================

set(SHARED_SOURCES 
    src/parse.cpp
    src/interAlignGPU.cu
    src/intraAlignGPU.cu
    src/alignCPU.cpp
    src/alignParasail.cpp
)

add_executable(benchmark 
    src/benchmark.cpp 
    ${SHARED_SOURCES}
)

target_include_directories(benchmark PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(benchmark PRIVATE parasail cudart)

# add_executable(run_tests 
#     test/test_main.cpp
#     test/test_parser.cpp
#     test/test_sw.cpp
#     test/test_nw.cpp
#     ${SHARED_SOURCES}
# )