cmake_minimum_required(VERSION 3.14)
project(protein-align LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 14)
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
    src/sw/sw_cpu.cpp
    src/sw/sw_optimised.cu
    src/nw/nw_cpu.cpp
    src/nw/nw_optimised.cu
    src/nw/nw_parasail.cpp
)

# add_executable(benchmark_sw 
#     src/benchmark_sw.cpp 
#     ${SHARED_SOURCES}
# )
# target_link_libraries(benchmark_sw PRIVATE parasail)

add_executable(benchmark_nw 
    src/benchmark_nw.cpp 
    ${SHARED_SOURCES}
)
target_link_libraries(benchmark_nw PRIVATE parasail)

# add_executable(run_tests 
#     test/test_main.cpp
#     test/test_parser.cpp
#     test/test_sw.cpp
#     test/test_nw.cpp
#     ${SHARED_SOURCES}
# )