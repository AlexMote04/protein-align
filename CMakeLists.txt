cmake_minimum_required(VERSION 3.18)

project(protein-align LANGUAGES CXX CUDA C)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenMP REQUIRED)

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
# Shared Sources
# ==========================================
set(SHARED_SOURCES 
    src/parse.cpp
    src/interAlignGPU.cu
    src/intraAlignGPU.cu
    src/naiveInterGPU.cu
    src/alignCPU.cpp
    src/alignParasail.cpp
)

# ==========================================
# Target: Benchmark Executable
# ==========================================
add_executable(benchmark 
    src/benchmark.cpp 
    ${SHARED_SOURCES}
)

target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(benchmark PRIVATE parasail OpenMP::OpenMP_CXX)

# ==========================================
# Target: Tests
# ==========================================
add_executable(run_tests 
    test/test_main.cpp
    test/test_parse.cpp
    ${SHARED_SOURCES}
)

target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(run_tests PRIVATE parasail OpenMP::OpenMP_CXX)