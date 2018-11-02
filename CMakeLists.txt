cmake_minimum_required(VERSION "3.1.0")
project(nn_introduction)

# compiler-related
set(CMAKE_CXX_STANDARD 17)

# TODO(equivalence1) -O3 and -march probably should be set elsewhere.
# left for now just to speed up Eigene (without it it's
# impossible to wait for matrix multiplication)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -O3 -march=native")

# TODO(equivalence1): move all dependency search to separate file

add_subdirectory(third_party)

set(PYTHON_PREFIX /usr/local/lib/python3.7)

set(CORE_SRCS
        cpp/include/nntree/buffer_info.h
        cpp/include/nntree/dataset.h
        cpp/include/nntree/nd_array.h
        cpp/include/nntree/cost_function.h
        cpp/include/nntree/model.h
        cpp/include/nntree/optimizer.h

        cpp/src/convolution.cpp
        cpp/src/convolution.h
        cpp/src/least_squares.cpp
        cpp/src/least_squares.h
        cpp/src/nd_array.cpp
        cpp/src/optimizer.cpp)

# TODO(equivalence1) Eigen + MKL (https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html)
# Eigen + OpenMP (but seems like clang and openmp are incompatible,
# see https://stackoverflow.com/questions/51656818/matrix-multiplication-performance-c-eigen-is-much-slower-than-python)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# sadly, it seems like CLION does not understand target_include_dir property
# and we have to just include_dir
include_directories(cpp/include)

add_library(nntree SHARED ${CORE_SRCS})

# mkl-dnn

## We don't include mkl-dnn as a submodule because it's huge.
## Instead just searching for locally installed one.

## library
message(STATUS "searching for mkl-dnn library.")
find_library(MKLDNN_LIB NAMES mkldnn)
if (MKLDNN_LIB)
    message(STATUS "ok, mkl-dnn was found (${MKLDNN_LIB}).")
else()
    message(FATAL_ERROR "mkl-dnn library was not found. Aborting.")
endif()

## include
message(STATUS "searching for mkl-dnn library.")
find_path(MKLDNN_PATH NAMES mkldnn.hpp)
if (MKLDNN_PATH)
    message(STATUS "ok, found mkldnn headers (${MKLDNN_PATH}).")
else()
    message(FATAL_ERROR "failed to find mkldnn headers.")
endif()

target_link_libraries(nntree PRIVATE ${MKLDNN_LIB} Eigen3::Eigen)
target_include_directories(nntree PRIVATE ${MKLDNN_PATH})

# python module
# TODO(equivalence1): move to separate file and only build if flags are set
# TODO(equivalence1): provide setup.py

set(PY_MODULE_SRCS
        cpp/src/py_bindings.cpp
        )
pybind11_add_module(nntreepy ${PY_MODULE_SRCS})
target_link_libraries(nntreepy PRIVATE nntree)

# TODO(equivalence1) just a simple exec to check least squares method
# left for now just because it was useful for debugging.
add_executable(main cpp/src/main.cpp)
target_link_libraries(main PRIVATE nntree)
