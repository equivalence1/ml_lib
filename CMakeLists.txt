cmake_minimum_required(VERSION "3.1.0")
project(nn_introduction)

add_subdirectory(third_party)

set(PYTHON_PREFIX /usr/local/lib/python3.7)

pybind11_add_module(example cpp/src/dataset.cpp
                            cpp/src/least_squares.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

message(STATUS "libs=${LIBS}, include_paths=${INCLUDE_PATHS}")

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

target_link_libraries(example PRIVATE ${MKLDNN_LIB} Eigen3::Eigen)
target_include_directories(example PRIVATE ${MKLDNN_PATH})