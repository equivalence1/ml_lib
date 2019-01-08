cmake_version()
project(nn_introduction)

# compiler-related
set(CMAKE_CXX_STANDARD 17)

option(USE_CUDA "Build with CUDA")
set(CUDA_COMPUTE_CAPABILITIES "" CACHE STRING
        "Space separated list of compute versions to be built against, e.g. '35 61'")


include(GNUInstallDirs)


# TODO(equivalence1) -O3 and -march probably should be set elsewhere.
# left for now just to speed up Eigene (without it it's
# impossible to wait for matrix multiplication)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -O3 -march=native")

# TODO(equivalence1): move all dependency search to separate file


set(PYTHON_PREFIX /usr/local/lib/python3.7)

set(CORE_SRCS
        include/nntree/dataset.h
        include/nntree/tensor.h
        include/nntree/cpu_tensor.h
        include/nntree/cost_function.h
        include/nntree/model.h
        include/nntree/optimizer.h

        src/convolution.cpp
        src/convolution.h
        src/least_squares.cpp
        src/least_squares.h
        src/optimizer.cpp)

# TODO(equivalence1) Eigen + MKL (https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html)
# Eigen + OpenMP (but seems like clang and openmp are incompatible,
# see https://stackoverflow.com/questions/51656818/matrix-multiplication-performance-c-eigen-is-much-slower-than-python)

find_package(Eigen3 REQUIRED NO_MODULE)

# sadly, it seems like CLION does not understand target_include_dir property
# and we have to just include_dir
include_directories(include)

add_library(nntree SHARED ${CORE_SRCS})

# mkl-dnn

## We don't include mkl-dnn as a submodule because it's huge.
## Instead just searching for locally installed one.

## library
find_mkldnn()

target_link_libraries(nntree PRIVATE ${MKLDNN_LIB} Eigen3::Eigen)
target_include_directories(nntree PRIVATE ${MKLDNN_PATH})

# python module
# TODO(equivalence1): move to separate file and only build if flags are set
# TODO(equivalence1): provide setup.py

set(PY_MODULE_SRCS
        src/py_bindings.cpp
        )
pybind11_add_module(nntreepy ${PY_MODULE_SRCS})
target_link_libraries(nntreepy PRIVATE nntree)

# TODO(equivalence1) just a simple exec to check least squares method
# left for now just because it was useful for debugging.
#add_executable(main cpp/src/main.cpp)
#target_link_libraries(main PRIVATE nntree)
