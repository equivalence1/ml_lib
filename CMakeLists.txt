cmake_minimum_required(VERSION "3.1.0")
include(cmake/custom_functions.cmake)
project(root)

# compiler-related
set(CMAKE_CXX_STANDARD 17)

option(USE_CUDA  "Build with CUDA")
set(CUDA_COMPUTE_CAPABILITIES "" CACHE STRING
        "Space separated list of compute versions to be built against, e.g. '35 61'")


include(GNUInstallDirs)


# TODO(equivalence1) -O3 and -march probably should be set elsewhere.
# left for now just to speed up Eigene (without it it's
# impossible to wait for matrix multiplication)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -O3 -march=native")

# TODO(equivalence1): move all dependency search to separate file

add_subdirectory(contrib/pybind11)
add_subdirectory(contrib/googletest)
add_subdirectory(cpp)

#TODO: get rid of this
set(PYTHON_PREFIX /usr/local/lib/python3.7)

