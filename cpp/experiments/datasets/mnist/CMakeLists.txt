cmake_version()
project(mnist)

include_directories(../../core)
include_directories(../../core/networks)

add_library(mnist
        mnist_reader.cpp
        mnist_reader.h)

target_link_libraries(mnist experiments_core)
