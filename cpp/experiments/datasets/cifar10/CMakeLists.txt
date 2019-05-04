cmake_version()
project(cifar10)

include_directories(../../core)
include_directories(../../core/networks)

add_library(cifar10
        cifar10_reader.cpp
        cifar10_reader.h)

target_link_libraries(cifar10 experiments_core)
