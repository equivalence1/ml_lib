cmake_minimum_required(VERSION "3.1.0")
project(nn_introduction)

add_subdirectory(third_party)

pybind11_add_module(example cpp/src/dataset.cpp)