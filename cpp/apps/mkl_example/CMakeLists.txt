cmake_version()
project(mkl_example)

add_executable(mkl_example main.cpp)


target_link_libraries(mkl_example mkldnn)
