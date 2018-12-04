cmake_version()
project(mkl_example)

add_executable(mkl_example main.cpp)
enable_cxx14(mkl_example)

find_mkldnn()

target_link_libraries(mkl_example mkldnn)
