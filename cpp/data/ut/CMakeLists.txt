cmake_version()
project(data_ut)

add_executable(data_ut data_ut.cpp)
target_link_libraries(data_ut core data vec_tools mx_tools trans funcs gtest_main)

add_test(data_ut data_ut COMMAND data_ut)
