cmake_version()
project(targets_ut)

add_executable(targets_ut targets_ut.cpp)
target_link_libraries(targets_ut core data vec_tools mx_tools trans funcs targets gtest_main)

add_test(targets_ut targets_ut COMMAND targets_ut)
