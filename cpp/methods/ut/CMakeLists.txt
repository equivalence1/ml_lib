cmake_version()
project(methods_ut)

add_executable(methods_ut methods_ut.cpp)
target_link_libraries(methods_ut core data vec_tools mx_tools trans funcs models methods gtest_main)

add_test(methods_ut methods_ut COMMAND methods_ut)
