cmake_version()
project(core_ut)

add_executable(vec_ut vec_ut.cpp func_ut.cpp trans_ut.cpp)
target_link_libraries(vec_ut core gtest_main)
add_test(vec_ut vec_ut COMMAND vec_ut)
