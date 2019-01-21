cmake_version()
project(vec_ut)

add_executable(vec_ut vec_ut.cpp)
target_link_libraries(vec_ut core vec_tools gtest_main)
add_test(vec_ut vec_ut COMMAND vec_ut)
