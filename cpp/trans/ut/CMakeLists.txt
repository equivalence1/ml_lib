cmake_version()
project(trans_ut)

add_executable(trans_ut trans_ut.cpp)
target_link_libraries(trans_ut core trans vec_tools mx_tools gtest_main)
add_test(trans_ut trans_ut COMMAND trans_ut)
