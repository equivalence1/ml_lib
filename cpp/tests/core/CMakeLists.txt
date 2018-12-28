cmake_version()
project(core_ut)

add_executable(core_ut vec_ut.cpp trans_ut.cpp func_ut.cpp optimizer_ut.cpp context_ut.cpp mx_ut.cpp)
target_link_libraries(core_ut core gtest_main)
add_test(core_ut core_ut COMMAND core_ut)
