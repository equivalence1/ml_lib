cmake_version()
project(methods_ut)

#cmake_policy(SET CMP0069 NEW)
#include(CheckIPOSupported)
#check_ipo_supported()


add_executable(methods_ut methods_ut.cpp)
target_link_libraries(methods_ut core data vec_tools mx_tools trans funcs models methods gtest_main gtest)
#set_target_properties(methods_ut PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

add_test(methods_ut methods_ut COMMAND methods_ut)
