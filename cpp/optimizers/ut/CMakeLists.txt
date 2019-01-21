cmake_version()
project(optimizers_ut)

add_executable(optimizer_ut optimizer_ut.cpp)
target_link_libraries(optimizer_ut core optimizers gtest_main)
add_test(optimizer_ut optimizer_ut COMMAND optimizer_ut)
