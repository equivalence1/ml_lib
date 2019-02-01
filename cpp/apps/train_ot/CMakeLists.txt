cmake_version()
project(train_it)

add_executable(train_ot main.cpp)

#cmake_policy(SET CMP0069 NEW)
#include(CheckIPOSupported)
#check_ipo_supported()
#set_target_properties(train_ot PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

target_link_libraries(train_ot methods  "${TORCH_LIBRARIES}" core util models targets data)
