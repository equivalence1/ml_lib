cmake_version()
project(catboost_ut)

#cmake_policy(SET CMP0069 NEW)
#include(CheckIPOSupported)
#check_ipo_supported()


add_executable(train_catboost train_catboost.cpp)
target_link_libraries(train_catboost core data vec_tools mx_tools trans funcs models methods ${CATBOOST})

