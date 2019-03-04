cmake_minimum_required(VERSION "3.1.0")
include(cmake/custom_functions.cmake)
project(root)

find_package(Torch REQUIRED)

option(USE_CUDA  "Build with CUDA")

include(GNUInstallDirs)
#add_compile_options(-Wall -Wextra -pedantic -Werror)


# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

add_subdirectory(contrib/pybind11)
enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DCUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#    SET(CMAKE_CUDA_FLAGS "–cudart static")
    set(CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION ON)
    include_directories(cub)

    set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr
                            -gencode arch=compute_30,code=compute_30
                            -gencode arch=compute_35,code=sm_35
                            -gencode arch=compute_50,code=compute_50
                            -gencode arch=compute_52,code=sm_52
                            -gencode arch=compute_60,code=compute_60
                            -gencode arch=compute_61,code=compute_61
                            -gencode arch=compute_61,code=sm_61
                            -gencode arch=compute_70,code=sm_70
                            -gencode arch=compute_70,code=compute_70
                            --ptxas-options=-v")
endif()

add_subdirectory(cpp)



