cmake_version()

project(core LANGUAGES CXX)

add_library(core
        include/core/object.h
        include/core/vec.h
        include/core/trans.h
        include/core/func.h
        include/core/funcs/linear.h
        include/core/vec_factory.h
        include/core/batch.h
        include/core/fwd.h
        include/core/trans/const_trans.h
#        include/coree/matrix.h

#
        include/core/vec_tools/fill.h
        include/core/vec_tools/distance.h
#        include/core/vec_tools/meta_info.h
        include/core/vec_tools/stats.h
        include/core/vec_tools/transform.h

        src/vec_impls.h
        src/vec_impls.cpp
        src/vec_ops.h
        src/vec_ref.cpp
        src/array_vec.h
        src/cuda_vec.h
        src/fixed_size_vec.h
        src/vec_ref.h
        src/vec.cpp
        src/vec_factory.cpp
        src/array_vec.cpp
        src/cuda_vec.cpp
        src/fixed_size_vec.cpp
        src/vec_tools/ops.cpp
        src/vec_tools/distance.cpp
        src/vec_tools/stats.cpp
        src/vec_tools/transform.cpp
        src/funcs/linear.cpp
        src/trans/const_trans.cpp
)

add_library(core_kernels
        src/kernels/fill.cuh
        src/kernels/fill.cu
)

maybe_enable_cuda(core_kernels)

enable_cxx17(core)
maybe_enable_cuda(core)

target_link_libraries(core util core_kernels)
target_link_libraries(core_kernels util)

target_include_directories(core INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(core_kernels INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/kernels>
)

install(TARGETS core EXPORT CoreConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS core_kernels EXPORT CoreKernelsConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
install(EXPORT CoreKernelsConfig DESTINATION share/CoreKernelsConfig/cmake)
export(TARGETS core FILE CoreConfig.cmake)
export(TARGETS core_kernels FILE CoreKernelsConfig.cmake)
