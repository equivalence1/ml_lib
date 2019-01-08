cmake_version()

project(core LANGUAGES CXX)

add_library(core
        context.h
        object.h
        vec.h
        scalar.h
        trans.h
        func.h
        funcs/linear.h
        funcs/lq.h
        vec_factory.h
        batch.h
        trans/linear_trans.h
        trans/fill.h
        trans/pow.h
        trans/identity.h
        trans/add_vec.h
        trans/pointwise_multiply.h

        trans/compose.h
        optimizer.h
        optimizers/gradient_descent.h
        index_transformation.h
        matrix.h
        mx_tools/ops.h

        #
        vec_tools/fill.h
        vec_tools/distance.h
        #        vec_tools/meta_info.h
        vec_tools/stats.h
        vec_tools/transform.h
        context.cpp
        trans.cpp
        func.cpp
        vec.cpp
        vec_factory.cpp
        vec_factory.cpp
        vec_tools/fill.cpp
        vec_tools/distance.cpp
        vec_tools/stats.cpp
        vec_tools/transform.cpp
        funcs/linear.cpp
        funcs/lq.cpp
        trans/fill.cpp
        trans/linear_trans.cpp
        mx_tools/ops.cpp
        trans/compose.cpp
        trans/pow.cpp
        trans/identity.cpp
        trans/add_vec.cpp
        matrix.cpp
        trans/pointwise_multiply.cpp
        optimizers/gradient_descent.cpp
        )

add_library(core_kernels
        kernels/fill.cuh
        kernels/fill.cu

        kernels/empty.h
        kernels/empty.cpp)


maybe_enable_cuda(core_kernels)

enable_cxx17(core)
maybe_enable_cuda(core)

target_link_libraries(core "${TORCH_LIBRARIES}" util core_kernels)
target_link_libraries(core_kernels util)


target_include_directories(core INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )

target_include_directories(core_kernels INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
        )

install(TARGETS core EXPORT CoreConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS core_kernels EXPORT CoreKernelsConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
#install(EXPORT CoreKernelsConfig DESTINATION share/CoreKernelsConfig/cmake)
#export(TARGETS core FILE CoreConfig.cmake)
#export(TARGETS core_kernels FILE CoreKernelsConfig.cmake)
