cmake_version()

project(core LANGUAGES CXX)

add_library(core
        include/core/context.h
        include/core/object.h
        include/core/vec.h
        include/core/scalar.h
        include/core/trans.h
        include/core/func.h
        include/core/funcs/linear.h
        include/core/funcs/lq.h
        include/core/vec_factory.h
        include/core/batch.h
        include/core/trans/fill.h
        include/core/trans/pow.h
        include/core/trans/identity.h
        include/core/trans/add_vec.h
        include/core/trans/pointwise_multiply.h

        include/core/trans/compose.h
        include/core/optimizer.h
        include/core/optimizers/gradient_descent.h
        include/core/index_transformation.h
        include/core/matrix.h

        #
        include/core/vec_tools/fill.h
        include/core/vec_tools/distance.h
        #        include/core/vec_tools/meta_info.h
        include/core/vec_tools/stats.h
        include/core/vec_tools/transform.h
        src/context.cpp
        src/trans.cpp
        src/func.cpp
        src/vec.cpp
        src/vec_factory.cpp
        src/vec_factory.cpp
        src/vec_tools/fill.cpp
        src/vec_tools/distance.cpp
        src/vec_tools/stats.cpp
        src/vec_tools/transform.cpp
        src/funcs/linear.cpp
        src/funcs/lq.cpp
        src/trans/fill.cpp
        src/trans/compose.cpp
        src/trans/pow.cpp
        src/trans/identity.cpp
        src/trans/add_vec.cpp
        src/matrix.cpp
        src/trans/pointwise_multiply.cpp
        src/optimizers/gradient_descent.cpp
        )

add_library(core_kernels
        src/kernels/fill.cuh
        src/kernels/fill.cu

        src/kernels/empty.h
        src/kernels/empty.cpp)


maybe_enable_cuda(core_kernels)

enable_cxx17(core)
maybe_enable_cuda(core)

target_link_libraries(core "${TORCH_LIBRARIES}" util core_kernels)
target_link_libraries(core_kernels util)


target_include_directories(core INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

target_include_directories(core_kernels INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/kernels>
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
