cmake_version()

project(trans LANGUAGES CXX)

add_library(trans
        add_vec.cpp
        add_vec.h
        compose.cpp
        compose.h
        fill.h
        fill.cpp
        identity.cpp
        identity.h
        linear_trans.h
        linear_trans.cpp
        pointwise_multiply.h
        pointwise_multiply.cpp
        pow.cpp
        pow.h
)



#maybe_enable_cuda(models_kernels)

enable_cxx17(trans)

target_link_libraries(trans "${TORCH_LIBRARIES}" core util vec_tools)
#target_link_libraries(models_kernels util)


target_include_directories(trans INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )

#target_include_directories(models_kernels INTERFACE
#        PUBLIC $<INSTALL_INTERFACE:include>
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
#        )

install(TARGETS trans EXPORT TransConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(TARGETS models_kernels EXPORT TreesKernelsConfig
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
#install(EXPORT CoreKernelsConfig DESTINATION share/CoreKernelsConfig/cmake)
#export(TARGETS core FILE CoreConfig.cmake)
#export(TARGETS core_kernels FILE CoreKernelsConfig.cmake)


add_subdirectory(ut)
