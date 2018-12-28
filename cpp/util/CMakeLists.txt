cmake_version()

project(util CXX)


add_library(util
            include/util/cuda_wrappers.h
            include/util/exception.h
            include/util/singleton.h
            include/util/guard.h

            src/cuda_wrappers.cpp
            src/exception.cpp
)

enable_cxx14(util)
maybe_enable_cuda(util)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

target_include_directories(util INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/src
        )


install(TARGETS util EXPORT UtilConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT UtilConfig DESTINATION share/UtilConfig/cmake)
export(TARGETS util FILE UtilConfig.cmake)
