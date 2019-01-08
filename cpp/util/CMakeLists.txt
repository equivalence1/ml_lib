cmake_version()

project(util CXX)


add_library(util
            cuda_wrappers.h
            exception.h
            singleton.h
            guard.h

            cuda_wrappers.cpp
            exception.cpp
)

enable_cxx14(util)
maybe_enable_cuda(util)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

target_include_directories(util INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/
        )


install(TARGETS util EXPORT UtilConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT UtilConfig DESTINATION share/UtilConfig/cmake)
export(TARGETS util FILE UtilConfig.cmake)
