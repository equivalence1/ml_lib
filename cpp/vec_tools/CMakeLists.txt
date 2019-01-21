cmake_version()

project(vec_tools LANGUAGES CXX)

add_library(vec_tools
        distance.cpp
        distance.h
        fill.cpp
        fill.h
        meta_info.h
        sort.cpp
        sort.h
        stats.cpp
        stats.h
        transform.cpp
        transform.h)




enable_cxx17(vec_tools)

target_link_libraries(vec_tools "${TORCH_LIBRARIES}" util core)


target_include_directories(vec_tools INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )


install(TARGETS vec_tools EXPORT VecToolsConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
#install(EXPORT CoreKernelsConfig DESTINATION share/CoreKernelsConfig/cmake)
#export(TARGETS core FILE CoreConfig.cmake)
#export(TARGETS core_kernels FILE CoreKernelsConfig.cmake)

add_subdirectory(ut)
