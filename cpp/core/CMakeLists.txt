cmake_version()

project(core)

add_library(core
        include/core/vec.h
        include/core/data_holder.h
        include/core/matrix.h
        include/core/vec_data.h
        include/core/vec_factory.h

        include/core/vec_tools/distance.h
        include/core/vec_tools/meta_info.h
        include/core/vec_tools/ops.h
        include/core/vec_tools/stats.h
        include/core/vec_tools/transform.h

        src/vec.cpp
        )

enable_cxx17(core)


target_include_directories(core INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core>
        PRIVATE src
        )


install(TARGETS core EXPORT CoreConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
export(TARGETS core FILE CoreConfig.cmake)
