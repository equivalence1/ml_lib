cmake_version()

project(core)

add_library(core
        include/core/object.h
        include/core/vec.h
        include/core/trans.h
        include/core/func.h
        include/core/funcs/linear.h
        include/core/vec_factory.h
        include/core/batch.h

        #        include/coree/matrix.h

#
        include/core/vec_tools/ops.h
#        include/util/vec_tools/meta_info.h
#        include/util/vec_tools/ops.h
#        include/util/vec_tools/stats.h
#        include/util/vec_tools/transform.h

        src/vec_ops.h
        src/vec_ref.cpp
        src/array_vec.h
        src/fixed_size_vec.h
        src/vec_ref.h
        src/vec.cpp
        src/vec_factory.cpp
        src/array_vec.cpp
        src/fixed_size_vec.cpp
        src/vec_tools/ops.cpp
        src/funcs/linear.cpp
)

enable_cxx17(core)


target_include_directories(core INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/src
        )


install(TARGETS core EXPORT CoreConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
export(TARGETS core FILE CoreConfig.cmake)
