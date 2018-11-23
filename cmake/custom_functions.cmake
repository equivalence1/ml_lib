function(enable_cxx17 TARGET)
    target_compile_features(${TARGET} PUBLIC cxx_std_17)
    set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction(enable_cxx17)


function(enable_cxx14 TARGET)
    target_compile_features(${TARGET} PUBLIC cxx_std_14)
endfunction(enable_cxx14)


function(cmake_version)
    cmake_minimum_required(VERSION 3.12)
endfunction(cmake_version)


function(maybe_enable_cuda TARGET)
    if (USE_CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        find_package(CUDA 10.0 REQUIRED)

        if(CUDA_FOUND)
            message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
            message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
            include(CheckLanguage)
            check_language(CUDA)
        endif()

        include_directories(${CUDA_INCLUDE_DIRS})
        target_link_libraries(${TARGET} ${CUDA_LIBRARIES})

        set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET ${TARGET} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()
    endif()

endfunction(maybe_enable_cuda)


function(maybe_enable_cuda TARGET)
    if (USE_CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        find_package(CUDA 10.0 REQUIRED)

        if(CUDA_FOUND)
            message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
            message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
            include(CheckLanguage)
            check_language(CUDA)
        endif()

        include_directories(${CUDA_INCLUDE_DIRS})
        target_link_libraries(${TARGET} ${CUDA_LIBRARIES})

#        set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#        set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET ${TARGET} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()
    endif()

endfunction(maybe_enable_cuda)


function(find_mkldnn)
    ## library
    message(STATUS "searching for mkl-dnn library.")
    find_library(MKLDNN_LIB NAMES mkldnn)
    if (MKLDNN_LIB)
        message(STATUS "ok, mkl-dnn was found (${MKLDNN_LIB}).")
    else ()
        message(FATAL_ERROR "mkl-dnn library was not found. Aborting.")
    endif ()

    ## include
    message(STATUS "searching for mkl-dnn library.")
    find_path(MKLDNN_PATH NAMES mkldnn.hpp)
    if (MKLDNN_PATH)
        message(STATUS "ok, found mkldnn headers (${MKLDNN_PATH}).")
    else ()
        message(FATAL_ERROR "failed to find mkldnn headers.")
    endif ()

endfunction(find_mkldnn)
