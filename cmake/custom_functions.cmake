function(enable_cxx17 TARGET)
	target_compile_features(${TARGET} PUBLIC cxx_std_17)
endfunction(enable_cxx17)


function(enable_cxx14 TARGET)
    target_compile_features(${TARGET} PUBLIC cxx_std_14)
endfunction(enable_cxx14)


function(cmake_version)
    cmake_minimum_required(VERSION 3.12)
endfunction(cmake_version)


# Generate nvcc compiler flags given a list of architectures
# Also generates PTX for the most recent architecture for forwards compatibility
function(format_gencode_flags flags out)
    # Set up architecture flags
    if(NOT flags)
        if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
            set(flags "35;50;52;60;61;70")
        else()
            set(flags "35;50;52;60;61")
        endif()
    endif()
    # Generate SASS
    foreach(ver ${flags})
        set(${out} "${${out}}-gencode arch=compute_${ver},code=sm_${ver};")
    endforeach()
    # Generate PTX for last architecture
    list(GET flags -1 ver)
    set(${out} "${${out}}-gencode arch=compute_${ver},code=compute_${ver};")

    set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)



function(find_mkldnn)
    ## library
    message(STATUS "searching for mkl-dnn library.")
    find_library(MKLDNN_LIB NAMES mkldnn)
    if (MKLDNN_LIB)
        message(STATUS "ok, mkl-dnn was found (${MKLDNN_LIB}).")
    else()
        message(FATAL_ERROR "mkl-dnn library was not found. Aborting.")
    endif()

    ## include
    message(STATUS "searching for mkl-dnn library.")
    find_path(MKLDNN_PATH NAMES mkldnn.hpp)
    if (MKLDNN_PATH)
        message(STATUS "ok, found mkldnn headers (${MKLDNN_PATH}).")
    else()
        message(FATAL_ERROR "failed to find mkldnn headers.")
    endif()
endfunction(find_mkldnn)
