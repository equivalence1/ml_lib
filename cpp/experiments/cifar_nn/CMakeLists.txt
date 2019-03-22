cmake_version()
project(cifar_nn)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(cifar_nn
        em_like_train.h
        model.h
        model.cpp
        tensor_pair_dataset.h
        tensor_pair_dataset.cpp
        loss.cpp
        loss.h
        optimizer.cpp
        optimizer.h
        initializer.cpp
        initializer.h
        linear_train.cpp
        linear_train.h
        linear_model.cpp
        linear_model.h
        linear_function.cpp
        linear_function.h
        cross_entropy_loss.cpp
        cross_entropy_loss.h
        lenet.cpp
        lenet.h
        vgg.cpp
        vgg.h
        )
target_link_libraries(cifar_nn "${TORCH_LIBRARIES}" Eigen3::Eigen)
enable_cxx14(cifar_nn)

pybind11_add_module(cifar_nn_py py_module.cpp)
target_link_libraries(cifar_nn_py PRIVATE cifar_nn)
enable_cxx14(cifar_nn_py)

install(TARGETS cifar_nn EXPORT CifarNnConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
