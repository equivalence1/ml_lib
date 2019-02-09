cmake_version()
project(cifar_nn)

add_library(cifar_nn
        em_like_train.h
        simple_conv_net.h
        simple_conv_net.cpp
        model.h
        model.cpp
        simple_net.h
        simple_net.cpp
        tensor_pair_dataset.h
        tensor_pair_dataset.cpp
        loss.cpp
        loss.h
        optimizer.cpp
        optimizer.h
        initializer.cpp
        initializer.h)
target_link_libraries(cifar_nn "${TORCH_LIBRARIES}")
enable_cxx14(cifar_nn)

pybind11_add_module(cifar_nn_py py_module.cpp)
target_link_libraries(cifar_nn_py PRIVATE cifar_nn)
enable_cxx14(cifar_nn_py)
