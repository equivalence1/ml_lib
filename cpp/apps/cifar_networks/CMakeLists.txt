cmake_version()
project(cifar_networks)

add_executable(lenet lenet.cpp)

target_link_libraries(lenet "${TORCH_LIBRARIES}" cifar_nn data)
