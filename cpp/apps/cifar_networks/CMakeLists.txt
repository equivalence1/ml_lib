cmake_version()
project(cifar_networks)

include_directories(../../experiments)

add_executable(lenet lenet.cpp)
target_link_libraries(lenet "${TORCH_LIBRARIES}" cifar_nn)

add_executable(vgg vgg.cpp)
target_link_libraries(vgg "${TORCH_LIBRARIES}" cifar_nn)

add_executable(resnet resnet.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}" cifar_nn)