cmake_version()
project(cifar_networks)

include_directories(../../experiments)

add_executable(lenet lenet.cpp common.h common.cpp)
target_link_libraries(lenet "${TORCH_LIBRARIES}" cifar_nn)

add_executable(vgg vgg.cpp common.h common.cpp)
target_link_libraries(vgg "${TORCH_LIBRARIES}" cifar_nn)

add_executable(resnet resnet.cpp common.h common.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}" cifar_nn)

add_executable(check_cuda_libtorch check_cuda_libtorch.cpp)
target_link_libraries(check_cuda_libtorch "${TORCH_LIBRARIES}")
