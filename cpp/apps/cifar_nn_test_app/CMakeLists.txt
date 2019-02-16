cmake_version()
project(cifar_nn_test_app)

add_executable(cifar_nn_test_app main.cpp)

target_link_libraries(cifar_nn_test_app "${TORCH_LIBRARIES}" cifar_nn)
