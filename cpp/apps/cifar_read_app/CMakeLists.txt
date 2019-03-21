cmake_version()
project(cifar_read_app)

add_executable(cifar_read_app main.cpp)

target_link_libraries(cifar_read_app cifar_nn)
