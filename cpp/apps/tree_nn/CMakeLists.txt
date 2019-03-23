cmake_version()
project(tree_nn)

include_directories(../../experiments)

add_executable(tree_nn tree_nn.cpp)
target_link_libraries(tree_nn "${TORCH_LIBRARIES}" cifar_nn)