cmake_version()
project(cifar_networks)

include_directories(../../experiments)

# train

add_executable(lenet lenet.cpp common.h common.cpp)
target_link_libraries(lenet "${TORCH_LIBRARIES}" experiments)

add_executable(lenet_em_exact_classifier lenet_em_exact_classifier.cpp common.h common.cpp common_em.h)
target_link_libraries(lenet_em_exact_classifier "${TORCH_LIBRARIES}" experiments)

add_executable(train_default train_default.cpp common.h common.cpp)
target_link_libraries(train_default "${TORCH_LIBRARIES}" experiments)

add_executable(train_default_em train_default_em.cpp common.h common.cpp common_em.h)
target_link_libraries(train_default_em "${TORCH_LIBRARIES}" experiments)

add_executable(train_with_catboost train_with_catboost.cpp common.h common.cpp catboost_nn.h catboost_nn.cpp)
target_link_libraries(train_with_catboost "${TORCH_LIBRARIES}" experiments polynom ${CATBOOST})

# misc

add_executable(svhn_read svhn_read.cpp common.h common.cpp)
target_link_libraries(svhn_read "${TORCH_LIBRARIES}" experiments)

add_executable(lenet_save_restore lenet_save_restore.cpp)
target_link_libraries(lenet_save_restore "${TORCH_LIBRARIES}" experiments)

add_executable(vgg_save_restore vgg_save_restore.cpp)
target_link_libraries(vgg_save_restore "${TORCH_LIBRARIES}" experiments)

add_executable(mobile_net_v2_save_restore mobile_net_v2_save_restore.cpp)
target_link_libraries(mobile_net_v2_save_restore "${TORCH_LIBRARIES}" experiments)

add_executable(resnet_save_restore resnet_save_restore.cpp)
target_link_libraries(resnet_save_restore "${TORCH_LIBRARIES}" experiments)

add_executable(check_cuda_libtorch check_cuda_libtorch.cpp)
target_link_libraries(check_cuda_libtorch "${TORCH_LIBRARIES}")
