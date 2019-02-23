cmake_version()

project(cpp_root)

add_subdirectory(apps/mkl_example)
add_subdirectory(apps/test_app)
add_subdirectory(apps/devices_info)
add_subdirectory(apps/train_ot)
add_subdirectory(apps/cifar_nn_test_app)

add_subdirectory(core)
add_subdirectory(vec_tools)
add_subdirectory(mx_tools)
add_subdirectory(models)
add_subdirectory(methods)
add_subdirectory(targets)
add_subdirectory(metrics)
add_subdirectory(trans)
add_subdirectory(funcs)
add_subdirectory(data)
add_subdirectory(util)
add_subdirectory(experiments)
#add_subdirectory(nn_trees)

