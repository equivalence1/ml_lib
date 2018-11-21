cmake_version()

project(cpp_root)

add_subdirectory(apps/mkl_example)
add_subdirectory(apps/test_app)

add_subdirectory(core)
add_subdirectory(nn_trees)
add_subdirectory(tests)

