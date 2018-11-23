cmake_version()
project(cuda_devices_info)

add_executable(device_info main.cpp)

maybe_enable_cuda(device_info)

target_link_libraries(device_info util)
