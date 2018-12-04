#include <util/cuda_wrappers.h>
#include <iostream>

#if defined(CUDA)

int main(int /*argc*/, char* /*argv*/[]) {
    int64_t devCount = Cuda::GetDeviceCount();
    for (int64_t dev = 0; dev < devCount; ++dev) {
        auto props = Cuda::DeviceProperties(dev);
        std::cout << dev << "." << props.GetName() << " (compute capability " << props.GetMajor() << "." << props.GetMinor() << ")" << std::endl;
    }
    return 0;
}

#else
int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "CUDA support was not compiled" << std::endl;
    return 0;
}
#endif
