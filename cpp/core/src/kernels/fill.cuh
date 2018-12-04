#pragma once
#if defined(CUDA)
#include <util/cuda_wrappers.h>

namespace Cuda {
    namespace Kernel {

        template <typename T>
        void FillBuffer(T* buffer, T value, int64_t size, StreamRef stream);

    }
}

#endif
