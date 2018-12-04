#include "fill.cuh"
#if defined(CUDA)
namespace Cuda {
    namespace Kernel {

        template <typename T>
        __global__ void FillBufferImpl(T* buffer, T value, int64_t size) {
            int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
            while (i < size) {
                buffer[i] = value;
                i += gridDim.x * blockDim.x;
            }
        }

        template <typename T>
        void FillBuffer(T* buffer, T value, int64_t size, StreamRef stream) {
            if (size > 0) {
                dim3 numBlocks;
                const int blockSize = 128;
                numBlocks.x = (size + blockSize - 1) / blockSize;
                numBlocks.y = 1;
                numBlocks.z = 1;
                FillBufferImpl<T> << < numBlocks, blockSize, 0, stream >> > (buffer, value, size);
            }
        }

        #define FILL_BUFFER(Type)\
        template void FillBuffer<Type>(Type* buffer, Type value, int64_t size, StreamRef stream);

        FILL_BUFFER(float);
        FILL_BUFFER(int);

    }

}
#endif
