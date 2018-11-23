#pragma once

#ifdef CUDA

#include "exception.h"
#include <cuda_runtime.h>
#include <cstdint>

#define CUDA_SAFE_CALL(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                                     \
         throw Exception() << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);              \
         }                                                                                                           \
}

namespace Cuda {

    namespace Impl {
        template <class S>
        class StreamImpl {
        public:
            void Synchronize() const {
                CUDA_SAFE_CALL(cudaStreamSynchronize(getStream()));
            }

            operator cudaStream_t() const {
                return getStream();
            }
        private:
            cudaStream_t getStream() const {
                return static_cast<const S*>(this)->stream();
            }
        };
    }

    class StreamRef : public Impl::StreamImpl<StreamRef>{
    private:
        cudaStream_t stream_ = 0;
    public:
        StreamRef(cudaStream_t stream)
        : stream_(stream) {
        }

        cudaStream_t stream() const {
            return stream_;
        }
    };


    class Stream: public Impl::StreamImpl<StreamRef> {
    private:
        cudaStream_t stream_ = 0;
    public:
        Stream() {
            CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        }

        Stream(Stream&& other)
            : stream_(other.stream_) {
            other.stream_ = 0;
        }

        Stream(const Stream& other) = delete;

        ~Stream() noexcept(false) {
            if (stream_) {
                CUDA_SAFE_CALL(cudaStreamDestroy(stream_));
            }
        }

        operator StreamRef() const {
            return StreamRef(stream_);
        }

        cudaStream_t stream() const {
            return stream_;
        }
    };


    template <class T>
    void CopyMemoryAsync(const T* from, T* to, int64_t size, const StreamRef& stream) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(static_cast<void*>(to),
                                       static_cast<void*>(const_cast<T*>(from)),
                                       sizeof(T) * size,
                                       cudaMemcpyDefault,
                                       stream));
    }

    template <class T>
    void CopyMemory(const T* from, T* to, int64_t size) {
        CUDA_SAFE_CALL(cudaMemcpy(static_cast<void*>(to),
                                  static_cast<void*>(const_cast<T*>(from)),
                                  sizeof(T) * size,
                                  cudaMemcpyDefault));
    }

    template <class T>
    struct Data {
    public:
        Data(int64_t size)
            : size_(size)  {
            CUDA_SAFE_CALL(cudaMalloc((void**) &ptr_, size * sizeof(T)));
        }

        ~Data()  {
            if (ptr_) {
//                CUDA_SAFE_CALL(cudaFree((void*) ptr_));
                cudaFree((void*) ptr_);
                ptr_ = nullptr;
                size_ = 0;
            }
        }

        Data(const Data& other) = delete;
        Data(Data&& other)
            : ptr_(other.ptr_)
              , size_(other._size) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        T* data() {
            return ptr_;
        }

        const T* data() const {
            return ptr_;
        };

        int64_t size() const {
            return size_;
        }

        void Read(int64_t offset, int64_t size, T* to) const {
            CopyMemory(ptr_ + offset, to, size);
        }

        void Write(const T* from, int64_t size) {
            CopyMemory(from, ptr_, size);
        }
    private:
        T* ptr_ = nullptr;
        int64_t size_ = 0;
    };

    inline uint64_t GetDeviceCount() {
        int deviceCount = 0;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
        return deviceCount;
    }

    class DeviceProperties {
    public:
        DeviceProperties(const DeviceProperties& other) = default;

        DeviceProperties() = default;

        explicit DeviceProperties(int64_t dev) {
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&props_, dev));
        }

        int64_t GetDeviceMemory() const {
            return props_.totalGlobalMem;
        }

        std::string GetName() const {
            return props_.name;
        }

        int64_t GetMajor() const {
            return props_.major;
        }

        int64_t GetMinor() const {
            return props_.minor;
        }
    private:
        cudaDeviceProp props_;
    };



}

#endif

