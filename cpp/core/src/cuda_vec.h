#pragma once

#include "vec_ops.h"
#include <vector>
#include <memory>
#include <cassert>
#include <core/vec.h>
#include <util/cuda_wrappers.h>

#if defined(CUDA)

namespace Impl {

    class CudaVec : public AnyVec {
    public:
        using Storage = Cuda::Data<float>;

        explicit CudaVec(int64_t size)
            : data_(std::make_shared<Storage>(size))
              , offset_(0) {
        }

        CudaVec(CudaVec&& other) = default;
        CudaVec(const CudaVec& other) = default;

        CudaVec(std::shared_ptr<Storage> ptr, int64_t offset)
            : data_(std::move(ptr))
              , offset_(offset) {

        }

        int64_t dim() const {
            return data_->size() - offset_;
        }

        float* data() const {
            return data_->data();
        }

        float* data() {
            return data_->data();
        }

    private:
        std::shared_ptr<Storage> data_;
        int64_t offset_;
    };

}

#endif


