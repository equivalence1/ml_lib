#pragma once

#include "vec_ops.h"
#include <core/vec.h>
#include <vector>
namespace Impl {
    class NonOwningVec : public VecOps<float, NonOwningVec>, public AnyVec {
    public:
        using ValueType = float;

        explicit NonOwningVec(float* data, int64_t size)
            : data_(data)
              , size_(size) {
        }

        NonOwningVec(NonOwningVec&& other) = default;
        NonOwningVec(const NonOwningVec& other) = default;

        int64_t size() const {
            return size_;
        }

        float* data() const {
            return data_;
        }
        float* data() {
            return data_;
        }

    private:
        float* data_;
        int64_t size_;
    };

}
