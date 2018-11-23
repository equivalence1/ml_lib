#pragma once

#include "vec_ops.h"
#include <vector>

class VecRef : public VecOps<float, VecRef> {
public:
    using ValueType = float;

    explicit VecRef(float* data, int64_t size)
    : data_(data)
    , size_(size) {
    }

    VecRef(VecRef&& other) = default;
    VecRef(const VecRef& other) = default;


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



