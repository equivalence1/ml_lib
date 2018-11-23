#pragma once

#include "vec_ops.h"
#include <vector>
#include <memory>

class ArrayVec  : public VecOps<float, ArrayVec> {
public:
    using DataContainer = std::vector<float>;

    explicit ArrayVec(int64_t size)
    : VecOps()
    , data_(std::make_shared<DataContainer>())
    , offset_(0) {
        data_->resize(size);
    }

    ArrayVec(ArrayVec&& other) = default;
    ArrayVec(const ArrayVec& other) = default;

    ArrayVec(std::shared_ptr<DataContainer> ptr, int64_t offset)
    : data_(std::move(ptr))
    , offset_(offset) {

    }

    int64_t size() const {
        return static_cast<int64_t>(data_->size() - offset_);
    }

    float* data() const {
        return data_->data();
    }
    float* data() {
        return data_->data();
    }
private:
    std::shared_ptr<DataContainer> data_;
    int64_t offset_;
};
