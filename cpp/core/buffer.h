#include <utility>

#pragma once

#include "torch_helpers.h"
#include <torch/torch.h>

template <class T>
class Buffer  {
public:

//    Buffer(uint64_t size, const ComputeDevice& device)
//    : data_(TorchHelpers::tensorOptionsOnDevice(device, torch::ScalarType.)) {
//
//    }

    ArrayRef<T> arrayRef() {
        return ArrayRef<T>(reinterpret_cast<T*>(data_.data<uint8_t>()), size());
    }

    void fill(const T& val) {
        auto dst = arrayRef();
        for (int64_t i = 0; i < dst.size(); ++i) {
            dst[i] = val;
        }
    }

    ConstArrayRef<T> arrayRef() const {
        return ConstArrayRef<T>(reinterpret_cast<const T*>(data_.data<uint8_t>()), size());
    }


    static Buffer create(int64_t size) {
        size *= sizeof(T);
        return Buffer(torch::empty({size}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice(), torch::ScalarType::Byte)));
    }

    ComputeDevice device() const {
        return TorchHelpers::getDevice(data_);
    }

    int64_t size() const {
        return  TorchHelpers::totalSize(data_) / sizeof(T);
    }
private:

    Buffer(torch::Tensor data)
    : data_(std::move(data)) {

}
private:
    torch::Tensor data_;
};
