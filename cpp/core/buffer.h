#pragma once
#include <utility>

#include "torch_helpers.h"
#include <torch/torch.h>

template <class T>
class Buffer  {
public:

    //TODO(noxoomo): should not clear memory  and this won't work with multiGPU
    explicit Buffer(int64_t size)
    : data_(torch::zeros({static_cast<int64_t>(size * sizeof(T))}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice(),
                                                                     torch::ScalarType::Byte))) {

    }


    Buffer()
    : Buffer(0) {

    }

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

    Buffer copy() const {
        return Buffer(data_.clone());
    }

    int64_t size() const {
        return  TorchHelpers::totalSize(data_) / sizeof(T);
    }

    void Swap(Buffer<T> other) {
        std::swap(data_, other.data_);
    }

private:

    Buffer(torch::Tensor data)
    : data_(std::move(data)) {

}
private:
    torch::Tensor data_;
};
