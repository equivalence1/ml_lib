#pragma once

#include <torch/torch.h>

class TensorPairDataset: public torch::data::datasets::Dataset<TensorPairDataset> {
public:
    TensorPairDataset(torch::Tensor x, torch::Tensor y): x_(std::move(x)), y_(std::move(y)) {
    }

    torch::data::Example<> get(size_t index) override {
        auto res = torch::data::Example<>(x_[index], y_[index]);
        return res;
    }

    c10::optional<size_t> size() const override {
        return x_.size(0);
    }

    torch::Tensor data() {
        return x_;
    }

    torch::Tensor targets() {
        return y_;
    }

    TensorPairDataset to(torch::DeviceType device) {
        return TensorPairDataset(x_.to(device), y_.to(device));
    }

protected:
    TensorPairDataset() = default;

    torch::Tensor x_;
    torch::Tensor y_;
};
