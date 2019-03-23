#pragma once

#include "loss.h"

#include <torch/torch.h>

class BCESigmoidLoss : public Loss {
public:
    BCESigmoidLoss() = default;

    torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
        return torch::binary_cross_entropy(torch::sigmoid(outputs), torch::_cast_Float(targets));
    }
};
