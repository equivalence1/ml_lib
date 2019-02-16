#pragma once

#include "model.h"

#include <torch/torch.h>

class SimpleConvNet : public Model {
public:
    SimpleConvNet() {
        conv1_ = register_module("conv1_", torch::nn::Conv2d(3, 6, 5));
        conv2_ = register_module("conv2_", torch::nn::Conv2d(6, 16, 5));
    }

    torch::Tensor forward(torch::Tensor x) override {
        x = conv1_->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2);
        x = conv2_->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2);
        return x.view({-1, 16 * 5 * 5});
    }

    ~SimpleConvNet() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
};
