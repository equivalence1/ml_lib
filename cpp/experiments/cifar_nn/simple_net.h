#pragma once

#include "simple_conv_net.h"

#include <torch/torch.h>

class SimpleNet : public SimpleConvNet {
public:
    SimpleNet() : SimpleConvNet() {
        fc1_ = register_module("fc1_", torch::nn::Linear(16 * 5 * 5, 120));
        fc2_ = register_module("fc2_", torch::nn::Linear(120, 84));
        fc3_ = register_module("fc3_", torch::nn::Linear(84, 10));
    }

    torch::Tensor forward(torch::Tensor x) override {
        x = SimpleConvNet::forward(x);
        x = torch::relu(fc1_->forward(x));
        x = torch::relu(fc2_->forward(x));
        x = fc3_->forward(x);
        return x;
    }

private:
    torch::nn::Linear fc1_ {nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
};
