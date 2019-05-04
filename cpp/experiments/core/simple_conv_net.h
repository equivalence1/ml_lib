#pragma once

#include "model.h"

#include <torch/torch.h>

class SimpleConvNet : public experiments::Model {
public:
    SimpleConvNet() {
        conv1_ = register_module("conv1_", torch::nn::Conv2d(3, 6, 5));
        conv2_ = register_module("conv2_", torch::nn::Conv2d(6, 16, 5));
        conv3_ = register_module("conv3_", torch::nn::Conv2d(16, 16, 5));
    }

    torch::Tensor forward(torch::Tensor x) override {
        x = conv1_->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2);
        x = conv2_->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2);
        x = conv3_->forward(x);
        return x.view({-1, 16});
    }
//
//    void init_weights() {
//        torch::nn::init::xavier_uniform_(conv1_.get()->weight);
//        torch::nn::init::xavier_uniform_(conv2_.get()->weight);
//        torch::nn::init::xavier_uniform_(conv3_.get()->weight);
//    }

    ~SimpleConvNet() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::Conv2d conv3_{nullptr};
};
