#include "lenet.h"

#include <torch/torch.h>

#include <memory>

namespace experiments::lenet {

// LeNetConv

LeNetConv::LeNetConv() {
    conv1_ = register_module("conv1_", torch::nn::Conv2d(3, 6, 5));
    conv2_ = register_module("conv2_", torch::nn::Conv2d(6, 16, 5));
}

torch::Tensor LeNetConv::forward(torch::Tensor x) {
    x = conv1_->forward(x);
    x = torch::max_pool2d(torch::relu(x), 2, 2);
    x = conv2_->forward(x);
    x = torch::max_pool2d(torch::relu(x), 2, 2);
    return x;
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    return std::make_shared<LeNetConv>();
}

}
