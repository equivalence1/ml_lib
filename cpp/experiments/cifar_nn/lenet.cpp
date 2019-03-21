#include "lenet.h"

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
    return x.view({-1, 16 * 5 * 5});
}

// LeNet

LeNet::LeNet() : LeNetConv() {
    fc1_ = register_module("fc1_", torch::nn::Linear(16 * 5 * 5, 120));
    fc2_ = register_module("fc2_", torch::nn::Linear(120, 84));
    fc3_ = register_module("fc3_", torch::nn::Linear(84, 10));
}

torch::Tensor LeNet::forward(torch::Tensor x) {
    x = LeNetConv::forward(x);
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = fc3_->forward(x);
    return x;
}
