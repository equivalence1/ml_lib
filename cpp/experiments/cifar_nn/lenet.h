#pragma once

#include "model.h"

#include <torch/torch.h>

// LeNetConv

class LeNetConv : public Model {
public:
    LeNetConv();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
};

// LeNet

class LeNet : public LeNetConv {
public:
    LeNet();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNet() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
};
