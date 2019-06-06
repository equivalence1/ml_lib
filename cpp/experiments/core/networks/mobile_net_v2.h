#pragma once

#include <experiments/core/model.h>
#include <util/json.h>

#include <torch/torch.h>

#include <vector>

// reference implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py

namespace experiments::mobile_net_v2 {

// BasicBlock

class BasicBlock : public experiments::Model {
public:
    BasicBlock(int inChannels, int outChannels, int expansion, int stride = 1);

    torch::Tensor forward(torch::Tensor x) override;

    ~BasicBlock() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::Conv2d conv3_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};
    torch::nn::BatchNorm bn2_{nullptr};
    torch::nn::BatchNorm bn3_{nullptr};

    torch::nn::Sequential shortcut_{nullptr};
};

// MobileNetV2Conv

class MobileNetV2Conv : public experiments::Model {
public:
    MobileNetV2Conv();

    torch::Tensor forward(torch::Tensor x) override;

    ~MobileNetV2Conv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};

    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::BatchNorm bn2_{nullptr};

    std::vector<experiments::ModelPtr> blocks_;
};

// Utils

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);

}
