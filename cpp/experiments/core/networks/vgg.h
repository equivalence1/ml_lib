#pragma once

#include <experiments/core/model.h>
#include <util/json.h>

#include <torch/torch.h>

#include <functional>
#include <vector>

namespace experiments::vgg {

enum class VggConfiguration {
    Vgg16,
};

// VggConv

// See https://arxiv.org/pdf/1409.1556.pdf
//
// We'll be comparing our accuracy with https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
// so in each configuration we use the same conv/maxpool/batchnorm layers as they do.
class VggConv : public experiments::Model {
public:
    VggConv() = default;

    torch::Tensor forward(torch::Tensor x) override = 0;

    ~VggConv() override = default;

protected:
    std::vector<std::function<torch::Tensor(torch::Tensor)>> layers_;
};

// Vgg16Conv

class Vgg16Conv : public VggConv {
public:
    Vgg16Conv();

    torch::Tensor forward(torch::Tensor x) override;

    ~Vgg16Conv() override = default;
};

// Utils

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);

}
