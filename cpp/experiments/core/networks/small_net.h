#pragma once

#include <experiments/core/model.h>

#include <torch/torch.h>

namespace experiments::small_net {

class SmallNetConv : public Model {
public:
    SmallNetConv();

    torch::Tensor forward(torch::Tensor x) override;

    ~SmallNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::Conv2d conv3_{nullptr};
    torch::nn::Conv2d conv4_{nullptr};
//    torch::nn::BatchNorm bn_{nullptr};
//    torch::nn::Linear l_{nullptr};
};

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);

}
