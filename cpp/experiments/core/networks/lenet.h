#pragma once

#include "experiments/core/layer_norm.h"
#include "experiments/core/model.h"

#include <torch/torch.h>

#include <memory>

namespace experiments {

// LeNetConfig

struct LeNetConfig {

};

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

// LeNetClassifier

class LeNetClassifier : public Model {
public:
    LeNetClassifier();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNetClassifier() override = default;

private:
    LayerNormPtr layerNorm_{nullptr};
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
};

// LeNet

class LeNet : public ConvModel {
public:
    explicit LeNet(ClassifierPtr classifier = makeClassifier<LeNetClassifier>());

    ModelPtr conv() override;

    ClassifierPtr classifier() override;

    ~LeNet() override = default;

private:
    std::shared_ptr<LeNetConv> conv_{nullptr};
    ClassifierPtr classifier_{nullptr};
};

}
