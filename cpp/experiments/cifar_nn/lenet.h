#pragma once

#include "layer_norm.h"
#include "model.h"

#include <torch/torch.h>

#include <memory>

// LeNetConv

class LeNetConv : public experiments::Model {
public:
    LeNetConv();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
};

// LeNetClassifier

class LeNetClassifier : public experiments::Model {
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

class LeNet : public experiments::ConvModel {
public:
    LeNet(experiments::ClassifierPtr classifier = makeClassifier<LeNetClassifier>());

    experiments::ModelPtr conv() override;

    experiments::ClassifierPtr classifier() override;

    ~LeNet() override = default;

private:
    std::shared_ptr<LeNetConv> conv_{nullptr};
    experiments::ClassifierPtr classifier_{nullptr};
};

