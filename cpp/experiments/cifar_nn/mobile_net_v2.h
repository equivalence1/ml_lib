#pragma once

#include "model.h"

#include <torch/torch.h>

#include <vector>

// reference implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py

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

// MobileNetV2Classifier

class MobileNetV2Classifier : public experiments::Model {
public:
    explicit MobileNetV2Classifier();

    torch::Tensor forward(torch::Tensor x) override;

    ~MobileNetV2Classifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};

// MobileNetV2

class MobileNetV2 : public experiments::ConvModel {
public:
    explicit MobileNetV2(std::shared_ptr<experiments::Model> classifier = std::make_shared<MobileNetV2Classifier>());

    torch::Tensor forward(torch::Tensor x) override;

    experiments::ModelPtr conv() override;

    experiments::ModelPtr classifier() override;

    ~MobileNetV2() override = default;

private:
    void init(std::vector<int> nBlocks, std::shared_ptr<experiments::Model> classifier);

private:
    std::shared_ptr<MobileNetV2Conv> conv_{nullptr};
    experiments::ModelPtr classifier_{nullptr};
};
