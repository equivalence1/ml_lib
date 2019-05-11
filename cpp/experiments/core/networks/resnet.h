#pragma once

#include "experiments/core/model.h"

#include <torch/torch.h>

#include <vector>

namespace experiments {

enum class ResNetConfiguration {
    ResNet18,
    ResNet34,
};



// BasicBlock

class BasicBlock : public experiments::Model {
public:
    BasicBlock(int inChannels, int outChannels, int stride = 1);

    torch::Tensor forward(torch::Tensor x) override;

    ~BasicBlock() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};
    torch::nn::BatchNorm bn2_{nullptr};

    torch::nn::Sequential shortcut_{nullptr};
};

// ResNetConv

class ResNetConv : public Model {
public:
    ResNetConv(std::vector<int> numBlocks,
               const std::function<ModelPtr(int, int, int)> &blocksBuilder);

    torch::Tensor forward(torch::Tensor x) override;

    ~ResNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};

    std::vector<ModelPtr> blocks_;
};

// ResNetClassifier

class ResNetClassifier : public Model {
public:
    explicit ResNetClassifier(int expansion);

    torch::Tensor forward(torch::Tensor x) override;

    ~ResNetClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};

// ResNet

class ResNet : public ConvModel {
public:
    explicit ResNet(ResNetConfiguration cfg,
                    ClassifierPtr classifier = makeClassifier<ResNetClassifier>(1));

    torch::Tensor forward(torch::Tensor x) override;

    ModelPtr conv() override;

    ClassifierPtr classifier() override;

    ~ResNet() override = default;

private:
    void init(std::vector<int> nBlocks, ClassifierPtr classifier);

private:
    std::shared_ptr<ResNetConv> conv_{nullptr};
    ClassifierPtr classifier_{nullptr};
};

}
