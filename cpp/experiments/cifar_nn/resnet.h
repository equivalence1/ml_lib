#pragma once

#include "model.h"

#include <torch/torch.h>

#include <vector>

enum class ResNetConfiguration {
    ResNet16,
};

// TODO(equivalence1) ResNet and Vgg don't support save/restore features, since
// they store layers using vectors, functions, etc. I need to figure out how to
// properly register submodules to be able to restore models. (Sequential might help).
// Also might be some trouble with private/public fields.


// BasicBlock

class BasicBlock : public Model {
public:
    BasicBlock(int id, int inChannels, int outChannels, int stride = 1);

    torch::Tensor forward(torch::Tensor x) override;

    ~BasicBlock() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};
    torch::nn::BatchNorm bn2_{nullptr};
};

// ResNetConv

class ResNetConv : public Model {
public:
    ResNetConv(torch::IntList numBlocks,
               std::function<ModelPtr(int, int, int, int)> blocksBuilder);

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

class ResNet : public Model {
public:
    explicit ResNet(ResNetConfiguration cfg);

    torch::Tensor forward(torch::Tensor x) override;

    ~ResNet() override = default;

private:
    std::shared_ptr<ResNetConv> conv_{nullptr};
    ModelPtr classifier_{nullptr};
};
