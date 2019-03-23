#include "resnet.h"

#include <string>
#include <functional>
#include <cassert>
#include <memory>

static torch::nn::ConvOptions<2> buildConvOptions(int inChannels,
        int outChannels, int kernelSize, int stride = 1, bool bias = false) {
    auto convOptions = torch::nn::ConvOptions<2>(inChannels, outChannels, kernelSize);
    convOptions.padding(kernelSize / 2);
    convOptions.stride(stride);
    convOptions.with_bias(bias);
    return convOptions;
}

// BasicBlock

#define MODULE_NAME(baseName, id, secondId) std::string(baseName) + "_" + std::to_string(id) + "_" + std::to_string(secondId)

BasicBlock::BasicBlock(int id, int inChannels, int outChannels, int stride) {
    auto options = buildConvOptions(inChannels, outChannels, 3, stride);
    conv1_ = register_module(MODULE_NAME("conv", id, 1), torch::nn::Conv2d(options));
    string bnName = "batch_norm_" + std::to_string(id) + "_1";
    bn1_ = register_module(bnName, torch::nn::BatchNorm(outChannels));

    options = buildConvOptions(outChannels, outChannels, 3, stride);
    conv2_ = register_module(MODULE_NAME("conv", id, 2), torch::nn::Conv2d(options));
    bnName = "batch_norm_" + std::to_string(id) + "_2";
    bn2_ = register_module(bnName, torch::nn::BatchNorm(outChannels));
}

#undef MODULE_NAME

torch::Tensor BasicBlock::forward(torch::Tensor x) {
    x = bn1_->forward(conv1_->forward(x));
    x = bn2_->forward(conv2_->forward(x));
    return x;
}

// ResNetConv

ResNetConv::ResNetConv(torch::IntList numBlocks,
                       std::function<ModelPtr(int, int, int, int)> blocksBuilder) {
    assert(numBlocks.size() == 4);

    static const int blockOutChannelSizes[] = {64, 128, 256, 512};
    static const int initStrides[] = {1, 2, 2, 2};

    int id = 0;
    int inChannels = 64;

    auto options = buildConvOptions(3, 64, 3);
    conv1_ = register_module("conv1", torch::nn::Conv2d(options));
    bn1_ = register_module("bn1", torch::nn::BatchNorm(64));

    for (int i = 0; i < (int)numBlocks.size(); ++i) {
        int stride = initStrides[i];
        for (int j = 0; j < numBlocks.at(std::size_t(i)); ++j) {
            int outChannels = blockOutChannelSizes[i];
            auto block = blocksBuilder(id, inChannels, outChannels, stride);
            auto b = register_module("block_" + std::to_string(id), block);
            blocks_.push_back(b);

            id++;
            inChannels = outChannels;
            stride = 1;
        }
    }
}

torch::Tensor ResNetConv::forward(torch::Tensor x) {
    x = torch::relu(bn1_->forward(conv1_->forward(x)));
    for (const auto& block : blocks_) {
        x = block->forward(x);
    }
    return x;
}

// ResNetClassifier

ResNetClassifier::ResNetClassifier(int expansion) {
    fc1_ = register_module("fc1", torch::nn::Linear(512 * expansion, 10));
}

torch::Tensor ResNetClassifier::forward(torch::Tensor x) {
    x = fc1_->forward(x.view({x.size(0), -1}));
    return x;
}

// ResNet

ResNet::ResNet(ResNetConfiguration cfg) {
    if (cfg == ResNetConfiguration::ResNet16) {
        std::function<ModelPtr(int, int, int, int)> blocksBuilder = [](int id, int inChannels, int outChannels, int stride){
            return std::make_shared<BasicBlock>(id, inChannels, outChannels, stride);
        };
        conv_ = std::make_shared<ResNetConv>(
                torch::IntList({3, 4, 6, 3}),
                blocksBuilder);
        classifier_ = std::make_shared<ResNetClassifier>(1);
    } else {
        throw "Unsupported configuration";
    }
}

torch::Tensor ResNet::forward(torch::Tensor x) {
    x = conv_->forward(x);
    x = classifier_->forward(x);
    return x;
}
