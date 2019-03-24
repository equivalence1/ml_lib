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

BasicBlock::BasicBlock(int inChannels, int outChannels, int stride) {
    auto options = buildConvOptions(inChannels, outChannels, 3, stride);
    conv1_ = register_module("conv1_", torch::nn::Conv2d(options));
    bn1_ = register_module("bn1_", torch::nn::BatchNorm(outChannels));

    options = buildConvOptions(outChannels, outChannels, 3, stride);
    conv2_ = register_module("conv2_", torch::nn::Conv2d(options));
    bn2_ = register_module("bn2_", torch::nn::BatchNorm(outChannels));
}

torch::Tensor BasicBlock::forward(torch::Tensor x) {
    x = bn1_->forward(conv1_->forward(x));
    x = bn2_->forward(conv2_->forward(x));
    return x;
}

// ResNetConv

ResNetConv::ResNetConv(torch::IntList numBlocks,
                       std::function<experiments::ModelPtr(int, int, int)> blocksBuilder) {
    assert(numBlocks.size() == 4);

    static const int blockOutChannelSizes[] = {64, 128, 256, 512};
    static const int initStrides[] = {1, 2, 2, 2};

    int inChannels = 64;

    auto options = buildConvOptions(3, 64, 3);
    conv1_ = register_module("conv1_", torch::nn::Conv2d(options));
    bn1_ = register_module("bn1_", torch::nn::BatchNorm(64));

    for (int i = 0; i < (int)numBlocks.size(); ++i) {
        int stride = initStrides[i];
        for (int j = 0; j < numBlocks.at(std::size_t(i)); ++j) {
            int outChannels = blockOutChannelSizes[i];
            auto block = blocksBuilder(inChannels, outChannels, stride);
            std::string blockId = "block_" + std::to_string(i) + "_" + std::to_string(j);
            auto b = register_module(blockId, block);
            blocks_.push_back(b);

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
    fc1_ = register_module("fc1_", torch::nn::Linear(512 * expansion, 10));
}

torch::Tensor ResNetClassifier::forward(torch::Tensor x) {
    x = fc1_->forward(x.view({x.size(0), -1}));
    return x;
}

// ResNet

ResNet::ResNet(ResNetConfiguration cfg) {
    if (cfg == ResNetConfiguration::ResNet16) {
        std::function<experiments::ModelPtr(int, int, int)> blocksBuilder = [](int inChannels, int outChannels, int stride){
            return std::make_shared<BasicBlock>(inChannels, outChannels, stride);
        };
        conv_ = std::make_shared<ResNetConv>(
                torch::IntList({3, 4, 6, 3}),
                blocksBuilder);
        classifier_ = std::make_shared<ResNetClassifier>(1);
        conv_ = register_module("conv_", conv_);
        classifier_ = register_module("classifier", classifier_);
    } else {
        throw "Unsupported configuration";
    }
}

torch::Tensor ResNet::forward(torch::Tensor x) {
    x = conv_->forward(x);
    x = classifier_->forward(x);
    return x;
}
