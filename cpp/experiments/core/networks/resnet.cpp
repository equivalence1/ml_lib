#include "resnet.h"

#include <string>
#include <functional>
#include <cassert>
#include <memory>
#include <stdexcept>

namespace experiments::resnet {

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

    options = buildConvOptions(outChannels, outChannels, 3, 1);
    conv2_ = register_module("conv2_", torch::nn::Conv2d(options));
    bn2_ = register_module("bn2_", torch::nn::BatchNorm(outChannels));

    if (stride != 1 || inChannels != outChannels) {
        options = buildConvOptions(inChannels, outChannels, 1, stride);
        shortcut_ = register_module("shortcut_", torch::nn::Sequential(
                torch::nn::Conv2d(options),
                torch::nn::BatchNorm(outChannels)
        ));
    }
}

torch::Tensor BasicBlock::forward(torch::Tensor x) {
    auto out = torch::relu(bn1_->forward(conv1_->forward(x)));
    out = bn2_->forward(conv2_->forward(out));
    if (!shortcut_.is_empty()) {
        out = torch::relu(out + shortcut_->forward(x));
    } else {
        out = torch::relu(out + x);
    }
    return out;
}

// ResNetConv

ResNetConv::ResNetConv(std::vector<int> numBlocks,
                       const std::function<ModelPtr(int, int, int)> &blocksBuilder) {
    assert(numBlocks.size() == 4);

    static const int blockOutChannelSizes[] = {64, 128, 256, 512};
    static const int initStrides[] = {1, 2, 2, 2};

    int inChannels = 64;

    auto options = buildConvOptions(3, 64, 3);
    conv1_ = register_module("conv1_", torch::nn::Conv2d(options));
    bn1_ = register_module("bn1_", torch::nn::BatchNorm(64));

    for (int i = 0; i < (int) numBlocks.size(); ++i) {
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
    for (const auto &block : blocks_) {
        x = block->forward(x);
    }
    x = torch::avg_pool2d(x, 4);
    return x;
}

// Utils

static ModelPtr createConvLayersHelper(std::vector<int> nBlocks) {
    std::function<ModelPtr(int, int, int)> blocksBuilder = [](int inChannels, int outChannels,
                                                                           int stride) {
        return std::make_shared<BasicBlock>(inChannels, outChannels, stride);
    };
    auto conv = std::make_shared<ResNetConv>(nBlocks,
            blocksBuilder);
    return conv;
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    std::string archVersion = params[ParamKeys::ModelArchVersionKey];

    if (archVersion == "18") {
        return createConvLayersHelper(std::vector<int>({2, 2, 2, 2}));
    } else if (archVersion == "34") {
        return createConvLayersHelper(std::vector<int>({3, 4, 6, 3}));
    }

    std::string errMsg("Unsupported ResNet Architecture version ResNet-");
    throw std::runtime_error(errMsg + " " + archVersion);
}

}
