#include "mobile_net_v2.h"

// BasicBlock

BasicBlock::BasicBlock(int inChannels, int outChannels, int expansion, int stride) {
    auto channels = expansion * inChannels;

    // expansion layer

    {
        auto convOptions = torch::nn::ConvOptions<2>(inChannels, channels, /*kernel_size=*/1);
        convOptions.padding(0);
        convOptions.stride(1);
        convOptions.with_bias(false);
        conv1_ = register_module("conv1_", torch::nn::Conv2d(convOptions));
        bn1_ = register_module("bn1_", torch::nn::BatchNorm(channels));
    }

    // depthwise convolution layer

    {
        auto convOptions = torch::nn::ConvOptions<2>(channels, channels, /*kernel_size=*/3);
        convOptions.padding(1);
        convOptions.stride(stride);
        convOptions.with_bias(false);
        convOptions.groups(channels);
        conv2_ = register_module("conv2_", torch::nn::Conv2d(convOptions));
        bn2_ = register_module("bn2_", torch::nn::BatchNorm(channels));
    }

    // pointwise convolution layer

    {
        auto convOptions = torch::nn::ConvOptions<2>(channels, outChannels, /*kernel_size=*/1);
        convOptions.padding(0);
        convOptions.stride(1);
        convOptions.with_bias(false);
        conv3_ = register_module("conv3_", torch::nn::Conv2d(convOptions));
        bn3_ = register_module("bn3_", torch::nn::BatchNorm(outChannels));
    }

    // shortcut layer

    if (stride == 1 && inChannels != outChannels) {
        auto convOptions = torch::nn::ConvOptions<2>(inChannels, outChannels, /*kernel_size=*/1);
        convOptions.padding(0);
        convOptions.stride(1);
        convOptions.with_bias(false);
        shortcut_ = register_module("shortcut_", torch::nn::Sequential(
                torch::nn::Conv2d(convOptions),
                torch::nn::BatchNorm(outChannels)
        ));
    }
}

torch::Tensor BasicBlock::forward(torch::Tensor x) {
    auto out = torch::relu(bn1_->forward(conv1_->forward(x)));
    out = torch::relu(bn2_->forward(conv2_->forward(out)));
    out = bn3_->forward(conv3_->forward(out));
    if (!shortcut_.is_empty()) {
        out = out + shortcut_->forward(x);
    }
    return out;
}

// MobileNetV2Conv

MobileNetV2Conv::MobileNetV2Conv() {
    {
        auto convOptions = torch::nn::ConvOptions<2>(3, 32, /*kernel_size=*/3);
        convOptions.padding(1);
        convOptions.stride(1);
        convOptions.with_bias(false);
        conv1_ = register_module("conv1_", torch::nn::Conv2d(convOptions));
        bn1_ = register_module("bn1_", torch::nn::BatchNorm(32));
    }

    static const int kIterations = 7;
    static const int expansion[kIterations]   = { 1,  6,  6,  6,  6,   6,   6};
    static const int outChannels[kIterations] = {16, 24, 32, 64, 96, 160, 320};
    static const int nBlocks[kIterations]     = { 1,  2,  3,  4,  3,   3,   1};
    static const int stride[kIterations]      = { 1,  1,  2,  2,  1,   2,   1};

    int inChannels = 32;
    for (int i = 0; i < kIterations; ++i) {
        int cur_stride = stride[i];
        for (int j = 0; j < nBlocks[i]; ++j) {
            auto block = std::make_shared<BasicBlock>(inChannels, outChannels[i], expansion[i], cur_stride);
            block = register_module("block_" + std::to_string(i) + "_" + std::to_string(j), block);
            blocks_.push_back(block);
            inChannels = outChannels[i];
            cur_stride = 1;
        }
    }

    {
        auto convOptions = torch::nn::ConvOptions<2>(320, 1280, /*kernel_size=*/1);
        convOptions.padding(0);
        convOptions.stride(1);
        convOptions.with_bias(false);
        conv2_ = register_module("conv2_", torch::nn::Conv2d(convOptions));
        bn2_ = register_module("bn2_", torch::nn::BatchNorm(1280));
    }
}

torch::Tensor MobileNetV2Conv::forward(torch::Tensor x) {
    x = torch::relu(bn1_->forward(conv1_->forward(x)));
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    x = torch::relu(bn2_->forward(conv2_->forward(x)));
    x = torch::avg_pool2d(x, 4);
    return x;
}

// MobileNetV2Classifier

MobileNetV2Classifier::MobileNetV2Classifier() {
    fc1_ = register_module("fc1_", torch::nn::Linear(1280, 10));
}

torch::Tensor MobileNetV2Classifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    x = fc1_->forward(x);
    return x;
}

// MobileNetV2

MobileNetV2::MobileNetV2(std::shared_ptr<experiments::Model> classifier) {
    conv_ = register_module("conv_", std::make_shared<MobileNetV2Conv>());
    classifier_ = register_module("classifier_", std::move(classifier));
}

torch::Tensor MobileNetV2::forward(torch::Tensor x) {
    x = conv_->forward(x);
    x = classifier_->forward(x);
    return x;
}

experiments::ModelPtr MobileNetV2::conv() {
    return conv_;
}

experiments::ModelPtr MobileNetV2::classifier() {
    return classifier_;
}
